# 02. Prefill and Attention Kernel | Prefill 与注意力计算内核

## 学习问题

同一个模型，输入从 2k token（词元）增加到 32k，首个字迟迟不出来。应该换 FlashAttention、拆开长输入，还是缓存公共前缀？三种优化都可能降低等待，却作用在不同位置。先沿着一次 prefill（预填充）看清数据怎样计算和保存，再决定改哪里。

## 核心机制

### 一次 prefill 既计算输出，也建立后续状态

假设输入有 2048 个 token。每层先生成 Q、K、V，用 Q 与 K 的相关性计算 attention（注意力），再汇总 V 并传给下一层。与此同时，K/V 写入 KV Cache（键值缓存）。之后每轮 decode（解码）只追加少量 token，读取已有 K/V，不必重新计算所有历史位置的表示。

分数矩阵是临时中间量，K/V 是后续生成仍需要的状态，两者不能混为一谈。例如 batch=1、32 个 query head、长度 32768，仅 BF16 的完整分数矩阵就需 `32 × 32768² × 2` 字节，即 64 GiB。这个估算不含模型权重、KV Cache 和其他张量，不能当成模型总显存。

![prefill 与注意力访存](../../docs/public/topic_discussion/inference_optimization/prefill_attention_zh.svg)

### FlashAttention：分块计算，避免保存完整分数矩阵

FlashAttention 每次只算一个 Q 块对一个 K/V 块的分数。麻烦在于 softmax 的分母覆盖整行，不能对每块各自归一化后简单平均。online softmax（在线 softmax）为每行维护最大值、指数和及累计输出；新块到来时，先把旧状态缩放到新的最大值基准，再合并新块。

例如分数 `[1,3,2,5]` 分成两块。第一块最大值是 3，第二块带来最大值 5，旧指数项便乘 `exp(3−5)`。不保存旧分数，也能得到整行 softmax 的结果。第 [20 节](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb)会手算并实现这个过程；已有基础可直接看[速查卡](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb#cheatsheet)。

这仍是精确 attention 的数学计算，不是稀疏近似。它避免物化平方规模的分数矩阵，但没有把计算量变成线性；浮点实现还可能因累加顺序不同而产生误差。模型权重和 KV Cache 的容量问题仍然存在。

### chunked prefill 与 prefix cache：何时算、是否需要再算

假设一个请求有 8192 个输入 token，另一个请求正等待生成。chunked prefill（分块预填充）可以把长输入拆成若干调度片段，例如每片 1024 个位置，给其他请求穿插执行的机会。它改变调度粒度，不保证减少长请求自身的总计算量，也不保证它的 TTFT（首 token 延迟）更低。

再假设 10 个请求都有相同的 1000-token 系统提示。prefix cache（前缀缓存）有机会复用这段已计算的 K/V，省去重复 prefill。是否命中要看 token、模型和缓存条件，不能只凭两段文字意思相近判断。两条路线分别在 [38 阶段调度](../../02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.ipynb)和 [24 前缀匹配](../../02_PyTorch_Algorithms/24_Prefix_Cache_Matching_and_Reuse.ipynb)展开。

### SDPA：调用 API 后，究竟执行了什么

`scaled_dot_product_attention`（SDPA）是 PyTorch 的 attention API。运行时根据设备、dtype（数据类型）、shape（形状）、mask 和可用 backend（执行后端）选择实现，可选路径还随版本变化。调用成功只能说明得到了结果，不能单凭 API 名称认定用了 FlashAttention。

例如同一份代码在受支持的 CUDA 低精度形状上可能走 flash，在另一配置上则走 math 等路径。先检查输出误差，再用 profiler（性能分析器）读取实际算子或 kernel（计算内核），最后比较延迟与峰值显存。[执行分派练习](../../02_PyTorch_Algorithms/INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.ipynb)用字典演示候选、能力检查和 fallback（回退）；真实选择仍由后端完成。

## 速查表

<details>
<summary>三类优化分别改变什么</summary>

| 优化 | 改变的对象 | 应查看什么 | 常见误判 |
|---|---|---|---|
| FlashAttention | attention 中间量读写与 kernel 组织 | 输出误差、实际路径、延迟、峰值显存 | 局部 attention 快就认为服务也快 |
| chunked prefill | 一轮调度处理的长输入片段 | TTFT、TPOT、P99、迭代组成 | 拆块必然减少总计算 |
| prefix cache | 重复前缀是否重新计算 | reused tokens、命中长度、TTFT、缓存占用 | 只看 TTFT 变化就认定命中 |

</details>

<details>
<summary>区分计算、搬运与启动开销</summary>

| 信号 | 进一步核对 | 可以尝试 |
|---|---|---|
| 大矩阵算子占时长、计算利用率高 | dtype、形状、Tensor Core 路径 | 改善计算实现与形状 |
| HBM 流量大、带宽利用率高 | 中间量大小、读写次数、算子时间 | 分块与片上复用 |
| 很多短 kernel，之间空隙明显 | CPU 调度、同步点、启动次数 | 融合、编译或 CUDA Graph |
| 长度改变时性能跳变 | padding、重编译、backend 切换 | 长度分桶与形状策略 |

</details>

这些只是排查线索，单个利用率不足以确定根因。对照时应保持模型、输入、硬件和请求压力一致。

## 选择条件

| 当前问题 | 先检查 | 下一步 |
|---|---|---|
| attention 临时显存过大 | 是否保存完整分数矩阵 | [20 速查卡](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb#cheatsheet) |
| 配置了高效路径但没有加速 | 实际 backend、shape、首调成本 | [分派速查卡](../../02_PyTorch_Algorithms/INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.ipynb#cheatsheet) |
| 长输入拖住正在生成的请求 | 调度片段与排队时间 | [07 Serving 调度](./07_serving_scheduling_and_pd.md) |
| 很多请求共享前缀 | 缓存命中是否真实发生 | [04 KV Cache](./04_kv_cache_lifecycle_and_reuse.md) |
| TTFT 高但 prefill 本身不慢 | 队列、传输与执行等待 | [01 请求链路](./01_request_path_and_metrics.md) |

## 相关阅读

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135) —— IO 分析与分块算法。
- [FlashAttention 官方实现](https://github.com/Dao-AILab/flash-attention) —— 硬件与版本支持。
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) —— 输入语义与后端约束。
