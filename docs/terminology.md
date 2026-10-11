# 术语表与中英文写作规范

> 统一审阅入口见 [维护与发布手册](./maintenance.md)。

本文统一教程的术语写法和常用缩写。目标读者是会写 PyTorch、日常用英文术语交流的工程师，所以**行业通用术语直接写英文**，每页首次出现时加一次中文括注，之后只写英文。代码、字段名、官方 API、模型名、库名和协议名保留英文。表中"中文"列用于括注和术语表检索，不是正文里的主称。

## 使用规则

- 行业通用术语（token、prefill、decode、KV Cache、logits、batch、workload、backend、kernel、decoder-only、draft / verify / accept、SDPA、TTFT、DPO、MoE 等）正文直接写英文；每页首次出现写成 `prefill（预填充）`，之后只写 `prefill`。
- 有稳定中文说法且中文更自然的概念（反向传播、梯度累积、混合精度、显存、吞吐、尾延迟、量化、校准）用中文主称，首次出现可括注英文。
- 不把库名、后端名或模型名翻译，例如 PyTorch、vLLM、SGLang、Triton、CUDA、Qwen。
- `workload`、`artifact`、`manifest`、`trace`、`profile`、`evidence_level` 等项目字段在代码和结果 JSON 中保留英文键名；正文第一次出现时解释含义（"workload 指写死的输入、配置和运行条件"）。
- `checkpoint` 是名词；表示"采用激活检查点机制"时统一写"activation checkpointing（激活检查点）"。
- `offload` 正文写 "offload 到 CPU / NVMe"；不单独译作"卸载"。
- 代码标识符用反引号与正文概念区分：写"调用 `backward()` 执行反向传播"，不写"backward 计算 gradient"。
- H1 保留 `# NN. English Title | 中文标题` 双语格式（转换脚本依赖 ASCII ` | `）；内部标题不重复堆叠中英文。
- 表格和图片遵循与正文相同的规范。

## 禁用词

避免用下面这些抽象表达代替技术解释，尤其是“证据出口”“收口”“准入结论”等项目内部说法。这里约束的是用法，不是字面词频：首次中文括注、原始资料标题、代码标识符与 JSON 字段保留原文；“固定随机种子”、接口契约、数学证明中的不变量等有明确含义的表达可以保留。不要为了替换词语而使句子更难读。

| 禁用 | 改用 |
| --- | --- |
| 证据出口 / 验证出口 | 去哪里验证："在 68 里用真实模型跑一遍" |
| 准入 / 准入结论 | 能不能上线 / 是否采用 |
| 收口 | 汇总 / 最后在 X 比较 |
| 口径 | 同样条件下 / 统一指标定义 |
| 账本 | 统计表 / 把 X 加起来 |
| 契约（接口定义之外） | 约定 / 必须满足的条件 |
| 复测 | 再跑一次 / 重新测量 |
| 机制页 / 项目页 | 本节 / 第 68 节（项目） |
| 工作负载 | workload（首次括注"写死的输入和配置"） |
| 词元 / 预填充 / 仅解码器 | token / prefill / decoder-only |
| 不变量（解析中） | 必须始终成立的条件，或直接写条件本身 |
| 固定（作动词） | 写死 / 保持不变 / 先确定 |
| 证据（泛用） | 结果 / 测量值 / 测试 |

## 基础张量、训练与数值

| 中文规范名 | English / 缩写 | 推荐写法 | 不建议混用 |
| --- | --- | --- | --- |
| 张量 | tensor | 张量 shape、张量布局 | 数组与张量不加区分 |
| 自动求导 | automatic differentiation / Autograd | PyTorch 自动求导（Autograd）记录计算依赖 | 将 Autograd、backward 和反向传播视为同一概念 |
| 计算图 | computation graph | 计算图记录张量与算子的依赖 | 执行顺序、代码顺序与计算图混用 |
| 叶子张量 | leaf tensor | 叶子张量默认接收 `.grad` | 正文直接使用 leaf、leaf node |
| 中间张量 | intermediate tensor | 中间张量调用 `retain_grad()` 后可观察梯度 | 中间结果、中间节点和激活不加区分 |
| 上游梯度 | upstream gradient | `backward(gradient)` 接收上游梯度 | 正文直接使用 upstream、grad output |
| 向量—雅可比积 | vector-Jacobian product / VJP | 非标量输出通过 VJP 回传梯度 | 只写 vector-Jacobian product 而不释义 |
| 反向保存张量 | saved tensors | 前向传播保存反向传播所需张量；API 写 `saved_tensors` | saved tensor、保存激活和中间梯度混用 |
| 梯度汇合 | gradient accumulation at branches | 多条反向路径的梯度在分支汇合点相加 | 与跨微批次的梯度累积混用 |
| 损失 | loss | 损失连接模型输出与监督目标；变量写 `loss` | 正文直接使用 loss |
| 激活 | activation | 激活由前向传播产生，可被反向传播消费 | 与所有中间张量完全等同 |
| 数据类型 | data type / dtype | dtype、低精度 dtype | 精度类型、数据精度混用 |
| 前向传播 | forward pass | 前向传播产生中间状态 | 前向、推理前向不加说明 |
| 反向传播 | backward pass | 反向传播计算梯度 | 反传、反向计算随意切换 |
| 优化器 | optimizer | 优化器读取参数梯度并更新参数；API 写 `optimizer.step()` | 正文直接使用 optimizer |
| 学习率调度器 | learning-rate scheduler | 学习率调度器按约定的优化器更新步推进 | 正文只写 scheduler 而不说明对象 |
| 优化器更新步 | optimizer step | scheduler 按 optimizer step 更新 | 与 micro-batch 混为同一“步” |
| 微批次 | micro-batch | 一个 micro-batch 的前后向 | 训练 batch、全局 batch 混用 |
| 梯度累积 | gradient accumulation | 多个 micro-batch 后执行一次更新 | 累计 batch、梯度堆积 |
| 混合精度 | mixed precision | BF16 / FP16 混合精度训练 | 将 BF16 支持直接等同于原生加速 |
| 参数量 | parameter count | 参数量、可训练参数量 | 参数规模与显存占用等同 |
| 浮点运算量 | FLOPs | 理论 FLOPs、实际吞吐 | FLOPs 与 FLOP/s 混用 |

## 模型架构与 Attention

| 中文规范名 | English / 缩写 | 推荐写法 | 不建议混用 |
| --- | --- | --- | --- |
| Transformer 块 | Transformer block | 一个 Block 的 Attention、MLP 与残差路径 | layer、block 不说明颗粒度 |
| 残差路径 | residual path | 残差路径连接子层输入与输出 | 残差连接、跳连随意混用 |
| 多头注意力 | multi-head attention / MHA | MHA 为每个 head 保存 K/V | 多头机制、注意力头不区分 |
| 分组查询注意力 | grouped-query attention / GQA | GQA 让多个 Q head 共享 KV head | grouped attention |
| 多头潜在注意力 | multi-head latent attention / MLA | MLA 压缩 K/V 的潜在表示 | latent attention、MLA 不释义 |
| 旋转位置编码 | rotary position embedding / RoPE | RoPE 对 Q/K 注入相对位置信息 | 位置编码与位置嵌入不加区分 |
| 均方根归一化 | RMS normalization / RMSNorm | RMSNorm 不执行去均值 | RMSNorm 与 LayerNorm 混写 |
| 前馈网络 | feed-forward network / FFN | FFN / MLP 子层 | FFN、MLP 不说明此处等价 |
| 混合专家 | mixture of experts / MoE | token 只激活部分专家 | 稀疏模型、MoE 不加区分 |
| 状态空间模型 | state space model / SSM | SSM / Mamba 是递推状态路线 | 把 SSM 直接称为 Attention 变体 |

## 后训练、偏好与对齐

| 中文规范名 | English / 缩写 | 推荐写法 | 不建议混用 |
| --- | --- | --- | --- |
| 监督微调 | supervised fine-tuning / SFT | SFT 学习指令格式与任务行为 | 微调默认等于 SFT |
| 偏好优化 | preference optimization | 离线 chosen / rejected 对上的偏好优化 | 对齐、偏好学习不加说明 |
| 直接偏好优化 | direct preference optimization / DPO | DPO 依赖 reference model 与偏好对 | 偏好损失、DPO 混用 |
| 近端策略优化 | proximal policy optimization / PPO | PPO 使用策略、奖励、价值等信号 | RLHF 与 PPO 完全等同 |
| 组相对策略优化 | group relative policy optimization / GRPO | GRPO 按同 prompt 的组内回报归一化 | group policy optimization |
| 基于人类反馈的强化学习 | reinforcement learning from human feedback / RLHF | RLHF 是一类对齐流程 | 人类偏好、奖励模型、PPO 等同 |
| 可验证奖励强化学习 | reinforcement learning with verifiable rewards / RLVR | RLVR 用规则或 verifier 提供奖励 | 把奖励直接等同于任务质量 |
| 参考模型 | reference model | 用于约束策略偏离的 reference model | baseline model 不说明角色 |
| rollout | rollout | 从当前策略采样的完整响应或轨迹 | 采样、生成、rollout 混用 |
| 奖励投机 | reward hacking | 奖励上升但目标能力未改善 | 把所有失败都称为 reward hacking |

## 推理、KV Cache 与服务

本节术语在正文中直接写英文，首次出现加中文括注。

| English / 缩写 | 中文（括注用） | 推荐写法 | 容易混淆之处 |
| --- | --- | --- | --- |
| token | 词元 | "每步生成一个 token"；不写"词元" | 与字符、词、subword 不加区分 |
| logits | 词表分数 | 模型最后一层输出的未归一化分数，形状 `[vocab_size]` | 与概率混用；logits 没有归一化 |
| decoder-only | 仅解码器 | "decoder-only 模型的自回归生成" | 与 encoder-decoder 的解码阶段混用 |
| prompt | 提示词 | prompt token、请求输入 | prompt 与完整 request 不加区分 |
| prefill | 预填充 | 处理输入 token 并建立 KV Cache，也可按调度预算分块 | 不等于 TTFT 的全部时间 |
| decode | 解码 | decode 每步读取 KV Cache 追加一个 token | 生成、推理阶段混用 |
| KV Cache | 键值缓存 | KV Cache 缓存历史 K/V | KV cache、kv cache 大小写漂移 |
| greedy / sampling | 贪心 / 采样 | greedy 取 argmax；sampling 从分布抽取 | 有限低温不等于显式 greedy；唯一最大值下趋零极限另论 |
| temperature | 温度 | logits 除以 T 再 softmax；T<1 更尖，T>1 更平 | 以为 temperature 改变排序 |
| top-k / top-p | 候选截断 / 核采样 | 按排名或累计概率质量过滤；21 的 top-k 包含阈值并列项，top-p 保留至累计首次 > p（p=1 不截断） | 等号边界、并列与最少保留数需核对具体实现 |
| speculative decoding | 投机解码 | draft 提案、target 一次验证并修正 | 推测采样、投机推理混用 |
| draft / target model | 草稿模型 / 目标模型 | draft 生成候选，target 验证 | baseline / candidate 取代模型角色 |
| draft / verify / accept / correct | 草稿 / 验证 / 接受 / 修正 | 接受概率 α = min(1, p/q)；拒绝后从 norm(max(0, p−q)) 重采样 | 把"拒绝后重采样"写成"交还 target 重新生成" |
| acceptance length | 接受长度 | 一轮验证中被接受的 draft token 数 | 与接受率（比例）混用 |
| lookahead / k | 草稿步数 | 每轮 draft 提出的候选数 | 与 beam width 混用 |
| MTP (multi-token prediction) | 多 token 预测 | checkpoint 内的预测模块提供额外位置的候选 | 真实多步执行依赖 backend，候选步数不总等于模块层数 |
| EAGLE | — | 复用 target 隐藏表示的轻量 drafter + 候选树 | 把 EAGLE 当成独立小模型 |
| candidate tree | 候选树 | 一轮验证多条候选路径 | 与 beam search 混用 |
| DFlash / DSpark | — | 块扩散草稿；DSpark 增加轻量顺序修正与接受置信度预测 | 逐行 argmax 的教学示例不是完整扩散模型 |
| verification budget | 验证预算 | 一轮验证允许的候选数或计算量 | 与 lookahead k 等同 |
| PagedAttention | 分页注意力 | 用 block table 管理 KV Cache | 分页 Cache、PagedAttention 混用 |
| prefix cache | 前缀缓存 | 复用共享前缀的已计算状态 | 共享前缀、公共前缀缓存混用 |
| continuous batching | 连续批处理 | 迭代级动态加入 / 移除请求 | dynamic batching 不加说明 |
| chunked prefill | 分块预填充 | 将长 prefill 分块与 decode 交错 | chunk prefill、分段推理混用 |
| online softmax | 在线 softmax | 分块扫描时维护 running max 和 running sum | 与 safe softmax（只减 max）混用 |
| SDPA | 缩放点积注意力 API | `torch.nn.functional.scaled_dot_product_attention` | 把 SDPA 当成 FlashAttention 本身 |
| attention backend (flash / mem-efficient / math) | 注意力后端 | SDPA 按 shape、dtype、mask 自动选择 backend | 以为开了 SDPA 就一定走 flash |
| fusion / dispatch / fallback | 融合 / 分派 / 回退 | 候选过 backend 能力检查才走编译 kernel，否则 fallback 到 eager | 把 fallback 当成失败 |
| eager / compiled | 即时执行 / 编译执行 | eager 逐算子执行；compiled 经 `torch.compile` 融合 | 与 CPU / GPU 混用 |
| backend | 后端 | vLLM、SGLang 等执行后端 | 模型、框架、runtime 混用 |

## 性能、显存、I/O 与 profiling

| 中文规范名 | English / 缩写 | 推荐写法 | 不建议混用 |
| --- | --- | --- | --- |
| workload | workload | 正文直接写 workload，首次括注"写死的输入、配置与运行条件" | 测试数据、任务、实验设置混用；不写"工作负载" |
| 基线 / 候选 | baseline / candidate | 相同 workload 下比较 baseline 与 candidate | 对照组、方案、模型角色不明确 |
| 探针 | probe | 验证局部接口或现象的 GPU probe | 将 probe 当成端到端 benchmark |
| 冒烟测试 | smoke test | 验证环境、接口或最小路径可运行 | 将 smoke 结果写成性能结论 |
| 基准测试 | benchmark | 固定 workload 的可比较测量 | 测试、benchmark 混用 |
| 性能分析 | profiling | 聚合热点、资源使用与瓶颈 | profile 与 trace 完全等同 |
| 时间线追踪 | trace | 带时间戳的事件与依赖关系 | trace、日志、profile 混用 |
| 性能回归 | performance regression | 同样条件下性能或质量变差 | 单次波动即称回归 |
| 激活检查点 | activation checkpointing | 以重算降低 activation 峰值显存 | checkpoint / checkpointing 混用 |
| offload | offload | offload 到 CPU / NVMe | 迁移、搬运、卸载不加方向 |
| 输入输出 | input/output / I/O | 输入搬运、artifact 加载、状态传输、输出回写 | 仅把磁盘读写称为 I/O |
| 首 token 延迟 | time to first token / TTFT | TTFT 衡量首次输出等待 | 与端到端延迟等同 |
| 单 token 时间 | time per output token / TPOT | TPOT 衡量后续生成节奏 | 与首 token 延迟混用 |

## 通信、并行与分布式

| 中文规范名 | English / 缩写 | 推荐写法 | 不建议混用 |
| --- | --- | --- | --- |
| 数据并行 | data parallelism / DP | 不同 rank 处理不同数据分片 | 与 batch 并行混用 |
| 张量并行 | tensor parallelism / TP | 在层内切分张量计算 | 与模型并行泛称混用 |
| 流水线并行 | pipeline parallelism / PP | 按 Block / stage 切分模型 | 与训练流水线、I/O 流水线混用 |
| 专家并行 | expert parallelism / EP | 将 MoE 专家分布在不同设备 | MoE 与 EP 等同 |
| 集体通信 | collective communication | all-reduce、all-gather、reduce-scatter 等 | 通信原语与网络拓扑混用 |
| 全归约 | all-reduce | 聚合并分发同一结果 | reduce 与 all-reduce 混用 |
| rank | rank | 分布式进程 / 设备的逻辑编号 | GPU id、node id 混用 |
| 全局进程数 | world size | 一个分布式作业中的总 rank 数 | 机器数、GPU 数混用 |
| 通信—计算重叠 | communication-computation overlap | 通信与可并行计算的时间重叠 | 把并发执行直接称为 overlap |
| 气泡 | pipeline bubble | 流水线未充分利用的空闲区间 | 任意 idle 都称 bubble |

## 算子、编译、Triton 与 CUDA

| 中文规范名 | English / 缩写 | 推荐写法 | 不建议混用 |
| --- | --- | --- | --- |
| 算子 | operator | 一个张量运算或计算图节点 | kernel 与 operator 等同 |
| 内核 | kernel | GPU 上执行的 kernel 实现 | 算子、函数、kernel 混用 |
| 算子融合 | operator fusion | 消除中间张量落地或减少 launch | 图优化与融合完全等同 |
| 图级优化 | graph optimization | 对计算图做重写、消除或融合选择 | 编译优化泛称 |
| 参考实现 | reference implementation | 用于正确性对照的 PyTorch 实现 | baseline 默认就是 reference |
| 自动调优 | autotuning | 搜索 block、warp、stage 等配置 | 手动参数试验等同 autotune |
| 占用率 | occupancy | 活跃 warp / block 对硬件资源的利用 | 利用率、吞吐或 SM busy 混用 |
| 合并访存 | coalesced memory access | 相邻线程访问相邻地址 | 连续内存与合并访存混用 |
| 共享内存 | shared memory | block 内可协作的片上存储 | L2、寄存器、shared memory 混用 |
| warp | warp | NVIDIA GPU 的线程执行组 | block、thread、warp 混用 |

## 量化、部署、数据工程、Agent 与多模态

| 中文规范名 | English / 缩写 | 推荐写法 | 不建议混用 |
| --- | --- | --- | --- |
| 量化 | quantization | 将权重、激活或 KV 表示映射到低比特 | 压缩默认等于量化 |
| 校准 | calibration | 用校准数据确定量化尺度或分组参数 | 评测、训练、校准混用 |
| 权重量化 | weight quantization | 只量化权重的方案 | W8A16 与全量低比特混用 |
| 量化产物 | quantized artifact | 可加载的 AWQ / GPTQ / GGUF 等部署产物 | QLoRA adapter 直接称量化产物 |
| 产物清单 | artifact manifest | 记录 artifact id、格式、来源、质量与兼容性 | result JSON 与 manifest 混用 |
| 后端 | backend | vLLM、SGLang 等执行后端 | 模型、框架、runtime 混用 |
| 数据血缘 | data lineage | 数据来源、变换与版本关系 | 数据路径、文件名等同血缘 |
| 去污染 | decontamination | 清除训练与评测集的重叠 | 普通去重等同去污染 |
| 检索增强生成 | retrieval-augmented generation / RAG | 检索、上下文构造与生成的组合 | RAG 与向量数据库等同 |
| Agent 轨迹 | agent trajectory | 多步推理、工具调用与环境反馈序列 | 单轮 prompt-response 等同轨迹 |
| 沙盒 | sandbox | 隔离、配额、可复现的 Agent 执行环境 | 普通开发环境等同沙盒 |
| 视觉语言模型 | vision-language model / VLM | 接收视觉与文本输入的模型 | 多模态模型、VLM 不加范围区分 |
| 视觉 token | visual token | 视觉编码器输出并输入语言模型的 token | 图像 patch 与 visual token 等同 |

## 维护流程

1. 新概念先搜索本表和现有页面，优先沿用已有规范名。
2. 术语确有新含义时，补充中文名、英文名、缩写、推荐写法和容易混淆的词。
3. 同步修正专题 `intro`、组页和首个机制页；不做全仓机械替换。
4. 代码变量、JSON 字段和官方 API 以兼容性优先，不强行翻译。
