# 03. Decoding Strategies | 解码策略

## 学习问题

首个 token（词元）已经出来，后面的回答却一字一顿。是采样设置的问题，还是模型每次只能推进一个位置？本页先解释怎样选择下一个 token，再比较一轮推进多个位置的方法。输出质量、分布和速度需要分别检查。

## 核心机制

### 从一张 logits 到下一个 token

模型每步给出 logits（未归一化词表分数）。greedy（贪心）取最大值；sampling（采样）先形成概率分布，再随机选择。temperature（温度）缩放分差，top-k 限制名额，top-p 限制候选集合覆盖的概率质量。它们作用于选择阶段，不会因为温度调低就减少模型层数或 KV Cache（键值缓存）。

例如概率 `[0.5,0.3,0.1,0.05,0.05]`、top-p=0.85，应保留累计达到 0.9 的前三项，而不是删掉让累计越过阈值的第三项。第 [21 节](../../02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb)实现这些操作，并解释 `cumsum > p` 后右移的等号边界约定；可从[速查卡](../../02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb#cheatsheet)快速回顾。

比较加速方案时，需要保持目标生成规则一致。改 temperature 或输出长度也可能改变耗时，但那不再是同一生成任务下的对照。

### draft 先猜，target 批量验证

小 batch 的 decode（解码）常受权重和 KV 读取限制：读取很多数据，却只推进一个位置。speculative decoding（投机解码）让 draft（草稿器）先猜 K 个 token，再由 target（目标模型）批量验证，以额外草稿成本换取更少的串行 target 前向。

例如候选 `[4,5,6]`，贪心 target 在对应位置给出 `[4,5,9,8]`。前两个匹配，第三个不同，于是输出 `[4,5,9]` 并结束本轮。后续候选建立在已经被拒的前文上，不能继续提交。全部匹配时还可输出一个 bonus token（额外 token）；EOS 和长度上限可能提前结束生成。

采样路径需要概率修正。若同一上下文下 draft 与 target 分布为 q、p，接受候选 x 的概率是 `min(1,p(x)/q(x))`；拒绝后从 `normalize(max(0,p−q))` 抽修正 token。例如 `q(B)=0.6`、`p(B)=0.3`，B 只接受一半，防止 draft 过度输出它。第 [23 节](../../02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb)用四词例子推导最终分布为什么仍是 p。

分布等价不保证两次随机运行生成相同字符串。实际低精度、累加顺序和近并列 argmax 还会影响逐 token 复现，不能因算法叫“无损”就跳过输出检查。

### 候选从哪里来：三个扩展分支

**原生 MTP（多 token 预测）** 把预测模块放进 checkpoint，复用主模型隐藏状态。第 [35 节](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.ipynb)从给定的各深度 logits 组成候选块，练习连续贪心验证。真实多步生成取决于模型和 backend（执行后端），不能把模块层数当成所有实现的候选上限。

**EAGLE** 为特定 target 训练轻量 drafter，复用隐藏特征和 token embedding（嵌入）。候选可组织成树，target 用 tree attention（树注意力）限制每条分支的可见上下文，再按相应算法验证。[EAGLE 练习](../../02_PyTorch_Algorithms/INF-EAGLE_Feature_Drafter_and_Candidate_Tree.ipynb)只演示简化融合和路径打分；draft 的高分不等于 target 接受率。

**DFlash / DSpark** 减少草稿主干的串行等待。DFlash 用块扩散网络并行出候选，DSpark 在并行主干后加轻量顺序修正和接受置信度预测。[并行草稿练习](../../02_PyTorch_Algorithms/INF-PARALLEL-DRAFTING_DFlash_and_DSpark.ipynb)从给定分数手算这些操作。置信度 `[0.9,0.8,0.3,0.95]`、阈值 0.5 时只送前两个位置验证，不能跨过中间缺口取最后一个。

这些是不同的候选来源，不必按顺序全部部署；无论选哪条路，target 验证都不可省略。

### 预算与成本：猜得多不一定快

假设 draft 每个候选花 3 ms，一次验证花 12 ms，K=5 就需 27 ms。如果只接受两个候选，而 baseline（基线）逐 token 花 10 ms，仅这两个候选节省的 20 ms 还不够支付成本。完整对照还要计入修正或 bonus 对应的输出量，以及调度、状态更新等耗时。这些是假设数字，说明接受率本身不能决定收益。

[动态预算练习](../../02_PyTorch_Algorithms/INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.ipynb)按近期接受率调整 K，并解释部分成本估算的边界。轮内置信度截断则在本轮候选不够可信时提前停止。前者看历史，后者看当下，最终都要用延迟、质量和资源占用检验。

## 速查表

<details>
<summary>候选来源与额外成本</summary>

| 来源 | 复用什么 | 新增成本或限制 | 快速查阅 |
|---|---|---|---|
| 外置 draft model | 自己的 KV Cache | 额外权重、缓存与串行草稿计算 | [23](../../02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb#cheatsheet) |
| 原生 MTP | 主模型隐藏状态 | checkpoint 模块与 backend 支持 | [35](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.ipynb#cheatsheet) |
| EAGLE | 配对 target 的特征 | drafter 训练、配对与树验证 | [INF-EAGLE](../../02_PyTorch_Algorithms/INF-EAGLE_Feature_Drafter_and_Candidate_Tree.ipynb#cheatsheet) |
| DFlash / DSpark | target 条件与并行主干输出 | 专用资产、顺序头与置信度校准 | [并行草稿](../../02_PyTorch_Algorithms/INF-PARALLEL-DRAFTING_DFlash_and_DSpark.ipynb#cheatsheet) |

</details>

## 选择条件

| 现象 | 优先检查 | 下一步 |
|---|---|---|
| 输出重复或采样不可复现 | 生成规则、种子与数值环境 | [21 解码策略](../../02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb) |
| 接受率高但延迟未降 | draft、verify、调度各自成本 | [动态预算速查卡](../../02_PyTorch_Algorithms/INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.ipynb#cheatsheet) |
| 首次拒绝总在前几个位置 | 模型配对、tokenizer、采样规则 | [23 验证规则](../../02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb#cheatsheet) |
| 长草稿后半段总浪费 | 顺序依赖与预算 | [并行草稿与置信前缀](../../02_PyTorch_Algorithms/INF-PARALLEL-DRAFTING_DFlash_and_DSpark.ipynb#cheatsheet) |
| 单请求快，高并发却慢 | 候选占用、批处理和 KV 预算 | [07 Serving 调度](./07_serving_scheduling_and_pd.md) |

真实模型对照进入 [68 投机解码基准](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)。保持 target、输入、生成规则和质量要求一致，同时记录延迟、有效推进量、额外显存及失败。各节是否已有测量、是否仅检查了配对条件，见[实验资产表](./experiment_assets.md)。

## 相关阅读

- [Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 接受与修正的推导。
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) —— 原生 MTP 训练与推理。
- [EAGLE-3](https://arxiv.org/abs/2503.01840) —— 多层特征与训练目标。
- [DFlash](https://arxiv.org/abs/2602.06036) —— 块扩散草稿。
- [DSpark 算法文档](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/) —— 顺序头与置信头。
