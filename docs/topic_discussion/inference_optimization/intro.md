# 推理优化（Inference Optimization）

> 专题类型：主学习路线　主服务目标：请求性能与 Serving 决策

假设你接手一个问答服务：1000-token（词元）输入很快，换成长文档后首个回答要等很久；增加并发后，生成又开始断断续续。这些症状可能分别来自 prefill（预填充）、decode（解码）、KV Cache（键值缓存）或排队调度。先定位时间和显存花在哪里，再选机制，比从工具名称开始更容易判断效果。

这条路线从请求指标出发，依次讨论一次前向怎样执行、一轮怎样推进更多 token、缓存和请求怎样共享资源，最后进入量化与分布式部署。每一步先用小例子理解，再到对应实验比较真实成本。已经熟悉基础时，可以按自己的问题直接进入。

## 学习路线

顺读可沿 Task0–6 的七个问题推进；想看连续解释可读[请求问题链](./walkthrough.md)，按现象查找用[判断手册](./casebook.md)，术语查阅用[专题术语表](./glossary.md)。推理执行与生成加速的练习集中在 [2.6 组](../../02_PyTorch_Algorithms/2_6.md)的 8 节。

| Task | 要回答的问题 | 先读这一节 | 专题正文 |
|---|---|---|---|
| Task0 · 请求与指标 | 慢在排队、首 token 还是后续生成？ | [INF-REQUEST 请求生命周期](../../02_PyTorch_Algorithms/INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.md) | [01 请求链路与指标](./01_request_path_and_metrics.md) |
| Task1 · Attention 与分派 | 怎样减少 attention 中间量，确认实际执行路径？ | [20 FlashAttention](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.md) | [02 Prefill 与 Attention](./02_prefill_and_attention_kernel.md) |
| Task2 · 解码与候选验证 | 怎样一轮推进多个 token 并保持目标生成规则？ | [21 解码策略](../../02_PyTorch_Algorithms/21_Decoding_Strategies.md) | [03 解码策略](./03_decoding_strategies.md) |
| Task3 · KV Cache | 缓存怎样分配、增长与复用？ | [22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md) | [04 KV Cache](./04_kv_cache_lifecycle_and_reuse.md) |
| Task4 · Serving 调度 | 长短请求怎样共享迭代与资源？ | [迭代预算](../../02_PyTorch_Algorithms/INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.md) | [07 Serving 与 PD](./07_serving_scheduling_and_pd.md) |
| Task5 · 低精度部署 | 压缩后能否正确加载，质量与速度怎样变化？ | [66 浮点基线](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) | [05 量化部署](./05_quantized_inference_and_deployment.md) |
| Task6 · 分布式推理 | 单卡不够时，副本或模型并行怎样取舍？ | [81 分布式推理](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.md) | [08 分布式推理](./08_distributed_inference_and_parallelism.md) |

<details>
<summary>完整学习链：项目、扩展与交叉阅读</summary>

**Task0 · 请求、workload 与指标**

核心：[Part 02 · INF-REQUEST 请求生命周期](../../02_PyTorch_Algorithms/INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.md)；项目：[Part 02 · 66 推理性能比较](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)

**Task1 · 注意力计算与执行分派**

核心：[Part 02 · 04 MHA / GQA](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md) → [Part 02 · 20 FlashAttention / SDPA](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.md) → [Part 02 · INF-EXECUTION-DISPATCH](../../02_PyTorch_Algorithms/INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.md)；扩展：[算子优化](../operator_optimization/intro.md)

**Task2 · 解码推进与候选验证**

核心：[Part 02 · 21 解码策略](../../02_PyTorch_Algorithms/21_Decoding_Strategies.md) → [Part 02 · 23 投机解码](../../02_PyTorch_Algorithms/23_Speculative_Decoding.md) → [Part 02 · 35 原生 MTP](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.md)或[Part 02 · INF-EAGLE](../../02_PyTorch_Algorithms/INF-EAGLE_Feature_Drafter_and_Candidate_Tree.md) → [Part 02 · INF-PROPOSAL 动态预算](../../02_PyTorch_Algorithms/INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.md)；项目：[Part 02 · 68 投机解码基准](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.md)；扩展：[Part 02 · INF-PARALLEL-DRAFTING 并行草稿](../../02_PyTorch_Algorithms/INF-PARALLEL-DRAFTING_DFlash_and_DSpark.md)

**Task3 · KV Cache 表示、管理与复用**

核心：[Part 02 · 04 MHA / GQA](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md) → [Part 02 · 22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md) → [Part 02 · 37 KV 生命周期](../../02_PyTorch_Algorithms/37_KV_Cache_Lifecycle_and_Capacity_Control.md) → [Part 02 · 24 Prefix Cache](../../02_PyTorch_Algorithms/24_Prefix_Cache_Matching_and_Reuse.md) → [Part 02 · 34 RadixAttention](../../02_PyTorch_Algorithms/34_SGLang_RadixAttention.md)；项目：[Part 02 · 69 Prefix Cache](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md)；扩展：[Part 02 · ARCH-MLA 紧凑 KV 表示](../../02_PyTorch_Algorithms/ARCH-MLA_Multihead_Latent_Attention.md) → [Part 02 · 71 MLA 基准](../../02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.md)，以及 [Part 02 · 54 KV Cache 量化](../../02_PyTorch_Algorithms/54_KV_Cache_Quantization_Strategies.md)

**Task4 · 服务迭代预算、调度与异构协同**

核心：[Part 02 · INF-ITERATION-BUDGET](../../02_PyTorch_Algorithms/INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.md) → [Part 02 · 36 请求调度](../../02_PyTorch_Algorithms/36_Decode_Scheduling.md) → [Part 02 · INF-CONTINUOUS-BATCHING](../../02_PyTorch_Algorithms/INF-CONTINUOUS-BATCHING_Continuous_Batching.md) → [Part 02 · 38 阶段调度](../../02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.md) → [Part 02 · INF-SERVING-OVERLAP](../../02_PyTorch_Algorithms/INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff.md) → [Part 02 · 39 异构 PD](../../02_PyTorch_Algorithms/39_Hetero_PD_and_Serving_Tiers.md)；项目：[Part 02 · 70 Serving 调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.md)；扩展：[Part 02 · INF-AFD](../../02_PyTorch_Algorithms/INF-AFD_Attention_FFN_Disaggregation.md)

**Task5 · 低精度执行与单卡部署检查**

核心：[Part 02 · 66 浮点基线](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) → [Part 02 · 52 校准与误差](../../02_PyTorch_Algorithms/52_Quantization_Calibration_and_Error.md) → [Part 02 · 25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md)，再按对象进入 [Part 02 · 40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md)、[Part 02 · 53 SmoothQuant / W8A8](../../02_PyTorch_Algorithms/53_Activation_Quantization_and_SmoothQuant.md)、[Part 02 · 41 FP8 执行路径](../../02_PyTorch_Algorithms/41_FP8_Runtime_Low_Precision_Paths.md)、[Part 02 · 54 KV Cache 量化](../../02_PyTorch_Algorithms/54_KV_Cache_Quantization_Strategies.md)或[Part 02 · QUANT-MIXED-BIT](../../02_PyTorch_Algorithms/QUANT-MIXED-BIT_Mixed_Bit_Allocation.md)，最后用 [Part 02 · INF-EXECUTION-DISPATCH](../../02_PyTorch_Algorithms/INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.md)检查能力门槛、实际执行路径与回退；后续项目入口：[Part 02 · 82 量化产物评估](../../02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project.md) → [Part 02 · 67 量化推理与部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md) → [Part 02 · 83 量化 Serving Profile](../../02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark.md) → [Part 02 · 72 单卡部署](../../02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison.md)

**Task6 · 分布式 Serving、可靠性与 SLO 决策**

核心：[Part 01 · 26 并行策略决策](../../01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.md) → 请求切分 [Part 02 · PAR-REPLICA](../../02_PyTorch_Algorithms/PAR-REPLICA_Serving_Replica_Routing.md)，或模型内部的 [Part 02 · 28 Pipeline Parallel](../../02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.md)、[Part 02 · 29 Tensor Parallel](../../02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.md)、[Part 02 · PAR-CONTEXT](../../02_PyTorch_Algorithms/PAR-CONTEXT_Context_and_Sequence_Parallelism.md)、[Part 02 · 46 Expert Parallel](../../02_PyTorch_Algorithms/46_MoE_Expert_Parallel.md)；通信证据：[Part 02 · 47 通信 Profiling](../../02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.md) → [Part 02 · 48 热点缓解](../../02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.md) → [Part 02 · 49 策略选型](../../02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.md)；恢复基础：[Part 01 · 28 故障容错](../../01_Hardware_Math_and_Systems/28_Fault_Tolerance_and_Checkpointing.md)；MoE 项目进入 [Part 02 · 80 专家并行](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.md)，最终由 [Part 02 · 81 分布式推理验证](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.md) 汇总比较

</details>

Task5 复用量化专题的机制 JSON 和版本化产物，不重复局部量化实验；当前 KV Cache 量化证据仅覆盖 Transformers `QuantizedCache` + HQQ，不能外推到 vLLM 或 SGLang。Task6 的 CPU 内容用于理解切分、路由和恢复决策，多卡通信、扩展效率与故障恢复证据暂缓到真实多 GPU 环境验证。

![推理优化学习路线：从请求链路到专项验证](../../public/topic_discussion/inference_optimization/inference_optimization_roadmap.svg)

![推理优化知识地图：从请求对象到验证决策](../../public/topic_discussion/inference_optimization/inference_optimization_knowledge_map.svg)

## 扩展与跨专题入口

### 按需回补：架构与共享前置

遇到架构、MoE、dtype 或精度方面的知识缺口时，从下表回补对应入口，再回到当前 Task 继续学习。

| 补充主题 | Notebook 入口 | 建议时机 |
|:---|:---|:---|
| Benchmark 证据与实验设计 | [Part 00 · 20 性能证据与实验设计](../../00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.md) | Task0 / Task6；需要保持输入与配置一致、基线、重复测量、失败记录和证据等级时 |
| 模型计算量与硬件资源 | [Part 01 · 02 参数量与 FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.md) → [Part 01 · 03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md) | Task0–1；需要估算计算、带宽与容量上界时 |
| 注意力工作集与显存 | [Part 01 · 04 注意力显存优化](../../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.md) → [Part 01 · 14 FlashAttention 显存模型](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md) | Task1；需要先建立完整分数矩阵、分块工作集和显存读写直觉时 |
| KV Cache 容量增长 | [Part 01 · 11 KV Cache 与显存增长](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md) | Task3；需要从层数、KV 头数、上下文长度和并发度估算容量上界时 |
| 架构基础 | [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md)、[Part 02 · 08 Architecture Tricks](../../02_PyTorch_Algorithms/08_Architecture_Tricks.md) | Task0 开始时需要确认 decoder-only 模型的数据流、持久状态或架构配置时 |
| 数据类型、量化表示与运行时约束 | [Part 01 · 01 数据类型与精度](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md) → [Part 01 · 21 量化理论](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md) → [Part 01 · 34 异构运行时约束](../../01_Hardware_Math_and_Systems/34_Heterogeneous_Accelerators_and_Runtime_Constraints.md) | Task5；需要解释数据类型、产物兼容性或执行回退时 |
| 异步执行与状态交接 | [Part 01 · 17 CUDA Stream](../../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.md) | Task4；需要解释准备、计算和搬运重叠时 |
| 通信拓扑与集合通信 | [Part 01 · 05 通信拓扑](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.md) → [Part 01 · 20 NCCL 与 AllReduce](../../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md) | Task6；需要解释多卡通信上界时 |
| 可靠性与恢复 | [Part 01 · 28 故障容错与检查点](../../01_Hardware_Math_and_Systems/28_Fault_Tolerance_and_Checkpointing.md) | Task6；需要检查副本失效、重试、状态恢复和降级条件时 |
| MoE 架构与计算 | [Part 02 · 06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.md)、[Part 02 · 07 MoE Load Balancing](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.md)、[Part 01 · 22 MoE 参数与计算](../../01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute.md) | Task4 / Task6；进入 AFD 或专家并行前 |

### 跨专题入口

需要定位延迟和容量来源时进入[性能分析](../performance_analysis/intro.md)，需要比较候选和瓶颈转移时进入[性能优化](../performance_optimization/intro.md)，量化产物与低精度路径进入[量化与压缩](../quantization_compression/intro.md)，多卡切分与通信证据进入[通信与并行](../communication_parallelism/intro.md)，计算内核与融合进入[算子优化](../operator_optimization/intro.md)。

## 去哪里验证

首次运行先按[实验 Quickstart](./quickstart.md)创建或恢复环境，再从[实验资产表](./experiment_assets.md)选择 Profile、模型、workload、候选和结果路径；跨专题的软件版本与硬件快照查询[GPU 环境资产表](../../gpu_environment_assets.md)。机制练习解释局部变化，后续项目再比较真实后端、请求队列和部署方案。

| 验证方式 | 要回答的问题 | 输出与入口 |
|:---|:---|:---|
| CPU 机制 | 状态转换、公式、预算或选择规则是否成立？ | 各机制页的题目、测试、答案与解析 |
| 单卡机制 GPU | 真实模型、算子或运行时路径是否改变局部质量、延迟或显存？ | [Part 02 · 20 FlashAttention / SDPA](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)、INF-ITERATION-BUDGET、INF-EXECUTION-DISPATCH、INF-PROPOSAL、INF-SERVING-OVERLAP，以及量化专题 25、40、41、53、54 和 Mixed-bit 的机制 JSON |
| 专项后端 | 缓存、调度或投机策略是否在真实请求中成立？ | [Part 02 · 68 投机解码](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.md)、[Part 02 · 69 前缀缓存](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md)、[Part 02 · 70 服务调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.md) |
| 单卡部署 | 合格产物是否被目标执行后端正确加载并产生可复现收益？ | 后续项目链：[Part 02 · 66 浮点基线](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) → [Part 02 · 82 产物评估](../../02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project.md) → [Part 02 · 67 量化部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md) → [Part 02 · 83 服务 Profile](../../02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark.md) → [Part 02 · 72 单卡部署](../../02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison.md)；是否完成以各项目的运行记录为准 |
| 分布式决策 | 多卡方案能否在通信代价下改善容量、延迟或吞吐？ | [Part 02 · 81 分布式推理验证](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.md) |
