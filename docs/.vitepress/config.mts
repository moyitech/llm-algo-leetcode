import { defineConfig } from 'vitepress'
import { createHash } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'

const mermaidCachePath = new URL('./cache/mermaid-cache.json', import.meta.url)
const mermaidCache = existsSync(mermaidCachePath)
  ? JSON.parse(readFileSync(mermaidCachePath, 'utf8')) as Record<string, string>
  : {}

const escapeHtml = (value: string) =>
  value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')

const renderMermaidSvg = (content: string) => {
  const hash = createHash('sha256').update(content, 'utf8').digest('hex')
  const svg = mermaidCache[hash]
  if (svg) {
    return `<div class="mermaid-snapshot">\n${svg}\n</div>\n`
  }

  return `<pre class="mermaid-fallback"><code>${escapeHtml(content)}</code></pre>\n`
}

const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/llm-algo-leetcode/'

export default defineConfig({
  lang: 'zh-CN',
  title: "大模型算法实战教程",
  description: "面向大模型入门到进阶的算法实战教程",
  base: baseConfig,
  ignoreDeadLinks: true,
  markdown: {
    math: true,
    config(md) {
      const defaultFence = md.renderer.rules.fence
      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        if (token.info.trim() === 'mermaid') {
          return renderMermaidSvg(token.content)
        }
        return defaultFence
          ? defaultFence(tokens, idx, options, env, self)
          : self.renderToken(tokens, idx, options)
      }
    }
  },
  themeConfig: {
    logo: '/datawhale-logo.png',
    nav: [
      { text: '第零部分\n前置知识与环境准备', link: '/00_Prerequisites/intro' },
      { text: '第一部分\n硬件与系统基础', link: '/01_Hardware_Math_and_Systems/intro' },
      { text: '第二部分\nPyTorch 核心算法', link: '/02_PyTorch_Algorithms/intro' },
      { text: '第三部分\nTriton 算子开发', link: '/03_Triton_Kernels/intro' },
      { text: '第四部分\nCUDA C++ 与系统优化', link: '/04_CUDA_and_System_Optimization/intro' },
      { text: '专题讨论', link: '/topic_discussion/intro' },
      { text: '组队学习', link: '/team_study/intro' },
    ],
    search: {
      provider: 'local',
      options: {
        detailedView: true,
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            displayDetails: '显示详细列表',
            resetButtonTitle: '清除',
            backButtonTitle: '关闭',
            noResultsText: '没有找到结果',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        },
        miniSearch: {
          options: {
            // Serialized and re-created in the browser, so it must stay self-contained.
            // Latin tokens keep '-' (top-p, decoder-only); CJK runs become bigrams
            // because minisearch would otherwise index a whole Chinese sentence as one token.
            tokenize: (text: string) => {
              const out: string[] = []
              for (const seg of text.toLowerCase().split(/[^\p{L}\p{N}-]+/u)) {
                if (!seg) continue
                for (const run of seg.split(/([\u3400-\u9fff]+)/u)) {
                  if (!run) continue
                  if (!/^[\u3400-\u9fff]+$/u.test(run)) { out.push(run); continue }
                  if (run.length === 1) out.push(run)
                  for (let i = 0; i + 1 < run.length; i++) out.push(run.slice(i, i + 2))
                }
              }
              return out
            }
          },
          searchOptions: {
            fuzzy: 0.1,
            prefix: true,
            combineWith: 'AND',
            boost: { title: 4, text: 2, titles: 1 }
          }
        }
      }
    },
    sidebar: [
      {
        text: '介绍',
        items: [
          { text: '项目概览', link: '/' },
          { text: '使用指南', link: '/guide' },
          { text: '贡献指南', link: '/contributing' },
          { text: '维护与发布手册', link: '/maintenance' }
        ]
      },
      {
        text: '专题讨论',
        items: [
          { text: '专题总览', link: '/topic_discussion/intro' },
          {
            text: '主学习路线：训练微调',
            collapsed: true,
            items: [
              { text: '专题导读', link: '/topic_discussion/fine_tuning_training/intro' },
              { text: '01. SFT 数据与 Loss', link: '/topic_discussion/fine_tuning_training/01_sft_data_and_loss' },
              { text: '02. LoRA / PEFT 设计', link: '/topic_discussion/fine_tuning_training/02_lora_peft_design' },
              { text: '03. 训练控制', link: '/topic_discussion/fine_tuning_training/03_training_control' },
              { text: '04. 端到端实验', link: '/topic_discussion/fine_tuning_training/04_end_to_end_experiment' },
              { text: '05. 项目交付与决策', link: '/topic_discussion/fine_tuning_training/05_project_delivery_decision' },
              { text: '06. 可视化资产', link: '/topic_discussion/fine_tuning_training/06_visual_assets' },
              { text: '正文手册', link: '/topic_discussion/fine_tuning_training/casebook' },
              { text: '深入阅读', link: '/topic_discussion/fine_tuning_training/walkthrough' },
              { text: '训练工程附录', link: '/topic_discussion/fine_tuning_training/training_engineering_appendix' },
              { text: '项目交付附录', link: '/topic_discussion/fine_tuning_training/project_delivery_appendix' }
            ]
          },
          {
            text: '主学习路线：推理优化',
            collapsed: true,
            items: [
              { text: '专题导读', link: '/topic_discussion/inference_optimization/intro' },
              { text: '问题链：从症状到决策', link: '/topic_discussion/inference_optimization/walkthrough' },
              { text: '01. 请求链路与指标', link: '/topic_discussion/inference_optimization/01_request_path_and_metrics' },
              { text: '02. Prefill 与 Attention Kernel', link: '/topic_discussion/inference_optimization/02_prefill_and_attention_kernel' },
              { text: '03. 解码策略', link: '/topic_discussion/inference_optimization/03_decoding_strategies' },
              { text: '04. KV Cache 表示、管理与复用', link: '/topic_discussion/inference_optimization/04_kv_cache_lifecycle_and_reuse' },
              { text: '05. 量化推理与部署', link: '/topic_discussion/inference_optimization/05_quantized_inference_and_deployment' },
              { text: '06. 端到端基准与决策', link: '/topic_discussion/inference_optimization/06_benchmark_and_decision' },
              { text: '07. Serving 调度与 PD 分离', link: '/topic_discussion/inference_optimization/07_serving_scheduling_and_pd' },
              { text: '08. 分布式推理与并行', link: '/topic_discussion/inference_optimization/08_distributed_inference_and_parallelism' },
              { text: '判断手册（速查）', link: '/topic_discussion/inference_optimization/casebook' },
              { text: '术语表', link: '/topic_discussion/inference_optimization/glossary' },
              { text: '实验 Quickstart', link: '/topic_discussion/inference_optimization/quickstart' },
              { text: '实验资产表', link: '/topic_discussion/inference_optimization/experiment_assets' }
            ]
          },
          {
            text: '主学习路线：显存优化',
            collapsed: true,
            items: [
              { text: '专题导读', link: '/topic_discussion/memory_performance_tuning/intro' },
              { text: '01. VRAM 账本与指标', link: '/topic_discussion/memory_performance_tuning/01_vram_ledger_and_metrics' },
              { text: '02. 训练显存压力', link: '/topic_discussion/memory_performance_tuning/02_training_memory_pressure' },
              { text: '03. Checkpointing 与 Offload', link: '/topic_discussion/memory_performance_tuning/03_checkpointing_and_offload' },
              { text: '04. 推理缓存与显存预算', link: '/topic_discussion/memory_performance_tuning/04_inference_cache_and_memory_budget' },
              { text: '05. 量化作为显存工具', link: '/topic_discussion/memory_performance_tuning/05_quantization_as_a_memory_tool' },
              { text: '06. Benchmark 与 Trade-off 决策', link: '/topic_discussion/memory_performance_tuning/06_benchmark_and_tradeoff_decision' },
              { text: '07. 可视化资产', link: '/topic_discussion/memory_performance_tuning/07_visual_assets' },
              { text: '正文手册', link: '/topic_discussion/memory_performance_tuning/casebook' },
              { text: '深入阅读', link: '/topic_discussion/memory_performance_tuning/walkthrough' }
            ]
          },
          {
            text: '横切支撑专题',
            collapsed: true,
            items: [
              { text: '量化与压缩', link: '/topic_discussion/quantization_compression/intro' },
              { text: '通信与并行', link: '/topic_discussion/communication_parallelism/intro' },
              { text: '性能分析', link: '/topic_discussion/performance_analysis/intro' },
              { text: '后训练优化', link: '/topic_discussion/post_training_optimization/intro' }
            ]
          },
          {
            text: '基础支撑专题',
            collapsed: true,
            items: [
              { text: '反向传播与训练机制', link: '/topic_discussion/backpropagation_training_mechanism/intro' },
              { text: '大模型架构演进', link: '/topic_discussion/llm_architecture_evolution/intro' },
              { text: '编译与图优化', link: '/topic_discussion/compiler_graph_optimization/intro' }
            ]
          }
        ]
      },
      { text: '组队学习', link: '/team_study/intro' },
      {
        text: '第零部分：前置知识',
        items: [
          { text: '部分导读', link: '/00_Prerequisites/intro' },
          {
            text: '0A 基础语言与数据表示',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/00_Prerequisites/0A' },
              { text: '01. Python 基础与数据表示', link: '/00_Prerequisites/01_Python_Essentials_for_LLM' },
              { text: '02. NumPy 与 Einsum', link: '/00_Prerequisites/02_NumPy_and_Einsum' },
              { text: '03. Python 面向对象与工具模式', link: '/00_Prerequisites/03_Python_OOP_and_Utility_Patterns' },
              { text: '04. Python 配置与数据入口', link: '/00_Prerequisites/04_Python_Config_and_Data_Entry' }
            ]
          },
          {
            text: '0B PyTorch 张量与自动求导',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/00_Prerequisites/0B' },
              { text: '05. PyTorch 张量基础操作', link: '/00_Prerequisites/05_PyTorch_Tensor_Fundamentals' },
              { text: '06. PyTorch 张量布局与索引', link: '/00_Prerequisites/06_PyTorch_Tensor_Layout_and_Indexing' },
              { text: '07. PyTorch 自动求导与反向传播', link: '/00_Prerequisites/07_PyTorch_Autograd_and_Backward' },
              { text: '08. PyTorch 梯度习惯与无梯度模式', link: '/00_Prerequisites/08_PyTorch_Grad_Hygiene_and_No_Grad' }
            ]
          },
          {
            text: '0C PyTorch 模型构建',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/00_Prerequisites/0C' },
              { text: '09. PyTorch nn.Module 基础', link: '/00_Prerequisites/09_PyTorch_nn_Module_Basics' },
              { text: '10. PyTorch 状态管理与持久化', link: '/00_Prerequisites/10_PyTorch_State_dict_and_Persistence' },
              { text: '11. PyTorch 优化器与损失函数', link: '/00_Prerequisites/11_PyTorch_Optimizers_and_Loss' },
              { text: '12. PyTorch 最小训练接口', link: '/00_Prerequisites/12_PyTorch_Minimal_Training_Interface' }
            ]
          },
          {
            text: '0D 训练与模型直觉',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/00_Prerequisites/0D' },
              { text: '13. 简单神经网络训练循环', link: '/00_Prerequisites/13_Simple_Neural_Network_Training' },
              { text: '14. 激活函数', link: '/00_Prerequisites/14_Activation_Functions' },
              { text: '15. 归一化技术', link: '/00_Prerequisites/15_Normalization_Techniques' },
              { text: '16. Attention 机制导论', link: '/00_Prerequisites/16_Attention_Mechanism_Intro' }
            ]
          },
          {
            text: '0E 调试与性能',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/00_Prerequisites/0E' },
              { text: '17. PyTorch 性能分析基础', link: '/00_Prerequisites/17_PyTorch_Profiling_Basics' },
              { text: '18. 显存分析与优化', link: '/00_Prerequisites/18_Memory_Profiling_and_Optimization' },
              { text: '19. 调试与异常定位', link: '/00_Prerequisites/19_Debugging_and_Anomaly_Localization' },
              { text: '20. 性能剖析与显存账本', link: '/00_Prerequisites/20_Profiling_and_Memory_Ledger' }
            ]
          }
        ]
      },
      {
        text: '第一部分：硬件与系统基础',
        items: [
          { text: '部分导读', link: '/01_Hardware_Math_and_Systems/intro' },
          {
            text: '1A 数值与规模',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/01_Hardware_Math_and_Systems/1A' },
              { text: '01. 大模型的数据格式与混合精度', link: '/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision' },
              { text: '02. 大模型参数量与 FLOPs', link: '/01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs' },
              { text: '21. 量化理论与 INT4/INT8', link: '/01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8' },
              { text: '22. MoE 参数量与计算量关系', link: '/01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute' }
            ]
          },
          {
            text: '1B 单设备执行、内存与访存',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/01_Hardware_Math_and_Systems/1B' },
              { text: '03. GPU 物理架构、内存层级与核心硬件单元', link: '/01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory' },
              { text: '04. Attention 显存优化', link: '/01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization' },
              { text: '11. KV Cache 与显存增长', link: '/01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth' },
              { text: '12. TensorCore 与混合精度', link: '/01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision' },
              { text: '13. 性能分析与瓶颈定位', link: '/01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis' },
              { text: '14. FlashAttention 显存模型', link: '/01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model' },
              { text: '23. TensorCore 深度剖析', link: '/01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive' },
              { text: '24. SRAM 优化技巧', link: '/01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques' },
              { text: '25. 稀疏计算与稀疏 Attention', link: '/01_Hardware_Math_and_Systems/25_Sparse_Computation_and_Sparse_Attention' }
            ]
          },
          {
            text: '1C 通信、并行与容错',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/01_Hardware_Math_and_Systems/1C' },
              { text: '05. 通信拓扑', link: '/01_Hardware_Math_and_Systems/05_Communication_Topologies' },
              { text: '06. 显存计算与 ZeRO', link: '/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO' },
              { text: '20. NCCL 与 AllReduce 基础', link: '/01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics' },
              { text: '26. 并行策略决策框架', link: '/01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework' },
              { text: '27. 通信调度优化', link: '/01_Hardware_Math_and_Systems/27_Communication_Scheduling_Optimization' },
              { text: '28. 容错与 Checkpointing', link: '/01_Hardware_Math_and_Systems/28_Fault_Tolerance_and_Checkpointing' }
            ]
          },
          {
            text: '1D 执行模型、异步调度与算子映射',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/01_Hardware_Math_and_Systems/1D' },
              { text: '07. CPU/GPU 异构调度', link: '/01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling' },
              { text: '08. CUDA/Triton 编程模型', link: '/01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton' },
              { text: '15. CUDA 执行模型', link: '/01_Hardware_Math_and_Systems/15_CUDA_Execution_Model' },
              { text: '16. Warp / Block / Shared Memory', link: '/01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics' },
              { text: '17. CUDA Stream 与异步', link: '/01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony' },
              { text: '18. Triton 块模型', link: '/01_Hardware_Math_and_Systems/18_Triton_Block_Model' },
              { text: '29. CUDA Stream 高级调度', link: '/01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling' }
            ]
          },
          {
            text: '1E 编译与后端',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/01_Hardware_Math_and_Systems/1E' },
              { text: '09. AI 编译器与图优化', link: '/01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization' },
              { text: '19. 算子融合导论', link: '/01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction' },
              { text: '30. 动态 Shape 处理', link: '/01_Hardware_Math_and_Systems/30_Dynamic_Shape_Handling' },
              { text: '32. TVM / MLIR 深度实践', link: '/01_Hardware_Math_and_Systems/32_TVM_MLIR_Deep_Practice' }
            ]
          },
          {
            text: '1F 硬件适配与系统约束',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/01_Hardware_Math_and_Systems/1F' },
              { text: '10. 国产 AI 芯片概览', link: '/01_Hardware_Math_and_Systems/10_Domestic_AI_Chips_Overview' },
              { text: '31. GPU 虚拟化与 MIG', link: '/01_Hardware_Math_and_Systems/31_GPU_Virtualization_and_MIG' },
              { text: '33. TCO 与成本模型', link: '/01_Hardware_Math_and_Systems/33_TCO_and_Cost_Model' },
              { text: '34. 异构加速器与运行时约束', link: '/01_Hardware_Math_and_Systems/34_Heterogeneous_Accelerators_and_Runtime_Constraints' }
            ]
          }
        ]
      },
      {
        text: '第二部分：PyTorch 核心算法',
        items: [
          { text: '部分导读', link: '/02_PyTorch_Algorithms/intro' },
          {
            text: '2.1 Decoder 组件与注意力',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_1' },
              { text: '00. PyTorch 热身', link: '/02_PyTorch_Algorithms/00_PyTorch_Warmup' },
              { text: '01. RMSNorm 教程', link: '/02_PyTorch_Algorithms/01_RMSNorm_Tutorial' },
              { text: '02. SwiGLU 激活', link: '/02_PyTorch_Algorithms/02_SwiGLU_Activation' },
              { text: '03. 旋转位置编码教程', link: '/02_PyTorch_Algorithms/03_RoPE_Tutorial' },
              { text: '04. 多头注意力', link: '/02_PyTorch_Algorithms/04_Attention_MHA_GQA' }
            ]
          },
          {
            text: '2.2 模型架构',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_2' },
              { text: '05. LLaMA3 Block 教程', link: '/02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial' },
              { text: '06. MoE 路由器', link: '/02_PyTorch_Algorithms/06_MoE_Router' },
              { text: '07. MoE 负载均衡损失', link: '/02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss' },
              { text: '08. 架构技巧', link: '/02_PyTorch_Algorithms/08_Architecture_Tricks' },
              { text: '44. 局部、滑动窗口与稀疏注意力', link: '/02_PyTorch_Algorithms/44_Local_Sliding_and_Sparse_Attention' },
              { text: '45. 线性注意力与递推状态', link: '/02_PyTorch_Algorithms/45_Linear_Attention_and_Recurrent_State' },
              { text: 'ARCH-MLA. MLA 与紧凑 KV 表示', link: '/02_PyTorch_Algorithms/ARCH-MLA_Multihead_Latent_Attention' },
              { text: 'ARCH-HYBRID. Attention、SSM 与混合记忆', link: '/02_PyTorch_Algorithms/ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid' }
            ]
          },
          {
            text: '2.3 监督微调与训练控制',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_3' },
              { text: '09. 监督微调训练循环', link: '/02_PyTorch_Algorithms/09_SFT_Training_Loop' },
              { text: '14. LoRA 教程', link: '/02_PyTorch_Algorithms/14_LoRA_Tutorial' },
              { text: '11. 优化器更新与学习率调度', link: '/02_PyTorch_Algorithms/11_Optimizer_Updates_and_Learning_Rate_Scheduling' },
              { text: '12. 梯度累积', link: '/02_PyTorch_Algorithms/12_Gradient_Accumulation' },
              { text: '13. 端到端微调实验', link: '/02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment' },
              { text: '30. 长上下文微调', link: '/02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning' },
              { text: '31. LoRA 变体原理', link: '/02_PyTorch_Algorithms/31_LoRA_Variants_Theory' },
              { text: '32. SFT 数据工程', link: '/02_PyTorch_Algorithms/32_Data_Engineering_for_SFT' },
              { text: '33. 微调 Readiness', link: '/02_PyTorch_Algorithms/33_Fine_Tuning_Readiness' }
            ]
          },
          {
            text: '2.4 偏好优化与对齐',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_4' },
              { text: '14. RLHF 与 PPO 显存占用与流转', link: '/02_PyTorch_Algorithms/14_RLHF_PPO_Memory' },
              { text: 'Reward Model、Value 与 Advantage', link: '/02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage' },
              { text: '15. 直接偏好优化损失教程', link: '/02_PyTorch_Algorithms/15_DPO_Loss_Tutorial' },
              { text: '16. 群体相对策略优化损失教程', link: '/02_PyTorch_Algorithms/16_GRPO_Loss_Tutorial' },
              { text: '50. 偏好数据与对齐评测', link: '/02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation' },
              { text: '51. 在线 DPO', link: '/02_PyTorch_Algorithms/51_Online_DPO' }
            ]
          },
          {
            text: '2.5 反向传播与训练内存',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_5' },
              { text: '17. 激活与损失反向', link: '/02_PyTorch_Algorithms/17_Activation_and_Loss_Backward' },
              { text: '18. Attention 反向传播', link: '/02_PyTorch_Algorithms/18_Attention_Backward_and_Custom_Autograd' },
              { text: '混合精度反向传播', link: '/02_PyTorch_Algorithms/TRAIN-MIXED-PRECISION-BACKWARD_Mixed_Precision_Backward' },
              { text: '19. 激活检查点', link: '/02_PyTorch_Algorithms/19_Activation_Checkpointing' },
              { text: '42. 激活卸载', link: '/02_PyTorch_Algorithms/42_Activation_Offload' },
              { text: '43. 统一训练内存账本', link: '/02_PyTorch_Algorithms/43_Unified_Training_Memory_Ledger' }
            ]
          },
          {
            text: '2.6 推理执行与生成加速',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_6' },
              { text: '20. FlashAttention 模拟', link: '/02_PyTorch_Algorithms/20_FlashAttention_Sim' },
              { text: '执行融合与后端分派', link: '/02_PyTorch_Algorithms/INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch' },
              { text: '21. 解码策略', link: '/02_PyTorch_Algorithms/21_Decoding_Strategies' },
              { text: '23. 投机解码', link: '/02_PyTorch_Algorithms/23_Speculative_Decoding' },
              { text: '35. 原生多 Token 预测', link: '/02_PyTorch_Algorithms/35_Multi_Token_Decoding' },
              { text: 'EAGLE 特征草稿与候选树', link: '/02_PyTorch_Algorithms/INF-EAGLE_Feature_Drafter_and_Candidate_Tree' },
              { text: 'DFlash / DSpark 并行草稿', link: '/02_PyTorch_Algorithms/INF-PARALLEL-DRAFTING_DFlash_and_DSpark' },
              { text: '动态提议与验证预算', link: '/02_PyTorch_Algorithms/INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget' }
            ]
          },
          {
            text: '2.7 推理策略与服务调度',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_7' },
              { text: '22. vLLM 分页注意力', link: '/02_PyTorch_Algorithms/22_vLLM_PagedAttention' },
              { text: '24. Prefix Cache 匹配与复用', link: '/02_PyTorch_Algorithms/24_Prefix_Cache_Matching_and_Reuse' },
              { text: '34. SGLang 基数注意力', link: '/02_PyTorch_Algorithms/34_SGLang_RadixAttention' },
              { text: '36. Decode Scheduling', link: '/02_PyTorch_Algorithms/36_Decode_Scheduling' },
              { text: '37. KV Cache Lifecycle and Capacity Control', link: '/02_PyTorch_Algorithms/37_KV_Cache_Lifecycle_and_Capacity_Control' },
              { text: 'Continuous Batching', link: '/02_PyTorch_Algorithms/INF-CONTINUOUS-BATCHING_Continuous_Batching' },
              { text: 'Serving Overlap and State Handoff', link: '/02_PyTorch_Algorithms/INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff' },
              { text: '38. Prefill/Decode 调度', link: '/02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling' },
              { text: '39. 异构 PD 与服务分层', link: '/02_PyTorch_Algorithms/39_Hetero_PD_and_Serving_Tiers' }
            ]
          },
          {
            text: '2.8 量化与低比特适配',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_8' },
              { text: '25. W8A16 量化', link: '/02_PyTorch_Algorithms/25_Quantization_W8A16' },
              { text: '26. QLoRA 与 4-bit 量化', link: '/02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization' },
              { text: '40. GPTQ 与 AWQ 权重量化', link: '/02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization' },
              { text: '41. FP8 与 KV Cache 量化', link: '/02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization' },
              { text: '52. 量化校准与误差控制', link: '/02_PyTorch_Algorithms/52_Quantization_Calibration_and_Error' },
              { text: '53. 激活量化与 SmoothQuant', link: '/02_PyTorch_Algorithms/53_Activation_Quantization_and_SmoothQuant' },
              { text: '54. KV Cache 量化策略', link: '/02_PyTorch_Algorithms/54_KV_Cache_Quantization_Strategies' }
            ]
          },
          {
            text: '2.9 通信与并行',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_9' },
              { text: '27. ZeRO 优化器模拟', link: '/02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim' },
              { text: '28. Pipeline 并行微批次', link: '/02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch' },
              { text: '29. Tensor 并行模拟', link: '/02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim' },
              { text: '46. NCCL 通信 profiling', link: '/02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL' },
              { text: '47. MoE 专家并行', link: '/02_PyTorch_Algorithms/47_MoE_Expert_Parallel' },
              { text: '48. 通信热点与缓解策略', link: '/02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation' },
              { text: '49. 并行策略选型', link: '/02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection' }
            ]
          },
          {
            text: '2.10 项目验证与工程收口',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_10' },
              { text: '60. Decoder Block 稳定性基准', link: '/02_PyTorch_Algorithms/60_Decoder_Block_Stability_Benchmark' },
              { text: '61. 架构验证项目', link: '/02_PyTorch_Algorithms/61_Model_Architecture_Exploration' },
              { text: '62. 指令微调项目', link: '/02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project' },
              { text: '63. LoRA 变体对比', link: '/02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark' },
              { text: '64. SFT 数据质量项目', link: '/02_PyTorch_Algorithms/64_SFT_Data_Quality_Project' },
              { text: '65. QLoRA 选型项目', link: '/02_PyTorch_Algorithms/65_QLoRA_Selection_Project' },
              { text: '66. 推理性能对比实验', link: '/02_PyTorch_Algorithms/66_Inference_Performance_Comparison' },
              { text: '67. 量化推理与部署项目', link: '/02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment' },
              { text: '68. 推测解码基准', link: '/02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark' },
              { text: '69. 前缀缓存基准', link: '/02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark' },
              { text: '70. 推理服务调度基准', link: '/02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark' },
              { text: '71. MLA 与 KV Cache 结构基准', link: '/02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark' },
              { text: '72. 单卡推理部署与 Backend 对比', link: '/02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison' },
              { text: '73. 训练性能分析', link: '/02_PyTorch_Algorithms/73_Training_Performance_Analysis' },
              { text: '74. Profiling 驱动端到端优化', link: '/02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization' },
              { text: '75. 显存预算压缩项目', link: '/02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project' },
              { text: '76. Activation / Checkpoint / Offload 对比项目', link: '/02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark' },
              { text: '77. 长序列记忆架构基准', link: '/02_PyTorch_Algorithms/77_Long_Sequence_Memory_Architecture_Benchmark' },
              { text: '78. Attention 访问模式基准', link: '/02_PyTorch_Algorithms/78_Attention_Access_Pattern_Benchmark' },
              { text: '79. 分布式并行基准', link: '/02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark' },
              { text: '80. MoE 专家并行 benchmark', link: '/02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark' },
              { text: '81. 分布式推理逻辑验证', link: '/02_PyTorch_Algorithms/81_Distributed_Inference_Project' },
              { text: '82. 量化产物评估项目', link: '/02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project' },
              { text: '83. 量化 Serving Profile 基准', link: '/02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark' },
              { text: '84. DPO 偏好优化项目', link: '/02_PyTorch_Algorithms/84_DPO_Preference_Project' },
              { text: '85. GRPO 组内对齐项目', link: '/02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project' },
              { text: '86. DPO 在线基准', link: '/02_PyTorch_Algorithms/86_DPO_Online_Benchmark' }
            ]
          }
        ]
      },
      {
        text: '第三部分：Triton 算子开发',
        items: [
          { text: '部分导读', link: '/03_Triton_Kernels/intro' },
          {
            text: '3.1 基础篇',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/03_Triton_Kernels/3_1' },
              { text: '01. Triton Vector Addition', link: '/03_Triton_Kernels/01_Triton_Vector_Addition' },
              { text: '02. Triton Fused SwiGLU', link: '/03_Triton_Kernels/02_Triton_Fused_SwiGLU' },
              { text: '03. Triton Fused RMSNorm', link: '/03_Triton_Kernels/03_Triton_Fused_RMSNorm' },
              { text: '04. Triton GEMM Tutorial', link: '/03_Triton_Kernels/04_Triton_GEMM_Tutorial' },
              { text: '05. Triton Autotune and Profiling', link: '/03_Triton_Kernels/05_Triton_Autotune_and_Profiling' }
            ]
          },
          {
            text: '3.2 过渡篇',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/03_Triton_Kernels/3_2' },
              { text: '06. Triton Fused Softmax', link: '/03_Triton_Kernels/06_Triton_Fused_Softmax' },
              { text: '06.5 Triton Design Patterns', link: '/03_Triton_Kernels/06_5_Triton_Design_Patterns' },
            ]
          },
          {
            text: '3.3 进阶A：Attention优化',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/03_Triton_Kernels/3_3' },
              { text: '07. Triton Fused RoPE', link: '/03_Triton_Kernels/07_Triton_Fused_RoPE' },
              { text: '08. Triton Flash Attention', link: '/03_Triton_Kernels/08_Triton_Flash_Attention' },
              { text: '09. Triton PagedAttention', link: '/03_Triton_Kernels/09_Triton_PagedAttention' }
            ]
          },
          {
            text: '3.4 进阶B：推理优化',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/03_Triton_Kernels/3_4' },
              { text: '10. Triton Quantization', link: '/03_Triton_Kernels/10_Triton_Quantization' },
              { text: '11. Triton Multi-LoRA', link: '/03_Triton_Kernels/11_Triton_Multi_LoRA' }
            ]
          },
          {
            text: '3.5 项目篇',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/03_Triton_Kernels/3_5' },
              { text: '12. Triton Memory Model and Debug', link: '/03_Triton_Kernels/12_Triton_Memory_Model_and_Debug' },
              { text: '13. Triton Llama3 Block Project', link: '/03_Triton_Kernels/13_Triton_Llama3_Block_Project' },
              { text: '14. Triton Best Practices and FAQ', link: '/03_Triton_Kernels/14_Triton_Best_Practices_and_FAQ' }
            ]
          }
        ]
      },
      {
        text: '第四部分：CUDA C++ 与系统优化',
        items: [
          { text: '部分导读', link: '/04_CUDA_and_System_Optimization/intro' },
          {
            text: '4.1 CUDA 编程基础（01-04）',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/04_CUDA_and_System_Optimization/4_1' },
              { text: '01. CUDA Custom Kernel Intro', link: '/04_CUDA_and_System_Optimization/15_CUDA_Custom_Kernel_Intro' },
              { text: '02. CUDA Shared Memory Optimization', link: '/04_CUDA_and_System_Optimization/16_CUDA_Shared_Memory_Optimization' },
              { text: '02.1 Bank Conflict Deep Dive', link: '/04_CUDA_and_System_Optimization/02_1_Bank_Conflict_Deep_Dive' },
              { text: '03. Tensor Core MMA Programming', link: '/04_CUDA_and_System_Optimization/03_Tensor_Core_MMA_Programming' },
              { text: '04. Warp-Level Primitives', link: '/04_CUDA_and_System_Optimization/04_Warp_Level_Primitives' }
            ]
          },
          {
            text: '4.2 系统级性能优化（05-08）',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/04_CUDA_and_System_Optimization/4_2' },
              { text: '05. CUDA Streams and Transfer', link: '/04_CUDA_and_System_Optimization/17_PyTorch_CUDA_Streams_and_Transfer' },
              { text: '06. CUDA Graph and JIT', link: '/04_CUDA_and_System_Optimization/18_CUDA_Graph_and_JIT_Compile' },
              { text: '07. 异步数据预取与 Double Buffering', link: '/04_CUDA_and_System_Optimization/07_Async_Data_Prefetch_and_Double_Buffering' },
              { text: '07.1 Double Buffering Deep Dive', link: '/04_CUDA_and_System_Optimization/07_1_Double_Buffering_Deep_Dive' },
              { text: '08. 内存池与显存管理', link: '/04_CUDA_and_System_Optimization/08_Memory_Pool_and_VRAM_Management' }
            ]
          },
          {
            text: '4.3 分布式训练工程（09-12）',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/04_CUDA_and_System_Optimization/4_3' },
              { text: '09. Distributed Communication Primitives', link: '/04_CUDA_and_System_Optimization/19_Distributed_Communication_Primitives' },
              { text: '09.1 Ring-AllReduce Deep Dive', link: '/04_CUDA_and_System_Optimization/09_1_Ring_AllReduce_Deep_Dive' },
              { text: '10. DeepSpeed ZeRO Config', link: '/04_CUDA_and_System_Optimization/20_DeepSpeed_Zero_Config' },
              { text: '11. 通信计算重叠高级调度', link: '/04_CUDA_and_System_Optimization/11_Communication_Computation_Overlap_Advanced_Scheduling' },
              { text: '12. 异构训练：CPU Offload 与 NVMe Offload', link: '/04_CUDA_and_System_Optimization/12_Heterogeneous_Training_CPU_Offload_NVMe_Offload' }
            ]
          },
          {
            text: '4.4 架构视野与总结（13-16）',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/04_CUDA_and_System_Optimization/4_4' },
              { text: '13. CUDA vs Triton vs PyTorch', link: '/04_CUDA_and_System_Optimization/21_CUDA_vs_Triton_vs_PyTorch' },
              { text: '14. TCO and Hardware Selection', link: '/04_CUDA_and_System_Optimization/22_TCO_and_Hardware_Selection' },
              { text: '15. 推理服务架构设计', link: '/04_CUDA_and_System_Optimization/15_Inference_Service_Architecture_Design' },
              { text: '16. 未来硬件趋势', link: '/04_CUDA_and_System_Optimization/16_Future_Hardware_Trends' }
            ]
          }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/llm-algo-leetcode' }
    ],
    editLink: {
      pattern: 'https://github.com/datawhalechina/llm-algo-leetcode/blob/main/docs/:path'
    },
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present'
    }
  }
})
