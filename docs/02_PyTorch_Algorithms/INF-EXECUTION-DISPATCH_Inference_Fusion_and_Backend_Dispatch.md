# INF-EXECUTION-DISPATCH. Inference Fusion and Backend Dispatch | 推理融合与后端分派

**难度：** Medium | **环境：** CPU-first + 可选单卡 GPU | **标签：** `推理优化`, `融合`, `执行分派`, `回退`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

把三次小计算合成一个 kernel（计算内核），有机会少启动两次 kernel、少写几份中间张量。但计算图里出现可融合模式，并不表示当前 backend（执行后端）一定支持它，也不表示融合后一定更快。shape（形状）、dtype（数据类型）和设备都会影响实际选择。

本节把这件事拆成三个函数：识别 fusion（融合）候选、检查能力、按优先级 dispatch（分派）。学完后你能解释候选为什么 fallback（回退）到 eager（逐算子执行），以及为什么 GPU 实验要分别看输出、执行路径和耗时。

## 前置阅读

- [20 FlashAttention / SDPA](./20_FlashAttention_Sim.md) —— 看懂同一个 attention API 为什么可能走不同 kernel。
- [Part 01 · 19 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md) —— 先理解减少中间张量读写的价值。

### Step 1：痛点与直觉——有融合机会，为什么还是走原来的路径

假设一段计算依次做 `linear_gate → silu → multiply`。在一个简化场景里，它们各启动一次 kernel，后两步还要读取前一步的中间结果。如果后端能把其中的计算合并，就有机会少付启动和搬运成本。但仅凭这三个名字，你还不知道输入是 FP16 还是 FP32、在 CPU 还是 GPU、形状是否受支持，更不能算出加速比。

可以把它理解成安排一条加工路线：先发现三道工序可以一起做，再确认车间的设备和材料规格是否匹配，最后决定送到哪个车间。任何一步都不能跳过。比如候选是 SwiGLU、输入长度 2048，某后端只支持 CUDA 上的 FP16 和长度不超过 4096，那么这次可以尝试；把 dtype 换成 FP32，同一个候选就应该留下明确的拒绝原因。

本节先在 CPU 上用字典模拟这种选择，便于逐条看清理由。Step 5 再比较真实的 eager 和 `torch.compile`：前者验证选择逻辑，后者才测执行与成本，两者的结论不能互相替代。

<img src="../public/02_PyTorch_Algorithms/inf_execution_dispatch_overview.svg" alt="推理图到执行路径的分派链" width="1200" height="520">

### Step 2：机制怎么走——从 9 个算子中找出 3 个候选

给定有序算子列表：

```text
0 embedding → 1 linear_qkv → 2 rope → 3 attention
→ 4 residual_add → 5 rms_norm → 6 linear_gate → 7 silu → 8 multiply
```

从下标 0 开始扫描。`embedding` 没有对应模式，前进一格；下标 1 命中 `qkv_rope`，记录区间 `[1, 3)`，直接跳到 3。按同样办法得到 `[4, 6)` 的 `residual_rmsnorm` 和 `[6, 9)` 的 `swiglu`。右端不包含在区间内，因此命中后把游标设为 `end`，已经归入候选的节点就不会再次使用。如果以后增加一个较短的重叠模式，应先尝试较长模式，避免它提前占走节点。

接着检查 `swiglu`。假设输入为 FP16、CUDA、长度 2048、静态形状；后端声明支持 `swiglu`、FP16、CUDA、最大长度 4096，不支持动态形状。五项都满足，返回 `supported=True`。只把 dtype 改为 FP32，就返回 `unsupported_dtype`；再把形状改为动态，会同时保留 `dynamic_shape_not_supported`，便于定位全部不匹配项。

最后按列表优先级选后端。例如 `backend-a` 不支持 `qkv_rope`，`backend-b` 支持，就选 b 并保留 a 的拒绝原因。如果都不支持，返回 `execution_path='eager'`。这只是“第一个兼容者”，不是自动找到最快者；是否值得优先用它，还要测量。

<details>
<summary>速查表：候选识别、能力检查与性能判断</summary>

| 步骤 | 回答什么 | 尚不能回答什么 |
|---|---|---|
| 模式扫描 | 哪些连续算子可能融合 | 真实图上能否合法融合 |
| 能力检查 | 当前输入是否符合后端声明 | 编译是否成功、实际调用什么 kernel |
| 按优先级分派 | 选谁，拒绝其他路径的原因是什么 | 是否比 eager 更快 |
| GPU 配对测量 | 输出是否一致、路径是否命中、成本是否改善 | 其他形状或完整服务是否也获益 |

</details>

<img src="../public/02_PyTorch_Algorithms/inf_execution_dispatch_candidates.svg" alt="融合候选及其主要约束" width="1200" height="520">

### Step 3：边界与陷阱——兼容、执行成功、更快是三个判断

**静态形状不需要动态能力。** 输入 `dynamic_shape=False` 时，即使后端不支持动态形状也能通过；只有输入确实动态，才要求 `supports_dynamic_shape=True`。因此判断是 `not dynamic_shape or supports_dynamic_shape`，不能无条件检查后一个字段。

**优先级不能被名字或分数偷偷改变。** `[a, b]` 表达调用者先试 a 的意图；两个都兼容时应该选 a。如果业务想先选更快的 b，应根据测量结果调整列表，而不是在此函数中猜测性能。全都不兼容时保留拒绝原因并回退 eager，调用者才知道下一步该改 dtype、形状还是后端。

**字符串扫描不是编译器。** 真实融合还依赖数据依赖、张量布局、alias 和数值语义；本题只用连续字符串模拟模式识别，不能用它证明 QKV/RoPE 等模式在任意实现里都可融合。GPU 实验也不会直接使用这张模式表生成 kernel。

**一次编译成功不等于值得使用。** 首次调用可能付出数百毫秒编译成本，稳态每次只省微秒。假设额外首次成本 500 ms、每次节省 0.02 ms，约需 25,000 次调用才能摊平；这些是假设数字，实际应读取 Step 5 的首调和稳态结果。图中断数、profiler 算子名和端到端耗时需要一起看。

### Step 4：任务——补上候选顺序、五项检查和路径选择

扫描循环、拒绝原因列表和 fallback 返回值已经给出。你只需补上决定每条路径的三段逻辑。

| TODO / 函数 | 要完成什么 | 需要注意 | 对应测试 |
|---|---|---|---|
| TODO 1 · `identify_fusion_candidates` | 将模式按长度降序排列 | 命中后跳到 end，不重复使用节点 | `test_candidate_identification` 检查候选及区间 |
| TODO 2 · `evaluate_backend_capability` | 分别判断模式、dtype、设备、长度与动态形状 | 静态输入无需动态能力；记录所有拒绝项 | `test_backend_capability_gate` 检查成功、dtype 与动态形状失败 |
| TODO 3 · `dispatch_candidate` | 选择第一个兼容报告，无则返回 None | 保持调用方给定的优先级 | `test_dispatch_and_fallback` 检查跳过不兼容后端与全拒绝回退 |


```python
from typing import Dict, List, Sequence

```



```python
FUSION_PATTERNS = {
    ('linear_qkv', 'rope'): 'qkv_rope',
    ('residual_add', 'rms_norm'): 'residual_rmsnorm',
    ('linear_gate', 'silu', 'multiply'): 'swiglu',
    ('logits', 'temperature', 'top_k'): 'sampling_chain',
}


def identify_fusion_candidates(ops: Sequence[str]) -> List[Dict[str, object]]:
    """从有序算子中识别不重叠的连续融合模式。"""
    # ==========================================
    # TODO 1：让长模式先参与匹配，避免短模式提前消费节点。
    # ==========================================
    # ordered_patterns = ???  # 按模式长度降序排列的 (ops, name) 列表。
    raise NotImplementedError('TODO 1：请按模式长度降序排列')
    candidates: List[Dict[str, object]] = []
    index = 0
    while index < len(ops):
        matched = None
        for pattern_ops, pattern_name in ordered_patterns:
            end = index + len(pattern_ops)
            if tuple(ops[index:end]) == pattern_ops:
                matched = {
                    'pattern': pattern_name,
                    'start': index,
                    'end': end,
                    'ops': list(pattern_ops),
                }
                break
        if matched is None:
            index += 1
        else:
            candidates.append(matched)
            index = matched['end']
    return candidates


def evaluate_backend_capability(
    candidate: Dict[str, object],
    workload: Dict[str, object],
    backend: Dict[str, object],
) -> Dict[str, object]:
    """检查候选模式和工作负载是否满足后端能力约束。"""
    # ==========================================
    # TODO 2：每个变量只判断一种后端约束。
    # ==========================================
    # pattern_supported = ???  # 候选模式是否在后端支持集合中。
    # dtype_supported = ???  # 工作负载 dtype 是否受支持。
    # device_supported = ???  # 工作负载设备是否受支持。
    # sequence_supported = ???  # 序列长度是否未超过后端上限。
    # dynamic_shape_supported = ???  # 静态 shape 始终通过；动态 shape 需后端显式支持。
    raise NotImplementedError('TODO 2：请分别判断五类能力条件')

    checks = {
        'unsupported_pattern': pattern_supported,
        'unsupported_dtype': dtype_supported,
        'unsupported_device': device_supported,
        'sequence_too_long': sequence_supported,
        'dynamic_shape_not_supported': dynamic_shape_supported,
    }
    reasons = [reason for reason, passed in checks.items() if not passed]
    return {
        'backend': str(backend.get('name', 'unknown')),
        'supported': not reasons,
        'reasons': reasons,
    }


def dispatch_candidate(
    candidate: Dict[str, object],
    workload: Dict[str, object],
    backends: List[Dict[str, object]],
) -> Dict[str, object]:
    """选择首个兼容后端；全部失败时回退 eager 并保留原因。"""
    capability_reports = [
        evaluate_backend_capability(candidate, workload, backend)
        for backend in backends
    ]
    # ==========================================
    # TODO 3：按列表优先级选择第一个 supported 报告。
    # ==========================================
    # selected = ???  # 能力报告中第一个 supported 项；不存在时为 None。
    raise NotImplementedError('TODO 3：请选择首个兼容后端')
    rejected = {
        report['backend']: list(report['reasons'])
        for report in capability_reports
        if not report['supported']
    }
    if selected is not None:
        return {
            'execution_path': selected['backend'],
            'decision': 'optimized',
            'rejected_backends': rejected,
        }
    return {
        'execution_path': 'eager',
        'decision': 'fallback',
        'rejected_backends': rejected,
    }
```


### 测试

- `test_candidate_identification` 检查三种候选与半开区间，确认没有重复消费节点。
- `test_backend_capability_gate` 检查支持路径，以及 dtype、动态形状的独立拒绝原因。
- `test_dispatch_and_fallback` 检查跳过不兼容后端、全部失败时返回 eager 并保留原因。

现有测试覆盖这些代表性分支；CPU 通过不代表真实编译器已生成融合 kernel。


```python
def test_candidate_identification():
    """验证连续模式、最长匹配和不重叠消费。"""
    ops = ['embedding', 'linear_qkv', 'rope', 'attention', 'residual_add', 'rms_norm', 'linear_gate', 'silu', 'multiply']
    candidates = identify_fusion_candidates(ops)
    assert [item['pattern'] for item in candidates] == ['qkv_rope', 'residual_rmsnorm', 'swiglu']
    assert candidates[0]['start'] == 1 and candidates[0]['end'] == 3
    assert candidates[-1]['ops'] == ['linear_gate', 'silu', 'multiply']


def test_backend_capability_gate():
    """验证 dtype 与动态 shape 会独立阻止候选命中。"""
    candidate = {'pattern': 'swiglu'}
    workload = {'dtype': 'float16', 'device': 'cuda', 'seq_len': 2048, 'dynamic_shape': False}
    backend = {'name': 'compiled', 'patterns': ['swiglu'], 'dtypes': ['float16'], 'devices': ['cuda'], 'max_seq_len': 4096, 'supports_dynamic_shape': False}
    assert evaluate_backend_capability(candidate, workload, backend)['supported'] is True
    dtype_failure = evaluate_backend_capability(candidate, {**workload, 'dtype': 'float32'}, backend)
    assert dtype_failure['supported'] is False and 'unsupported_dtype' in dtype_failure['reasons']
    dynamic_failure = evaluate_backend_capability(candidate, {**workload, 'dynamic_shape': True}, backend)
    assert 'dynamic_shape_not_supported' in dynamic_failure['reasons']


def test_dispatch_and_fallback():
    """验证跳过不兼容 backend，并在全部失败时回退 eager。"""
    candidate = {'pattern': 'qkv_rope'}
    workload = {'dtype': 'float16', 'device': 'cuda', 'seq_len': 1024, 'dynamic_shape': False}
    unsupported = {'name': 'backend-a', 'patterns': ['swiglu'], 'dtypes': ['float16'], 'devices': ['cuda'], 'max_seq_len': 4096, 'supports_dynamic_shape': True}
    supported = {**unsupported, 'name': 'backend-b', 'patterns': ['qkv_rope']}
    selected = dispatch_candidate(candidate, workload, [unsupported, supported])
    assert selected['execution_path'] == 'backend-b'
    assert 'backend-a' in selected['rejected_backends']
    fallback = dispatch_candidate(candidate, workload, [unsupported])
    assert fallback['execution_path'] == 'eager' and fallback['decision'] == 'fallback'


def test_inference_execution_dispatch():
    """汇总融合候选、能力门槛和执行回退测试。"""
    try:
        test_candidate_identification()
        test_backend_capability_gate()
        test_dispatch_and_fallback()
        print('✅ 推理融合与后端分派测试通过：候选、能力门槛和回退均通过。')
    except NotImplementedError:
        raise
    except (NameError, AttributeError, TypeError, ValueError, AssertionError) as error:
        raise NotImplementedError('请先完成 TODO 代码或检查字段名！') from error


test_inference_execution_dispatch()
```


---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果遇到困难，再向下查看参考代码与解析。
<br><br><br><br><br><br><br><br><br><br>

---

## 参考代码与解析

### 代码


```python
FUSION_PATTERNS = {
    ('linear_qkv', 'rope'): 'qkv_rope',
    ('residual_add', 'rms_norm'): 'residual_rmsnorm',
    ('linear_gate', 'silu', 'multiply'): 'swiglu',
    ('logits', 'temperature', 'top_k'): 'sampling_chain',
}


# TODO 1：识别不重叠融合候选。
def identify_fusion_candidates(ops: Sequence[str]) -> List[Dict[str, object]]:
    """从有序算子中识别不重叠的连续融合模式。"""
    ordered_patterns = sorted(FUSION_PATTERNS.items(), key=lambda item: len(item[0]), reverse=True)
    candidates: List[Dict[str, object]] = []
    index = 0
    while index < len(ops):
        matched = None
        for pattern_ops, pattern_name in ordered_patterns:
            end = index + len(pattern_ops)
            if tuple(ops[index:end]) == pattern_ops:
                matched = {
                    'pattern': pattern_name,
                    'start': index,
                    'end': end,
                    'ops': list(pattern_ops),
                }
                break
        if matched is None:
            index += 1
        else:
            candidates.append(matched)
            index = matched['end']
    return candidates


# TODO 2：逐项检查后端能力。
def evaluate_backend_capability(
    candidate: Dict[str, object],
    workload: Dict[str, object],
    backend: Dict[str, object],
) -> Dict[str, object]:
    """检查候选模式和工作负载是否满足后端能力约束。"""
    pattern_supported = candidate.get('pattern') in backend.get('patterns', [])
    dtype_supported = workload.get('dtype') in backend.get('dtypes', [])
    device_supported = workload.get('device') in backend.get('devices', [])
    sequence_supported = int(workload.get('seq_len', 0)) <= int(backend.get('max_seq_len', 0))
    dynamic_shape_supported = (
        not bool(workload.get('dynamic_shape', False))
        or bool(backend.get('supports_dynamic_shape', False))
    )
    checks = {
        'unsupported_pattern': pattern_supported,
        'unsupported_dtype': dtype_supported,
        'unsupported_device': device_supported,
        'sequence_too_long': sequence_supported,
        'dynamic_shape_not_supported': dynamic_shape_supported,
    }
    reasons = [reason for reason, passed in checks.items() if not passed]
    return {
        'backend': str(backend.get('name', 'unknown')),
        'supported': not reasons,
        'reasons': reasons,
    }


# TODO 3：选择首个兼容执行路径。
def dispatch_candidate(
    candidate: Dict[str, object],
    workload: Dict[str, object],
    backends: List[Dict[str, object]],
) -> Dict[str, object]:
    """选择首个兼容后端；全部失败时回退 eager 并保留原因。"""
    capability_reports = [
        evaluate_backend_capability(candidate, workload, backend)
        for backend in backends
    ]
    selected = next((report for report in capability_reports if report['supported']), None)
    rejected = {
        report['backend']: list(report['reasons'])
        for report in capability_reports
        if not report['supported']
    }
    if selected is not None:
        return {
            'execution_path': selected['backend'],
            'decision': 'optimized',
            'rejected_backends': rejected,
        }
    return {
        'execution_path': 'eager',
        'decision': 'fallback',
        'rejected_backends': rejected,
    }

```


### 解析

#### TODO 1：先尝试长模式

**思路：** 把模式表变成有序列表，每个扫描位置先尝试包含更多算子的模式。

**关键一行：**
```python
ordered_patterns = sorted(FUSION_PATTERNS.items(), key=lambda item: len(item[0]), reverse=True)
```

**为什么这样写：** `item[0]` 是算子元组，长度就是模式包含的节点数。若同时有 `linear_gate → silu` 和三节点的 SwiGLU 模式，先匹配短模式就会把前两项消费掉。后面的循环在命中时跳到 `end`，负责保证不重叠。

**常见错法：** 按模式名称排序，或命中后仍只将 index 加一。前者没有表达长度优先，后者可能让同一节点重复归属。

#### TODO 2：独立判断能力条件

**思路：** 每个布尔变量只回答一个问题，然后汇总所有失败项。

**关键一行：**
```python
dynamic_shape_supported = not bool(workload.get('dynamic_shape', False)) or bool(backend.get('supports_dynamic_shape', False))
```

**为什么这样写：** 静态输入不要求动态支持；动态输入才看后端声明。其他条件分别做集合成员检查和长度上限比较。这样 FP32 和动态形状同时不支持时，可以一次返回两个原因。

**常见错法：** 用五个条件的 `or` 决定整体是否支持。那样只要设备正确，即使 dtype 不支持也会通过；整体支持必须是所有条件均满足。

#### TODO 3：选择第一个兼容后端

**思路：** 保持输入顺序，找到第一个 `supported` 为真的报告；没有就用 None 表示。

**关键一行：**
```python
selected = next((report for report in capability_reports if report['supported']), None)
```

**为什么这样写：** `next` 的默认值让空列表和全部拒绝走同一条 eager 回退路径。拒绝原因已由骨架保存，选择成功也不需要删掉其他后端的失败信息。

**常见错法：** 循环中每遇到兼容报告就覆盖 selected，最后得到的是最后一个兼容者；或者全部失败时抛异常，丢掉本题要求的 fallback 结果。

### Step 5：GPU 机制实验——形状变化下的 eager / compile 分派

CPU 练习只验证选择逻辑。这里用真实的 Residual + RMSNorm + SwiGLU 合成块比较 eager 与编译执行，测输出、首次调用和稳态耗时；它不能直接证明完整模型或在线服务也加速。

环境创建与预检见[推理优化实验 Quickstart](../topic_discussion/inference_optimization/quickstart.md#inf-cuda-mechanism-quickstart)，页面和结果字段见[推理优化实验资产表](../topic_discussion/inference_optimization/experiment_assets.md)。实验在固定数据类型、隐藏维度和块深度下改变 Token 数，对同一 Residual + RMSNorm + SwiGLU 计算进行 eager / `torch.compile` 配对测量。

<details>
<summary>速查表：GPU 实验分别回答什么</summary>

| 对照目标 | 固定条件 | 观察结果 |
|---|---|---|
| 路径正确性 | 同一权重、输入和容差 | 输出一致性、编译失败 |
| 形状敏感性 | 数据类型、隐藏维度、块深度、预热和重复次数 | Median、P95、加速比 |
| 分派条件 | CPU 题目使用相同的模式、数据类型、设备与形状条件 | 图数量、图中断、主要 GPU 算子、接受或回退的形状 |
| 执行路径 | 同一编译对象按形状触发编译或缓存命中 | 首次调用成本、稳态延迟、峰值显存 |
| 资源代价 | 同一进程和设备 | 峰值已分配显存 |

</details>

#### 5.1 环境与 workload

先按 [Quickstart](../topic_discussion/inference_optimization/quickstart.md#inf-cuda-mechanism-quickstart) 创建并预检 `INF-CUDA-MECHANISM`。默认只登记计划；真实运行时仅修改 `RUN_MODE` 和 `RUN_LABEL`，其余配置共同定义同一组分派条件。


| 配置 | 本节设置 |
|---|---|
| workload（输入与运行条件） | 同一 FP16 合成块，四层，token 数 1/32/256/1024 |
| 对照 | 相同权重和输入的 eager / torch.compile |
| 测量 | 首次调用、预热后的 Median/P95、输出误差、峰值显存和 profiler 算子 |
| 实际参数 | 以下配置单元为运行依据；修改形状或 dtype 后应保存为新的实验记录 |


```python
# GPU 实验默认关闭；完成 INF-CUDA-MECHANISM 预检后再启用。
RUN_MODE = 'dry_run'  # 默认 'dry_run'；真实执行时改为 'real_gpu'。
RUN_LABEL = 'default'  # 默认 'default'；正式复测改为唯一标签，避免覆盖结果。
PROFILE_ID = 'INF-CUDA-MECHANISM'  # 默认且唯一支持的单卡机制环境。
WORKLOAD_ID = 'inference-execution-dispatch'  # 稳定工作负载身份。
WORKLOAD_VERSION = 'v3'  # v3 分离首次编译成本与稳态执行证据。
GPU_RESULT_PATH = f'benchmarks/results/inf_execution_dispatch_compile_{RUN_LABEL}.json'

GPU_DTYPE = 'float16'  # 默认 float16；同组候选必须保持一致。
GPU_TOKEN_COUNTS = [1, 32, 256, 1024]  # 默认覆盖解码、小批、预填充与较大预填充负载。
GPU_HIDDEN_SIZE = 1024  # 默认隐藏维度；调大前先检查显存。
GPU_BLOCK_DEPTH = 4  # 默认堆叠四个相同结构块，使融合收益能够摊薄分派开销。
GPU_WARMUP = 10  # 默认每条路径预热 10 次，不计入延迟。
GPU_REPEATS = 30  # 默认每条路径测量 30 次，用于 median 和 P95。
GPU_ATOL = 2e-2  # 默认绝对误差容限，适配 float16 累积误差。
GPU_RTOL = 2e-2  # 默认相对误差容限，eager 与 compile 共用。
GPU_MIN_SPEEDUP = 1.05  # 默认至少提升 5%，才把该形状视为稳定收益候选。
GPU_MAX_P95_RATIO = 1.05  # 默认允许编译路径 P95 最多回退 5%。

```


#### 5.2 执行并保存 JSON


```python
import json
import math
import statistics
import time
from pathlib import Path

from tools.project_runtime import resolve_project_root

GPU_DISPATCH_REPORT = None
if RUN_MODE == 'real_gpu':
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        raise RuntimeError('本实验需要 CUDA GPU。')
    if not hasattr(torch, 'compile'):
        raise RuntimeError('当前 PyTorch 不提供 torch.compile。')
    dtype = getattr(torch, GPU_DTYPE)

    class ResidualSwiGLUBlock(nn.Module):
        """构造可观察归一化、门控激活与残差写回的推理片段。"""

        def __init__(self, hidden_size):
            super().__init__()
            self.norm = nn.RMSNorm(hidden_size)
            self.gate = nn.Linear(hidden_size, hidden_size, bias=False)
            self.up = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, x):
            normalized = self.norm(x)
            return x + torch.nn.functional.silu(self.gate(normalized)) * self.up(normalized)

    class ResidualSwiGLUStack(nn.Module):
        """堆叠固定深度的计算片段，观察编译收益能否摊薄分派开销。"""

        def __init__(self, hidden_size, depth):
            super().__init__()
            self.blocks = nn.ModuleList([ResidualSwiGLUBlock(hidden_size) for _ in range(depth)])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    def percentile(values, ratio):
        """返回离散样本的最近秩分位数。"""
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, math.ceil(ratio * len(ordered)) - 1))
        return ordered[index]

    def measure(fn, x):
        """使用 CUDA Event 测量稳态延迟，并记录该路径的显存峰值。"""
        with torch.inference_mode():
            for _ in range(GPU_WARMUP):
                fn(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            values = []
            for _ in range(GPU_REPEATS):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn(x)
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end))
        return {
            'median_ms': statistics.median(values),
            'p95_ms': percentile(values, 0.95),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        }

    def profile_operator_keys(fn, x):
        """返回具有设备执行时间的主要算子，排除编译控制事件。"""
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.inference_mode(), torch.profiler.profile(activities=activities) as profile:
            fn(x)
            torch.cuda.synchronize()
        events = sorted(
            profile.key_averages(),
            key=lambda event: getattr(event, 'self_device_time_total', getattr(event, 'self_cuda_time_total', 0.0)),
            reverse=True,
        )
        excluded = ('dynamo', 'compile', 'cache lookup')
        return [event.key for event in events if not any(term in event.key.lower() for term in excluded)][:10]

    model = ResidualSwiGLUStack(GPU_HIDDEN_SIZE, GPU_BLOCK_DEPTH).cuda().to(dtype).eval()
    compiled = torch.compile(model, fullgraph=True, dynamic=False)
    rows = []
    failures = []
    for token_count in GPU_TOKEN_COUNTS:
        x = torch.randn(token_count, GPU_HIDDEN_SIZE, device='cuda', dtype=dtype)
        try:
            with torch.inference_mode():
                eager_output = model(x)
                compile_start = time.perf_counter()
                compiled_output = compiled(x)
                torch.cuda.synchronize()
                first_call_ms = (time.perf_counter() - compile_start) * 1000
                output_close = torch.allclose(
                    eager_output, compiled_output, atol=GPU_ATOL, rtol=GPU_RTOL
                )

                explanation = torch._dynamo.explain(model)(x)
                graph_count = int(getattr(explanation, 'graph_count', 0))
                graph_break_count = int(getattr(explanation, 'graph_break_count', 0))

                eager_metrics = measure(model, x)
                compiled_metrics = measure(compiled, x)
                eager_operator_keys = profile_operator_keys(model, x)
                compiled_operator_keys = profile_operator_keys(compiled, x)

            speedup = eager_metrics['median_ms'] / compiled_metrics['median_ms']
            p95_ratio = compiled_metrics['p95_ms'] / eager_metrics['p95_ms']
            rows.append({
                'token_count': token_count,
                'output_close': bool(output_close),
                'graph_count': graph_count,
                'graph_break_count': graph_break_count,
                'first_call_ms': first_call_ms,
                'eager_operator_keys': eager_operator_keys,
                'compiled_operator_keys': compiled_operator_keys,
                'eager': eager_metrics,
                'compiled': compiled_metrics,
                'speedup': speedup,
                'p95_ratio': p95_ratio,
                'benefit_gate_passed': speedup >= GPU_MIN_SPEEDUP and p95_ratio <= GPU_MAX_P95_RATIO,
            })
        except Exception as error:
            failures.append({'token_count': token_count, 'type': type(error).__name__, 'message': str(error)})

    correctness_passed = bool(rows) and all(row['output_close'] for row in rows)
    captured_without_breaks = bool(rows) and all(
        row['graph_count'] >= 1 and row['graph_break_count'] == 0 for row in rows
    )
    accepted_cases = sum(row['benefit_gate_passed'] for row in rows)
    decision = (
        'accept_for_selected_shapes'
        if correctness_passed and captured_without_breaks and accepted_cases > 0 and not failures
        else 'tune_or_fallback'
    )
    GPU_DISPATCH_REPORT = {
        'schema_version': 'inference-mechanism-gpu/v3',
        'semantic_id': 'INF-EXECUTION-DISPATCH',
        'profile_id': PROFILE_ID,
        'run_label': RUN_LABEL,
        'workload': {
            'id': WORKLOAD_ID,
            'version': WORKLOAD_VERSION,
            'token_counts': GPU_TOKEN_COUNTS,
            'hidden_size': GPU_HIDDEN_SIZE,
            'block_depth': GPU_BLOCK_DEPTH,
            'dtype': GPU_DTYPE,
            'warmup': GPU_WARMUP,
            'repeats': GPU_REPEATS,
        },
        'hardware': {
            'device': torch.cuda.get_device_name(0),
            'torch_version': torch.__version__,
            'cuda': torch.version.cuda,
        },
        'candidate_matrix': rows,
        'summary': {
            'correctness_passed': correctness_passed,
            'captured_without_breaks': captured_without_breaks,
            'accepted_cases': accepted_cases,
            'total_cases': len(GPU_TOKEN_COUNTS),
            'minimum_speedup': GPU_MIN_SPEEDUP,
            'maximum_p95_ratio': GPU_MAX_P95_RATIO,
        },
        'failure': failures or None,
        'evidence_level': 'local_mechanism_benchmark',
        'decision': decision,
    }
    output = resolve_project_root(Path.cwd()) / '02_PyTorch_Algorithms' / GPU_RESULT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(GPU_DISPATCH_REPORT, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'结果已保存：{output}')
else:
    print("GPU 实验默认关闭；INF-CUDA-MECHANISM 预检通过后将 RUN_MODE 改为 'real_gpu'。")

```


#### 5.3 读取结果


```python
import json
from pathlib import Path

from tools.project_runtime import resolve_project_root

result_path = resolve_project_root(Path.cwd()) / '02_PyTorch_Algorithms' / GPU_RESULT_PATH
if result_path.exists():
    report = json.loads(result_path.read_text(encoding='utf-8'))
    print({key: report.get(key) for key in ('workload', 'hardware', 'summary', 'failure', 'evidence_level', 'decision')})
    print()
    print('形状 × 执行路径摘要')
    for row in report.get('candidate_matrix', []):
        print({
            'token_count': row['token_count'],
            'output_close': row['output_close'],
            'graph_break_count': row['graph_break_count'],
            'first_call_ms': round(row['first_call_ms'], 2),
            'eager_median_ms': round(row['eager']['median_ms'], 4),
            'compiled_median_ms': round(row['compiled']['median_ms'], 4),
            'compiled_p95_ms': round(row['compiled']['p95_ms'], 4),
            'speedup': round(row['speedup'], 4),
            'p95_ratio': round(row['p95_ratio'], 4),
            'benefit_gate_passed': row['benefit_gate_passed'],
            'compiled_operator_keys': row['compiled_operator_keys'],
        })
else:
    print(f'尚未找到结果文件：{result_path}。先运行 5.2。')

```


#### 5.4 解释结果并作出决策

下表保留已登记的 `rtx5070ti_dispatch_v3` 历史摘要，复用结论时需核对当前环境与输入。

先检查输出一致性和失败项，再比较各 Token 规模的首次调用成本、图中断、Median、P95 与峰值显存。`torch.compile` 成功只证明编译分派可用；只有稳态收益通过门槛的形状才进入候选分桶。

| 观察项 | 读取字段 | 当前结果 | 解释与动作 |
|:---|:---|:---|:---|
| 运行与条件 | `failure`、workload、Token 数、块深度 | `rtx5070ti_dispatch_v3`：四层、Token `1/32/256/1024`，无运行失败 | 结论只适用于当前 FP16 合成块和固定形状 |
| 正确性与路径 | `output_close`、图数量、图中断、算子名 | 四档输出一致，均为单图且无图中断；编译路径出现 Triton 融合算子 | 编译成功证明路径可用，但不能单独证明值得分派 |
| 资源与性能 | Median、P95、`speedup`、`p95_ratio` | Token 1/32 为 `1.12×/1.33×` 且通过门槛；256 仅 `1.03×`；1024 的 P95 回退约 `30%` | 只把 1/32 作为稳态候选，256/1024 保留 eager |
| 首次调用与决策 | `first_call_ms`、`accepted_cases`、`decision` | 首次调用约 `454–658 ms`；2/4 形状通过；`accept_for_selected_shapes` | 该接受结论仅指高复用稳态路径；1/32 约需 3.1 万/1.8 万次调用摊平首次成本 |

<a id="cheatsheet"></a>
## 速查卡

| | |
|:--|:--|
| **一句话机制** | 从算子序列找候选，逐项检查输入与后端能力，选择第一个兼容者；都不兼容则回退 eager 并保留原因。 |
| **关键公式与数量级** | 候选区间是 `[start,end)`；整体兼容要求五项检查均通过。摊平调用数约为额外首次成本 / 单次稳态节省，只有稳态确有节省时才有意义。 |
| **常见坑** | 1. 字符串命中就当作真实融合。2. 静态输入误要求动态支持。3. 选择最后而非第一个兼容者。4. 编译成功就认定更快，忽略首调成本与 P95。 |
| **延伸阅读** | 专题 [02 Prefill](../topic_discussion/inference_optimization/02_prefill_and_attention_kernel.md) · [机制索引](../topic_discussion/inference_optimization/casebook.md#机制速查索引) · 下一步 [21 解码策略](./21_Decoding_Strategies.md) |

## 相关阅读

- [PyTorch torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html) —— 编译参数与使用限制。
- [PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) —— 注意力后端选择。
- [TorchInductor GPU Profiling](https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html) —— 确认实际执行路径。