# 20. FlashAttention Sim | FlashAttention 模拟

**难度：** Hard | **环境：** CPU + 单卡 GPU | **标签：** `推理优化`, `Attention`, `FlashAttention`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

标准 attention 要先算出完整的分数矩阵 $S = QK^T$，再对每一行做 softmax。$S$ 的大小随序列长度 $N$ 平方增长：32k 上下文、BF16 下，**每个 head** 的分数矩阵就是 2 GiB，32 个 head 就是 64 GiB——而它只是个中间结果，算完就扔。更糟的是这块矩阵要先写进显存（HBM）再读回来做 softmax，带宽被白白消耗两遍。

FlashAttention 的做法是不让这个矩阵完整出现：把 Q、K、V 切成小块（tile），一小块一小块地算分数，边算边更新 softmax 需要的统计量，直到所有 K/V 块都看过，输出和标准 attention 在精确算术下相等，浮点实现允许舍入误差。本节用纯 PyTorch 把这条路径模拟出来——核心是 online softmax（在线 softmax）：在只看到一部分分数时，怎样维护"最大值"和"指数和"这两个量，使得新块到来时能把旧结果修正过来而不必重算。学完后你能解释为什么分块不丢信息、GPU 上的 SDPA 什么时候真的走到了 FlashAttention kernel，以及本节历史实验在长度 4096 的指定配置下为何减少约 95% 的峰值已分配显存。

## 前置阅读

- [Part 01 · 14 FlashAttention 显存模型](../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md) —— 先在纸上算一遍完整分数矩阵和一个 tile 各占多少字节，本节的代码就是把这张账变成循环。
- [Part 01 · 03 GPU 物理架构与内存层级](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md) —— HBM 和片上 SRAM 差一个数量级的带宽，这是"不落地分数矩阵"值钱的根本原因。

### Step 1：痛点与直觉——那张算完就扔的 $N \times N$ 矩阵

想象你在给一个 32 头、32k 上下文的模型做 prefill（预填充）。标准 attention 对每个 head 先算 $S = QK^T$，这是一个 $32768 \times 32768$ 的矩阵，BF16 下 2 GiB；32 个 head 同时算就是 64 GiB，快赶上一张 80 GB H100 的全部显存。它不是模型要输出的东西，softmax 一做完、乘上 $V$ 就没用了，却要先完整写进 HBM，再读回来做 softmax，softmax 得到的概率矩阵还要读入来乘 $V$。FlashAttention 论文给的数字：A100 的 HBM 带宽 1.5–2 TB/s，而片上 SRAM 约 19 TB/s——差了十倍。这解释了减少中间写回的动机；具体有多少时间用于搬运，还要结合形状和 profiler 判断。

直觉上的解法很简单：别把整个矩阵算出来，把 K/V 切成小块，一次只算一个 Q 块对一个 K 块的局部分数——比如 128 × 128，仅分数 tile 在 BF16 下就需 32 KiB；真实 kernel 还要为 Q/K/V 和统计量安排片上空间。可问题立刻来了：softmax 是按整行归一化的，分母 $\sum_j e^{s_j}$ 要把一行所有分数加完才知道。只看到前 128 个分数，怎么做 softmax？

答案是 online softmax：你不需要一开始就知道正确的分母，只要记住到目前为止的"行最大值"和"指数和"，等新的一块分数来了，用一个指数因子把旧的统计量修正到新基准上，再把新块加进去。换个说法，这像是你在分批收简历打分，每批到了都重新定一下"目前最高分"，把之前算好的相对分按新标准折一下——不用把旧简历再看一遍。全部块看完，在精确算术下，分母和输出就与一次性计算一致；浮点累加顺序会带来误差。

本节你要实现的就是这个循环：外层遍历 Q 块，内层遍历 K/V 块，每一步维护三个逐行状态——最大值 $m$、指数和 $l$、累计输出 $O$——并在因果 mask 和尾块这些边界上保持正确。

<img src="../public/02_PyTorch_Algorithms/20_flashattention_tiling.svg" alt="FlashAttention 分块图" width="1600" height="780">

### Step 2：机制怎么走——一行 4 个分数、切成两块手算

把问题缩到最小：某一个 query 行对 4 个 key 的分数是 $s = [1, 3, 2, 5]$，对应的 $V$ 取标量 $[10, 20, 30, 40]$，分块大小 2。

**直接算（参照）.** 行最大值 5，$l = e^{-4} + e^{-2} + e^{-3} + e^{0} = 1.2034$，softmax 概率 $[0.015, 0.112, 0.041, 0.831]$，输出 $O = \sum p_j v_j = 36.88$。

**分块算.** 先看第一块 $[1, 3]$：
- $m_1 = 3$，$l_1 = e^{1-3} + e^{3-3} = 1.1353$，$O_1 = (e^{-2} \cdot 10 + 1 \cdot 20)/1.1353 = 18.81$。

这时 $O_1$ 是"只看过前两个 key"的 attention 输出，分母是错的（少了后两项）。再来第二块 $[2, 5]$：
- 这一块的最大值是 5，比 $m_1 = 3$ 大，所以新基准 $m_{new} = 5$；
- 旧统计量要折到新基准：history_scale $= e^{m_1 - m_{new}} = e^{-2} = 0.1353$；
- 新块相对新基准的权重：tile_weights $= [e^{2-5}, e^{5-5}] = [0.0498, 1]$；
- 新分母：$l_{new} = l_1 \cdot 0.1353 + (0.0498 + 1) = 0.1536 + 1.0498 = 1.2034$——和直接算的分母**完全一样**；
- 新输出：$O_{new} = O_1 \cdot \dfrac{l_1 \cdot 0.1353}{l_{new}} + \dfrac{0.0498 \cdot 30 + 1 \cdot 40}{l_{new}} = 2.40 + 34.48 = 36.88$——也一样。

为什么能对上？因为 $e^{s_j - m_1} \cdot e^{m_1 - m_{new}} = e^{s_j - m_{new}}$：旧的指数项乘上一个统一的因子，就等于以新基准重新算了一遍，但不需要再碰旧的分数。这就是 online softmax 的全部数学。

**如果不这么做会怎样.** 一个常见的错误直觉是"每块各自 softmax，最后平均"：第一块 softmax 后加权得 18.81，第二块得 39.52，平均 29.17——和 36.88 差得远。两块各自归一化时分母不同，简单平均仍能形成合法分布，但它强行给每块一半质量，不是整行 softmax 所定义的分布。

**三个逐行状态.** 实现时对 Q 块里的每一行都要维护 $m$、$l$、$O$，其中 $m$、$l$ 是 `[rows, 1]` 的列向量，$O$ 是 `[rows, value_dim]`；每一行有自己的最大值、分母和一个输出向量。

<details>
<summary>速查表：合并一个新 K/V 块时三个状态怎么变</summary>

| 状态 | 含义 | 合并新块时 |
|---|---|---|
| $m$（行最大值） | 到目前为止看过的最大分数，防 $e^{x}$ 溢出 | $m_{new} = \max(m_{old}, \max(S_{tile}))$ |
| $l$（指数和） | softmax 分母的累计 | $l_{new} = l_{old} \cdot e^{m_{old} - m_{new}} + \sum e^{S_{tile} - m_{new}}$ |
| $O$（累计输出） | 已看过 key 的加权 $V$ | $O_{new} = O_{old} \cdot \dfrac{l_{old} e^{m_{old}-m_{new}}}{l_{new}} + \dfrac{e^{S_{tile}-m_{new}} V_{tile}}{l_{new}}$ |

</details>

<img src="../public/02_PyTorch_Algorithms/20_online_softmax_state.svg" alt="在线 Softmax 状态更新图" width="1200" height="520">

### Step 3：边界与陷阱——因果 mask、尾块和 $-\infty$

**陷阱一：因果 mask 用了块内下标.** decoder 的 attention 只能看当前位置及之前的 key。分块后 Q 块从位置 $i$ 开始、K 块从位置 $j$ 开始，判断"这个 key 在未来吗"必须用**全局**位置 $j + c > i + r$，而不是块内的 $c > r$。比如 Q 块覆盖位置 2–3、K 块覆盖位置 0–1：块内下标会错误地遮掉 key 1 对 query 0 的贡献，而实际上位置 1 在位置 2 之前，完全可见。`test_causal_equivalence` 用 6 个位置、块大小 2 正好跨了块边界，用局部下标一定过不了。

**陷阱二：尾块比 `block_size` 短.** 序列长度 5、块大小 3 时，最后一块只有 2 行。代码里所有切片都用 `q[i:i+block_size]`，PyTorch 会自动截断，但如果你手写了 `torch.zeros(block_size, ...)` 之类的固定形状缓冲，尾块就会错位。`test_noncausal_equivalence` 里 `(5, 3, 3)` 这一组专门测这个。

**陷阱三：$-\infty - (-\infty) = \text{NaN}$.** $m$ 初始化为 $-\infty$。如果某一行在某个 K 块里所有 key 都被因果 mask 遮掉，那一块的 tile_max 也是 $-\infty$；此时若 $m_{old}$ 还没被任何有限值更新过，history_scale $= e^{-\infty - (-\infty)}$ 就是 NaN，污染整行。本节的循环按 $j = 0, 1, 2, \dots$ 顺序遍历 K 块，第一块总包含位置 0，对任何 query 都可见，所以 $m$ 在第一块后一定变成有限值——但如果你改成倒序遍历或并行归约，就要显式处理这个情况。真实 FlashAttention kernel 里有专门的分支。

**顺带一提：FlashAttention-2 的小改动.** 本节每合并一块都把 $O$ 除以 $l_{new}$ 归一化一次。FA-2 的做法是循环里只维护**未归一化**的 $\tilde O_{new} = \tilde O_{old} \cdot e^{m_{old}-m_{new}} + e^{S_{tile}-m_{new}} V_{tile}$，最后一次性除以 $l$。数学上等价，但每块少一次除法和一次缩放——这些是跑在 CUDA core 上的非矩阵乘运算，可能限制 kernel 的效率。

<details>
<summary>延伸：FlashAttention 1 → 2 → 3 分别解决了什么</summary>

| 版本 | 年份 | 主要改动 | 直观效果 |
|---|---|---|---|
| FlashAttention-1 | 2022 | tiling + online softmax，反向用重计算代替存分数矩阵 | 避免保存 $O(N^2)$ attention 中间量；Q/K/V 与输出仍随 $N$ 增长，其他显存开销仍可能 OOM |
| FlashAttention-2 | 2023 | 减少非矩阵乘运算（如上面的"最后归一化"）；沿序列维度并行；重新划分 warp 间工作 | 论文报告在其测试配置下可接近 FA-1 的 2×；实际收益依赖形状与硬件 |
| FlashAttention-3 | 2024 | 面向 Hopper：WGMMA 异步矩阵乘、TMA 硬件搬运、ping-pong 流水把 softmax 和矩阵乘重叠，支持 FP8 | H100 上进一步提升利用率 |

本节模拟的是 FA-1 的算法骨架；版本之间改的是 kernel 组织，不是数学。

</details>

<img src="../public/02_PyTorch_Algorithms/20_q_tile_lifecycle.svg" alt="Q 分块生命周期" width="1200" height="520">

### Step 4：任务——补全双层循环里的状态更新

题目区已经写好输入校验、外层 Q 块循环、内层 K/V 块循环和状态写回，你要补的是循环体里的四段逻辑，正好对应 Step 2 手算的每一步。

| TODO | 要做什么 | 必须满足 | 对应测试 |
|:---|:---|:---|:---|
| TODO 1 · `m`、`l` 初始化 | 建 `m`（`[seq_len, 1]`，填 $-\infty$）和 `l`（`[seq_len, 1]`，填 0） | 保留列维以便逐行广播；dtype / device 与 `q` 一致 | `test_noncausal_equivalence` |
| TODO 2 · `scores`、`future_mask` | `scores = q_block @ k_block.T`；因果时用**全局**位置判断 `key_pos > query_pos` | 不能用块内下标 | `test_causal_equivalence`（跨块边界） |
| TODO 3 · `new_max`、`history_scale`、`tile_weights` | 算 `tile_max`、`new_max`、`history_scale = exp(m_i - new_max)`、`tile_weights = exp(scores - new_max)` | 历史和当前都对齐到同一个 `new_max` | `test_dtype_and_stability`（分数全为 100 时输出仍有限） |
| TODO 4 · `new_l`、`out_i` | `new_l = l_i * history_scale + tile_weights.sum(-1)`；`out_i` 按 Step 2 的公式合并 | 分子分母用同一个 `history_scale` | `test_noncausal_equivalence`（多组形状含尾块） |


```python
import torch
import math
```



```python
def flash_attention_forward_sim(q, k, v, block_size=2, causal=False):
    """计算二维输入上的 FlashAttention 前向模拟。

    Args:
        q, k, v: [seq_len, dim] 张量，device 和 dtype 应保持一致。
        block_size: Q/K/V 的分块大小，必须为正数。
        causal: 是否只允许关注当前位置及之前的 K 位置。

    Returns:
        [seq_len, dim] 的注意力输出。

    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError('q、k、v 必须是 [seq_len, dim] 二维张量')
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError('q、k、v 的形状必须一致')
    if q.device != k.device or k.device != v.device:
        raise ValueError('q、k、v 必须位于同一 device')
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise TypeError('q、k、v 必须使用相同 dtype')
    if block_size <= 0:
        raise ValueError('block_size 必须为正数')

    seq_len, dim = q.shape
    
    out = torch.zeros((seq_len, dim), device=q.device, dtype=q.dtype)
    # ==========================================
    # TODO 1：初始化跨分块状态
    # ==========================================
    # m = ???  # 每行最大值基准；形状 [seq_len, 1]，初始值为负无穷。
    # l = ???  # 每行指数和；形状 [seq_len, 1]，初始值为零。
    raise NotImplementedError("TODO 1：请初始化 m 与 l")
    
    scale = 1.0 / math.sqrt(dim)
    
    # 外层循环：遍历 Q 的分块
    for i in range(0, seq_len, block_size):
        q_block = q[i:i+block_size] * scale
        m_i = m[i:i+block_size]
        l_i = l[i:i+block_size]
        out_i = out[i:i+block_size]
        
        # 内层循环：遍历 K, V 的分块
        for j in range(0, seq_len, block_size):
            k_block = k[j:j+block_size]
            v_block = v[j:j+block_size]
            
            # ==========================================
            # TODO 2：形成局部分数与因果约束
            # ==========================================
            # scores = ???  # 当前 Q/K 分块的局部分数；形状 [Q 分块长度, K 分块长度]。
            if causal:
                query_pos = torch.arange(i, i + q_block.shape[0], device=q.device)[:, None]
                key_pos = torch.arange(j, j + k_block.shape[0], device=q.device)[None, :]
                # future_mask = ???  # K 的全局位置晚于 Q 时为真。
                scores = scores.masked_fill(future_mask, -float('inf'))
            raise NotImplementedError("TODO 2：请计算 scores 与 future_mask")
            
            # ==========================================
            # TODO 3：执行数值重标定
            # ==========================================
            # tile_max = ???  # 当前分块逐行最大值；保留列维。
            # new_max = ???  # 历史与当前分块共享的新最大值基准。
            # history_scale = ???  # 历史状态迁移到新基准的缩放因子。
            # tile_weights = ???  # 当前分块相对新基准的指数权重。
            raise NotImplementedError("TODO 3：请完成最大值基准与指数权重的重标定")

            # ==========================================
            # TODO 4：合并归一化输出
            # ==========================================
            # new_l = ???  # 同一最大值基准下的历史与当前指数和。
            # out_i = ???  # 使用同一归一化比例合并历史与当前输出。
            raise NotImplementedError("TODO 4：请更新指数和与归一化输出")
            m_i = new_max
            l_i = new_l
        
        out[i:i+block_size] = out_i
        m[i:i+block_size] = m_i
        l[i:i+block_size] = l_i
            
    return out

```


### 测试

`test_noncausal_equivalence` 用三组不同形状（含 `seq_len=5, block_size=3` 的尾块）对照完整 attention，误差要小于 $10^{-5}$；`test_causal_equivalence` 用 6 个位置、块大小 2 检查跨块的因果 mask；`test_dtype_and_stability` 看 float64 输入是否保持 dtype，以及分数全是 100 时是否还能算出有限值（这就是减最大值的意义）；`test_working_set_proxy` 不跑模型，只用算术确认固定块大小下局部分数的元素数不随 $N^2$ 增长；`test_input_contract` 确认 `block_size=0` 会被拒绝。


```python
def _reference_attention(q, k, v, causal=False):
    """构造完整分数矩阵，作为分块注意力的数值参考。"""
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.triu(torch.ones(scores.shape, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -float('inf'))
    return torch.softmax(scores, dim=-1) @ v

def test_noncausal_equivalence():
    """TODO 1–4：非因果分块路径应与完整注意力等价，含尾块。"""
    for seq_len, dim, block_size, seed in [(8, 4, 2, 42), (5, 3, 3, 7), (3, 2, 1, 123)]:
        torch.manual_seed(seed)
        q, k, v = (torch.randn(seq_len, dim) for _ in range(3))
        diff = (_reference_attention(q, k, v) - flash_attention_forward_sim(q, k, v, block_size)).abs().max()
        assert diff < 1e-5, f"非因果结果失配：seq={seq_len}, block={block_size}"

def test_causal_equivalence():
    """TODO 2：因果掩码必须屏蔽未来 K/V。"""
    torch.manual_seed(11)
    q, k, v = (torch.randn(6, 4) for _ in range(3))
    assert torch.allclose(_reference_attention(q, k, v, causal=True), flash_attention_forward_sim(q, k, v, 2, causal=True), atol=1e-5, rtol=1e-5)

def test_dtype_and_stability():
    """TODO 3：重标定应保留数据类型，并让大分数输出保持有限。"""
    torch.manual_seed(9)
    q, k, v = (torch.randn(4, 3, dtype=torch.float64) for _ in range(3))
    out = flash_attention_forward_sim(q, k, v, 2)
    assert out.dtype == torch.float64
    assert torch.allclose(_reference_attention(q, k, v), out, atol=1e-10, rtol=1e-10)
    stable_out = flash_attention_forward_sim(torch.full((3, 2), 100.0), torch.full((3, 2), 100.0), torch.randn(3, 2), 2)
    assert torch.isfinite(stable_out).all()

def test_working_set_proxy():
    """验证局部分数工作集在固定分块下不随总序列长度平方增长。"""
    block_size = 8
    proxies = []
    for seq_len in [64, 256, 1024]:
        full_score_elements = seq_len * seq_len
        tile_score_elements = block_size * block_size
        proxies.append((full_score_elements, tile_score_elements))
        assert tile_score_elements == 64
        assert tile_score_elements < full_score_elements
    assert [full // tile for full, tile in proxies] == [64, 1024, 16384]

    seq_len = 256
    assert [block * block for block in [4, 8, 16]] == [16, 64, 256]

def test_input_contract():
    """输入形状、数据类型、设备与分块大小契约应拒绝非法值。"""
    try:
        flash_attention_forward_sim(torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2), block_size=0)
    except ValueError:
        return
    raise AssertionError('block_size <= 0 应被拒绝')

def run_flash_attention_sim_tests():
    try:
        test_noncausal_equivalence()
        test_causal_equivalence()
        test_dtype_and_stability()
        test_working_set_proxy()
        test_input_contract()
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    print("✅ FlashAttention CPU 机制测试通过：数值等价、因果约束、稳定性、工作集代理与输入契约均符合预期。")

run_flash_attention_sim_tests()

```


---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

## 参考代码与解析

### 代码


```python
def flash_attention_forward_sim(q, k, v, block_size=2, causal=False):
    """计算二维输入上的 FlashAttention 前向模拟。

    Args:
        q, k, v: [seq_len, dim] 张量，device 和 dtype 应保持一致。
        block_size: Q/K/V 的分块大小，必须为正数。
        causal: 是否只允许关注当前位置及之前的 K 位置。

    Returns:
        [seq_len, dim] 的注意力输出。

    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError('q、k、v 必须是 [seq_len, dim] 二维张量')
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError('q、k、v 的形状必须一致')
    if q.device != k.device or k.device != v.device:
        raise ValueError('q、k、v 必须位于同一 device')
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise TypeError('q、k、v 必须使用相同 dtype')
    if block_size <= 0:
        raise ValueError('block_size 必须为正数')

    seq_len, dim = q.shape
    
    out = torch.zeros((seq_len, dim), device=q.device, dtype=q.dtype)
    # TODO 1：初始化跨分块状态
    m = torch.full((seq_len, 1), -float('inf'), device=q.device, dtype=q.dtype)
    l = torch.zeros((seq_len, 1), device=q.device, dtype=q.dtype)
    
    scale = 1.0 / math.sqrt(dim)
    
    # 外层循环：遍历 Q 的分块
    for i in range(0, seq_len, block_size):
        q_block = q[i:i+block_size] * scale
        m_i = m[i:i+block_size]
        l_i = l[i:i+block_size]
        out_i = out[i:i+block_size]
        
        # 内层循环：遍历 K, V 的分块
        for j in range(0, seq_len, block_size):
            k_block = k[j:j+block_size]
            v_block = v[j:j+block_size]
            
            # TODO 2：形成局部分数与因果约束
            scores = q_block @ k_block.transpose(-2, -1)
            if causal:
                query_pos = torch.arange(i, i + q_block.shape[0], device=q.device)[:, None]
                key_pos = torch.arange(j, j + k_block.shape[0], device=q.device)[None, :]
                future_mask = key_pos > query_pos
                scores = scores.masked_fill(future_mask, -float('inf'))
            
            # TODO 3：执行数值重标定
            tile_max = torch.max(scores, dim=-1, keepdim=True)[0]
            new_max = torch.maximum(m_i, tile_max)
            history_scale = torch.exp(m_i - new_max)
            tile_weights = torch.exp(scores - new_max)

            # TODO 4：合并归一化输出
            new_l = l_i * history_scale + tile_weights.sum(dim=-1, keepdim=True)
            out_i = out_i * (l_i * history_scale / new_l) + (tile_weights @ v_block) / new_l
            m_i = new_max
            l_i = new_l
        
        out[i:i+block_size] = out_i
        m[i:i+block_size] = m_i
        l[i:i+block_size] = l_i
            
    return out
```


### 解析

#### TODO 1：初始化 $m$ 和 $l$

**思路：** 每个 query 行一个最大值、一个指数和，还没看任何 key 时最大值是 $-\infty$、指数和是 0。

**关键一行：**
```python
m = torch.full((seq_len, 1), -float('inf'), device=q.device, dtype=q.dtype)
l = torch.zeros((seq_len, 1), device=q.device, dtype=q.dtype)
```

**为什么这样写：** 形状 `[seq_len, 1]` 而不是 `[seq_len]`，是为了后面和 `[rows, cols]` 的 `scores` 做广播时自动沿列展开。$-\infty$ 作初值保证第一块的 `new_max = maximum(-inf, tile_max) = tile_max`，history_scale $= e^{-\infty} = 0$，旧状态（本来就是空的）贡献为 0。

**常见错法：** 将 `m` 随意设成 0。在精确算术且 `l=O=0` 时，这不一定产生错误的历史贡献，但若所有分数都很负，`exp(scores - 0)` 可能全部下溢，分母成为零。初始化为负无穷能让第一块按自己的最大值缩放。现有稳定性测试检查有限输出与参考值一致性，但并不穷尽所有负分数极端情况。

#### TODO 2：局部分数与因果 mask

**思路：** 当前 Q 块乘当前 K 块的转置；因果时用全局位置比较。

**关键一行：**
```python
scores = q_block @ k_block.transpose(-2, -1)
future_mask = key_pos > query_pos
```

**为什么这样写：** `q_block` 在循环开头已经乘过 `scale = 1/sqrt(dim)`，这里不用再乘。`query_pos` 是 `arange(i, i+rows)[:, None]`、`key_pos` 是 `arange(j, j+cols)[None, :]`，广播后得到 `[rows, cols]` 的全局位置比较——这正是 Step 3 陷阱一的修法。被遮掉的位置填 $-\infty$，`exp` 后变成 0，自动从分母里消失。

**常见错法：** 写 `torch.triu(torch.ones(rows, cols), diagonal=1)`。这是块内下标的上三角，只在 $i = j$ 的对角块上碰巧正确；$i > j$ 的块（Q 在后、K 在前）应该全部可见，却被遮掉了一半。

#### TODO 3：重标定

**思路：** 新块的最大值和旧的比一比，取大的做新基准；旧状态和新块都折算到这个基准上。

**关键一行：**
```python
tile_max = torch.max(scores, dim=-1, keepdim=True)[0]
new_max = torch.maximum(m_i, tile_max)
history_scale = torch.exp(m_i - new_max)
tile_weights = torch.exp(scores - new_max)
```

**为什么这样写：** `keepdim=True` 保住列维，`[rows, 1]` 才能和 `m_i` 逐元素取 `maximum`。`history_scale` 和 `tile_weights` 的指数里减的是同一个 `new_max`，所以两者在同一尺度上，可以直接相加。`new_max ≥ m_i` 且 `new_max ≥ scores`，两个指数的参数都 ≤ 0，永远不会溢出——这是分数全为 100 的测试仍能得到有限输出的原因。

**常见错法：** 用 `tile_max` 而不是 `new_max` 做基准算 `tile_weights`。如果 `tile_max < m_i`，新块权重会被高估，和按 `m_i` 折算的历史对不上。

#### TODO 4：合并输出

**思路：** 分母 $l$ 和分子 $O$ 都要用同一个 history_scale 折算旧值，再加新块。

**关键一行：**
```python
new_l = l_i * history_scale + tile_weights.sum(dim=-1, keepdim=True)
out_i = out_i * (l_i * history_scale / new_l) + (tile_weights @ v_block) / new_l
```

**为什么这样写：** `out_i` 存的是**已归一化**的输出（除过 $l_i$ 了），要并入新块必须先"反归一化"回未除状态——乘 $l_i$——再折到新基准、再除以 $l_{new}$，这就是 `l_i * history_scale / new_l` 这个因子的来历。新块的贡献 `tile_weights @ v_block` 还没除过任何分母，直接除 $l_{new}$。

**常见错法：** 只更新了 `new_l`，`out_i` 还是用旧的 `l_i` 归一化。分子分母比例不一致，输出不再是合法的加权平均。另一种是写成 `out_i * history_scale + ...`，漏了 `l_i / new_l`——这其实是 FA-2 的未归一化形式，但那样循环结束后必须再整体除一次 `l`，本节的骨架没有这一步。

### Step 5：GPU 实验——显式 attention 对比 PyTorch SDPA

CPU 题目区证明了分块 + online softmax 在数学上和完整 attention 等价，但它回答不了"在真 GPU 上到底省了多少显存、快了多少"——纯 Python 的双层 `for` 循环比显式矩阵乘慢得多，拿它测性能毫无意义。GPU 实验换成 PyTorch 的 `scaled_dot_product_attention`（SDPA）：它会根据设备、dtype、形状自动选择 flash / memory-efficient / math 三个 backend 之一，命中 flash 时跑的就是 FlashAttention kernel。对照组是显式物化分数矩阵的朴素实现，两条路径共用同一组 Q/K/V，在 512 到 4096 四个长度上比较输出误差、Median/P95 延迟、峰值显存，并用 profiler 读出实际执行的算子名——"SDPA 可用"和"真的走了 flash kernel"是两回事，只有后者才算证据。

#### 5.1 环境与 workload

本实验使用 `INF-CUDA-MECHANISM`。首次运行先完成[推理机制 GPU 环境 Quickstart](../topic_discussion/inference_optimization/quickstart.md#inf-cuda-mechanism-quickstart)，通过 Profile 预检，并在[推理优化实验资产表](../topic_discussion/inference_optimization/experiment_assets.md#机制实验注册表)核对当前记录状态后，再把 `RUN_MODE` 改为 `real_gpu`。

| 配置层 | 本实验的设定 | 修改规则 |
|---|---|---|
| 运行环境 | `INF-CUDA-MECHANISM`；单张 NVIDIA GPU；参考显存为 12 GB 级 | 参考显存不是最低门槛；不与 vLLM、SGLang 等服务环境混装，预检失败时停止实验 |
| workload | batch=1、8 个 head、head_dim 64、序列长度 512/1024/2048/4096；默认自动选择 BF16 或 FP16 | 改变形状、dtype 或因果条件后递增 workload 版本 |
| 对照 | 显式分数矩阵 baseline → PyTorch SDPA candidate | 两条路径共用 Q/K/V、随机种子和测量方法，只改变 attention 执行路径 |
| 测量 | 输出误差、Median/P95、峰值已分配/保留显存和 profiler 算子名 | 不用 backend 能力开关代替实际算子命中 |
| 结果文件 | `benchmarks/results/20_flashattention/` 下的版本化 JSON；运行标签标识一次运行 | 正式运行使用唯一标签，旧协议结果不混入新协议 |


```python
# 运行入口与结果身份
RUN_MODE = "dry_run"  # 默认值 dry_run：只登记计划；改为 real_gpu 才执行真实 GPU 对照。
RUN_LABEL = "default"  # 默认值 default；正式复测改为唯一标签，避免覆盖同名结果。

ENVIRONMENT_PROFILE = "INF-CUDA-MECHANISM"  # 固定环境身份；运行前必须通过同名 Profile 预检。
WORKLOAD_ID = "inference-attention-sdpa"  # 稳定工作负载身份。
WORKLOAD_VERSION = "v3"  # v3 使用长度矩阵、dtype 感知容差与分路径显存口径。

GPU_DTYPE = "auto"  # 默认值 auto：优先原生 BF16，否则使用 FP16；也可指定 float16 / bfloat16。
GPU_BATCH_SIZE = 1  # 默认值 1；batch 维度，两条路径保持一致。
GPU_NUM_HEADS = 8  # 默认值 8；Q head 数，两条路径共用。
GPU_SEQ_LENGTHS = [512, 1024, 2048, 4096]  # 默认长度矩阵；增大会放大显式分数矩阵的二次增长。
GPU_HEAD_DIM = 64  # 默认值 64；单头维度，增大会提高每次注意力计算量。
GPU_CAUSAL = False  # 默认值 False；改为 True 可验证因果掩码路径。
GPU_ATOL = 1e-2  # 默认绝对容差；用于 FP16/BF16 两条执行路径的数值比较。
GPU_RTOL = 1e-2  # 默认相对容差；避免只按单点最大误差误判低精度结果。

GPU_WARMUP = 5  # 默认值 5 次；预热不进入统计。
GPU_ITERS = 20  # 默认值 20 次；用于计算 Median 与 P95。
GPU_SEED = 42  # 默认值 42；固定 Q/K/V 随机输入。
OUTPUT_RELATIVE_PATH = f"benchmarks/results/20_flashattention/{WORKLOAD_ID}_{RUN_LABEL}.json"  # 仓库相对结果路径。
```


#### 5.2 执行显式 attention / SDPA 配对实验

保持 5.1 的 workload 和对照条件不变。`dry_run` 只生成执行计划；`real_gpu` 构造共享 Q/K/V，依次测量两条路径并保存 JSON。先检查输出误差，再比较延迟、显存和 profiler 算子路径。


```python
"""运行显式 Attention 与 PyTorch SDPA 的固定 workload GPU 对照。"""

from datetime import datetime, timezone
import json
import math
import platform
import statistics

import torch

from tools.project_runtime import resolve_project_root

def _select_gpu_dtype():
    """根据配置选择 dtype，并区分原生 BF16 与模拟支持。"""
    if GPU_DTYPE == 'float16':
        return torch.float16
    if GPU_DTYPE == 'bfloat16':
        return torch.bfloat16
    if GPU_DTYPE != 'auto':
        raise ValueError('GPU_DTYPE 只能是 auto / float16 / bfloat16')
    # including_emulation=False 避免把软件模拟误当成硬件 BF16 支持。
    try:
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        major, _ = torch.cuda.get_device_capability()
        native_bf16 = major >= 8
    return torch.bfloat16 if native_bf16 else torch.float16

def _naive_attention(q, k, v, causal=False):
    """显式物化 score 的 Attention baseline，用于同 workload 对照。"""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(scores.shape[-2:], device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -float('inf'))
    return torch.softmax(scores, dim=-1) @ v

def _measure_attention(fn, q, k, v, causal):
    """预热后使用 CUDA Event 记录 Median、P95 与该路径的显存峰值。"""
    with torch.inference_mode():
        for _ in range(GPU_WARMUP):
            fn(q, k, v, causal)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(GPU_ITERS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(q, k, v, causal)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        'median_ms': round(statistics.median(samples), 4),
        'p95_ms': round(ordered[p95_index], 4),
        'peak_allocated_mb': round(torch.cuda.max_memory_allocated() / 2**20, 2),
        'peak_reserved_mb': round(torch.cuda.max_memory_reserved() / 2**20, 2),
    }

def _profile_attention_path(fn, q, k, v, causal):
    """读取 SDPA 的算子名，区分能力可用与实际执行路径。"""
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.inference_mode(), torch.profiler.profile(activities=activities) as profile:
        fn(q, k, v, causal)
        torch.cuda.synchronize()
    keys = [event.key for event in profile.key_averages()]
    return [key for key in keys if 'attention' in key.lower() or 'scaled_dot_product' in key.lower()]

def _run_gpu_attention_experiment():
    """执行两条 Attention 路径，并保存可复查的环境与结果记录。"""
    if RUN_MODE != 'real_gpu':
        print("GPU 对照未启动：将 RUN_MODE 从 dry_run 改为 real_gpu 后重新运行。")
        return None
    # 先做环境和 workload 校验，避免在 CPU 环境中静默产生伪 GPU 结果。
    if not torch.cuda.is_available():
        raise RuntimeError('RUN_MODE=real_gpu，但当前环境没有可用 CUDA。')
    if not GPU_SEQ_LENGTHS or min(GPU_SEQ_LENGTHS) <= 0:
        raise ValueError('GPU_SEQ_LENGTHS 必须包含正整数。')
    if min(GPU_BATCH_SIZE, GPU_NUM_HEADS, GPU_HEAD_DIM, GPU_WARMUP, GPU_ITERS) <= 0:
        raise ValueError('batch、heads、head_dim、warmup 和 iters 必须为正数。')
    if min(GPU_ATOL, GPU_RTOL) < 0:
        raise ValueError('GPU_ATOL 与 GPU_RTOL 不能为负数。')
    torch.manual_seed(GPU_SEED)
    device = torch.device('cuda')
    dtype = _select_gpu_dtype()
    methods = {
        'naive': _naive_attention,
        'SDPA': lambda a, b, c, causal: torch.nn.functional.scaled_dot_product_attention(
            a, b, c, is_causal=causal
        ),
    }
    cases = []
    failures = []
    for seq_len in GPU_SEQ_LENGTHS:
        shape = (GPU_BATCH_SIZE, GPU_NUM_HEADS, seq_len, GPU_HEAD_DIM)
        q = torch.randn(shape, device=device, dtype=dtype)
        k = torch.randn(shape, device=device, dtype=dtype)
        v = torch.randn(shape, device=device, dtype=dtype)
        path_metrics = {}
        for name, fn in methods.items():
            try:
                path_metrics[name] = {'implementation': name, **_measure_attention(fn, q, k, v, GPU_CAUSAL), 'status': 'ok'}
            except RuntimeError as exc:
                if 'out of memory' not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                path_metrics[name] = {'implementation': name, 'median_ms': None, 'p95_ms': None, 'peak_allocated_mb': None, 'peak_reserved_mb': None, 'status': 'OOM'}
                failures.append({'seq_len': seq_len, 'implementation': name, 'status': 'OOM'})
        correctness = None
        speedup = None
        allocated_reduction = None
        sdpa_operator_keys = []
        if all(path_metrics[name]['status'] == 'ok' for name in methods):
            with torch.inference_mode():
                naive_output = methods['naive'](q, k, v, GPU_CAUSAL)
                sdpa_output = methods['SDPA'](q, k, v, GPU_CAUSAL)
            difference = naive_output.float() - sdpa_output.float()
            reference_norm = torch.linalg.vector_norm(naive_output.float()).clamp_min(torch.finfo(torch.float32).eps)
            correctness = {
                'output_close': bool(torch.allclose(naive_output, sdpa_output, atol=GPU_ATOL, rtol=GPU_RTOL)),
                'max_abs_error': float(difference.abs().max().item()),
                'relative_l2_error': float((torch.linalg.vector_norm(difference) / reference_norm).item()),
                'atol': GPU_ATOL,
                'rtol': GPU_RTOL,
            }
            speedup = path_metrics['naive']['median_ms'] / path_metrics['SDPA']['median_ms']
            allocated_reduction = 1.0 - path_metrics['SDPA']['peak_allocated_mb'] / path_metrics['naive']['peak_allocated_mb']
            sdpa_operator_keys = _profile_attention_path(methods['SDPA'], q, k, v, GPU_CAUSAL)
            del naive_output, sdpa_output, difference
        cases.append({
            'seq_len': seq_len,
            'baseline': path_metrics['naive'],
            'candidate': path_metrics['SDPA'],
            'correctness': correctness,
            'speedup': speedup,
            'allocated_memory_reduction': allocated_reduction,
            'sdpa_operator_keys': sdpa_operator_keys,
        })
        del q, k, v
        torch.cuda.empty_cache()
    # evidence_level 说明这是单 GPU、固定 workload 的对照，不是稳定 benchmark。
    result = {
        'schema_version': 'inference-mechanism-gpu/v2',
        'semantic_id': 'INF-SDPA-ATTENTION',
        'environment_profile': ENVIRONMENT_PROFILE,
        'run_mode': RUN_MODE,
        'run_label': RUN_LABEL,
        'workload': {'id': WORKLOAD_ID, 'version': WORKLOAD_VERSION},
        'created_at_utc': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
        'config': {
            'dtype': str(dtype), 'batch_size': GPU_BATCH_SIZE, 'num_heads': GPU_NUM_HEADS,
            'seq_lengths': GPU_SEQ_LENGTHS, 'head_dim': GPU_HEAD_DIM, 'causal': GPU_CAUSAL,
            'atol': GPU_ATOL, 'rtol': GPU_RTOL,
            'warmup': GPU_WARMUP, 'iters': GPU_ITERS, 'seed': GPU_SEED,
        },
        'environment': {
            'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'python': platform.python_version(),
        },
        'cases': cases,
        'evidence_level': 'local_mechanism_benchmark',
        'note': 'naive 与 SDPA 的对照不能直接命名为 FlashAttention-3/4；稳定结论需要重复运行。',
    }
    output_path = resolve_project_root() / OUTPUT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result['artifact'] = {'result_path': str(output_path), 'artifact_type': 'json_result'}
    result['failure'] = failures or None
    comparable = [case for case in cases if case['correctness'] is not None]
    all_correct = bool(comparable) and all(case['correctness']['output_close'] for case in comparable)
    has_efficiency_gain = any(
        (case['speedup'] is not None and case['speedup'] > 1.0)
        or (case['allocated_memory_reduction'] is not None and case['allocated_memory_reduction'] > 0.0)
        for case in comparable
    )
    if not comparable:
        result['decision'] = {'decision': 'reject', 'reason': '没有形成可比较的显式 Attention / SDPA 配对结果'}
    elif not all_correct:
        result['decision'] = {'decision': 'reject', 'reason': '至少一档长度未通过 dtype 感知的数值一致性检查'}
    elif failures:
        result['decision'] = {'decision': 'tune_capacity_boundary', 'reason': '已形成有效低长度证据，同时观察到容量边界'}
    elif has_efficiency_gain:
        result['decision'] = {'decision': 'accept_for_mechanism_evidence', 'reason': '长度矩阵通过正确性检查，并观察到延迟或显存收益'}
    else:
        result['decision'] = {'decision': 'tune', 'reason': '正确性通过，但当前长度矩阵未观察到效率收益'}
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'已保存 GPU 结果: {output_path}')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

gpu_result = _run_gpu_attention_experiment()
```


#### 5.3 读取配对实验结果

读取 5.2 保存的 JSON，先核对 Profile、workload 版本、形状和失败状态，再比较显式 attention 与 SDPA；没有通过数值检查的结果不进入性能判断。


```python
# 读取固定运行标签对应的 JSON，不重复执行注意力计算。
import json
from tools.project_runtime import resolve_project_root

output_path = resolve_project_root() / OUTPUT_RELATIVE_PATH
if not output_path.exists():
    print(f"等待 GPU 配对实验结果：{output_path}")
else:
    record = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps({
        key: record.get(key)
        for key in ("schema_version", "environment_profile", "workload", "config", "environment", "failure", "evidence_level", "decision")
    }, ensure_ascii=False, indent=2))
    print("\n长度 × 执行路径摘要")
    for case in record.get("cases", []):
        print({
            'seq_len': case['seq_len'],
            'output_close': case['correctness']['output_close'] if case['correctness'] else None,
            'max_abs_error': case['correctness']['max_abs_error'] if case['correctness'] else None,
            'relative_l2_error': case['correctness']['relative_l2_error'] if case['correctness'] else None,
            'naive_median_ms': case['baseline']['median_ms'],
            'sdpa_median_ms': case['candidate']['median_ms'],
            'speedup': round(case['speedup'], 4) if case['speedup'] is not None else None,
            'allocated_memory_reduction': round(case['allocated_memory_reduction'], 4) if case['allocated_memory_reduction'] is not None else None,
            'sdpa_operator_keys': case['sdpa_operator_keys'],
        })
```


#### 5.4 解释正确性、显存与执行路径

先看 `failure`、`output_close` 和相对误差；只有两条路径都成功且输出可比，才读延迟和峰值显存。SDPA 的收益依赖 GPU、dtype、形状和实际 backend，本节结果只支持这组长度下的局部判断。

本节保留的历史结果表记录 `rtx5070ti_sdpa_matrix_v3`：四个长度上输出都在容差内（最大绝对误差 0.00195–0.00586，相对 L2 误差约 0.5%，在本次 BF16 容差内；两条路径的累加顺序可能不同）；加速比从 512 长度的 2.13× 一路涨到 4096 长度的 6.18×，已分配显存的降幅从 44.0% 涨到 95.5%——长度越长，那张 $N^2$ 矩阵占的比例越大，省掉它的收益也越大。profiler 看到的算子是 `_scaled_dot_product_flash_attention` / `_flash_attention_forward`，说明 SDPA 确实走进了 FlashAttention kernel，而不是 math 回退。

| 观察项 | 读取字段 | 当前结果 | 解释与动作 |
|:---|:---|:---|:---|
| 运行与条件 | `config`、`workload`、`failure` | `rtx5070ti_sdpa_matrix_v3`：RTX 5070 Ti、BF16、长度 512–4096、无失败 | 结论只适用于当前形状、精度和 PyTorch backend |
| 正确性 | `output_close`、`max_abs_error`、`relative_l2_error` | 四档均通过；最大绝对误差 `0.00195–0.00586`，相对 L2 误差约 `0.48%–0.50%` | `atol=rtol=1e-2` 适配当前 BF16 配对实验 |
| 资源与性能 | Median、P95、峰值显存、`speedup` | 加速比由 `2.13×` 增至 `6.18×`；已分配显存降幅由 `44.0%` 增至 `95.5%` | 长度增大后显式分数矩阵的时间和显存代价更突出 |
| 执行路径与决策 | 算子名、`evidence_level`、`decision` | 命中 `_scaled_dot_product_flash_attention` / `_flash_attention_forward`；`accept_for_mechanism_evidence` | 机制层面成立；不外推为完整模型或服务吞吐 |

<a id="cheatsheet"></a>
## 速查卡

| | |
|:--|:--|
| **一句话机制** | 把 Q、K、V 切块，一次只算一个 Q 块对一个 K 块的局部分数；每行维护最大值 $m$、指数和 $l$、累计输出 $O$ 三个状态，新块到来时用 $e^{m_{old}-m_{new}}$ 把旧状态折到新基准再加新块。全部块看完，结果与完整 attention 在精确算术下相等，浮点实现允许舍入误差，而 $N \times N$ 分数矩阵从未完整出现。 |
| **关键公式与数量级** | $m_{new}=\max(m_{old}, \max S_{tile})$；$l_{new}=l_{old}e^{m_{old}-m_{new}}+\sum e^{S_{tile}-m_{new}}$；$O_{new}=O_{old}\frac{l_{old}e^{m_{old}-m_{new}}}{l_{new}}+\frac{e^{S_{tile}-m_{new}}V_{tile}}{l_{new}}$。分数矩阵 BF16 每 head：4k 上下文 32 MiB，32k 上下文 2 GiB，128k 上下文 32 GiB。本节实验（RTX 5070 Ti，8 头，4096 长度）：SDPA 比显式实现快 6.18×，已分配显存省 95.5%。 |
| **常见坑** | 1. 因果 mask 用块内下标而不是全局位置——跨块时把可见的 key 遮掉。2. `m` 用 0 而不是 $-\infty$ 初始化——分数为负时失去防溢出效果，还混入不存在的"历史"。3. 分子 $O$ 和分母 $l$ 没用同一个 history_scale 折算。4. $-\infty-(-\infty)$ 得 NaN：K 块顺序遍历时第一块总含位置 0，不会触发；改并行或倒序要显式处理。5. "SDPA 可用"≠"走了 flash kernel"，要看 profiler 的算子名。 |
| **延伸阅读** | 专题正文 [02 Prefill 与 Attention Kernel](../topic_discussion/inference_optimization/02_prefill_and_attention_kernel.md) · 判断手册 [按现象选择机制](../topic_discussion/inference_optimization/casebook.md#按现象选择机制) · 下一步 [INF-EXECUTION-DISPATCH](./INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.md)：SDPA 什么时候回退到 math · [22 PagedAttention](./22_vLLM_PagedAttention.md)：decode 阶段 K/V 怎么存 |

## 相关阅读

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) —— 原论文；Section 3.1 的 Algorithm 1 就是本节代码的伪码版，Section 2 有 HBM / SRAM 带宽数字
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) —— Step 3 提到的"最后归一化"和 warp 分工
- [FlashAttention 官方实现](https://github.com/Dao-AILab/flash-attention) —— `csrc/flash_attn/` 下的 CUDA kernel
- [PyTorch `scaled_dot_product_attention` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention) —— 三个 backend 的选择规则和 `sdpa_kernel` 上下文管理器
- [Part 02 · 22 vLLM 分页注意力](./22_vLLM_PagedAttention.md) —— prefill 之后，K/V 在 decode 阶段怎么分页存放