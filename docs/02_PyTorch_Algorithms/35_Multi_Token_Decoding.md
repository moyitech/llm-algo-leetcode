# 35. Multi Token Decoding | 原生多 Token 预测

**难度：** Hard | **环境：** CPU + GPU | **标签：** `推理优化`, `投机解码`, `MTP`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/35_Multi_Token_Decoding.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

23 节的投机解码要额外养一个 draft model（草稿模型）：多一份权重、多一份 KV Cache、tokenizer 还得对得上，而且一个通用小模型和大模型的"口味"未必一致，接受率上不去。MTP（multi-token prediction，多 token 预测）换了个思路：**把草稿能力做进模型自身**。checkpoint 里多带一两个很轻的预测模块，它们复用主模型刚算好的隐藏状态，顺手猜出再往后一两个 token；主模型下一次前向时把这些候选一并验证。DeepSeek-V3 和 Qwen3.5 系列都带这样的模块。

本节实现这条路径的三个关节：从各预测深度的 logits 形成有序候选块、按位置和主模型结果做贪心验证、在首次不匹配处修正并停止。学完后你能解释 MTP 和外置 draft model 在"候选从哪来"上的本质区别、为什么贪心下"候选 == argmax"就足够无损，以及 vLLM 启用 MTP 时为什么必须先确认 checkpoint 里真有这些层。

## 前置阅读

- [Part 02 · 21 解码策略](./21_Decoding_Strategies.md) —— 本节的候选就是各深度 logits 的 argmax，验证也只看 argmax；先熟悉 logits → token 这一步。
- [Part 02 · 23 投机解码](./23_Speculative_Decoding.md) —— draft / verify / 首次拒绝即停 / bonus token 的循环骨架在这里不变，变的只是"draft 从哪来"。

### Step 1：痛点与直觉——草稿不用另请一个模型

想象你已经按 23 节的办法给一个 70B 模型配了个 7B 的 draft。好处是真的，麻烦也是真的：7B 的 BF16 权重就是 14 GB 显存，它自己的 KV Cache 也要占地方；本节的直接 token-ID 验证路径要求两个模型使用兼容的 tokenizer；最关键的是，7B 模型是独立训练的，它对"下一个词"的判断和 70B 经常不一致，例如在一个假设场景中，每轮猜 5 个却平均只收下两三个，而 draft 自己跑 5 步的时间已经花出去了。

MTP 的观察是：主模型算完位置 $t$ 之后，最后一层的隐藏状态里其实已经携带了关于 $t+1$、$t+2$ 的大量信息——否则它怎么能准确预测 $t+1$？那就别浪费：在模型末尾挂一个很轻的模块（DeepSeek-V3 里是一个 Transformer block 加共享的输出头），输入是这个隐藏状态拼上刚选出来的 $t+1$ token 的 embedding，输出 $t+2$ 的 logits；要猜更远就再串一个模块。这些模块和主模型一起训练，看的是同一份表示，有机会更好地拟合主模型的分布，但并不保证接受——DeepSeek-V3 论文报告第二个 token 的接受率在 85%–90%，解码吞吐约 1.8×。换个说法，这不是请一个实习生来打草稿，而是作者自己写完一个词时顺手在页边记下接下来想写的词，下一句落笔前瞄一眼就行。

代价是每多猜一个 token 只多算一个轻量 block，而不是再跑一个几 B 的模型；但本节练习只使用传入的各深度 logits，候选数不能超过这些行数。真实 backend 有时会递推复用 MTP 模块，因此候选步数不必等于 MTP 层数；能否这样运行要查模型与 backend 的具体支持。

本节你要实现的是推理侧的三步：从各深度 logits 选出候选块、和主模型的 argmax 逐位比对算出连续接受长度、在首次不匹配处用主模型的 token 修正并结束本轮。

<img src="../public/02_PyTorch_Algorithms/35_multi_token_overview.svg" alt="原生 MTP 与投机解码机制链" width="1440" height="650">

### Step 2：机制怎么走——3 个深度、3 个词的例子

把词表缩到 3 个 token，假设我们已经获得 3 个有序预测位置的 logits（这里不实现生成这些 logits 的模型）。主模型算完当前位置后，三个深度各给出一行 logits：

```
depth 1: [0.1, 0.9, 0.0]   → argmax = 1
depth 2: [0.8, 0.1, 0.1]   → argmax = 0
depth 3: [0.2, 0.3, 0.5]   → argmax = 2
```

**形成候选块.** 每个深度各取 argmax，按深度顺序排好：`[1, 0, 2]`。如果本轮预算 `max_candidates=2`，只取前两个 `[1, 0]`——深度 3 的候选扔掉，不是重选。顺序不能动：深度 1 对应紧接着的下一个位置，深度 2 对应再下一个，位置和深度一一绑定。

**验证.** 主模型下一次前向把这 $K$ 个候选当作输入一并算，得到 $K+1$ 个位置上它自己的 argmax（多出来的那个是全部接受时的 bonus 位置）。换一组数字看：候选 `[4, 5, 6]`，主模型给出 `[4, 5, 9, 8]`。从左到右比：4 = 4 ✓，5 = 5 ✓，6 ≠ 9 ✗，停。连续接受长度 2。

**本轮输出.** 接受的前缀 `[4, 5]` 加上主模型在不匹配位置给的 token 9，输出 `[4, 5, 9]`，共 3 个 token，一次主模型前向推进了 3 步。候选 6 以及它后面的（如果有）全部作废。如果候选是 `[4, 5]` 而主模型给 `[4, 5, 8]`，两个全中，输出 `[4, 5, 8]`——最后那个 8 就是 bonus，白送的。

**为什么贪心下这样就够了.** 23 节的接受规则是 $\min(1, p/q)$，因为那里 target 在采样。本节走贪心：主模型无论如何都会输出它的 argmax，候选恰好等于 argmax 就等价于"主模型自己也会选它"，不等就按主模型的来——在 target 数值计算与并列取值规则一致的前提下，结果和逐 token 贪心生成相同。不需要概率，不需要掷硬币。

<details>
<summary>速查表：MTP 与外置 draft model 的对照</summary>

| | 外置 draft model（23 节） | 原生 MTP（本节） |
|---|---|---|
| 候选从哪来 | 独立的小模型自回归跑 $K$ 步 | checkpoint 内的 MTP 模块，复用主模型隐藏状态 |
| 额外显存 | 小模型权重 + 它的 KV Cache | 几个轻量 block 的权重 |
| 候选上限 | 由运行预算和实现限制共同决定 | 本题受输入深度数限制；真实 backend 可能递推复用模块 |
| 口味一致性 | 独立训练，接受率看配对 | 联合训练可改善分布匹配，仍需测量接受情况 |
| 验证规则 | 采样：$\min(1, p/q)$ + residual | 贪心：候选 == argmax；采样时仍需 23 节的规则 |
| 一轮推进 | $1 \sim K+1$ 个 token | $1 \sim K+1$ 个 token |

</details>

### Step 3：边界与陷阱——接受长度不是加速比

**陷阱一：候选块长不等于推进量.** 预算 $K = 3$ 只说明最多试 3 个位置。真正能提交的是**连续**接受长度：`[4, 5, 6]` 对 `[4, 5, 9, 8]` 接受 2 个，但 `[4, 7, 6]` 对 `[4, 5, 6, 8]` 只接受 1 个——哪怕第三个候选 6 碰巧和主模型一致，它前面的 7 已经错了，6 是在"7 之后"这个错误前提下猜的，不能算。首次不匹配越靠前，后面的候选越是白算。

**陷阱二：接受长度高也不等于提速.** 每轮多推进的 token 要用 MTP 模块的额外计算、更长的验证前向和 backend 调度开销来换。一个极端例子：如果 MTP 模块本身很慢、或者 batch 很大导致验证前向不再是"白送"，接受率 90% 也可能净收益为负。本节 CPU 部分只算接受长度，不算时间；时间要到 Step 5 和 [INF-PROPOSAL](./INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.md) 里看。

**陷阱三：贪心验证用在了采样上.** 如果主路径开了 temperature / top-p 采样，"候选 == argmax"就不再无损——主模型自己采样也不一定选 argmax。这时必须换回 23 节的概率规则。vLLM 等 backend 会根据 `SamplingParams` 自动切换，手写时别混。

**陷阱四：把 MTP 层数直接当成通用候选上限。** 本节 CPU 函数只能返回已有 logits 行中的候选；GPU 示例选择 MTP-1，并记录 Qwen3.5-0.8B 配置含 1 个 MTP 层。这不意味着所有 backend 的多步复用都不成立。增加 `SPECULATIVE_TOKENS` 前，要确认对应模型和 backend 是否支持递推、怎样维护状态，再验证输出与性能。

<img src="../public/02_PyTorch_Algorithms/35_acceptance_budget.svg" alt="多 Token 解码的接受长度与收益" width="1320" height="600">

### Step 4：任务——实现候选形成与贪心验证

题目区已经写好输入校验和返回字典的字段，你要补的是 Step 2 走过的三步。

| TODO | 要做什么 | 必须满足 | 对应测试 |
| --- | --- | --- | --- |
| TODO 1 · `select_mtp_candidates` | 每个深度取 argmax，按深度顺序排列，截到 `max_candidates` | 顺序不变；截断是"丢后面的"，不是重选 | `test_select_mtp_candidates`：预算 2 得 `[1, 0]`，预算 8 得 `[1, 0, 2]` |
| TODO 2 · `continuous_match_length` | 逐位比较候选与主模型 token，首次不同即停 | 返回连续前缀长度 | `test_continuous_match_length`：`[4,5,6]` vs `[4,5,9,8]` → 2；首位就错 → 0 |
| TODO 3 · `build_mtp_round` | 输出 = 接受前缀 + `target_tokens[accepted_length]` | 不匹配时那个 token 是修正，全中时是 bonus；后缀不进输出 | `test_build_mtp_round`：`[4,5,9]`、`[4,5,8]` |


```python
from typing import List, Sequence
import torch

```



```python
def _validate_depth_logits(depth_logits: torch.Tensor) -> torch.Tensor:
    """校验 MTP 预测深度与词表维度。"""
    scores = torch.as_tensor(depth_logits)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("depth_logits 必须是非空二维张量 [depth, vocab]")
    if not torch.isfinite(scores).all():
        raise ValueError("词表分数不能包含 NaN 或 Inf")
    return scores


def select_mtp_candidates(depth_logits: torch.Tensor, max_candidates: int) -> List[int]:
    """按预测深度形成受预算约束的有序候选块。"""
    scores = _validate_depth_logits(depth_logits)
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates < 1:
        raise ValueError("max_candidates 必须是正整数")
    # ==========================================
    # TODO 1：从各预测深度形成有序候选块。
    # ==========================================
    # candidates = ???  # 各深度最高分 Token，长度不超过 max_candidates。
    raise NotImplementedError("TODO 1：请形成 MTP 候选块")
    return candidates


def continuous_match_length(
    candidates: Sequence[int],
    target_tokens: Sequence[int],
) -> int:
    """返回候选与目标结果从首位置开始连续匹配的长度。"""
    if len(target_tokens) < len(candidates):
        raise ValueError("目标结果必须覆盖所有候选位置")
    accepted_length = 0
    for candidate, target in zip(candidates, target_tokens):
        # ==========================================
        # TODO 2：在首次不匹配位置停止连续接受。
        # ==========================================
        # matches = ???  # 当前 MTP 候选与目标首选 Token 相同时为 True。
        raise NotImplementedError("TODO 2：请比较当前位置")
        if not matches:
            break
        accepted_length += 1
    return accepted_length


def build_mtp_round(candidates: Sequence[int], target_tokens: Sequence[int]) -> dict:
    """根据连续匹配边界形成本轮输出和下一轮状态。"""
    if len(target_tokens) != len(candidates) + 1:
        raise ValueError("目标结果必须包含候选位置和一个额外位置")
    accepted_length = continuous_match_length(candidates, target_tokens)
    # ==========================================
    # TODO 3：提交接受前缀，并追加修正或额外目标 Token。
    # ==========================================
    # emitted_tokens = ???  # 接受前缀后追加 target_tokens[accepted_length]。
    # all_candidates_accepted = ???  # 接受长度等于候选长度时为 True。
    raise NotImplementedError("TODO 3：请形成本轮输出")
    return {
        "candidate_count": len(candidates),
        "accepted_length": accepted_length,
        "emitted_tokens": emitted_tokens,
        "all_candidates_accepted": all_candidates_accepted,
        "first_mismatch": None if all_candidates_accepted else accepted_length,
    }

```


### 测试

`test_select_mtp_candidates` 用 3 深度 × 3 词表的 logits 看候选顺序和预算截断，并确认空输入、一维输入会被拒绝；`test_continuous_match_length` 覆盖中途不匹配、首位不匹配、全部匹配三种情况；`test_build_mtp_round` 检查修正路径和 bonus 路径各自的输出以及 `first_mismatch` 字段。


```python
def test_select_mtp_candidates():
    """候选应按预测深度排列，并受最大候选数限制。"""
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    assert select_mtp_candidates(logits, 2) == [1, 0]
    assert select_mtp_candidates(logits, 8) == [1, 0, 2]
    for invalid in (torch.empty(0, 3), torch.tensor([0.1, 0.9])):
        try:
            select_mtp_candidates(invalid, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("空输入或非二维输入应失败")


def test_continuous_match_length():
    """连续匹配必须在首个不同位置停止。"""
    assert continuous_match_length([4, 5, 6], [4, 5, 9, 8]) == 2
    assert continuous_match_length([4, 5], [9, 5, 8]) == 0
    assert continuous_match_length([4, 5], [4, 5, 8]) == 2


def test_build_mtp_round():
    """中途不匹配使用目标修正，全部匹配使用额外目标 Token。"""
    rejected = build_mtp_round([4, 5, 6], [4, 5, 9, 8])
    assert rejected == {
        "candidate_count": 3,
        "accepted_length": 2,
        "emitted_tokens": [4, 5, 9],
        "all_candidates_accepted": False,
        "first_mismatch": 2,
    }
    accepted = build_mtp_round([4, 5], [4, 5, 8])
    assert accepted["emitted_tokens"] == [4, 5, 8]
    assert accepted["all_candidates_accepted"] is True
    assert accepted["first_mismatch"] is None


test_select_mtp_candidates()
test_continuous_match_length()
test_build_mtp_round()
print("✅ 原生 MTP 候选与贪心验证测试通过")

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
def _validate_depth_logits(depth_logits: torch.Tensor) -> torch.Tensor:
    """校验 MTP 预测深度与词表维度。"""
    scores = torch.as_tensor(depth_logits)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("depth_logits 必须是非空二维张量 [depth, vocab]")
    if not torch.isfinite(scores).all():
        raise ValueError("词表分数不能包含 NaN 或 Inf")
    return scores


# ==========================================
# TODO 1：从各预测深度形成有序候选块。
# ==========================================
def select_mtp_candidates(depth_logits: torch.Tensor, max_candidates: int) -> List[int]:
    """按预测深度形成受预算约束的有序候选块。"""
    scores = _validate_depth_logits(depth_logits)
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates < 1:
        raise ValueError("max_candidates 必须是正整数")
    candidates = scores.argmax(dim=-1)[:max_candidates].tolist()
    return candidates


# ==========================================
# TODO 2：在首次不匹配位置停止连续接受。
# ==========================================
def continuous_match_length(
    candidates: Sequence[int],
    target_tokens: Sequence[int],
) -> int:
    """返回候选与目标结果从首位置开始连续匹配的长度。"""
    if len(target_tokens) < len(candidates):
        raise ValueError("目标结果必须覆盖所有候选位置")
    accepted_length = 0
    for candidate, target in zip(candidates, target_tokens):
        matches = int(candidate) == int(target)
        if not matches:
            break
        accepted_length += 1
    return accepted_length


# ==========================================
# TODO 3：提交接受前缀，并追加修正或额外目标 Token。
# ==========================================
def build_mtp_round(candidates: Sequence[int], target_tokens: Sequence[int]) -> dict:
    """根据连续匹配边界形成本轮输出和下一轮状态。"""
    if len(target_tokens) != len(candidates) + 1:
        raise ValueError("目标结果必须包含候选位置和一个额外位置")
    accepted_length = continuous_match_length(candidates, target_tokens)
    emitted_tokens = list(candidates[:accepted_length]) + [int(target_tokens[accepted_length])]
    all_candidates_accepted = accepted_length == len(candidates)
    return {
        "candidate_count": len(candidates),
        "accepted_length": accepted_length,
        "emitted_tokens": emitted_tokens,
        "all_candidates_accepted": all_candidates_accepted,
        "first_mismatch": None if all_candidates_accepted else accepted_length,
    }

```


### 解析

#### TODO 1：形成候选块

**思路：** 每个深度一行 logits，各取 argmax，按深度顺序排，再截到预算。

**关键一行：**
```python
candidates = scores.argmax(dim=-1)[:max_candidates].tolist()
```

**为什么这样写：** `argmax(dim=-1)` 一次对所有深度求最大值位置，得到形状 `[depth]` 的张量，天然就是按深度排列的；`[:max_candidates]` 在深度维上截断，预算大于深度数时自然取全部。转成 Python `list[int]` 是为了后面和 `target_tokens` 逐个比较时不混入 0 维张量。

**常见错法：** 先按分数高低给候选排序再截断。候选的顺序就是位置的顺序，深度 1 永远对应最近的那个位置，按分数重排会把"第二个 token"放到"第一个"的位置上去验证，必然不匹配。

#### TODO 2：连续匹配长度

**思路：** 从左到右比，相同就计数，不同就停。

**关键一行：**
```python
matches = int(candidate) == int(target)
```

**为什么这样写：** 两边都转 `int`，是因为 `candidates` 可能来自 `tolist()` 的 Python 整数，`target_tokens` 可能来自张量切片；`torch.tensor(5) == 5` 返回的是 0 维布尔张量而不是 `bool`，`if not matches` 虽然能跑，但显式转成整数可以让这一教学接口统一使用 Python 标量。`break` 保证一旦不匹配就不再往后看——这是陷阱一的实现。

**常见错法：** 用 `sum(c == t for c, t in zip(...))` 数相同的个数。`[4, 7, 6]` 对 `[4, 5, 6]` 会得到 2，而正确答案是 1：第三个位置的"碰巧相同"不能算。

#### TODO 3：本轮输出

**思路：** 接受的前缀，加上 `target_tokens` 在第 `accepted_length` 个位置的那个 token。

**关键一行：**
```python
emitted_tokens = list(candidates[:accepted_length]) + [int(target_tokens[accepted_length])]
all_candidates_accepted = accepted_length == len(candidates)
```

**为什么这样写：** `target_tokens` 的长度是 $K+1$，下标 `accepted_length` 恰好统一了两种情况：中途不匹配时它指向主模型在不匹配位置给的修正 token；全部接受时 `accepted_length == K`，它指向多出来的 bonus 位置。一行代码两种语义，不需要 `if`。

**常见错法：** 全部接受时忘了加 bonus，只返回 `candidates`；或者不匹配时把候选的后缀也带上。前者少赚一个 token，后者直接输出了未经验证的内容——测试里 `emitted_tokens == [4, 5, 9]` 对应的候选是 `[4, 5, 6]`，6 必须不在输出里。

### Step 5：GPU 实验——真实 backend 上的原生 MTP

CPU 题目区验证的是候选形成和贪心验证的逻辑，它不知道一个真实 checkpoint 里到底有几个 MTP 层、vLLM 会不会真的走 MTP 路径、也测不出时间。GPU 实验要回答的是：同一个 checkpoint，"只用主模型"和"主模型 + MTP-1"两条路径能不能成功配对，输出是否一致，接受长度、Median/P95、吞吐和峰值显存各是多少。注意实验的前半段——确认 checkpoint 含 MTP 层、让主模型路径先完整跑通——和 MTP 本身一样重要：环境或 kernel 的问题会在这一步暴露，而不该被记成 MTP 的性能结论。

#### 5.1 环境与 workload

本实验使用隔离的 `INF-VLLM-MTP` 环境。首次运行先完成[推理优化实验 Quickstart](../topic_discussion/inference_optimization/quickstart.md#inf-vllm-mtp-quickstart)；已有环境先通过同名 Profile 预检。只有主模型路径完成模型加载、kernel 初始化、预热和写死输入的生成之后，才启用 MTP-1。

| 配置层 | 本节的设定 | 修改规则 |
| --- | --- | --- |
| 运行环境 | `INF-VLLM-MTP`；隔离的 vLLM、PyTorch、CUDA wheel 与 Triton 组合 | 更换驱动、wheel 或 Triton 后使用新运行标签，并先重跑主模型路径 |
| 12 GB 配置 | `Qwen/Qwen3.5-0.8B`；MTP-1；BF16；batch 1；`max_model_len=2048`；显存利用率 0.75 | checkpoint 必须包含原生 MTP 层；候选深度不超过可用层数 |
| 24 GB 配置 | 先复用 0.8B 配对，再扩大上下文、并发或重复次数 | 9B BF16 不作为默认配置；更换模型后单独登记 workload 版本 |
| 对照 | 同一 checkpoint 的主模型 → 主模型 + MTP-1 | 两条路径共用输入、dtype、生成长度、随机种子和执行模式 |
| 测量 | 输出一致性、接受长度、Median/P95、吞吐、峰值显存和失败阶段 | 初始化和下载不计入正式延迟；两条路径分别预热 |
| 结果文件 | `benchmarks/results/inf_mtp_<run_label>.json` | 正式运行使用唯一标签；失败结果不覆盖 |


```python
RUN_MODE = 'real_backend'  # 默认 real_backend；只做配置检查时改为 dry_run。
RUN_LABEL = 'rtx5070ti_qwen35_08b_mtp1_v4'  # 默认不覆盖实测结果；真实复测改为唯一标签。
PROFILE_ID = 'INF-VLLM-MTP'  # 原生 MTP 使用独立 vLLM 环境。
SEMANTIC_ID = 'INF-MTP'  # 稳定机制身份。
WORKLOAD_ID = 'inf-native-mtp-qwen35-08b'  # 稳定工作负载身份。
WORKLOAD_VERSION = 'v2'  # v2 固定原生 MTP-1 与隔离 worker 配对协议。
MODEL_ID = 'Qwen/Qwen3.5-0.8B'  # 默认原生 MTP 小模型。
MODEL_REVISION = 'main'  # 运行时自动解析并记录不可变 commit。
DTYPE = 'bfloat16'  # baseline 与 candidate 共用 BF16。
MAX_NEW_TOKENS = 64  # 每条提示固定生成 64 Token。
MAX_MODEL_LEN = 2048  # 单卡机制实验只保留短上下文容量。
GPU_MEMORY_UTILIZATION = 0.75  # 为 Notebook 与 backend 保留显存余量。
ENFORCE_EAGER = False  # 两条路径共用 vLLM 默认编译/CUDA Graph 路径。
WARMUP_STEPS = 2  # 每条路径独立预热 2 次。
REPEAT_STEPS = 5  # 每条路径正式重复 5 次。
SPECULATIVE_TOKENS = 1  # checkpoint 只有一个原生 MTP 层。
SEED = 42  # 固定贪心生成与 backend 随机状态。
PROMPTS = [
    'Explain why speculative decoding can reduce latency in two concise paragraphs.',
    'Write a Python function that checks whether a string is a palindrome, then explain its complexity.',
    'A shop applies a 20 percent discount and then a 10 percent tax. Explain the final price for an item costing 100 dollars.',
]
OUTPUT_RELATIVE_PATH = f'02_PyTorch_Algorithms/benchmarks/results/inf_mtp_{RUN_LABEL}.json'

```


#### 5.2 执行真实 backend 配对并保存 JSON

本单元先读模型配置确认原生 MTP 层的数量，再用两个独立子进程顺序执行主模型路径与 MTP-1 路径。初始化和模型下载不计入正式延迟；两条路径分别预热后才采集结果。


```python
import importlib
from tools.project_runtime import resolve_project_root
import tools.mtp_mechanism_runtime as mtp_runtime

mtp_runtime = importlib.reload(mtp_runtime)
RESULT_PATH = resolve_project_root() / OUTPUT_RELATIVE_PATH
GPU_REPORT = mtp_runtime.run_native_mtp_benchmark(
    semantic_id=SEMANTIC_ID, profile_id=PROFILE_ID, run_mode=RUN_MODE, run_label=RUN_LABEL,
    workload_id=WORKLOAD_ID, workload_version=WORKLOAD_VERSION,
    model_id=MODEL_ID, model_revision=MODEL_REVISION, dtype=DTYPE, prompts=PROMPTS,
    max_new_tokens=MAX_NEW_TOKENS, max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION, enforce_eager=ENFORCE_EAGER,
    warmup_steps=WARMUP_STEPS, repeat_steps=REPEAT_STEPS,
    speculative_tokens=SPECULATIVE_TOKENS, seed=SEED, output_path=RESULT_PATH,
)
GPU_REPORT

```


#### 5.3 读取结果

读取模型信息、两条路径和配对比较。`failure` 非空时先处理环境、模型或 backend 的问题，不解释性能字段。


```python
import json
from pathlib import Path

report = json.loads(Path(RESULT_PATH).read_text(encoding='utf-8'))
{
    'semantic_id': report.get('semantic_id'),
    'workload': report.get('workload'),
    'model_contract': report.get('model_contract'),
    'comparison': report.get('comparison'),
    'failure': report.get('failure'),
    'evidence_level': report.get('evidence_level'),
    'decision': report.get('decision'),
}

```


#### 5.4 解释结果并作出决策

RTX 5070 Ti 上已完成一次真实 backend 尝试。Qwen3.5-0.8B 的模型配置和一个原生 MTP 层都被正确识别，但主模型路径在 vLLM 初始化 Triton RoPE kernel 时失败（`device kernel image is invalid`），因此本次运行没有产生 baseline / candidate 配对结果。

失败发生在启用 MTP **之前**，说明是当前驱动、CUDA 12.9 wheel、vLLM 与 Triton 的组合跑不起来——这既不是 MTP 的性能结论，也不是显存不足。下一次运行要先让主模型路径完整生成写死的输出，再启用 MTP-1。

| 观察项 | 读取字段 | 当前结果 | 解释与动作 |
| --- | --- | --- | --- |
| 模型信息 | `model_contract`、resolved commit | Qwen3.5-0.8B commit `2fc0636`；识别 1 个 MTP 层 | checkpoint 支持 MTP-1 |
| backend 启动 | `failure`、`evidence_level` | Triton RoPE kernel 报 `device kernel image is invalid` | 主模型路径没跑起来，保留失败时的环境快照 |
| 正确性与推进 | `comparison`、接受统计 | 未产生 | 修复执行栈后重新生成配对结果 |
| 性能与资源 | Median/P95、吞吐、峰值显存 | 未产生 | 不用初始化失败的耗时推断性能 |
| 决策 | `evidence_level`、`decision` | `failed_real_backend_attempt`；`reject_until_fixed` | 在兼容环境从主模型路径开始再跑 |

<a id="cheatsheet"></a>
## 速查卡

| | |
|:--|:--|
| **一句话机制** | 把 draft 做进模型自身：checkpoint 内的 MTP 模块复用主模型隐藏状态，每个深度猜一个更远的 token，按深度排成候选块；主模型一次前向验证全部，贪心下"候选 == argmax"即接受，首次不匹配处用主模型的 token 修正并停止；全中时第 $K+1$ 个位置白送一个 bonus。 |
| **关键公式与数量级** | 候选块 = 各深度 argmax 的前 $\min(\text{budget}, \text{depth})$ 个；接受长度 $L \in [0, K]$，每轮推进 $L+1$ 个 token。DeepSeek-V3 论文报告第二个 token 接受率 85%–90%、解码 TPS 约 1.8×。本节实验 Qwen3.5-0.8B 含 1 个 MTP 层；一次真实 backend 尝试因 Triton RoPE kernel 失败未产生配对结果。 |
| **常见坑** | 1. 候选按分数重排——顺序就是位置，不能动。2. 数"相同的个数"而不是"连续前缀长度"：`[4,7,6]` vs `[4,5,6]` 是 1 不是 2。3. 贪心验证用在采样路径上——不再无损，要换 23 节的概率规则。4. 把层数当成所有 backend 的候选上限；本题只截取已有 logits 行。5. 接受长度高 ≠ 提速：MTP 模块的额外计算和验证前向要一起算。 |
| **延伸阅读** | 专题正文 [03 解码策略](../topic_discussion/inference_optimization/03_decoding_strategies.md) · 判断手册 [按现象选择机制](../topic_discussion/inference_optimization/casebook.md#按现象选择机制) · 另一条路 [INF-EAGLE](./INF-EAGLE_Feature_Drafter_and_Candidate_Tree.md)：给没有 MTP 层的模型外挂一个复用隐藏状态的 drafter · 预算怎么定 [INF-PROPOSAL](./INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.md) · 真实后端 [68](./68_Speculative_Decoding_Benchmark.md) |

## 相关阅读

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) —— Section 2.2 的 MTP 模块结构（隐藏状态 + 下一 token embedding → 一个 block → 共享输出头），以及推理侧接受率的报告
- [vLLM Speculators：MTP](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/decision_guide/) —— 哪些模型带 MTP 层、`num_speculative_tokens` 怎么配
- [Part 02 · INF-EAGLE 特征草稿器与候选树](./INF-EAGLE_Feature_Drafter_and_Candidate_Tree.md) —— 同样复用隐藏状态，但 drafter 是训练后外挂的
- [Part 02 · INF-PARALLEL-DRAFTING 并行草稿与置信度调度](./INF-PARALLEL-DRAFTING_DFlash_and_DSpark.md) —— 不串行递推、一次并行出一块候选