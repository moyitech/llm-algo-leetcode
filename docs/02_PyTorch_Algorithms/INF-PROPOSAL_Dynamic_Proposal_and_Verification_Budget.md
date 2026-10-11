# INF-PROPOSAL. Dynamic Proposal and Verification Budget | 动态候选生成与验证预算

**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `投机解码`, `动态候选`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

23 和 35 节都默认了一个参数：每轮猜几个 token（记作 $k$）。这个数字远比看起来重要。$k$ 太小，target 一次验证只带回一两个 token，批量验证的便宜没占到；$k$ 太大，draft 多猜的那几个大概率被拒，白跑的 draft 前向照样要付钱。更棘手的是，最优的 $k$ 不固定——不同输入的可预测性不同，同一段生成中的分布也会变化，适合的候选数需要随运行情况调整。

本节把这件事拆成三个可以单独实现、单独测试的小机制：一个根据近期接受率上调或下调 $k$ 的控制器，一个找出连续接受边界的验证函数，和一张把节省的时间与 draft、verify、修正成本放在一起算的账。学完后你能解释本节 GPU 实验里那个反直觉的结果——固定 $k=5$ 让 target 前向从 192 次降到 61 次，端到端却**变慢**了 6%——以及为什么按置信度提前停止 draft 反而赢了。

## 前置阅读

- [Part 02 · 23 投机解码](./23_Speculative_Decoding.md) —— draft / verify / accept / correct 的循环；本节在它外面加一层"每轮猜几个"的控制。
- [Part 02 · 35 原生 MTP](./35_Multi_Token_Decoding.md) —— 连续接受长度的概念；本节的验证函数和它的 `continuous_match_length` 是同一件事，多返回一个拒绝位置。

### Step 1：痛点与直觉——target 调用少了，怎么反而更慢

想象你把 23 节的方案搬到了真机上：Qwen2.5-1.5B 做 target，0.5B 做 draft，每轮固定猜 5 个，三条提示各生成 64 个 token。结果 target 的前向次数从 192 降到 61——平均每次 target 前向推进 3 个 token，看起来投机解码完全起效了。但一掐表，端到端延迟比不用 draft 还慢了 5.9%。

这份历史记录还显示，draft 做了 289 次前向，而 target 只省下 131 次。模型参数少三分之二，不代表每次前向就便宜三分之二：草稿计算、采样、调度、KV 管理和更长的验证都会计入总时间。当前汇总表不足以确定其中谁占主导，不能据此断言“Python 开销就是根因”；要做这种归因，还需要分阶段计时或 profiler。

换个说法：投机解码是在"用 draft 的时间换 target 的时间"，划不划算取决于汇率。接受率告诉你能换回多少，但不告诉你付出了多少——猜了 5 个只收下 2 个，那 3 个白猜的 draft 前向照付。所以 $k$ 不是越大越好，而是要让"多猜一个的期望收益"刚好盖住"多跑一次 draft 的成本"。而这个平衡点随内容变：当 draft 和 target 更一致时可以尝试多猜；当分歧增多时应缩短候选，并用耗时检验是否划算。

本节把这件事做成三个函数：`choose_proposal_budget` 根据最近几轮的接受率把 $k$ 调大、调小或保持；`verify_candidate_prefix` 找到连续接受的边界和首个拒绝位置；`estimate_net_gain` 把节省的时间减去 draft、verify 和修正的成本，算出这一轮到底赚了还是亏了。

<img src="../public/02_PyTorch_Algorithms/inf_proposal_loop.svg" alt="动态候选与验证闭环" width="1200" height="520">

### Step 2：机制怎么走——一张账、一个控制器

**先算账.** 假设 target 逐 token 生成是 10 ms/token，draft 一次前向 3 ms，target 一次验证前向 12 ms（这是教学假设，不是 Step 5 的计时拆分），拒绝后的修正忽略不计。本轮猜 $k = 5$：

- 5 个全部接受：省下 $5 \times 10 = 50$ ms（这 5 个 token 不用 target 逐个生成了），花了 $5 \times 3 + 12 = 27$ ms，净赚 23 ms。
- 只接受 2 个：省下 20 ms，花的还是 27 ms（5 次 draft 一次都少不了），净亏 7 ms。
- 刚好接受 3 个：省 30、花 27，净赚 3 ms——这是整数接受数中开始盈利的最小值；连续意义的平衡点为 27/10=2.7。

现在把 draft 的单价从 3 ms 改成 6 ms（另一个假设）：5 个全接受才赚 8 ms，接受 4 个就亏 2 ms。也就是说 draft 不够便宜时，$k = 5$ 几乎只有满分才不亏——这说明单价会改变判断，但不能用假设单价反推出真机各阶段耗时。

本题成本估算的规则很简单：节省 = 接受数 × target 逐 token 成本；成本 = **全部**候选数 × draft 单价 + 一次验证 + （若有拒绝）一次修正。注意成本里乘的是"猜了几个"而不是"收了几个"。

**再调预算.** 既然最优 $k$ 随内容变，就让它跟着最近的接受率走。控制器只做三件事：最近几轮平均接受率 ≥ `high`（默认 0.8）就把 $k$ 加 1；≤ `low`（默认 0.4）就减 1；在中间就不动。再用 `min_k` / `max_k` 夹住。比如当前 $k = 4$，最近三轮接受率 `[1, 1, 1]`，平均 1.0 ≥ 0.8，下一轮 $k = 5$；最近三轮 `[0, 0, 1]`，平均 0.33 ≤ 0.4，下一轮 $k = 3$；最近三轮 `[0.5, 0.7, 0.6]`，落在 0.4 到 0.8 之间，保持 4 不变。

为什么是 ±1 而不是直接按接受率算一个"理想 $k$"？因为接受率是噪声很大的信号——三轮里一轮全拒就能把平均值砸到 0.67。每次只挪一步，加上中间那段不动的区间（这里是无调整区间），减少小幅噪声导致的调整，但并不保证控制器永不振荡。

<details>
<summary>速查表：三种预算策略分别在哪里起作用</summary>

| 策略 | 调整发生在 | 依据的信号 | Transformers 里对应 |
|---|---|---|---|
| 固定预算 | 不调 | — | `num_assistant_tokens` 固定，`num_assistant_tokens_schedule='constant'` |
| 跨轮启发式 | 每轮结束后 | 上一轮是否全部接受：全接受则 +2，否则 −1 | `num_assistant_tokens_schedule='heuristic'` |
| 轮内置信度停止 | 每猜一个 token 时 | draft 对当前 token 的最大概率低于阈值就不再往下猜 | `assistant_confidence_threshold` |
| 本节的控制器 | 每轮结束后 | 最近 $n$ 轮的平均接受率，带无调整区间 | 教学控制器，规则与 Transformers heuristic 不同 |

</details>

**这个估算省略了什么？** 函数只把接受的 draft token 计入节省，没有把修正或 bonus token 对应的 baseline 时间计入，因此是部分成本模型，不是整轮端到端收益。完整比较应按实际输出的 token 数计算 baseline，再扣除所有阶段耗时；若 `verify_ms` 已含修正成本，也不能重复相加。

### Step 3：边界与陷阱——三个会悄悄算错的地方

**陷阱一：成本按"接受数"算.** 最常见的错。`estimate_net_gain(accepted=2, proposed=5, ...)` 里 draft 成本必须乘 `proposed=5`，不是 `accepted=2`——那 3 个被拒的候选，draft 也是老老实实一个一个算出来的。按接受数算成本会让每轮都显得赚钱，和真机结果对不上。

**陷阱二：draft 比 target 验证范围更长.** `verify_candidate_prefix([1, 2, 3], [1, 2])`：draft 猜了 3 个，但 target 只给了 2 个位置的结果（比如生成到了长度上限）。前两个一致，第三个没法验证——没被验证的不能算接受，所以 `accepted = 2`，并且 `rejected_at = 2`：第 3 个候选的命运是"被截断"，对下一轮来说和被拒绝一样，要从位置 2 重新开始。如果这里返回 `rejected_at=None`，调用方会以为全部通过，把没验证过的 token 3 提交出去。

**陷阱三：控制器不夹边界.** 接受率连续走高时 $k$ 会一直加；没有 `max_k` 它会涨到几十，draft 跑几十步却只被收下几个。反过来没有 `min_k` 它会降到 0，等于关掉投机解码且再也回不来——因为 $k = 0$ 时不会再产生任何接受率样本。`min(max(current_k + adjustment, min_k), max_k)` 这一行不是防御性编程，是控制器能长期工作的前提。

**顺带一提：两种"动态"是不同层面的事.** 本节的控制器在轮与轮之间调 $k$；Transformers 的 `assistant_confidence_threshold` 是在一轮内部、draft 每猜一个 token 就看一眼自己的置信度，低于阈值就停。前者看历史，后者看当下。Step 5 的真机结果里赢的是后者——它把每轮 draft 前向从 4.7 次压到 2.5 次，target 调用从 61 增到 73 次；相对普通解码 baseline 的延迟约少 14.6%，相对固定预算候选约少 19.4%。两者可以叠加。

<img src="../public/02_PyTorch_Algorithms/inf_proposal_budget.svg" alt="候选预算的收益与浪费" width="1200" height="520">

### Step 4：任务——实现控制器、验证边界和账本

题目区已经写好所有输入校验和返回结构，三个 TODO 各只有一两行。

| TODO | 要做什么 | 必须满足 | 对应测试 |
| --- | --- | --- | --- |
| TODO 1 · `choose_proposal_budget` | 平均接受率 ≥ `high` 则 +1，≤ `low` 则 −1，否则 0 | 结果夹在 `[min_k, max_k]` | `test_budget_adaptation`：`[1,1,1]` 4→5，`[0,0,1]` 4→3，`[1]` 在 8 保持 8 |
| TODO 2 · `verify_candidate_prefix` | 找到第一个 draft ≠ target 的下标；全一致则 `None` | 后续逻辑会处理"draft 更长"的截断情形 | `test_contiguous_verification`：`[1,2,3,4]` vs `[1,2,9,4]` → 接受 2、拒绝于 2 |
| TODO 3 · `estimate_net_gain` | `saved = accepted × baseline`；`extra = proposed × draft + verify + 修正（仅有拒绝时）` | draft 成本乘的是 `proposed` | `test_net_gain_ledger`：全接受净赚 6.2 ms；全拒绝为负 |


```python
from statistics import mean

```



```python
def choose_proposal_budget(
    recent_acceptance,
    current_k,
    min_k=1,
    max_k=8,
    high=0.8,
    low=0.4,
):
    """根据近期接受率证据调整下一轮候选长度。"""
    if not (1 <= min_k <= current_k <= max_k):
        raise ValueError('候选预算边界非法')
    if not (0.0 <= low < high <= 1.0):
        raise ValueError('接受率阈值非法')
    if any(not 0.0 <= value <= 1.0 for value in recent_acceptance):
        raise ValueError('接受率必须位于 [0, 1]')
    if not recent_acceptance:
        return current_k
    rate = mean(recent_acceptance)
    # TODO 1（动态预算）：根据 high / low 选择 +1、-1 或 0。
    # adjustment = ???  # 根据 high/low 区间取 +1、-1 或 0。
    raise NotImplementedError('TODO 1：请根据近期接受率选择预算调整量')
    return min(max(current_k + adjustment, min_k), max_k)


def verify_candidate_prefix(draft_tokens, target_tokens):
    """比较草稿与目标 token，返回连续接受数和首个拒绝位置。"""
    # TODO 2（连续验证）：找到第一个 token 不同的位置；全部相同则为 None。
    # rejected_at = ???  # 首个不匹配索引；可比较前缀完全一致时为 None。
    raise NotImplementedError('TODO 2：请找到首个不匹配位置')
    accepted = min(len(draft_tokens), len(target_tokens)) if rejected_at is None else rejected_at
    if rejected_at is None and len(draft_tokens) > len(target_tokens):
        rejected_at = accepted
    return {'accepted': accepted, 'rejected_at': rejected_at}


def estimate_net_gain(
    accepted,
    proposed,
    baseline_token_ms,
    draft_token_ms,
    verify_ms,
    correction_ms=0.0,
):
    """返回候选路径节省时间、额外成本与净收益。"""
    if not 0 <= accepted <= proposed:
        raise ValueError('accepted 必须位于 0..proposed')
    if min(baseline_token_ms, draft_token_ms, verify_ms, correction_ms) < 0:
        raise ValueError('时间成本不得为负')
    # TODO 3（收益账本）：分别计算基线节省和候选路径额外成本。
    # saved_ms = ???  # 已接受 token 按基线逐 token 执行可节省的时间。
    # extra_cost_ms = ???  # 全部草稿、一次验证和必要修正的成本。
    raise NotImplementedError('TODO 3：请计算节省与额外成本')
    return {
        'saved_ms': saved_ms,
        'extra_cost_ms': extra_cost_ms,
        'net_gain_ms': saved_ms - extra_cost_ms,
    }

```


### 测试

`test_budget_adaptation` 看控制器在高、低、空和到达上界四种情况下的行为；`test_budget_rejects_invalid_acceptance` 确认接受率超出 $[0,1]$ 会被拒绝；`test_contiguous_verification` 覆盖中途拒绝、全部一致和 draft 比 target 长三种情况；`test_net_gain_ledger` 用一组全接受和一组全拒绝的数字检查账本正负。


```python
def test_budget_adaptation():
    """预算应随近期接受率增减，并保持上下界。"""
    assert choose_proposal_budget([1, 1, 1], 4) == 5
    assert choose_proposal_budget([0, 0, 1], 4) == 3
    assert choose_proposal_budget([], 4) == 4
    assert choose_proposal_budget([1], 8) == 8


def test_budget_rejects_invalid_acceptance():
    """接受率超出 [0, 1] 时应拒绝输入。"""
    for invalid in ([1.2], [-0.1]):
        try:
            choose_proposal_budget(invalid, 4)
        except ValueError:
            pass
        else:
            raise AssertionError('接受率必须位于 [0, 1]')


def test_contiguous_verification():
    """验证只能接受首个分歧之前的连续前缀。"""
    assert verify_candidate_prefix([1, 2, 3, 4], [1, 2, 9, 4]) == {'accepted': 2, 'rejected_at': 2}
    assert verify_candidate_prefix([1, 2], [1, 2]) == {'accepted': 2, 'rejected_at': None}
    assert verify_candidate_prefix([1, 2, 3], [1, 2]) == {'accepted': 2, 'rejected_at': 2}


def test_net_gain_ledger():
    """收益账本应区分有效推进与拒绝修正成本。"""
    assert estimate_net_gain(4, 4, 2.0, 0.2, 1.0)['net_gain_ms'] == 6.2
    assert estimate_net_gain(0, 4, 2.0, 0.5, 2.0, 1.0)['net_gain_ms'] < 0


test_budget_adaptation()
test_budget_rejects_invalid_acceptance()
test_contiguous_verification()
test_net_gain_ledger()
print('✅ 动态候选机制测试通过')

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
# TODO 1：根据近期接受率调整候选预算。
def choose_proposal_budget(
    recent_acceptance,
    current_k,
    min_k=1,
    max_k=8,
    high=0.8,
    low=0.4,
):
    """根据近期接受率证据调整下一轮候选长度。"""
    if not (1 <= min_k <= current_k <= max_k):
        raise ValueError('候选预算边界非法')
    if not (0.0 <= low < high <= 1.0):
        raise ValueError('接受率阈值非法')
    if any(not 0.0 <= value <= 1.0 for value in recent_acceptance):
        raise ValueError('接受率必须位于 [0, 1]')
    if not recent_acceptance:
        return current_k
    rate = mean(recent_acceptance)
    adjustment = 1 if rate >= high else -1 if rate <= low else 0
    return min(max(current_k + adjustment, min_k), max_k)


# TODO 2：找到连续接受边界。
def verify_candidate_prefix(draft_tokens, target_tokens):
    """比较草稿与目标 token，返回连续接受数和首个拒绝位置。"""
    rejected_at = next(
        (index for index, pair in enumerate(zip(draft_tokens, target_tokens)) if pair[0] != pair[1]),
        None,
    )
    accepted = min(len(draft_tokens), len(target_tokens)) if rejected_at is None else rejected_at
    if rejected_at is None and len(draft_tokens) > len(target_tokens):
        rejected_at = accepted
    return {'accepted': accepted, 'rejected_at': rejected_at}


# TODO 3：计算候选路径净收益。
def estimate_net_gain(
    accepted,
    proposed,
    baseline_token_ms,
    draft_token_ms,
    verify_ms,
    correction_ms=0.0,
):
    """返回候选路径节省时间、额外成本与净收益。"""
    if not 0 <= accepted <= proposed:
        raise ValueError('accepted 必须位于 0..proposed')
    if min(baseline_token_ms, draft_token_ms, verify_ms, correction_ms) < 0:
        raise ValueError('时间成本不得为负')
    saved_ms = accepted * baseline_token_ms
    extra_cost_ms = proposed * draft_token_ms + verify_ms + (correction_ms if accepted < proposed else 0.0)
    return {
        'saved_ms': saved_ms,
        'extra_cost_ms': extra_cost_ms,
        'net_gain_ms': saved_ms - extra_cost_ms,
    }

```


### 解析

#### TODO 1：预算控制器

**思路：** 三段式——高于上阈值加一，低于下阈值减一，中间不动。

**关键一行：**
```python
adjustment = 1 if rate >= high else -1 if rate <= low else 0
```

**为什么这样写：** 用两个阈值而不是一个，中间那段 `(low, high)` 就是无调整区间：接受率在 0.5–0.7 之间小幅波动时 $k$ 不会跟着抖。每次只挪 1 是刻意的——接受率只有最近几轮的样本，噪声大，一步到位容易过冲。夹边界那行在骨架里已经写好。

**常见错法：** `adjustment = round((rate - 0.6) * 10)` 之类按比例调。接受率从 1.0 掉到 0.3 时 $k$ 会一次跳 7，下一轮接受率又回升再跳回来，控制器自己制造震荡。

#### TODO 2：连续接受边界

**思路：** 从左到右配对比较，第一个不同的下标就是拒绝位置；走完没遇到不同就是 `None`。

**关键一行：**
```python
rejected_at = next(
    (index for index, pair in enumerate(zip(draft_tokens, target_tokens)) if pair[0] != pair[1]),
    None,
)
```

**为什么这样写：** `zip` 自动停在较短的那个序列末尾，所以 draft 比 target 长时不会越界；`next(..., None)` 在生成器耗尽时返回默认值，恰好表达"没有分歧"。骨架里接下来两行把"draft 更长但前缀全对"的情况改写成 `rejected_at = accepted`——没被验证的不算通过。

**常见错法：** 用 `sum(a == b for a, b in zip(...))` 数相同的个数。`[1, 9, 3]` vs `[1, 2, 3]` 会得到 2，但第三个 3 是在"第二个是 9"这个错误前提下猜的，不能算；正确答案是接受 1、拒绝于 1。

#### TODO 3：候选成本估算

**思路：** 节省只记接受的 token；成本记全部 draft、一次验证、以及有拒绝时的一次修正。

**关键一行：**
```python
saved_ms = accepted * baseline_token_ms
extra_cost_ms = proposed * draft_token_ms + verify_ms + (correction_ms if accepted < proposed else 0.0)
```

**为什么这样写：** `accepted < proposed` 在本题中表示有候选未被接受，包括验证范围不足的截断——全部接受时两者相等，没有修正这一步（有的是 bonus，但那是验证前向顺手产出的，成本已含在 `verify_ms` 里）。测试 `estimate_net_gain(4, 4, 2.0, 0.2, 1.0)`：省 8.0，花 $4 \times 0.2 + 1.0 = 1.8$，净 6.2。

**常见错法：** `extra_cost_ms = accepted * draft_token_ms + ...`。Step 3 陷阱一——被拒的候选 draft 也算过了，成本不会因为结果被扔掉而消失。这个错会低估候选路径成本，和 Step 5 里"target 调用少了但更慢"的真机结果矛盾。

CPU 练习只验证控制器、边界和算式本身。真实的 `baseline_token_ms`、`draft_token_ms`、`verify_ms` 要在写死的模型、输入和生成规则下测出来，不能拿随机数或论文里别人的数字代入。

### Step 5：GPU 实验——三种预算策略的真机对照

CPU 部分验证了控制器和账本的逻辑，但账本里的三个单价是编的。真机上要回答的是：同一对模型、同一组提示，固定预算、跨轮启发式和轮内置信度停止三种策略，各自让 target 和 draft 跑了多少次前向、端到端快了还是慢了、输出是否和不用 draft 时逐字一致。这是本节最有信息量的部分——结果和"target 调用越少越快"的直觉相反。

#### 5.1 环境与 workload

使用 `INF-CUDA-MECHANISM` Profile；参考环境为单卡 12 GB 以上显存，首次运行需要下载目标模型与助手模型，不需要外部数据集。环境创建和预检见[推理优化实验 Quickstart](../topic_discussion/inference_optimization/quickstart.md#inf-cuda-mechanism-quickstart)，实验登记见[推理优化实验资产表](../topic_discussion/inference_optimization/experiment_assets.md)。运行前的 Profile 预检必须返回 `status=ok`。

| 配置层 | 本实验的设定 | 修改规则 |
| --- | --- | --- |
| 模型 | Qwen2.5 1.5B 目标模型 + 0.5B 助手模型；同 tokenizer 与 dtype | 更换任一模型或 revision 后不与旧结果合并 |
| workload | 三条写死的提示；贪心生成 64 token | 提示、长度或采样语义变化后递增 workload version |
| 对照 | 无助手 baseline → constant / heuristic / dynamic 三个候选 | 共用输入、目标模型、生成参数、预热与重复规则 |
| 测量 | 输出一致性、目标/助手前向次数、median/P95、峰值显存 | 先过一致性，再解释调用次数和延迟 |
| 结果 | 候选矩阵与 JSON 路径 | 正式运行使用唯一 `RUN_LABEL` |


```python
RUN_MODE = 'real_gpu'  # 默认值 dry_run：只登记计划；改为 real_gpu 执行真实辅助生成矩阵。
RUN_LABEL = 'rtx5070ti_proposal_v2'  # 默认值 default；正式复测改为唯一标签，避免覆盖旧结果。
ENVIRONMENT_PROFILE = 'INF-CUDA-MECHANISM'  # 默认单卡推理机制 Profile；预检必须返回 status=ok。
WORKLOAD_ID = 'inf-dynamic-assisted-generation'  # 稳定实验身份；不要在 ID 中编码运行标签。
WORKLOAD_VERSION = 'v1'  # 默认 v1：比较无助手、固定、启发式与动态预算。
TARGET_MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'  # 默认目标模型；负责最终验证和输出。
TARGET_MODEL_REVISION = 'main'  # 默认 main；5.2 自动解析并记录实际 commit。
ASSISTANT_MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # 默认助手模型；与目标模型共享 tokenizer。
ASSISTANT_MODEL_REVISION = 'main'  # 默认 main；5.2 自动解析并记录实际 commit。
MODEL_DTYPE = 'float16'  # 默认 float16；更换 dtype 后结果不可直接合并。
PROMPTS = (  # 默认三类短提示；修改文本后递增 workload version。
    '请用三点解释 KV Cache 为什么随上下文增长。',
    'Write a Python function that merges overlapping intervals.',
    'A serving system has high TPOT but low queue wait. Diagnose likely causes.',
)
MAX_NEW_TOKENS = 64  # 默认生成 64 token；增大会提高复测时间与显存占用。
INITIAL_ASSISTANT_TOKENS = 5  # 默认初始候选预算 5 token。
DYNAMIC_CONFIDENCE_THRESHOLD = 0.4  # 默认动态停止阈值 0.4；只用于置信度候选。
ASSISTED_CANDIDATES = (  # 三个候选分别隔离固定预算、启发式预算和置信度提前停止。
    {'name': 'constant', 'schedule': 'constant', 'confidence_threshold': 0.0},
    {'name': 'heuristic_transient', 'schedule': 'heuristic_transient', 'confidence_threshold': 0.0},
    {'name': 'dynamic_confidence', 'schedule': 'constant', 'confidence_threshold': DYNAMIC_CONFIDENCE_THRESHOLD},
)
WARMUP_STEPS = 1  # 默认每条路径预热 1 次；不进入统计。
REPEAT_STEPS = 5  # 默认重复 5 次；用于计算 median 与 P95。
SEED = 42  # 默认随机种子 42；贪心生成仍记录以固定环境状态。
OUTPUT_RELATIVE_PATH = f'02_PyTorch_Algorithms/benchmarks/results/inf_proposal_{RUN_LABEL}.json'  # 默认仓库相对结果路径。

```


#### 5.2 执行并保存 JSON

执行单元只读 5.1 的配置。它先生成无助手的参考输出，再依次运行三种助手预算策略；每次测量都重置候选预算，记录目标/助手前向次数、输出一致性、median/P95、峰值显存和失败原因。


```python
from tools.notebook_runtime import package_version, resolve_model_commit
from tools.project_runtime import resolve_project_root

import copy
import json
import math
import os
import platform
import statistics
import time

import torch


REPO_ROOT = resolve_project_root()
OUTPUT_PATH = REPO_ROOT / OUTPUT_RELATIVE_PATH


def percentile_95(values):
    """返回离散样本的最近秩 P95。"""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def greedy_generation_config(model, tokenizer):
    """从模型默认配置复制一份无采样、无重复惩罚的贪心配置。"""
    config = copy.deepcopy(model.generation_config)
    config.do_sample = False
    config.temperature = None
    config.top_p = None
    config.top_k = None
    config.repetition_penalty = 1.0
    config.max_new_tokens = MAX_NEW_TOKENS
    config.min_new_tokens = 0
    config.use_cache = True
    config.pad_token_id = tokenizer.eos_token_id
    return config


def run_generation(model, tokenizer, prompts, assistant_model=None, generation_config=None):
    """运行完整提示集并记录输出、前向次数与 CUDA 延迟。"""
    target_calls = 0
    assistant_calls = 0

    def count_target(_module, _inputs):
        nonlocal target_calls
        target_calls += 1

    def count_assistant(_module, _inputs):
        nonlocal assistant_calls
        assistant_calls += 1

    target_hook = model.register_forward_pre_hook(count_target)
    assistant_hook = assistant_model.register_forward_pre_hook(count_assistant) if assistant_model is not None else None
    outputs = []
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        started.record()
        with torch.inference_mode():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
                generated = model.generate(
                    **inputs,
                    assistant_model=assistant_model,
                    generation_config=generation_config,
                )
                outputs.append(generated[0].detach().cpu().tolist())
        finished.record()
        torch.cuda.synchronize()
    finally:
        target_hook.remove()
        if assistant_hook is not None:
            assistant_hook.remove()
    generated_tokens = sum(max(0, len(ids) - len(tokenizer(prompt).input_ids)) for ids, prompt in zip(outputs, prompts))
    return {
        'outputs': outputs,
        'latency_ms': started.elapsed_time(finished),
        'target_forward_calls': target_calls,
        'assistant_forward_calls': assistant_calls,
        'generated_tokens': generated_tokens,
        'tokens_per_target_call': generated_tokens / max(1, target_calls),
        'peak_memory_bytes': int(torch.cuda.max_memory_allocated()),
    }


result = {
    'schema_version': 'inference-mechanism-gpu/v1',
    'semantic_id': 'INF-PROPOSAL',
    'environment_profile': ENVIRONMENT_PROFILE,
    'run_mode': RUN_MODE,
    'run_label': RUN_LABEL,
    'framework': {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'transformers': package_version('transformers'),
    },
    'hardware': {
        'cuda_available': torch.cuda.is_available(),
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
    'workload': {
        'id': WORKLOAD_ID,
        'version': WORKLOAD_VERSION,
        'target_model_id': TARGET_MODEL_ID,
        'target_revision': TARGET_MODEL_REVISION,
        'target_commit': None,
        'assistant_model_id': ASSISTANT_MODEL_ID,
        'assistant_revision': ASSISTANT_MODEL_REVISION,
        'assistant_commit': None,
        'dtype': MODEL_DTYPE,
        'prompt_count': len(PROMPTS),
        'max_new_tokens': MAX_NEW_TOKENS,
        'assisted_candidates': list(ASSISTED_CANDIDATES),
        'initial_assistant_tokens': INITIAL_ASSISTANT_TOKENS,
        'dynamic_confidence_threshold': DYNAMIC_CONFIDENCE_THRESHOLD,
        'warmup_steps': WARMUP_STEPS,
        'repeat_steps': REPEAT_STEPS,
        'seed': SEED,
    },
    'baseline': None,
    'candidates': [],
    'failure': None,
    'evidence_level': 'dry_run',
    'decision': 'not_measured',
}

if RUN_MODE == 'real_gpu':
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('RUN_MODE=real_gpu，但 CUDA 不可用')
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_dir = os.getenv('HF_HUB_CACHE') or None
        local_only = os.getenv('HF_HUB_OFFLINE') == '1'
        target_commit, _ = resolve_model_commit(TARGET_MODEL_ID, TARGET_MODEL_REVISION, cache_dir, local_only)
        assistant_commit, _ = resolve_model_commit(ASSISTANT_MODEL_ID, ASSISTANT_MODEL_REVISION, cache_dir, local_only)
        result['workload']['target_commit'] = target_commit
        result['workload']['assistant_commit'] = assistant_commit

        dtype = getattr(torch, MODEL_DTYPE)
        torch.manual_seed(SEED)
        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, revision=target_commit, cache_dir=cache_dir, local_files_only=local_only)
        target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, revision=target_commit, cache_dir=cache_dir, local_files_only=local_only, dtype=dtype).to('cuda').eval()
        target_generation_config = greedy_generation_config(target, tokenizer)
        target.generation_config = copy.deepcopy(target_generation_config)
        for _ in range(WARMUP_STEPS):
            run_generation(target, tokenizer, PROMPTS[:1], generation_config=target_generation_config)
        baseline_runs = [run_generation(target, tokenizer, PROMPTS, generation_config=target_generation_config) for _ in range(REPEAT_STEPS)]
        reference_outputs = baseline_runs[0]['outputs']
        result['baseline'] = {
            'name': 'target_only',
            'median_latency_ms': statistics.median(item['latency_ms'] for item in baseline_runs),
            'p95_latency_ms': percentile_95([item['latency_ms'] for item in baseline_runs]),
            'median_target_forward_calls': statistics.median(item['target_forward_calls'] for item in baseline_runs),
            'peak_memory_bytes': max(item['peak_memory_bytes'] for item in baseline_runs),
        }

        assistant = AutoModelForCausalLM.from_pretrained(ASSISTANT_MODEL_ID, revision=assistant_commit, cache_dir=cache_dir, local_files_only=local_only, dtype=dtype).to('cuda').eval()
        assistant_generation_config = greedy_generation_config(assistant, tokenizer)
        assistant.generation_config = copy.deepcopy(assistant_generation_config)

        for candidate in ASSISTED_CANDIDATES:
            assistant.generation_config.num_assistant_tokens_schedule = candidate['schedule']
            assistant.generation_config.num_assistant_tokens = INITIAL_ASSISTANT_TOKENS
            assistant.generation_config.assistant_confidence_threshold = candidate['confidence_threshold']
            runs = []
            for _ in range(REPEAT_STEPS):
                assistant.generation_config.num_assistant_tokens = INITIAL_ASSISTANT_TOKENS
                runs.append(run_generation(target, tokenizer, PROMPTS, assistant_model=assistant, generation_config=target_generation_config))
            median_ms = statistics.median(item['latency_ms'] for item in runs)
            outputs_match = all(item['outputs'] == reference_outputs for item in runs)
            result['candidates'].append({
                'candidate': candidate['name'],
                'schedule': candidate['schedule'],
                'confidence_threshold': candidate['confidence_threshold'],
                'outputs_match_baseline': outputs_match,
                'median_latency_ms': median_ms,
                'p95_latency_ms': percentile_95([item['latency_ms'] for item in runs]),
                'latency_ratio_vs_baseline': median_ms / result['baseline']['median_latency_ms'],
                'median_target_forward_calls': statistics.median(item['target_forward_calls'] for item in runs),
                'median_assistant_forward_calls': statistics.median(item['assistant_forward_calls'] for item in runs),
                'median_tokens_per_target_call': statistics.median(item['tokens_per_target_call'] for item in runs),
                'peak_memory_bytes': max(item['peak_memory_bytes'] for item in runs),
                'final_num_assistant_tokens': assistant.generation_config.num_assistant_tokens,
                'path_confirmed': True,
            })

        valid = [item for item in result['candidates'] if item['outputs_match_baseline'] and item['path_confirmed']]
        faster = [item for item in valid if item['latency_ratio_vs_baseline'] < 0.95]
        result['evidence_level'] = 'local_mechanism_benchmark'
        result['decision'] = 'accept' if faster else ('tune' if valid else 'reject')
    except Exception as error:
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
        result['evidence_level'] = 'failed_real_gpu_attempt'
        result['decision'] = 'reject_until_fixed'

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
result

```


#### 5.3 读取结果

只读取 5.2 保存的 JSON。先看环境、workload、失败、结果等级和决策，再比较三种预算策略的输出一致性、目标/助手前向次数、median/P95、峰值显存和相对延迟。


```python
import json
from tools.project_runtime import resolve_project_root

result_path = resolve_project_root() / OUTPUT_RELATIVE_PATH
report = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {'failure': {'type': 'not_run', 'message': str(result_path)}, 'candidates': []}
summary = {
    'schema_version': report.get('schema_version'),
    'framework': report.get('framework'),
    'hardware': report.get('hardware'),
    'workload': report.get('workload'),
    'failure': report.get('failure'),
    'evidence_level': report.get('evidence_level'),
    'decision': report.get('decision'),
}
matrix = [
    {
        'candidate': item.get('candidate'),
        'schedule': item.get('schedule'),
        'confidence_threshold': item.get('confidence_threshold'),
        'outputs_match_baseline': item.get('outputs_match_baseline'),
        'latency_ratio_vs_baseline': item.get('latency_ratio_vs_baseline'),
        'median_latency_ms': item.get('median_latency_ms'),
        'p95_latency_ms': item.get('p95_latency_ms'),
        'median_target_forward_calls': item.get('median_target_forward_calls'),
        'median_assistant_forward_calls': item.get('median_assistant_forward_calls'),
        'median_tokens_per_target_call': item.get('median_tokens_per_target_call'),
        'peak_memory_bytes': item.get('peak_memory_bytes'),
        'path_confirmed': item.get('path_confirmed'),
    }
    for item in report.get('candidates', [])
]
summary, report.get('baseline'), matrix

```


#### 5.4 解释结果并作出决策

先确认目标/助手模型 commit、提示、生成长度和贪心语义一致，再确认三条候选路径都启用了、输出都和无助手 baseline 逐字相同。然后比较目标前向次数、每次目标调用推进的 token 数、median/P95 和峰值显存。

`rtx5070ti_proposal_v2` 在 RTX 5070 Ti Laptop 12 GiB 上完成了配对。三种候选都保持贪心输出与仅目标模型一致，所以差别全在时间上：

- **固定预算 $k=5$**：目标前向中位数从 192 降到 61，但助手前向 289 次，延迟比 **1.059**——比不用 draft 还慢。每轮平均 4.7 次 draft 前向，而 0.5B 在 Python 循环里一次前向远不止 1.5B 的三分之一。
- **跨轮启发式**：目标前向 68 次、助手 256 次，延迟比 1.006——基本打平。少猜了一些，但还不够。
- **轮内置信度停止（阈值 0.4）**：目标前向 73 次、助手只有 186 次，延迟比 **0.854**，Median 从 2594.6 ms 降到 2216.9 ms。它多付了 12 次 target 前向，换来少跑 103 次 draft——这正是 Step 2 账本里"成本乘的是猜了几个"那条规则在真机上的体现。

| 观察项 | 读取字段 | 当前结果 | 解释与动作 |
| --- | --- | --- | --- |
| 运行与条件 | `failure`、模型 commit、workload、生成参数 | 无失败；目标 1.5B、助手 0.5B、贪心生成 64 token | 条件变化后不与旧结果合并 |
| 固定预算 | 目标/助手前向次数、相对延迟 | 61 / 289；1.059 | 调用减少但端到端更慢，拒绝当前配置 |
| 启发式预算 | 目标/助手前向次数、相对延迟 | 68 / 256；1.006 | 尚无稳定收益，继续调参 |
| 置信度动态候选 | 输出一致性、目标/助手前向次数、相对延迟 | 一致；73 / 186；0.854 | 接受为本机的候选策略 |
| 结果等级与下一步 | `evidence_level`、`decision` | `local_mechanism_benchmark`、`accept` | 扩大输入和负载后再下 backend 层面的结论 |

<a id="cheatsheet"></a>
## 速查卡

| | |
|:--|:--|
| **一句话机制** | 每轮猜几个（$k$）要跟着接受率走：最近几轮接受率 ≥ `high` 则 $k$+1，≤ `low` 则 $k$−1，中间不动，再夹在 `[min_k, max_k]`；本题部分成本估算 = 接受数 × target 单价 − （**全部**候选数 × draft 单价 + 一次验证 + 有拒绝时的修正）。 |
| **关键公式与数量级** | $\text{net} = a \cdot c_{\text{target}} - (k \cdot c_{\text{draft}} + c_{\text{verify}} + [a<k]\, c_{\text{corr}})$；忽略修正与 bonus 收益时的平衡接受数 $a^* = (k c_{\text{draft}} + c_{\text{verify}})/c_{\text{target}}$。本节真机（Qwen2.5-1.5B / 0.5B，RTX 5070 Ti Laptop，Transformers）：固定 $k=5$ 目标前向 192→61 但延迟比 1.059；置信度停止 73 次目标 / 186 次助手，延迟比 0.854。 |
| **常见坑** | 1. draft 成本乘接受数而不是候选数——低估额外成本。2. draft 比 target 验证范围长时把未验证的 token 当作接受。3. 控制器不夹 `[min_k, max_k]`：$k$ 涨到几十或掉到 0 回不来。4. 按比例一步调到"理想 $k$"——接受率噪声大，控制器自己震荡。5. 把"target 调用变少"当成"变快"。 |
| **延伸阅读** | 专题正文 [03 解码策略](../topic_discussion/inference_optimization/03_decoding_strategies.md) · 判断手册 [按现象选择机制](../topic_discussion/inference_optimization/casebook.md#按现象选择机制) · 候选怎么来 [INF-EAGLE](./INF-EAGLE_Feature_Drafter_and_Candidate_Tree.md) / [INF-PARALLEL-DRAFTING](./INF-PARALLEL-DRAFTING_DFlash_and_DSpark.md) · 真实后端 [68](./68_Speculative_Decoding_Benchmark.md) |

## 相关阅读

- [Transformers Assisted Decoding](https://huggingface.co/docs/transformers/main/assisted_decoding) —— `num_assistant_tokens_schedule` 和 `assistant_confidence_threshold` 两个参数的含义，Step 5 三种策略的来源
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 推导了给定接受率 $\alpha$ 和 draft/target 成本比 $c$ 时该怎么选 $\gamma$（即本节的 $k$）
- [vLLM Speculators：算法选择](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/decision_guide/) —— 服务端在不同候选来源之间怎么选