# INF-PARALLEL-DRAFTING. DFlash and DSpark | 并行草稿与置信度调度

**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `投机解码`, `DFlash`, `DSpark`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/INF-PARALLEL-DRAFTING_DFlash_and_DSpark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

自回归 draft（草稿）要先猜出第一个 token，才能猜第二个。候选越长，这条串行路径越贵。DFlash 用轻量 block diffusion（块扩散）草稿网络并行提出多个位置；DSpark 在这样的并行主干后增加轻量顺序修正，并预测每个位置的 acceptance confidence（接受置信度），决定哪些连续候选值得验证。

本节不训练扩散模型，也不实现完整论文系统。我们从已经给定的 logits（词表分数）出发，手算并行选择、低秩 Markov 偏置和连续置信前缀。学完后你能区分“候选怎么形成”“送几个去验证”和“target 最终接受几个”，并解释首位置不可信时为什么回到普通 decode（解码）。

## 前置阅读

- [23 投机解码](./23_Speculative_Decoding.md) —— 先理解候选需要 target 验证，首个拒绝之后不能继续提交。
- [INF-PROPOSAL 动态预算](./INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.md)（可选）—— 对照跨轮预算与本节轮内截断的区别。

### Step 1：痛点与直觉——猜 4 个位置，能不能只跑一次草稿主干

假设一个自回归草稿器每步花 2 ms，连续猜 4 个位置就需要约 8 ms，还没开始 target 验证。若能让一次主干前向同时产生 4 行分数，就能减少串行等待；但不能据此把时间直接除以 4，因为并行块的计算量、内存访问和 kernel 效率也会变化。这组毫秒数只是教学假设。

困难在于条件依赖：第二个位置应该知道第一个位置选了什么，而纯并行候选不能等待这个结果。像同时请四个人各写一句话，他们都读过前文，却还不知道其他人刚写了什么。DFlash 用训练好的块扩散草稿网络产生候选；DSpark 则在并行主干之后，让一个较轻的顺序头根据前一候选修正当前分数，再用置信度决定提交多长的前缀。

本节把主干输出当作已知输入，只实现其后的三个小动作。你会看到 `[1,0,2]` 怎样从三行分数独立选出，前一 token 怎样改变下一行的选择，以及 `[0.9,0.8,0.3,0.95]` 为什么只能提交前两个位置。

<img src="../public/02_PyTorch_Algorithms/inf_parallel_drafting_overview.svg" alt="DFlash 与 DSpark 的候选生成差异" width="1120" height="500">

### Step 2：机制怎么走——先并行选，再加一条轻量依赖

**独立选候选。** 先看三行、三词的分数：

```text
位置 0: [0.1, 0.8, 0.1] → token 1
位置 1: [0.6, 0.2, 0.2] → token 0
位置 2: [0.1, 0.2, 0.7] → token 2
```

每行各取 argmax，得到 `[1,0,2]`。把中间行改成 `[0.1,0.2,0.9]`，结果变为 `[1,2,2]`，其他位置不受影响。这里的分数按 logits 使用，不要求和为 1；逐行 argmax 只演示输出选择，不是完整的 DFlash 扩散前向。

**投影前驱状态。** 假设前一 token 7 的低维状态是 `[2,-1]`，秩 $r=2$，投影矩阵为 `[[0.5,0,1],[1,-1,0.5]]`。逐列内积得到偏置：

$$
b=[2\times0.5-1\times1,\;2\times0-1\times(-1),\;2\times1-1\times0.5]=[0,1,1.5].
$$

把它加到当前位置的基础 logits 上，就能让前一候选影响下一位置。真实 DSpark 提供不同顺序头，本题只模拟 vanilla 低秩偏置；词表映射和训练都省略了。

**顺序修正如何传下去。** 再用测试里的两词例子：基础分数为 `[[0.4,0.6],[0.7,0.3],[0.6,0.4]]`，投影是 `[[0.5,-0.5]]`，anchor token 9 与 token 1 的状态为 `[1]`，token 0 的状态为 `[-1]`。第一行加 `[0.5,-0.5]` 后选 0；第二行因此加 `[-0.5,0.5]`，选 1；第三行再加 `[0.5,-0.5]`，选 0，最终 `[0,1,0]`。基础分数已经并行算好，只有这个轻量修正按顺序进行。

<details>
<summary>速查表：完整方法与本题练习的边界</summary>

| 路径 | 完整方法关注什么 | 本题实现什么 |
|---|---|---|
| DFlash | 由 target 特征辅助的块扩散草稿 | 对已给定分数逐位置选 argmax |
| DSpark 顺序头 | 在并行主干后恢复块内依赖 | 前驱 embedding 乘低秩投影，再加到 logits |
| DSpark 置信头 | 学习预测各位置的接受概率 | 读取已给定置信度，选择连续前缀 |

</details>

### Step 3：边界与陷阱——可信的后缀也不能跳过前面的缺口

设置信度为 `[0.9,0.8,0.3,0.95]`、阈值 0.5、上限 4。从左向右，前两个通过，第三个低于阈值就停止，所以预算为 2。不能因为最后一个是 0.95 就把它拼进来：它对应的是第四个位置，缺少第三个位置便失去了连续上下文。等于阈值时本题允许通过，因为停止条件是 `< threshold`。

如果第一项只有 0.1，预算就是 0，`build_draft_record` 应返回空的 `verify_tokens` 并令 `fallback_to_decode=True`。这是“本轮不送草稿”的正常选择；空置信序列的预算也为 0，但创建完整记录时仍要求候选与置信度非空、等长。

**置信度不是接受结论。** 真实 DSpark 的置信头学习预测接受概率；本题直接接收外部给定值，没有训练或校准该头。不能随手把任意 logits 最大值当成置信度，更不能跳过 target 验证。过了阈值只代表值得尝试。

**只验证访问到的前缀。** 函数在首个低置信位置立即停止，也只扫描 `max_tokens` 范围。它会拒绝访问到的非法置信度，但不会继续检查已经丢弃的后缀；如果调用方需要验证整份数据，应在调用前单独检查。

<img src="../public/02_PyTorch_Algorithms/inf_parallel_drafting_budget.svg" alt="DSpark 置信度预算与目标验证的分工" width="1120" height="500">

### Step 4：任务——补全候选、偏置和停止条件

输入校验、顺序调用和结果记录已经给出。完成下面三处后，测试会把它们组合成一轮草稿记录。

| TODO / 函数 | 要完成什么 | 需要注意 | 对应测试 |
|---|---|---|---|
| TODO 1 · `dflash_parallel_candidates` | 每行独立选择最高分 token | 输出顺序就是位置顺序 | `test_dflash_parallel_candidates` 检查单行变化不影响其他行 |
| TODO 2 · `project_markov_bias` | 对每个词表列求低维内积 | 返回长度为 vocab_size 的偏置；未知前驱由骨架返回零 | `test_project_markov_bias` 检查 `[0,1,1.5]` 与维度错误 |
| TODO 3 · `choose_confident_prefix` | 当前置信度低于阈值就停止 | 允许零预算；不可跳过低置信位置 | `test_choose_confident_prefix` 与 `test_unified_draft_record` 检查截断和 fallback |


```python
from typing import Dict, List, Sequence

```



```python
def _validate_block_logits(block_logits: Sequence[Sequence[float]]) -> int:
    """校验位置数和词表维度，并返回词表大小。"""
    if not block_logits or not block_logits[0]:
        raise ValueError("候选块分数不能为空")
    vocab_size = len(block_logits[0])
    if any(len(row) != vocab_size for row in block_logits):
        raise ValueError("所有位置必须共享词表维度")
    return vocab_size


def dflash_parallel_candidates(block_logits: Sequence[Sequence[float]]) -> List[int]:
    """对候选块中的每个位置独立选择最高分 Token。"""
    vocab_size = _validate_block_logits(block_logits)
    # ==========================================
    # TODO 1：形成位置相互独立的并行候选块。
    # ==========================================
    # candidates = ???  # 每个位置独立选择的最高分 Token；列表长度等于位置数。
    raise NotImplementedError('TODO 1：请逐位置选择候选')
    return candidates


def project_markov_bias(
    previous_token: int,
    markov_embeddings: Dict[int, Sequence[float]],
    markov_projection: Sequence[Sequence[float]],
    vocab_size: int,
) -> List[float]:
    """把前驱 Token 的低维状态投影为词表偏置。"""
    if vocab_size <= 0 or not markov_projection:
        raise ValueError("词表大小和 Markov 投影不能为空")
    rank = len(markov_projection)
    if any(len(row) != vocab_size for row in markov_projection):
        raise ValueError("Markov 投影的每一行必须匹配词表维度")
    latent = markov_embeddings.get(previous_token)
    if latent is None:
        return [0.0] * vocab_size
    if len(latent) != rank:
        raise ValueError("前驱状态维度必须匹配 Markov 秩")
    # ==========================================
    # TODO 2：执行低秩状态到词表偏置的投影。
    # ==========================================
    # bias = ???  # 长度为 vocab_size；第 v 项为各秩分量与投影第 v 列的内积。
    raise NotImplementedError('TODO 2：请计算低秩投影偏置')
    return bias


def dspark_markov_candidates(
    block_logits: Sequence[Sequence[float]],
    anchor_token: int,
    markov_embeddings: Dict[int, Sequence[float]],
    markov_projection: Sequence[Sequence[float]],
) -> List[int]:
    """按候选顺序使用低秩 Markov 偏置修正并行主干分数。"""
    vocab_size = _validate_block_logits(block_logits)
    previous = anchor_token
    candidates: List[int] = []
    for row in block_logits:
        bias = project_markov_bias(previous, markov_embeddings, markov_projection, vocab_size)
        corrected = [float(score) + delta for score, delta in zip(row, bias)]
        previous = max(range(vocab_size), key=lambda token: corrected[token])
        candidates.append(previous)
    return candidates


def choose_confident_prefix(
    confidences: Sequence[float],
    threshold: float,
    max_tokens: int = 8,
) -> int:
    """返回值得交给目标模型验证的连续候选长度。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("置信度阈值必须位于 [0, 1]")
    if max_tokens < 1:
        raise ValueError("最大预算必须大于零")
    prefix_length = 0
    for confidence in confidences[:max_tokens]:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("每个置信度必须位于 [0, 1]")
        # ==========================================
        # TODO 3：在首个低置信位置停止连续前缀。
        # ==========================================
        # should_stop = ???  # 当前置信度不足以进入验证前缀时为 True。
        raise NotImplementedError('TODO 3：请判断连续前缀何时停止')
        if should_stop:
            break
        prefix_length += 1
    return prefix_length


def build_draft_record(mode, candidates, confidences, threshold, max_tokens=8):
    """把候选块和验证预算整理为统一记录。"""
    if mode not in {"dflash", "dspark"}:
        raise ValueError("mode 必须是 dflash 或 dspark")
    if not candidates or len(candidates) != len(confidences):
        raise ValueError("候选与置信度必须非空且等长")
    budget = choose_confident_prefix(confidences, threshold, max_tokens=max_tokens)
    return {
        "mode": mode,
        "block_tokens": list(candidates),
        "verify_tokens": list(candidates[:budget]),
        "proposed": len(candidates),
        "verify_budget": budget,
        "fallback_to_decode": budget == 0,
    }
```


### 测试

- 并行候选测试检查位置独立性、空输入和词表维度。
- 偏置测试检查手算结果及未知前驱回退；顺序候选测试检查 `[0,1,0]` 的传播过程。
- 预算测试检查中途停止、首位置失败、空序列和上限；完整记录测试确认零预算不提交候选。

这些 CPU 测试验证算式与记录逻辑，不测真实 DFlash / DSpark 的速度或接受率。


```python
def test_dflash_parallel_candidates():
    """并行候选应逐位置选择，修改一行不能影响其他位置。"""
    logits = [[0.1, 0.8, 0.1], [0.6, 0.2, 0.2], [0.1, 0.2, 0.7]]
    assert dflash_parallel_candidates(logits) == [1, 0, 2]
    changed = [logits[0], [0.1, 0.2, 0.9], logits[2]]
    assert dflash_parallel_candidates(changed) == [1, 2, 2]
    for invalid in ([], [[0.1, 0.9], [0.3]]):
        try:
            dflash_parallel_candidates(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("空块或词表维度失配时应失败")


def test_project_markov_bias():
    """低秩状态应按投影列还原为词表偏置。"""
    embeddings = {7: [2.0, -1.0]}
    projection = [[0.5, 0.0, 1.0], [1.0, -1.0, 0.5]]
    assert project_markov_bias(7, embeddings, projection, 3) == [0.0, 1.0, 1.5]
    assert project_markov_bias(9, embeddings, projection, 3) == [0.0, 0.0, 0.0]
    try:
        project_markov_bias(7, {7: [1.0]}, projection, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("前驱状态与 Markov 秩不匹配时应失败")


def test_dspark_markov_candidates():
    """前一候选应通过低秩状态改变下一位置的修正方向。"""
    logits = [[0.4, 0.6], [0.7, 0.3], [0.6, 0.4]]
    embeddings = {9: [1.0], 0: [-1.0], 1: [1.0]}
    projection = [[0.5, -0.5]]
    assert dspark_markov_candidates(logits, 9, embeddings, projection) == [0, 1, 0]


def test_choose_confident_prefix():
    """连续预算应支持中途停止、零预算、空序列与上限截断。"""
    assert choose_confident_prefix([0.9, 0.8, 0.3, 0.95], 0.5, 4) == 2
    assert choose_confident_prefix([0.1, 0.9], 0.5, 4) == 0
    assert choose_confident_prefix([], 0.5, 4) == 0
    assert choose_confident_prefix([0.9] * 10, 0.5, 4) == 4
    for args in (([1.1], 0.5, 4), ([0.9], -0.1, 4), ([0.9], 0.5, 0)):
        try:
            choose_confident_prefix(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("非法置信度、阈值或预算上限应失败")


def test_unified_draft_record():
    """零预算记录不得向目标模型提交候选，并应显式触发回退。"""
    record = build_draft_record("dspark", [4, 5], [0.2, 0.9], 0.5, 2)
    assert record["verify_tokens"] == []
    assert record["verify_budget"] == 0
    assert record["fallback_to_decode"] is True
    try:
        build_draft_record("unknown", [1], [0.9], 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("未知草稿模式应失败")


test_dflash_parallel_candidates()
test_project_markov_bias()
test_dspark_markov_candidates()
test_choose_confident_prefix()
test_unified_draft_record()
print("✅ 并行草稿、低秩修正与连续预算测试通过")

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
def _validate_block_logits(block_logits: Sequence[Sequence[float]]) -> int:
    """校验位置数和词表维度，并返回词表大小。"""
    if not block_logits or not block_logits[0]:
        raise ValueError("候选块分数不能为空")
    vocab_size = len(block_logits[0])
    if any(len(row) != vocab_size for row in block_logits):
        raise ValueError("所有位置必须共享词表维度")
    return vocab_size


# ==========================================
# TODO 1：形成位置相互独立的并行候选块。
# ==========================================
def dflash_parallel_candidates(block_logits: Sequence[Sequence[float]]) -> List[int]:
    """对候选块中的每个位置独立选择最高分 Token。"""
    vocab_size = _validate_block_logits(block_logits)
    candidates = [max(range(vocab_size), key=lambda token: row[token]) for row in block_logits]
    return candidates


# ==========================================
# TODO 2：执行低秩状态到词表偏置的投影。
# ==========================================
def project_markov_bias(
    previous_token: int,
    markov_embeddings: Dict[int, Sequence[float]],
    markov_projection: Sequence[Sequence[float]],
    vocab_size: int,
) -> List[float]:
    """把前驱 Token 的低维状态投影为词表偏置。"""
    if vocab_size <= 0 or not markov_projection:
        raise ValueError("词表大小和 Markov 投影不能为空")
    rank = len(markov_projection)
    if any(len(row) != vocab_size for row in markov_projection):
        raise ValueError("Markov 投影的每一行必须匹配词表维度")
    latent = markov_embeddings.get(previous_token)
    if latent is None:
        return [0.0] * vocab_size
    if len(latent) != rank:
        raise ValueError("前驱状态维度必须匹配 Markov 秩")
    bias = [
        sum(float(latent[index]) * float(markov_projection[index][token]) for index in range(rank))
        for token in range(vocab_size)
    ]
    return bias


def dspark_markov_candidates(
    block_logits: Sequence[Sequence[float]],
    anchor_token: int,
    markov_embeddings: Dict[int, Sequence[float]],
    markov_projection: Sequence[Sequence[float]],
) -> List[int]:
    """按候选顺序使用低秩 Markov 偏置修正并行主干分数。"""
    vocab_size = _validate_block_logits(block_logits)
    previous = anchor_token
    candidates: List[int] = []
    for row in block_logits:
        bias = project_markov_bias(previous, markov_embeddings, markov_projection, vocab_size)
        corrected = [float(score) + delta for score, delta in zip(row, bias)]
        previous = max(range(vocab_size), key=lambda token: corrected[token])
        candidates.append(previous)
    return candidates


# ==========================================
# TODO 3：在首个低置信位置停止连续前缀。
# ==========================================
def choose_confident_prefix(
    confidences: Sequence[float],
    threshold: float,
    max_tokens: int = 8,
) -> int:
    """返回值得交给目标模型验证的连续候选长度。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("置信度阈值必须位于 [0, 1]")
    if max_tokens < 1:
        raise ValueError("最大预算必须大于零")
    prefix_length = 0
    for confidence in confidences[:max_tokens]:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("每个置信度必须位于 [0, 1]")
        should_stop = confidence < threshold
        if should_stop:
            break
        prefix_length += 1
    return prefix_length


def build_draft_record(mode, candidates, confidences, threshold, max_tokens=8):
    """把候选块和验证预算整理为统一记录。"""
    if mode not in {"dflash", "dspark"}:
        raise ValueError("mode 必须是 dflash 或 dspark")
    if not candidates or len(candidates) != len(confidences):
        raise ValueError("候选与置信度必须非空且等长")
    budget = choose_confident_prefix(confidences, threshold, max_tokens=max_tokens)
    return {
        "mode": mode,
        "block_tokens": list(candidates),
        "verify_tokens": list(candidates[:budget]),
        "proposed": len(candidates),
        "verify_budget": budget,
        "fallback_to_decode": budget == 0,
    }

```


### 解析

#### TODO 1：逐位置独立选择

**思路：** 每行只看自己的分数，返回最大值所在的词表下标。

**关键一行：**
```python
candidates = [max(range(vocab_size), key=lambda token: row[token]) for row in block_logits]
```

**为什么这样写：** 外层保留位置顺序，内层比较词表分数。两行之间没有读写依赖，所以改中间一行只改变中间一个候选。Python 的 max 在并列时选择先遇到的下标，本题据此得到稳定的选择规则。

**常见错法：** 把所有行展平再取最大值，最后只得到一个 token；或者将前一候选用于改写下一行，那已经进入顺序修正路径。

#### TODO 2：按词表列做内积

**思路：** 一个 token 对应投影矩阵的一列，偏置是这列与前驱低维状态的内积。

**关键一行：**
```python
bias = [sum(float(latent[i]) * float(markov_projection[i][token]) for i in range(rank)) for token in range(vocab_size)]
```

**为什么这样写：** 外层遍历词表保证输出长度为 V，内层遍历 rank。状态 `[2,-1]` 乘投影的第三列 `[1,0.5]` 就得到 `2-0.5=1.5`。未知前驱的零偏置已由骨架处理，意味着暂不修正基础分数。

**常见错法：** 对投影的每一行求和，输出长度就成了 rank；或把偏置当作概率归一化，它实际是要加到 logits 上的分数。

#### TODO 3：遇到低置信位置立即停止

**思路：** 只有从头开始连续通过的候选才能组成验证前缀。

**关键一行：**
```python
should_stop = confidence < threshold
```

**为什么这样写：** 等于阈值可以继续，低于阈值触发骨架中的 break。上限截断由 `confidences[:max_tokens]` 完成，函数返回的是长度而不是候选列表。

**常见错法：** 用 `sum(c >= threshold for c in confidences)` 数通过个数。`[0.9,0.8,0.3,0.95]` 会被算成 3，实际连续前缀只有 2；首项不通过时也不能强行至少返回 1。

<a id="cheatsheet"></a>
## 速查卡

| | |
|:--|:--|
| **一句话机制** | 并行主干一次给出候选块的基础分数，轻量顺序头补前驱依赖；置信度只控制送去验证的连续前缀，target 决定接受与否。 |
| **关键公式与数量级** | 低秩偏置 `b = e(previous) @ P`，形状 `[r] @ [r,V] → [V]`。置信度 `[0.9,0.8,0.3,0.95]`、阈值 0.5 得预算 2；首项不通过得 0。 |
| **常见坑** | 1. 把逐行 argmax 当成完整扩散模型。2. 将偏置误作概率。3. 跳过低置信位置拼后缀。4. 把草稿置信度当成 target 接受结果。 |
| **延伸阅读** | 专题 [03 解码策略](../topic_discussion/inference_optimization/03_decoding_strategies.md) · [机制索引](../topic_discussion/inference_optimization/casebook.md#机制速查索引) · 下一步 [INF-PROPOSAL 动态预算](./INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.md) · 实验 [68](./68_Speculative_Decoding_Benchmark.md) |

## 相关阅读

- [DFlash 论文](https://arxiv.org/abs/2602.06036) —— 块扩散草稿与 target 特征。
- [DSpark 论文](https://arxiv.org/abs/2607.05147) —— 顺序修正与置信度调度。
- [vLLM Speculators：DSpark](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/) —— Markov head、confidence head 与支持配置。