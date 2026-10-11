# 21. Decoding Strategies | 解码策略

**难度：** Medium | **环境：** CPU-first | **标签：** `推理优化`, `解码`, `采样`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

模型每生成一个 token（词元），最后一层给出的并不是"一个词"，而是整张词表的分数：Qwen2.5 的词表有 151,936 个 token，模型就吐出 151,936 个数。这组未归一化的分数叫 logits。从 logits 到"下一个 token 是谁"，中间这一步就是解码策略（decoding strategy）。

同一组 logits，选法不同结果完全不同：永远拿最高分叫 greedy（贪心），同样输入下通常更稳定，但也可能重复；按概率随机抽叫 sampling（采样），有变化但可能抽到胡话。temperature、top-k、top-p 这三个参数就是在这两端之间调节的旋钮——你在 OpenAI、vLLM、Transformers 的 API 里天天见到它们。

本节用纯 PyTorch 实现这三个旋钮并串成一步解码。学完后你能解释：为什么 temperature 不改变排序只改变"尖锐度"，为什么 top-p 的边界 token 必须保留，以及低温采样、趋零极限与 greedy 的区别。

## 前置阅读

- [04. Attention MHA GQA | 注意力机制：MHA、MQA、GQA](./04_Attention_MHA_GQA.md) —— 看清 attention 怎样产生"当前位置的表示"；logits 就是这个表示过了 LM head 之后的样子。
- [05. LLaMA3 Block Tutorial | LLaMA3 Block 实现](./05_LLaMA3_Block_Tutorial.md)（可选）—— 一个完整 decoder-only（仅解码器）Block 的数据流。知道 logits 从哪来即可，不必先读完。

---

### Step 1：痛点与直觉——从 151,936 个分数里挑一个

想象你在用一个 Qwen2.5 模型写诗。每生成一个 token，模型都给词表里的 151,936 个位置打分。最朴素的做法是每次取最高分，也就是 greedy。它便于复现，但只保留一条局部最优路径：如果高分词反复落在同一种句式里，诗就容易重复。这里的 token 也不等于一个汉字，具体切分由 tokenizer 决定。

反过来，直接按 softmax 后的分布采样，可以探索更多表达，也可能抽中不合适的尾部候选。例如假设很多低概率 token 合计占 5% 概率质量，单看每个都不起眼，合起来却不能忽略。这个 5% 是说明问题的假设，不是 Qwen2.5 的测量值；实际分布随上下文而变。

解码策略的思路是：别在"全选最高分"和"全靠运气"之间二选一，而是先修一下这张分布，再抽。换个说法，就像从一摞简历里招人：temperature 决定你多看重分差（T 越小越只盯着第一名），top-k 说"只看前 K 份"，top-p 说"从高往低翻，翻到累计覆盖 85% 的概率质量就停"。三个旋钮都在抽签之前起作用，它们不碰模型、不碰 KV Cache，只改这一步谁还在候选名单里、各自概率多大。

本节你要实现的正是这三个旋钮：`apply_temperature`、`apply_top_k`、`apply_top_p`，然后把它们接到一个最小的自回归循环里。

<img src="../public/02_PyTorch_Algorithms/21_decoding_pipeline.svg" alt="解码策略流程" width="1500" height="560">

### Step 2：机制怎么走——用 5 个 token 的词表走一遍

先把词表缩到 5 个 token，logits 是 `[4.0, 3.1, 2.3, 1.2, 0.1]`，这样每一步都能手算。

**temperature.** 把 logits 整体除以 $T$ 再做 softmax。$T=1$ 时概率约为 `[0.60, 0.24, 0.11, 0.04, 0.01]`；$T=0.5$ 时第一名约占 0.83，$T=2$ 时约占 0.41。正温度不会改变排序，只改变概率的集中程度。对这组分数，$T=0.01$ 时第二名与第一名的概率比为 $e^{(3.1-4.0)/0.01}=e^{-90}\approx8.2\times10^{-40}$，不能说采样一万次就会抽到几次第二名。数学上，最高分唯一时，$T\to0^+$ 的分布趋向该 token；最高分并列时则趋向并列项上的均匀分布。实际浮点计算还可能把极小概率舍入成零。需要 greedy 时应直接指定 `do_sample=False`，而不是依赖极小温度。

**top-k.** 找出第 K 大的分数当门槛，低于门槛的一律设为 $-\infty$（softmax 后变成 0）。K=3 时门槛是 2.3，`[4.0, 3.1, 2.3]` 留下，后两个出局。用 $-\infty$ 而不是真的把元素删掉，是为了张量形状不变，后面 softmax 和采样的代码一行都不用改。有一个细节：如果恰好有几个 token 和门槛同分（比如 `[2.0, 2.0, 2.0, 1.0]` 取 K=2），它们一起保留，候选会多于 K——这是刻意的，同分的 token 没有理由厚此薄彼。

**top-p.** 把概率从大到小累加，本节实现累计到首次严格大于 $p$ 的那个 token 为止（$p=1$ 时不截断），之后的全部出局。换一组更整的数字：排序后的概率是 `[0.5, 0.3, 0.1, 0.05, 0.05]`，取 $p=0.85$，累计是 `0.5 → 0.8 → 0.9`，第三个 token 让累计首次越过 0.85，所以保留前三个。分布越平（模型越犹豫），top-p 留下的 token 就越多；分布越尖，留下的越少。这正是它比 top-k 灵活的地方：K 控制名额，p 控制概率质量；保持 p 不变，留下的候选数也会随分布变化。

**顺序与最后一步.** 本节的流水线是 temperature → top-k → top-p → softmax → 选择。三个旋钮只负责改 logits；最后 softmax 把幸存者的概率重新归一化到和为 1，`torch.multinomial` 按这个分布抽一个（或 `argmax` 做 greedy）。

<details>
<summary>速查表：三个旋钮各改什么</summary>

| 旋钮 | 输入 | 怎么改 logits | 不改什么 | 示例设置（需按任务调整） |
|---|---|---|---|---|
| temperature $T$ | logits | 整体除以 $T$ | 排序 | 0.5 / 1 / 2，用于比较分布 |
| top-k | 缩放后的 logits，$K$ | 低于第 K 大值的设为 $-\infty$（同分一起留） | 候选内部的相对大小 | 20–100；0 表示关闭 |
| top-p | 候选分布，$p$ | 排序累加，首次 > $p$ 之后的设为 $-\infty$ | 候选内部的相对大小 | 0.8–0.95；1.0 表示关闭 |
| 选择 | 最终分布 | softmax 归一化后 `multinomial` 或 `argmax` | — | `do_sample=False` 即 greedy |

</details>

### Step 3：边界与陷阱——top-p 的那一个 token

top-p 的实现有一个几乎人人都踩过的坑。还是用排序后的概率 `[0.5, 0.3, 0.1, 0.05, 0.05]`、$p=0.85$：

1. `cumsum` 得到 `[0.5, 0.8, 0.9, 0.95, 1.0]`；
2. 直觉上写 `remove = cumsum > 0.85`，得到 `[F, F, T, T, T]`；
3. 按这个掩码屏蔽，只剩前两个 token，累计概率 0.8——**没有达到 0.85**。

错在哪？第三个 token 正是让累计"首次越过 0.85"的那一个，它应该是候选集合的最后一员，而 `cumsum > p` 把它一起删了。修法是把掩码整体右移一位：`remove[1:] = remove[:-1]`，再把第一位强制设为 `False`（哪怕第一个 token 自己就超过了 $p$，也至少要留一个）。右移后掩码变成 `[F, F, F, T, T]`，前三个 token 保留，累计 0.9 ≥ 0.85，这才对。

**恰好等于阈值怎么办？** 本节代码使用 `cumsum > p` 后右移，因此若 `[0.5, 0.3, 0.2]` 的累计值恰好等于 `p=0.5`，还会保留下一个 token。它与“累计首次 ≥ p 的最小前缀”在等号处不同。浮点舍入也会影响边界，比较不同实现时应核对比较符和最少保留数。

第二个容易忽略的点是**顺序**。top-p 是在排序后的空间里做的，掩码对应的是"第几大"而不是"词表第几号"。屏蔽完必须用 `sorted_indices` 把 logits 散回原来的词表位置（`scatter_`），否则后面 `multinomial` 抽到的下标就是错的。测试 `test_top_p_boundary` 同时检查这两件事：p=0.8 时留下的恰好是 3 个 token，而且它们在词表里的位置是 `[5, 6, 1]`，与原 logits 一一对应。

<img src="../public/02_PyTorch_Algorithms/21_top_p_boundary.svg" alt="Top-p 边界保留示意" width="1200" height="500">

### Step 4：任务——实现三个旋钮

下面的题目区已经写好了参数校验、`decode_next_token` 和自回归循环，你只需要补三处核心逻辑。每一处都对应 Step 2 / Step 3 里走过的一步。

| TODO | 要做什么 | 必须满足 | 对应测试 |
|---|---|---|---|
| TODO 1 · `apply_temperature` | 校验 temperature 是有限正数，然后 logits 整体除以它 | 形状、dtype、设备、排序都不变 | `test_temperature_contract`：分差放大 2 倍、排序不变、`T=0` 抛 `ValueError` |
| TODO 2 · `apply_top_k` | 取第 K 大值做门槛，低于门槛的设为 $-\infty$ | 同分 token 一起保留，候选可能多于 K | `test_top_k_filtering`：K=3 剩 3 个；`[2, 2, 2, 1]` 取 K=2 剩 3 个 |
| TODO 3 · `apply_top_p` | 把 `remove_mask` 右移一位，屏蔽后用 `sorted_indices` 散回词表顺序 | 首次越过 $p$ 的 token 必须保留 | `test_top_p_boundary`：p=0.8 剩 3 个，位置 `[5, 6, 1]` |


```python
import torch
import torch.nn.functional as F
```



```python
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """在概率归一化前缩放词表分数。"""
    # ==========================================
    # TODO 1：校验温度并缩放词表分数。
    # ==========================================
    # temperature_is_valid = ???  # temperature 是有限正数时为 True。
    # scaled_logits = ???  # 保持形状、数据类型、设备和排序的缩放结果。
    raise NotImplementedError("TODO 1：请校验温度并缩放")
    if not temperature_is_valid:
        raise ValueError("temperature 必须是有限正数")
    return scaled_logits


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """按第 K 大值筛选词表分数，同分边界一起保留。"""
    if top_k is None:
        return logits
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError("top_k 必须为非负整数")
    if top_k == 0 or top_k >= logits.size(-1):
        return logits
    # ==========================================
    # TODO 2：按第 K 大值形成候选掩码。
    # ==========================================
    # kth_values = ???  # 每个样本的第 K 大阈值，保留最后一维。
    # filtered_logits = ???  # 小于阈值的位置设为负无穷。
    raise NotImplementedError("TODO 2：请实现 Top-K 过滤")
    return filtered_logits


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """保留累计概率首次达到阈值的最小候选集合。"""
    if top_p is None or top_p == 1.0:
        return logits
    if not 0.0 < top_p < 1.0:
        raise ValueError("top_p 必须满足 0 < top_p <= 1")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove_mask = cumulative_probs > top_p
    # ==========================================
    # TODO 3：保留边界 Token，并恢复原词表顺序。
    # ==========================================
    # shifted_mask = ???  # remove_mask 右移一位后的边界保留掩码。
    # restored_logits = ???  # 屏蔽候选后恢复到原词表顺序的结果。
    raise NotImplementedError("TODO 3：请实现 Top-p 边界过滤")
    remove_mask[..., 1:] = shifted_mask
    remove_mask[..., 0] = False
    sorted_logits[remove_mask] = float("-inf")
    return restored_logits


def decode_next_token(logits, temperature=0.7, top_k=50, top_p=0.9, do_sample=True, generator=None):
    """应用候选策略并选择一个 Token。"""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    elif logits.dim() != 2:
        raise ValueError("logits 必须是 [vocab] 或 [batch, vocab]")
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    probabilities = F.softmax(logits, dim=-1)
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def autoregressive_decode(prompt_ids, logits_fn, max_new_tokens, **decode_kwargs):
    """逐 Token 调用词表分数函数并追加结果。"""
    tokens = prompt_ids.clone()
    for _ in range(max_new_tokens):
        step_logits = logits_fn(tokens)
        if step_logits.dim() == 3:
            step_logits = step_logits[:, -1, :]
        next_token = decode_next_token(step_logits, **decode_kwargs)
        tokens = torch.cat([tokens, next_token], dim=-1)
    return tokens

```


### 测试

五个测试各管一件事：`test_temperature_contract` 看缩放是否只改尖锐度不改排序，并拒绝 `T=0`；`test_top_k_filtering` 看 K=3 是否恰好剩 3 个、同分是否一起保留；`test_top_p_boundary` 看边界 token 有没有被误删、词表位置有没有散回去；`test_selection_and_autoregressive_loop` 看 greedy 是否取到最高分、固定种子的两次采样是否一致、自回归循环是否每步追加一个 token；`test_batch_and_input_contract` 看 `[batch, vocab]` 输入是否沿最后一维处理。


```python
def _fixture_logits():
    """返回包含明确排序关系的固定词表分数。"""
    return torch.tensor([[0.1, 2.3, 0.4, 1.2, -0.5, 4.0, 3.1, 0.0, 1.1, -1.0]])


def test_temperature_contract():
    """验证温度缩放保持排序、形状和数据类型，并拒绝非法温度。"""
    logits = _fixture_logits()
    scaled = apply_temperature(logits, 0.5)
    assert torch.allclose(scaled[0, 5] - scaled[0, 6], (logits[0, 5] - logits[0, 6]) * 2)
    assert scaled.shape == logits.shape and scaled.dtype == logits.dtype
    assert torch.equal(torch.argsort(logits), torch.argsort(scaled))
    try:
        apply_temperature(logits, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("非法 temperature 应失败")


def test_top_k_filtering():
    """验证 Top-K 阈值、同分边界和张量属性。"""
    logits = _fixture_logits().to(dtype=torch.float64)
    filtered = apply_top_k(logits, 3)
    assert torch.isfinite(filtered).sum().item() == 3
    assert filtered.shape == logits.shape and filtered.dtype == logits.dtype and filtered.device == logits.device
    tied = torch.tensor([[2.0, 2.0, 2.0, 1.0]])
    assert torch.isfinite(apply_top_k(tied, 2)).sum().item() == 3


def test_top_p_boundary():
    """验证 Top-p 保留首次越过阈值的边界 Token。"""
    logits = _fixture_logits().to(dtype=torch.float64)
    filtered = apply_top_p(logits, 0.8)
    assert torch.isfinite(filtered).sum().item() == 3
    assert torch.equal(filtered[0, [5, 6, 1]], logits[0, [5, 6, 1]])
    for invalid_p in (0.0, 1.1):
        try:
            apply_top_p(logits, invalid_p)
        except ValueError:
            continue
        raise AssertionError("非法 top_p 应失败")


def test_selection_and_autoregressive_loop():
    """验证贪心选择、可复现采样和逐 Token 状态推进。"""
    logits = _fixture_logits()
    assert decode_next_token(logits, 1.0, 0, 1.0, do_sample=False).item() == 5
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    assert torch.equal(decode_next_token(logits, generator=g1), decode_next_token(logits, generator=g2))
    prompt = torch.tensor([[1, 2]])

    def fake_logits(tokens):
        result = torch.zeros(tokens.size(0), tokens.size(1), 10)
        result[..., 3] = 2.0
        return result

    generated = autoregressive_decode(prompt, fake_logits, 3, temperature=1.0, top_k=0, top_p=1.0, do_sample=False)
    assert generated.shape == (1, 5) and torch.equal(generated[0, -3:], torch.tensor([3, 3, 3]))


def test_batch_and_input_contract():
    """验证批量输入沿词表维处理，并保持批量维。"""
    logits = _fixture_logits().repeat(2, 1)
    assert apply_top_k(logits, 3).shape == logits.shape
    assert apply_top_p(logits, 0.8).shape == logits.shape
    assert decode_next_token(logits[0], 1.0, 0, 1.0, do_sample=False).shape == (1, 1)


test_temperature_contract()
test_top_k_filtering()
test_top_p_boundary()
test_selection_and_autoregressive_loop()
test_batch_and_input_contract()
print("✅ 解码策略测试通过")

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
# ==========================================
# TODO 1：校验温度并缩放词表分数。
# ==========================================
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """在概率归一化前缩放词表分数。"""
    temperature_is_valid = torch.isfinite(torch.tensor(temperature)).item() and temperature > 0
    if not temperature_is_valid:
        raise ValueError("temperature 必须是有限正数")
    scaled_logits = logits / temperature
    return scaled_logits


# ==========================================
# TODO 2：按第 K 大值形成候选掩码。
# ==========================================
def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """按第 K 大值筛选词表分数，同分边界一起保留。"""
    if top_k is None:
        return logits
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError("top_k 必须为非负整数")
    if top_k == 0 or top_k >= logits.size(-1):
        return logits
    kth_values = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    filtered_logits = torch.where(logits < kth_values, torch.full_like(logits, float("-inf")), logits)
    return filtered_logits


# ==========================================
# TODO 3：保留边界 Token，并恢复原词表顺序。
# ==========================================
def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """保留累计概率首次达到阈值的最小候选集合。"""
    if top_p is None or top_p == 1.0:
        return logits
    if not 0.0 < top_p < 1.0:
        raise ValueError("top_p 必须满足 0 < top_p <= 1")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove_mask = cumulative_probs > top_p
    shifted_mask = remove_mask[..., :-1].clone()
    remove_mask[..., 1:] = shifted_mask
    remove_mask[..., 0] = False
    sorted_logits[remove_mask] = float("-inf")
    restored_logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    return restored_logits


def decode_next_token(logits, temperature=0.7, top_k=50, top_p=0.9, do_sample=True, generator=None):
    """应用候选策略并选择一个 Token。"""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    elif logits.dim() != 2:
        raise ValueError("logits 必须是 [vocab] 或 [batch, vocab]")
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    probabilities = F.softmax(logits, dim=-1)
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def autoregressive_decode(prompt_ids, logits_fn, max_new_tokens, **decode_kwargs):
    """逐 Token 调用词表分数函数并追加结果。"""
    tokens = prompt_ids.clone()
    for _ in range(max_new_tokens):
        step_logits = logits_fn(tokens)
        if step_logits.dim() == 3:
            step_logits = step_logits[:, -1, :]
        next_token = decode_next_token(step_logits, **decode_kwargs)
        tokens = torch.cat([tokens, next_token], dim=-1)
    return tokens

```


### 解析

#### TODO 1：temperature 缩放

**思路：** 三个旋钮里最简单的一个——logits 整体除以 $T$。关键不在算，在校验：$T$ 必须是有限正数。

**关键一行：**
```python
scaled_logits = logits / temperature
```

**为什么这样写：** 除法对所有位置用同一个比例，所以任意两个 token 的分差都被放大 $1/T$ 倍，排序不动；`test_temperature_contract` 核对的 `scaled[5] - scaled[6] == 2 × (logits[5] - logits[6])` 就是这个性质。形状、dtype、设备由 PyTorch 的广播规则自然保住，不需要额外处理。

**常见错法：** 老版本用 `temp = max(temperature, 1e-6)` 把 `T=0` 悄悄改成一个正数，掩盖了非法参数。稳定 softmax 会先减最大值，并不会因为有限 logits 较大就必然溢出；风险在于前面的除法已超出 dtype 范围，例如 FP16 的 `4 / 1e-6`。本节要求有限正温度；想用 greedy 就传 `do_sample=False`。

#### TODO 2：top-k 过滤

**思路：** 第 K 大的分数就是门槛，低于门槛的设为 $-\infty$。

**关键一行：**
```python
kth_values = torch.topk(logits, top_k, dim=-1).values[..., -1:]
filtered_logits = torch.where(logits < kth_values, torch.full_like(logits, float("-inf")), logits)
```

**为什么这样写：** `topk(...).values` 已经按降序排好，取最后一个就是第 K 大；`[..., -1:]` 保留最后一维是为了能和 `logits` 广播比较，批量输入也能一行搞定。比较用的是 `<` 而不是 `<=`，所以和门槛同分的 token 全部留下——`[2.0, 2.0, 2.0, 1.0]` 取 K=2 会剩 3 个，这是刻意的。

**常见错法：** 用 `torch.topk(...).indices` 直接去索引，再构造一个更短的新张量。形状变了，后面的 softmax 和 `multinomial` 就得跟着改，批量维也容易丢。用 $-\infty$ 屏蔽是整条流水线形状不变的前提。

#### TODO 3：top-p 边界与还原顺序

**思路：** 在排序空间里找到累计概率首次 > $p$ 的位置，把它之后的屏蔽掉，再把结果散回词表原来的位置。

**关键一行：**
```python
shifted_mask = remove_mask[..., :-1].clone()
remove_mask[..., 1:] = shifted_mask
remove_mask[..., 0] = False
restored_logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
```

**为什么这样写：** `cumsum > p` 标记的是"累计已经超过 p"的位置，而我们要删的是"超过 p 之后"的位置，两者正好差一位，所以把掩码右移。`.clone()` 是因为 `remove_mask[..., 1:]` 和 `remove_mask[..., :-1]` 是同一块内存上两个重叠的视图，PyTorch 可能拒绝这种重叠写入。`remove_mask[..., 0] = False` 保证至少留一个 token。最后 `scatter_` 用 `sorted_indices` 把排序后的 logits 放回原位置——`torch.sort` 返回的 `indices[i]` 表示"排序后第 i 个来自原来的第几号"，正好是 `scatter_` 需要的目标下标。

**常见错法：** 不右移掩码。按 Step 3 的例子，`[0.5, 0.3, 0.1, 0.05, 0.05]`、p=0.85 只会剩两个 token，累计 0.8 没达到 p。这个错不会报任何异常，只会让生成结果比预期更"保守"一点，线上很难被发现。另一个错法是忘记 `scatter_` 回去，直接返回 `sorted_logits`——测试里 `filtered[0, [5, 6, 1]]` 就对不上了。

<a id="cheatsheet"></a>
## 速查卡

| | |
|:--|:--|
| **一句话机制** | 每步解码先拿到整张词表的 logits，按 temperature → top-k → top-p → softmax → 选择 的顺序改分布、砍候选、再抽一个 token；三个旋钮都不碰模型和 KV Cache。 |
| **关键公式与数量级** | $p_i=\mathrm{softmax}(z_i/T)$；$T<1$ 更尖、$T>1$ 更平，排序不变。top-k 保留前 K（同分一起留）；本节 top-p 保留到累计概率首次 > $p$ 的位置，$p=1$ 不截断。词表量级：Llama 2 32,000，Llama 3 128,256，Qwen2.5 151,936。温度和阈值需按任务比较，没有通用最优组合。 |
| **常见坑** | 1. top-p 掩码忘记右移，边界 token 被误删（`[0.5, 0.3, 0.1, …]`、p=0.85 时第 3 个必须留）。2. 有限低温不等于显式 greedy；唯一最大值下趋零极限与 greedy 一致，并列最大值另论。3. 过滤用 $-\infty$ 屏蔽而不是删元素，否则形状变了、`scatter_` 散不回去。4. `remove_mask[1:] = remove_mask[:-1]` 要先 `.clone()`，避免重叠视图写入错误。 |
| **延伸阅读** | 专题正文 [03 解码策略](../topic_discussion/inference_optimization/03_decoding_strategies.md) · 判断手册 [按现象选择机制](../topic_discussion/inference_optimization/casebook.md#按现象选择机制) · 下一步 [23 投机解码](./23_Speculative_Decoding.md)：一次验证多个候选 token |

## 相关阅读

- [The Curious Case of Neural Text Degeneration（Nucleus Sampling 原论文）](https://arxiv.org/abs/1904.09751) —— 为什么 greedy / beam 会退化，top-p 的动机
- [Transformers 文本生成策略文档](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation) —— `generate()` 里这三个参数怎么配
- [vLLM SamplingParams 文档](https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html) —— 服务端的同一组旋钮
- [Part 02 · 23 投机解码](./23_Speculative_Decoding.md) —— 用 draft 模型一次提出多个候选
- [Part 02 · 35 原生多 Token 预测](./35_Multi_Token_Decoding.md) —— 模型自带预测头出候选
- [Part 02 · 36 解码调度](./36_Decode_Scheduling.md) —— 多请求同时 decode 时怎么排