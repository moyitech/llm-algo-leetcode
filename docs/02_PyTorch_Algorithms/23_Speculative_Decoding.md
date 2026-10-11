# 23. Speculative Decoding | 投机解码

**难度：** Hard | **环境：** CPU + GPU | **标签：** `推理优化`, `解码`, `投机解码`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

在小 batch、权重读取占主导的 decode（解码）场景中，大模型往往每轮只推进一个 token，却要读取大量权重。speculative decoding（投机解码）让便宜的 draft model（草稿模型）先猜几个 token，再让 target model（目标模型）批量验证，以额外草稿计算换取更少的串行目标前向。它是否更快，还取决于 batch、上下文长度和两个模型的成本差。

它的精妙之处在于"无损"：接受、拒绝、修正的规则设计得恰到好处，最终输出的分布和只用 target 逐 token 采样完全一样，一点都不打折。本节把这条验证循环拆出来亲手实现：接受概率怎么算、拒绝后从哪里重新抽、全部接受时为什么能白送一个 token。学完后你能解释 Transformers 的 assisted generation 和 vLLM 的 speculative decoding 在验证这一步到底做了什么，以及为什么"接受率高"不等于"一定提速"。

## 前置阅读

- [Part 02 · 21 解码策略](./21_Decoding_Strategies.md) —— 本节的 $p$、$q$ 都是 21 里那种 softmax 之后的词表分布；验证循环本质上是在两个这样的分布之间做接受—拒绝采样。
- [Part 01 · 11 KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md) —— 理解 decode 为什么是"读权重"而不是"算"占主导，才能明白一次验证多个 token 几乎不多花时间。

### Step 1：痛点与直觉——大模型一次只走一步太亏了

想象一个 3B 参数的稠密模型，以 batch=1、BF16 生成回答。仅权重约占 6 GB；忽略 attention 随上下文增长的开销，单 token 的权重相关计算量约为 $2\times3\text{B}=6$ GFLOPs。如果权重不能驻留片上缓存，每轮需要从显存搬运大量数据。批量验证 6 个位置可以让一次权重读取服务更多计算，但计算量和 KV 访问也会增加，耗时并不保证不变。机会在于：在带宽受限的配置里，这通常比让 target 串行执行 6 次更划算。困难在于自回归依赖——我们还不知道后面几个 token。

投机解码的想法是：那就先猜。找一个小得多的 draft model（比如 0.5B），让它快速连猜 $K$ 个 token；然后 target model 用**一次**前向同时拿到这 $K$ 个位置（外加第 $K+1$ 个位置）的分布，从左到右逐个检查：这个 token 我也会这么选吗？能接受就继续，碰到第一个不接受的就停下来，由 target 自己在这个位置改一个，然后开始下一轮。比如说，像资深编辑审稿：让实习生先写一段 5 句话的草稿，编辑一遍读完，看到第一句不对就从那里接手重写，而不是自己一个字一个字地口述。

这里最要紧的是"无损"。只要接受和修正的规则设计正确，最终每个位置的输出分布和 target 自己逐 token 采样完全一致——不是"差不多"，是数学上严格相等。本节 Step 5 保留的历史实验记录（Qwen2.5-3B 做 target、0.5B 做 draft、每轮猜 5 个）里，target 前向次数从 192 降到 64，平均每次前向推进 3 个 token，端到端 Median 延迟降了约 22%。

本节你要实现的是这条验证循环的四个关节：接受概率 `_accept_probability`、拒绝修正分布 `_normalized_residual`、逐位接受判断，以及全部接受后的追加 token。

<img src="../public/02_PyTorch_Algorithms/23_speculative_decoding_flow.svg" alt="投机解码流程" width="1440" height="650">

### Step 2：机制怎么走——用 4 个 token 的词表算一遍

把词表缩到 4 个 token `[A, B, C, D]`，只看某一个位置。draft 在这个位置的分布是 $q = [0.2, 0.6, 0.1, 0.1]$，它实际猜了 B；target 在同一位置的分布是 $p = [0.5, 0.3, 0.15, 0.05]$。

**第一步：算接受概率.** 规则是 $\alpha(x) = \min(1, p(x)/q(x))$。对 B 来说 $p/q = 0.3/0.6 = 0.5$，所以 B 有一半的机会被接受。直觉上：draft 对 B 的信心（0.6）是 target（0.3）的两倍，说明 draft 在这个 token 上"过于乐观"，必须打个折扣，折多少？恰好折到 target 的水平。反过来如果 draft 猜的是 A，$p/q = 0.5/0.2 = 2.5$，截断到 1——target 比 draft 更喜欢 A，没有理由拒绝。

**第二步：掷硬币.** 抽一个 $u \sim U(0,1)$，$u < \alpha$ 就接受。接受了就把 B 记下来，去看下一个位置。

**第三步：拒绝时从哪里重抽.** 如果 B 被拒绝了，不能简单地从 $p$ 重抽——那样 B 会被抽到两次（draft 一次、重抽一次），总概率超过 0.3。正确做法是从 residual（残差）分布抽：$\text{residual} \propto \max(0, p - q)$。算一下：$p - q = [0.3, -0.3, 0.05, -0.05]$，负的归零得到 $[0.3, 0, 0.05, 0]$，归一化后是 $[6/7, 0, 1/7, 0]$。注意 B 的位置是 0——**拒绝了 B 就绝不会再抽到 B**，修正只会落在 target 比 draft 更看好的 token 上。

**验证一下无损.** B 最终被输出的概率 = draft 提出 B 的概率 × 接受概率 = $0.6 \times 0.5 = 0.3 = p(B)$。A 的概率 = draft 提出 A 且接受（$0.2 \times 1 = 0.2$）+ 拒绝（总拒绝概率 $0.6 \times 0.5 + 0.1 \times 0.5 = 0.35$）后从 residual 抽到 A（$6/7$）= $0.2 + 0.35 \times 6/7 = 0.5 = p(A)$。四个 token 算下来都和 $p$ 严丝合缝。这不是巧合——这就是接受—拒绝采样的经典构造。

**第四步：一轮结束的两种方式.** 从左到右验证 $K$ 个候选，要么在第 $i$ 个位置首次拒绝（输出前 $i-1$ 个接受的 token + 1 个修正 token，共 $i$ 个），要么 $K$ 个全部接受。全部接受时还有一个甜头：target 这次前向顺手算出了第 $K+1$ 个位置的分布，直接从它抽一个 bonus token，本轮推进 $K+1$ 个。所以每一轮 target 只做一次前向，推进 1 到 $K+1$ 个 token 不等。

<details>
<summary>速查表：一轮里的三条路径</summary>

| 验证结果 | 用到的量 | 本轮输出 | 下一轮从哪开始 |
|---|---|---|---|
| 候选 $x_i$ 接受 | $u < \min(1, p_i(x_i)/q_i(x_i))$ | 把 $x_i$ 记下，继续看 $i+1$ | — |
| 第 $i$ 个首次拒绝 | residual $\propto \max(0, p_i - q_i)$ | 前 $i-1$ 个接受 token + 1 个修正 token | 修正 token 之后 |
| $K$ 个全部接受 | 第 $K+1$ 个位置的 $p_{K+1}$ | $K$ 个候选 + 1 个 bonus token | bonus 之后 |

</details>

### Step 3：边界与陷阱——三个最容易写错的地方

**陷阱一：$q(x) = 0$ 时除零.** draft 用了 top-k / top-p 截断时，很多 token 的 $q$ 是精确的 0。但 draft 只会提出 $q > 0$ 的 token，所以正常路径不会撞上这个分母；真正会撞上的是测试里刻意构造的边界：`test_zero_q_acceptance` 让 draft 提出一个 $q=0$ 的 token。约定是 $p > 0$ 就接受（target 认可、draft 没信心——没理由拒绝），$p = 0$ 就拒绝。先判零再做除法，不要指望 `float('inf')` 替你兜底。

**陷阱二：residual 总质量为 0.** 如果 $p$ 和 $q$ 完全一样，$\max(0, p-q)$ 全是 0，归一化会除零。数学上这种情况不该发生——$p = q$ 时每个 $\alpha = 1$，根本不会走到拒绝分支。但代码里要显式检查并 `raise ValueError`，`test_zero_residual_failure` 就是用一组 $p = q$ 的分布配上一个 $q = 0$ 的候选硬把它逼出来。这类"理论上到不了"的分支，宁可报错也不要静默返回全 0 向量——下游 `multinomial` 会因概率总和为零而报错。

**陷阱三：拒绝后没有立刻停.** draft 猜第 $i+1$ 个 token 时，前提是第 $i$ 个就是它猜的那个。一旦第 $i$ 个被拒绝、换成了修正 token，后面所有候选的前提都崩了，再验证它们毫无意义，继续接受甚至会破坏无损性。所以首次拒绝必须 `return`，`target_positions_checked` 也只记到 $i+1$。同理，bonus token 只能来自第 $K+1$ 行的分布（`target_probs[K]`），不能偷懒复用第 $K$ 行——那一行是用来验证第 $K$ 个候选的，描述的是另一个位置。

<img src="../public/02_PyTorch_Algorithms/23_speculative_acceptance_flow.svg" alt="投机解码验证结果" width="1440" height="660">

### Step 4：任务——补全验证循环的四个关节

题目区已经写好了输入校验（形状、归一化、token 范围）和循环骨架，你要补的是 Step 2 / Step 3 讨论过的四个判断点。

| TODO | 要做什么 | 必须满足 | 对应测试 |
|---|---|---|---|
| TODO 1 · `_accept_probability` | 返回 $\alpha = \min(1, p/q)$ | $q = 0$ 时不除零：$p > 0$ 返回 1，否则 0 | `test_zero_q_acceptance`、`test_rejection_and_residual` |
| TODO 2 · `_normalized_residual` | 取 $\max(0, p - q)$ 并归一化 | 总质量为 0 时 `raise ValueError` | `test_rejection_and_residual`、`test_zero_residual_failure` |
| TODO 3 · `speculative_decode_step` | 比较随机数与 $\alpha$ 决定接受 | 接受则继续；首次拒绝立刻 `return` | `test_rejection_and_residual`：`tokens == [0, 3]`，只检查了 2 个位置 |
| TODO 4 · `speculative_decode_step` | 全部接受后从 `target_probs[K]` 抽 bonus | 用第 $K+1$ 行，不是第 $K$ 行 | `test_all_accept_and_bonus`：`tokens == [0, 3, 2, 4]` |


```python
import torch
```



```python
def _validate_probability_contract(draft_probs, target_probs, draft_tokens):
    """检查草稿、目标和候选 token 是否能按位置一一对应。"""
    if draft_probs.dim() != 2 or target_probs.dim() != 2:
        raise ValueError('概率张量必须是二维 [K, vocab]')
    K, vocab_size = draft_probs.shape
    if target_probs.shape != (K + 1, vocab_size) or len(draft_tokens) != K:
        raise ValueError('draft/target/token 的长度或词表维度不匹配')
    if not (torch.isfinite(draft_probs).all() and torch.isfinite(target_probs).all()):
        raise ValueError('概率分布不能包含 NaN 或 Inf')
    if (draft_probs < 0).any() or (target_probs < 0).any():
        raise ValueError('概率分布不能包含负数')
    expected_draft = torch.ones(K, device=draft_probs.device, dtype=draft_probs.dtype)
    expected_target = torch.ones(K + 1, device=target_probs.device, dtype=target_probs.dtype)
    if not (torch.allclose(draft_probs.sum(-1), expected_draft, atol=1e-5) and torch.allclose(target_probs.sum(-1), expected_target, atol=1e-5)):
        raise ValueError('输入必须是归一化概率分布')
    if any(not isinstance(token_id, int) or not 0 <= token_id < vocab_size for token_id in draft_tokens):
        raise ValueError('draft token id 必须是词表范围内的整数')
    return K, vocab_size

def _accept_probability(target_prob, draft_prob):
    """返回当前候选的接受概率 alpha。"""
    # ==========================================
    # TODO 1：计算候选接受概率
    # 作用：处理 q=0，并把 p/q 截断到 [0, 1]。
    # alpha = ???  # 当前候选的接受概率
    # ==========================================
    raise NotImplementedError('TODO 1：请计算接受概率')

def _normalized_residual(target_row, draft_row):
    """构造拒绝时用于 correction 的归一化 residual 分布。"""
    # ==========================================
    # TODO 2：构造拒绝修正分布
    # 作用：保留 target-draft 的正概率质量并归一化。
    # residual = ???  # 未归一化的正概率差
    # residual_mass = ???  # residual 的总概率质量
    # ==========================================
    raise NotImplementedError('TODO 2：请构造 residual 分布')

def speculative_decode_step(draft_probs, target_probs, draft_tokens, generator=None):
    """
    验证 K 个草稿，并在拒绝或全接受时生成修正 token。
    
    Args:
        draft_probs: 草稿模型在 K 个位置上的概率分布, shape [K, vocab_size]
        target_probs: 目标模型在 K 个验证位置及 1 个 bonus 位置的概率分布, shape [K+1, vocab_size]
        draft_tokens: 草稿模型实际提出的 K 个 token_id, shape [K]
        
    Returns:
        dict: 包含最终 token 序列、接受数量、拒绝状态和检查位置。
    """
    K, vocab_size = _validate_probability_contract(draft_probs, target_probs, draft_tokens)
    accepted_tokens = []
    
    for i in range(K):
        token_id = draft_tokens[i]
        
        if not 0 <= int(token_id) < vocab_size:
            raise ValueError('draft token id 超出词表范围')
        p = target_probs[i, token_id]
        q = draft_probs[i, token_id]
        
        alpha = _accept_probability(p, q)
        random_draw = torch.rand((), generator=generator).item()
        # ==========================================
        # TODO 3：判断当前候选是否接受
        # 作用：比较随机样本与接受概率。
        # accepted = ???  # 当前候选的布尔接受状态
        # ==========================================
        raise NotImplementedError('TODO 3：请判断当前候选是否接受')
        if accepted:
            accepted_tokens.append(int(token_id))
            continue
        residual = _normalized_residual(target_probs[i], draft_probs[i])
        correction = torch.multinomial(residual, 1, generator=generator).item()
        return {'tokens': accepted_tokens + [correction], 'accepted_count': len(accepted_tokens), 'rejected': True, 'target_positions_checked': i + 1}
    
    # ==========================================
    # TODO 4：采样全部接受后的追加 token
    # 作用：从额外的第 K 行目标分布采样，而不是复用验证位置。
    # bonus = ???  # 追加 token 的整数 ID
    # ==========================================
    raise NotImplementedError('TODO 4：请采样全部接受后的追加 token')
    return {'tokens': accepted_tokens + [bonus], 'accepted_count': K, 'rejected': False, 'target_positions_checked': K}


```


### 测试

测试夹具是一个 5 词词表、$K = 3$ 的小例子，draft 和 target 都是 one-hot 分布，这样接受与拒绝都是确定的，便于看清路径。`test_rejection_and_residual` 看第 2 个候选被拒绝后是否输出 `[0, 3]` 并停在第 2 个位置；`test_all_accept_and_bonus` 让 draft 和 target 完全一致，看是否接受 3 个并追加第 4 个；`test_zero_q_acceptance` 和 `test_zero_residual_failure` 分别逼出 Step 3 的陷阱一和陷阱二；`test_probability_contract` 确认负概率等非法输入会被明确拒绝。


```python
# 按机制拆分测试，分别定位输入契约、接受、residual、bonus 和边界状态。
def _build_speculative_fixture():
    draft_tokens = [0, 1, 2]
    draft_probs = torch.zeros(3, 5)
    target_probs = torch.zeros(4, 5)
    draft_probs[0, 0] = 1.0; target_probs[0, 0] = 1.0
    draft_probs[1, 1] = 1.0; target_probs[1, 3] = 1.0
    draft_probs[2, 2] = 1.0; target_probs[2, 2] = 1.0
    target_probs[3, 4] = 1.0
    return draft_tokens, draft_probs, target_probs

def test_probability_contract():
    """输入概率、形状和 token 位置不合法时应明确拒绝。"""
    draft_tokens, draft_probs, target_probs = _build_speculative_fixture()
    invalid = draft_probs.clone()
    invalid[0, 0] = -0.1
    try:
        speculative_decode_step(invalid, target_probs, draft_tokens)
    except ValueError:
        return
    raise AssertionError('负概率应明确被拒绝')

def test_rejection_and_residual():
    """首次拒绝时应从修正分布采样并停止本轮验证。"""
    draft_tokens, draft_probs, target_probs = _build_speculative_fixture()
    result = speculative_decode_step(draft_probs, target_probs, draft_tokens)
    assert result['tokens'] == [0, 3]
    assert result['accepted_count'] == 1 and result['rejected']
    assert result['target_positions_checked'] == 2

def test_all_accept_and_bonus():
    """全部候选接受后应从额外目标位置采样追加 token。"""
    _, _, target_probs = _build_speculative_fixture()
    all_accepted = speculative_decode_step(target_probs[:3], target_probs, [0, 3, 2])
    assert all_accepted['tokens'] == [0, 3, 2, 4]
    assert all_accepted['accepted_count'] == 3
    assert not all_accepted['rejected']

def test_zero_q_acceptance():
    """草稿概率为零而目标概率为正时，候选应被接受。"""
    zero_q_draft = torch.tensor([[0.0, 1.0, 0.0]])
    zero_q_target = torch.tensor([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])
    result = speculative_decode_step(zero_q_draft, zero_q_target, [0])
    assert result['accepted_count'] == 1 and not result['rejected']

def test_zero_residual_failure():
    """拒绝位置没有正修正质量时应明确报错。"""
    draft = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    try:
        speculative_decode_step(draft, target, [1])
    except ValueError:
        return
    raise AssertionError('residual 概率质量为 0 时应明确报错')

def test_speculative_decoding():
    """汇总运行单轮投机解码的机制测试。"""
    try:
        torch.manual_seed(42)
        test_probability_contract()
        test_rejection_and_residual()
        test_all_accept_and_bonus()
        test_zero_q_acceptance()
        test_zero_residual_failure()
        print("✅ 投机解码机制测试通过：输入契约、接受、拒绝修正和追加 Token 均通过。")
    except NotImplementedError:
        print("请先完成 TODO 代码。")
        raise
    except Exception as error:
        print(f"❌ 投机解码测试失败: {type(error).__name__}: {error}")
        raise

test_speculative_decoding()

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
def _validate_probability_contract(draft_probs, target_probs, draft_tokens):
    """检查草稿、目标和候选 token 是否能按位置一一对应。"""
    if draft_probs.dim() != 2 or target_probs.dim() != 2:
        raise ValueError('概率张量必须是二维 [K, vocab]')
    K, vocab_size = draft_probs.shape
    if target_probs.shape != (K + 1, vocab_size) or len(draft_tokens) != K:
        raise ValueError('draft/target/token 的长度或词表维度不匹配')
    if not (torch.isfinite(draft_probs).all() and torch.isfinite(target_probs).all()):
        raise ValueError('概率分布不能包含 NaN 或 Inf')
    if (draft_probs < 0).any() or (target_probs < 0).any():
        raise ValueError('概率分布不能包含负数')
    ones_draft = torch.ones(K, device=draft_probs.device, dtype=draft_probs.dtype)
    ones_target = torch.ones(K + 1, device=target_probs.device, dtype=target_probs.dtype)
    if not (torch.allclose(draft_probs.sum(-1), ones_draft, atol=1e-5) and torch.allclose(target_probs.sum(-1), ones_target, atol=1e-5)):
        raise ValueError('输入必须是归一化概率分布')
    if any(not isinstance(token_id, int) or not 0 <= token_id < vocab_size for token_id in draft_tokens):
        raise ValueError('draft token id 必须是词表范围内的整数')
    return K, vocab_size

def _accept_probability(target_prob, draft_prob):
    """返回当前候选的接受概率 alpha。"""
    # TODO 1（接受概率）：处理 q=0，并计算 alpha = min(1, p / q)。
    p, q = float(target_prob), float(draft_prob)
    if q == 0.0:
        return 1.0 if p > 0.0 else 0.0
    return min(1.0, p / q)

def _normalized_residual(target_row, draft_row):
    """构造拒绝时用于 correction 的归一化 residual 分布。"""
    # TODO 2（拒绝修正）：保留 target-draft 的正概率质量并归一化。
    residual = torch.clamp(target_row - draft_row, min=0)
    residual_mass = residual.sum()
    if residual_mass <= 0:
        raise ValueError('residual 概率质量必须大于 0')
    return residual / residual_mass

def speculative_decode_step(draft_probs, target_probs, draft_tokens, generator=None):
    """完成一次 speculative decoding step。

    `target_probs[:K]` 验证草稿，`target_probs[K]` 生成 bonus token。
    """
    K, vocab_size = _validate_probability_contract(draft_probs, target_probs, draft_tokens)
    accepted_tokens = []
    
    for i in range(K):
        token_id = draft_tokens[i]
        alpha = _accept_probability(target_probs[i, token_id], draft_probs[i, token_id])
        random_draw = torch.rand((), generator=generator).item()
        # TODO 3（接受判断）：比较随机样本与接受概率。
        accepted = random_draw < alpha
        if accepted:
            accepted_tokens.append(int(token_id))
            continue
        residual = _normalized_residual(target_probs[i], draft_probs[i])
        correction = torch.multinomial(residual, 1, generator=generator).item()
        return {'tokens': accepted_tokens + [correction], 'accepted_count': len(accepted_tokens), 'rejected': True, 'target_positions_checked': i + 1}
                
    # TODO 4（全部接受）：从额外的第 K 行目标分布采样追加 token。
    bonus = torch.multinomial(target_probs[K], 1, generator=generator).item()
    return {'tokens': accepted_tokens + [bonus], 'accepted_count': K, 'rejected': False, 'target_positions_checked': K}


```


### 解析

#### TODO 1：接受概率

**思路：** $\alpha = \min(1, p/q)$，但先处理 $q = 0$。

**关键一行：**
```python
if q == 0.0:
    return 1.0 if p > 0.0 else 0.0
return min(1.0, p / q)
```

**为什么这样写：** 先把张量标量转成 Python `float` 再比较，避免 0 维张量和 Python 数字混用时的类型意外。$q = 0$ 的分支按"target 认可就接受"处理——这与 $\min(1, p/0) = \infty \to 1$ 的极限一致，只是不让除法真的发生。

**常见错法：** 直接写 `min(1.0, p / q)`，$q = 0$ 时 Python 抛 `ZeroDivisionError`；如果 `p`、`q` 还是张量，`p / q` 得到 `inf` 或 `nan`，`min(1.0, nan)` 的结果取决于参数顺序——这种 bug 只在 draft 用了激进截断时偶发，极难复现。

#### TODO 2：拒绝修正分布

**思路：** 只保留 target 比 draft 更看好的那部分概率质量，归一化成一个新分布。

**关键一行：**
```python
residual = torch.clamp(target_row - draft_row, min=0)
residual_mass = residual.sum()
if residual_mass <= 0:
    raise ValueError('residual 概率质量必须大于 0')
return residual / residual_mass
```

**为什么这样写：** `clamp(min=0)` 就是 $\max(0, p - q)$ 的向量化写法。被拒绝的那个 token 一定满足 $p < q$（否则 $\alpha = 1$ 不会被拒绝），所以它在 residual 里恰好是 0，不可能被重抽——这正是无损性成立的原因。

**常见错法：** 拒绝后直接从 `target_row` 重抽。按 Step 2 的例子，B 会以 $0.6 \times 0.5 + 0.35 \times 0.3 = 0.405$ 的概率出现，而 target 只给它 0.3。生成结果不会报错，只是悄悄偏向 draft 的口味——这种偏差要跑几千个样本做分布检验才看得出来。

#### TODO 3：接受判断与首次拒绝

**思路：** 掷硬币，`random_draw < alpha` 就接受；不接受就抽修正 token 并立刻返回。

**关键一行：**
```python
accepted = random_draw < alpha
```

**为什么这样写：** $u \sim U(0,1)$，$P(u < \alpha) = \alpha$，这就是接受概率的实现。用 `<` 而不是 `<=`：$\alpha = 0$ 时必须拒绝，而 `torch.rand` 可能返回精确的 0。首次拒绝后的 `return` 由题目骨架提供，但要明白它不是"提前优化"，而是正确性要求——后面的候选前缀已经不成立了。

**常见错法：** 把判断写成 `alpha >= 0.5` 之类的阈值。那样接受与否变成确定性的，$\alpha = 0.5$ 的 token 永远被接受，分布立刻就不对了。接受必须是随机的，概率恰好等于 $\alpha$。

#### TODO 4：bonus token

**思路：** $K$ 个全部接受，从第 $K+1$ 个位置的 target 分布再抽一个。

**关键一行：**
```python
bonus = torch.multinomial(target_probs[K], 1, generator=generator).item()
```

**为什么这样写：** `target_probs` 的形状是 `[K+1, vocab]`，下标 `K` 正是多出来的那一行。这一行是 target 在"看到全部 $K$ 个候选"之后对下一个位置的预测，白送的，不用额外前向。

**常见错法：** 写成 `target_probs[K-1]` 或 `target_probs[-2]`，复用了第 $K$ 个验证位置的分布。测试里第 3 行（下标 3）是 one-hot 在 4，第 2 行是 one-hot 在 2，`test_all_accept_and_bonus` 期望末尾是 4，复用错行会得到 2。

### Step 5：GPU 实验——真实的 3B / 0.5B 配对

CPU 题目区验证的是概率规则本身：给定 $p$、$q$，接受和修正是否正确。它回答不了的问题是——换成真实模型，draft 自己的耗时、验证开销和接受率加在一起，到底还剩多少净收益？这只能在 GPU 上用真实的目标模型和草稿模型跑出来。本实验用 Transformers 的 assisted generation 对照"只用 target 生成"，看 target 前向次数、Median / P95 延迟、峰值显存和输出是否一致；结果会进入 [68 投机解码基准](./68_Speculative_Decoding_Benchmark.md) 做真实 backend 的对照。

#### 5.1 环境与 workload

使用 `INF-CUDA-MECHANISM` Profile；首次运行需要下载目标模型与草稿模型，不需要外部数据集。环境创建和预检见[推理优化实验 Quickstart](../topic_discussion/inference_optimization/quickstart.md#inf-cuda-mechanism-quickstart)。真实实验必须完成 draft–verify 并对照仅 target 生成，不能用 CPU 的概率夹具代替。

| 配置层 | 本实验的设定 | 修改规则 |
| --- | --- | --- |
| 模型 | Qwen2.5 3B 目标模型 + 0.5B 草稿模型 | 与历史 1.5B/0.5B 结果分开登记 |
| workload | 三条写死的提示，贪心生成 64 token；两条路径各预热 3 次、正式重复 10 次 | 修改提示、长度或测量协议时递增 workload 版本 |
| 对照 | 仅目标模型 → 每轮 5 个候选的 assisted generation | 两条路径共用输入、dtype、预热和重复次数 |
| 结果 | 输出一致性、前向次数、每次目标调用推进量、Median/P95、TPOT、峰值显存和失败状态 | 逐候选接受率只有 backend 直接暴露时才登记 |


```python
RUN_MODE = 'dry_run'  # 默认 dry_run；复测时改为 real_gpu。
RUN_LABEL = 'dry_run'  # 默认不覆盖已登记结果；正式复测必须改为唯一标签。
PROFILE_ID = 'INF-CUDA-MECHANISM'  # 默认单卡推理机制环境。
SEMANTIC_ID = 'INF-SPECULATIVE-DECODING'  # 稳定机制身份。
WORKLOAD_ID = 'inf-classic-draft-verify'  # 稳定工作负载身份。
WORKLOAD_VERSION = 'v6'  # v6 为 baseline 与 candidate 分别预热，并记录 BF16 ULP 诊断。
TARGET_MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'  # 默认 3B 目标模型，扩大 target/assistant 成本差。
TARGET_MODEL_REVISION = 'main'  # 5.2 自动解析实际 commit。
ASSISTANT_MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # 默认草稿模型。
ASSISTANT_MODEL_REVISION = 'main'  # 5.2 自动解析实际 commit。
MODEL_DTYPE = 'bfloat16'  # 默认 BF16；用于复测 FP16 批量验证路径的数值漂移。
PROMPTS = ('请解释 KV Cache 的增长原因。', 'Write a Python binary search.', 'Diagnose high TPOT in an LLM service.')
MAX_NEW_TOKENS = 64  # 默认每条提示最多生成 64 token。
ASSISTANT_TOKENS = 5  # 默认每轮最多提出 5 个候选。
WARMUP_STEPS = 3  # 默认每条路径独立预热 3 次，不进入统计。
REPEAT_STEPS = 10  # 默认重复 10 次，计算 Median/P95。
SEED = 42  # 默认随机种子；贪心生成仍保留记录。
OUTPUT_RELATIVE_PATH = f'02_PyTorch_Algorithms/benchmarks/results/inf_speculative_decoding_{RUN_LABEL}.json'

```


#### 5.2 执行并保存 JSON

执行单元只读 5.1 的配置，先测 target-only，再测每轮 5 个候选的真实 assistant 路径。Transformers 没有暴露逐候选的接受日志，所以本实验不从前向次数反推接受率——"每次目标前向平均推进几个 token"是一个相关但不同的量。


```python
import importlib
from tools.project_runtime import resolve_project_root
import tools.speculative_mechanism_runtime as speculative_runtime

speculative_runtime = importlib.reload(speculative_runtime)  # 确保 Notebook 使用仓库中的最新运行时。
RESULT_PATH = resolve_project_root() / OUTPUT_RELATIVE_PATH
GPU_REPORT = speculative_runtime.run_transformers_assisted_benchmark(
    semantic_id=SEMANTIC_ID, profile_id=PROFILE_ID, run_mode=RUN_MODE, run_label=RUN_LABEL,
    workload_id=WORKLOAD_ID, workload_version=WORKLOAD_VERSION,
    target_model_id=TARGET_MODEL_ID, target_revision=TARGET_MODEL_REVISION,
    assistant_model_id=ASSISTANT_MODEL_ID, assistant_revision=ASSISTANT_MODEL_REVISION,
    dtype_name=MODEL_DTYPE, prompts=PROMPTS, max_new_tokens=MAX_NEW_TOKENS,
    assistant_tokens=ASSISTANT_TOKENS, warmup_steps=WARMUP_STEPS,
    repeat_steps=REPEAT_STEPS, seed=SEED, output_path=RESULT_PATH,
)
GPU_REPORT

```


#### 5.3 读取结果

只读取 5.2 保存的 JSON。先看有没有失败、模型 commit 是否一致、两条路径的 10 次结果是否稳定；再看每条提示的首个分歧 token、目标 top-1/top-2 之间的 margin 以及它折算成几个 BF16 ULP；最后比较目标前向次数、Median/P95 和显存。


```python
from tools.speculative_mechanism_runtime import read_mechanism_result

report = read_mechanism_result(RESULT_PATH)
summary = {key: report.get(key) for key in ('semantic_id', 'framework', 'hardware', 'workload', 'failure', 'evidence_level', 'decision')}
summary, report.get('baseline'), report.get('candidates', [])

```


#### 5.4 解释结果并作出决策

`rtx5070ti_qwen3b_05b_v2` 在两条路径分别预热 3 次、正式重复 10 次后的历史记录显示局部收益：Median 由 4661.6 ms 降至 3634.8 ms，P95 由 4742.8 ms 降至 3699.8 ms，约 1.28× 加速；目标模型前向次数由 192 降至 64（3 条提示 × 64 token，平均每次目标前向推进 3 个 token），峰值显存由 5.87 GiB 增至 6.81 GiB——多出来的约 0.94 GiB 包含新增草稿权重、缓存及其他运行时分配，不能只归因于权重。三条提示中两条逐 token 一致；第三条在第 4 个生成 token 处稳定分歧，此时目标模型 top-1/top-2 的 margin 为 0.125，恰好等于一个 BF16 ULP：两个 token 的分数在 BF16 精度下几乎并列，批量验证路径与逐 token 路径的累加顺序不同就足以让 argmax 翻转。这一观察支持“近并列位置对数值差异敏感”的解释，但仅凭 margin 不能排除所有实现差异；该运行记录没有通过严格逐 token 复现检查。

| 观察项 | 读取字段 | 当前结果 | 解释与动作 |
| --- | --- | --- | --- |
| 运行与条件 | `failure`、模型 commit、workload | v2 成功；模型 commit 与 v1 一致 | baseline/candidate 各自预热，10 次结果稳定 |
| 正确性 | stability、output hashes、first mismatch、margin in ULPs | 两条一致；一条在 1 BF16 ULP 处分歧 | 保留严格复现失败的记录，并标记为近并列数值边界 |
| 推进机制 | target/assistant calls、tokens per target call | 目标前向 192 → 64；每次目标前向推进 3 token | 不把该字段等同于逐候选接受率 |
| 性能与资源 | Median/P95、峰值显存、延迟比 | Median/P95 均降低约 22.0%；显存增加约 0.94 GiB | 3B/0.5B 成本差已形成稳定净收益 |
| 决策 | `evidence_level`、`decision` | `tune_numeric_equivalence` | 接受机制层面的结论；严格复现与真实 backend 能否上线分开判断 |

<a id="cheatsheet"></a>
## 速查卡

| | |
|:--|:--|
| **一句话机制** | draft 模型连猜 $K$ 个 token，target 一次前向验证全部：按 $\alpha = \min(1, p/q)$ 逐个接受，首次拒绝时从 $\text{norm}(\max(0, p-q))$ 抽一个修正 token 并停止；$K$ 个全接受时用第 $K+1$ 个位置白送一个 bonus token。输出分布与只用 target 严格相同。 |
| **关键公式与数量级** | $\alpha(x) = \min(1, p(x)/q(x))$；residual $\propto \max(0, p-q)$，被拒绝的 token 在其中恒为 0。每轮 target 前向 1 次，推进 $1 \sim K+1$ 个 token。本节实验（Qwen2.5-3B / 0.5B，$K=5$）：目标前向 192 → 64，平均每次推进 3 个，Median 延迟 −22%，显存 +0.94 GiB。 |
| **常见坑** | 1. 拒绝后从 $p$ 直接重抽——分布偏向 draft 的口味，且不会报错。2. $q = 0$ 时除零：先判零，$p > 0$ 则接受。3. 首次拒绝后不停止，继续验证后面的候选——它们的前缀已经不成立。4. bonus 复用第 $K$ 行而不是第 $K+1$ 行的分布。5. 接受率高 ≠ 提速：draft 自身耗时和验证开销要一起算，见 INF-PROPOSAL。 |
| **延伸阅读** | 专题正文 [03 解码策略](../topic_discussion/inference_optimization/03_decoding_strategies.md) · 判断手册 [按现象选择机制](../topic_discussion/inference_optimization/casebook.md#按现象选择机制) · 下一步 [35 原生 MTP](./35_Multi_Token_Decoding.md)：不用外置 draft，模型自带预测头出候选 · [INF-PROPOSAL](./INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.md)：$K$ 该取多少 · 真实后端 [68](./68_Speculative_Decoding_Benchmark.md) |

## 相关阅读

- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)（DeepMind）—— 接受规则与无损性证明，本节公式的出处之一
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)（Google）—— 同期独立提出；给出了期望每轮推进长度 $\frac{1-\alpha^{K+1}}{1-\alpha}$ 的推导
- [Transformers Assisted Generation 文档](https://huggingface.co/docs/transformers/main/en/generation_strategies#speculative-decoding) —— Step 5 用到的 `assistant_model` 接口
- [Part 02 · 35 原生多 Token 预测](./35_Multi_Token_Decoding.md) —— 把 draft 做进模型自身
- [Part 02 · INF-PROPOSAL 动态候选与验证预算](./INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.md) —— 每轮猜几个才划算