# INF-EAGLE. Feature Drafter and Candidate Tree | 特征草稿器与候选树

**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `投机解码`, `EAGLE`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/INF-EAGLE_Feature_Drafter_and_Candidate_Tree.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

外置 draft model 需要运行自己的网络并维护自己的 KV Cache；完成 prefill 后，它也会复用缓存，并不是每一步重算整段上下文。EAGLE 则训练一个与特定 target 配对的轻量 drafter（草稿器），复用 target 的 hidden features（隐藏特征），并结合已选 token 的 embedding 生成候选。不同 EAGLE 版本在特征构造和训练目标上有所变化，需要按版本理解。

EAGLE 还换了候选的组织方式：不是一条链，而是一棵树——drafter 不确定时同时保留两三个分支，target 一次前向把整棵树验证掉，取从根开始连续通过的最长路径。本节实现两个小机制：多层特征的加权融合（受 EAGLE-3 启发的教学简化），和候选树的路径打分与回溯。学完后你能解释 EAGLE 的 drafter 为什么必须和 target 严格配对、树上的分数为什么只是"验证优先级"而不是接受结论、以及为什么本节 GPU 实验在 12 GB 显卡上只做了条件检查。

## 前置阅读

- [Part 02 · 23 投机解码](./23_Speculative_Decoding.md) —— 验证规则不变；EAGLE 改的是候选从哪来、怎么组织。
- [Part 02 · 35 原生 MTP](./35_Multi_Token_Decoding.md) —— 同样复用隐藏状态；对照着看"训练时内置"和"事后外挂"的差别。

### Step 1：痛点与直觉——草稿器不该从零读上下文

想象 target 是一个 70B 模型，draft 是另一个 7B 模型。外置 draft 虽然比 target 小，仍要用自己的网络处理输入，并在生成时维护独立的状态。target 已经算出的隐藏特征能否复用？这就像给助手看主模型已经整理好的上下文笔记，减少从头建立表示的工作。复用是否划算，取决于传递哪些特征、drafter 的开销和分布匹配程度。

EAGLE 的观察有两条。第一，**原始 EAGLE 尝试在特征空间做预测**：target 倒数第二层的输出（论文叫 feature）是连续向量，携带上下文信息；而 token 是从分布里采样出来的离散结果，带随机性。让 drafter 在 feature 层面做自回归——给它当前位置的 feature，预测下一个位置的 feature，再借 target 自己的输出头变成 token 分布——旨在降低直接预测离散 token 的难度；这是原始 EAGLE 的设计动机，不是所有模型都成立的定理。第二，光给 feature 还不够：同一个 feature 之后可能采样出"I"也可能采样出"Always"，两条路的下一个 feature 完全不同，drafter 只看 feature 会学到一个模糊的平均。所以 drafter 的输入是 feature **拼上**已经采样出来的下一个 token 的 embedding——把采样的随机性当作已知条件喂进去。有了这两条，EAGLE 的 drafter 只需要一层 Transformer 加一个线性层（70B 的 drafter 不到 1B 参数），论文报告在 LLaMA-2-Chat 70B 上达到 2.7–3.5× 的解码加速。

EAGLE 还换了候选的形状。一条链上 drafter 对第一个位置犹豫在 A 和 B 之间，赌了 A，错了整轮作废；一棵树可以同时保留 A 和 B 两个分支，各往下再猜几步，target 一次前向用 tree attention（每个节点可看共享前缀、自身和祖先，不能看兄弟分支）把所有节点一起验证，然后按对应验证算法提交连续路径；采样模式还需要概率修正，不能只按长度选分支。换个说法：链是"押一注"，树是"分散下注，哪条中了算哪条"。代价是验证前向要处理更多位置，所以树不能无限长——哪些分支值得保留，要靠 drafter 自己给的置信度打分排序，这就是候选树上的累计分数。

本节你要实现的是两个最小机制：把 target 多层的隐藏状态按归一化权重融合成 drafter 的输入（EAGLE-3 把低、中、高三层拼在一起，本节用加权和做简化），以及给候选树的每个节点算从根到它的累计 log 分数、找出最高分叶子并回溯整条路径。

<img src="../public/02_PyTorch_Algorithms/inf_eagle_overview.svg" alt="EAGLE 从目标表示到候选验证的机制链" width="1200" height="520">

### Step 2：机制怎么走——融合两层特征，给一棵四节点的树打分

**融合特征.** 假设 target 有两层被选中，隐藏维度缩到 2：第一层输出 `[2.0, 4.0]`，第二层 `[6.0, 0.0]`，融合权重 `[1, 3]`。先把权重归一化成 `[0.25, 0.75]`（和为 1），再逐维加权：第 0 维 $0.25 \times 2 + 0.75 \times 6 = 5.0$，第 1 维 $0.25 \times 4 + 0.75 \times 0 = 1.0$，融合结果 `[5.0, 1.0]`。为什么要归一化？为了让 `[1, 3]` 和 `[2, 6]` 表达同一组相对权重，避免权重整体放大时输出也随之放大。它不会消除特征本身的层间尺度差异；若要处理这种差异，需要另行归一化特征或学习投影。真实 EAGLE-3 是把三层拼接起来过一个可训练的线性层，让模型自己学该听谁的；本节的归一化加权和是这个线性层的手工简化版。

**给树打分.** 候选树用一张"父节点表"表示，节点按父在前、子在后的顺序排列。教学函数在每个节点接收一个可相加的 `log_score`。以下是用于检查累计与回溯的人工分数，不是归一化概率（两个兄弟节点的指数和甚至大于 1）：

```
节点 0：parent=-1，token 10，log_score=-0.10   ← 根
节点 1：parent=0， token 11，log_score=-0.20   ← 根的第一个分支
节点 2：parent=0， token 12，log_score=-1.00   ← 根的第二个分支
节点 3：parent=1， token 13，log_score=-0.05   ← 节点 1 的孩子
```

累计分数 = 父节点累计分数 + 自己的 log 分数，根的父分数是 0：节点 0 得 $-0.10$，节点 1 得 $-0.30$，节点 2 得 $-1.10$，节点 3 得 $-0.35$。如果输入是真实条件 log 概率，路径相加可得到 draft 路径概率的对数；本例只把它用作排序分数。即使输入是合法概率，也不能把 draft 路径概率解释成 target 接受率。叶子是没有孩子的节点——这里是节点 2 和节点 3——最高分叶子是节点 3，沿 parent 回溯到根得到路径 `[10, 11, 13]`，这就是优先送去验证的候选。

**分数在整条流水线里的位置.** 累计分数回答的是"drafter 自己觉得哪条路最靠谱"，用来决定树怎么长（EAGLE-2 的动态树：只展开高分节点，再按分数保留前 $m$ 个）和哪些路径先验证。它**不是**接受结论——target 验证时还是逐位置比，`10 → 11` 通过、`13` 被拒，提交的就只有 `[10, 11]` 加 target 的修正 token，分数再高也没用。

<details>
<summary>速查表：三种候选来源对照</summary>

| | 外置 draft model（23） | 原生 MTP（35） | EAGLE（本节） |
|---|---|---|---|
| drafter 输入 | token 序列 | target 隐藏状态 | target 隐藏状态 + 下一 token 的 embedding |
| drafter 大小 | 独立小模型（几 B） | checkpoint 内的轻量 block | 一层 Transformer + 线性层（70B 配 <1B） |
| 需要重新训练 target 吗 | 否 | 是（训练时内置） | 否（事后训练 drafter） |
| 候选形状 | 链 | 链 | 树（tree attention 一次验证） |
| 配对约束 | 同 tokenizer | 同 checkpoint | 同 target 权重版本、层索引、隐藏维度 |

</details>

### Step 3：边界与陷阱——分数不是结论，配对不能将就

**陷阱一：把树上的高分当成接受。** `score_candidate_tree` 只负责累计分数和回溯路径，没有读取 target 分布，因此不可能计算 target 的接受概率。即使节点分数来自真实 draft 概率，也必须执行 target 验证；采样模式要使用相应的树验证与概率修正规则。

**陷阱二：验证后拼接不连续的节点.** 节点的预测依赖自己的祖先路径，提交必须是从根开始的**连续**前缀。`10` 通过、`11` 被拒、`13` 碰巧和 target 一致，也不能提交 `13`——它是 drafter 在"11 之后"这个前提下猜的，前提已经不成立。这和 23、35 节"首次拒绝即停"是同一条规则在树上的版本。

**陷阱三：drafter 和 target 名字像就拿来配.** EAGLE 的 drafter 是对着某个具体 target 的隐藏状态训练的——它学的是"这个模型第 $k$ 层的向量长什么样"。换一个同系列不同尺寸的 target，隐藏维度都对不上；换同尺寸的另一个微调版本，向量分布变了，接受率可能变化；甚至把 target 换成它自己的量化版本，隐藏状态的微小偏移也可能影响配对表现，需要重新验证。所以 Step 5 把"配对资产齐全"列为实验的第一道门：目标模型 ID + 不可变 revision + 官方声明配对的 speculator，三者缺一不可。

**陷阱四：父节点表乱序.** `score_candidate_tree` 要求节点按父在前、子在后排列（`parent < index`），这样算到子节点时父节点的累计分数已经在 `scores` 里。如果允许子在父前面，`scores[parent]` 会越界或读到错值。`test_candidate_tree_contract` 里 `parent=0` 出现在节点 0 自己身上（指向自己或未来）就必须被拒绝。

<img src="../public/02_PyTorch_Algorithms/inf_eagle_tree.svg" alt="EAGLE 候选树评分与连续验证" width="1200" height="520">

### Step 4：任务——实现特征融合与候选树打分

题目区已经写好输入校验、叶子查找和路径回溯，三个 TODO 各一到两行。

| TODO | 要做什么 | 必须满足 | 对应测试 |
| --- | --- | --- | --- |
| TODO 1 · `normalize_fusion_weights` | 求权重总和，逐项除以总和 | 总和必须为正，否则 `ValueError` | `test_normalize_fusion_weights`：`[1, 3]` → `[0.25, 0.75]`；`[0, 0]` 被拒 |
| TODO 2 · `fuse_layer_features` | 每个隐藏维度做跨层加权和 | 用归一化后的权重 | `test_fuse_layer_features`：`[[2,4],[6,0]]`、`[1,3]` → `[5.0, 1.0]` |
| TODO 3 · `score_candidate_tree` | 根的父分数为 0，其余继承 `scores[parent]` | 父在子之前已有分数 | `test_score_candidate_tree`：分数 `[-0.1, -0.3, -1.1, -0.35]`，路径 `[10, 11, 13]` |


```python
from typing import Dict, List, Sequence, Tuple

```



```python
def normalize_fusion_weights(weights: Sequence[float]) -> List[float]:
    """将非负层权重归一化为和为 1 的融合系数。"""
    if not weights or any(weight < 0 for weight in weights):
        raise ValueError('weights 必须非空且非负')
    # TODO 1（层表示融合）：计算正的权重总和并逐项归一化。
    # total = ???  # 所有层权重之和，必须为正。
    raise NotImplementedError('TODO 1：请计算权重总和并逐项归一化')
    if total <= 0:
        raise ValueError('权重总和必须为正')
    # normalized = ???  # 与输入等长、总和为 1 的融合系数。
    return normalized


def fuse_layer_features(
    layer_features: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> List[float]:
    """按层权重逐维融合隐藏表示；所有层共享隐藏维度。"""
    if not layer_features or len(layer_features) != len(weights):
        raise ValueError('特征层数必须非零且与权重数相同')
    hidden_size = len(layer_features[0])
    if hidden_size == 0 or any(len(feature) != hidden_size for feature in layer_features):
        raise ValueError('所有层必须共享非零隐藏维度')
    normalized = normalize_fusion_weights(weights)
    # TODO 2（草稿输入）：对每个隐藏维度计算跨层加权和。
    # fused = ???  # 每个隐藏维度的跨层加权和。
    raise NotImplementedError('TODO 2：请对每个隐藏维度做跨层加权和')
    return fused


def score_candidate_tree(nodes: Sequence[Dict[str, float]]) -> Tuple[List[float], List[int]]:
    """累计候选树路径分数，并返回最高分叶节点的 token 路径。"""
    if not nodes:
        raise ValueError('候选树不能为空')
    scores: List[float] = []
    for index, node in enumerate(nodes):
        parent = int(node['parent'])
        if parent >= index or parent < -1:
            raise ValueError('parent 必须是 -1 或更早的节点')
        # TODO 3（候选树）：根分支从 0 开始，其余节点继承父路径分数。
        # parent_score = ???  # 根节点为 0，其余节点继承父路径累计分数。
        raise NotImplementedError('TODO 3：请确定父路径累计分数')
        scores.append(parent_score + float(node['log_score']))

    parent_nodes = {int(node['parent']) for node in nodes if int(node['parent']) >= 0}
    leaves = [index for index in range(len(nodes)) if index not in parent_nodes]
    best = max(leaves, key=lambda index: scores[index])
    path: List[int] = []
    while best != -1:
        path.append(int(nodes[best]['token']))
        best = int(nodes[best]['parent'])
    return scores, list(reversed(path))

```


### 测试

`test_normalize_fusion_weights` 看归一化结果和空、负、零和三种非法输入；`test_fuse_layer_features` 用 Step 2 的两层例子核对逐维结果，并确认隐藏维度不一致会被拒；`test_score_candidate_tree` 用四节点树核对累计分数和最高分叶子的路径；`test_candidate_tree_contract` 检查空树、指向自己或未来的 parent、以及小于 −1 的 parent 都会报错。


```python
def test_normalize_fusion_weights():
    """权重应归一化，并拒绝空、负数和零总和输入。"""
    assert normalize_fusion_weights([1.0, 3.0]) == [0.25, 0.75]
    for invalid in ([], [0.0, 0.0], [1.0, -1.0]):
        try:
            normalize_fusion_weights(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'非法权重未被拒绝：{invalid}')


def test_fuse_layer_features():
    """多层特征应逐维融合，并拒绝隐藏维度不一致。"""
    fused = fuse_layer_features([[2.0, 4.0], [6.0, 0.0]], [1.0, 3.0])
    assert all(abs(a - b) < 1e-9 for a, b in zip(fused, [5.0, 1.0]))
    try:
        fuse_layer_features([[1.0], [1.0, 2.0]], [0.5, 0.5])
    except ValueError:
        pass
    else:
        raise AssertionError('隐藏维度不一致时应失败')


def test_score_candidate_tree():
    """候选分数应包含父路径，最佳结果应从叶节点回溯。"""
    nodes = [
        {'parent': -1, 'token': 10, 'log_score': -0.1},
        {'parent': 0, 'token': 11, 'log_score': -0.2},
        {'parent': 0, 'token': 12, 'log_score': -1.0},
        {'parent': 1, 'token': 13, 'log_score': -0.05},
    ]
    scores, path = score_candidate_tree(nodes)
    assert scores == [-0.1, -0.30000000000000004, -1.1, -0.35000000000000003]
    assert path == [10, 11, 13]


def test_candidate_tree_contract():
    """空树、未来父节点和非法父编号都应被拒绝。"""
    invalid_cases = [
        [],
        [{'parent': 0, 'token': 1, 'log_score': -0.1}],
        [{'parent': -2, 'token': 1, 'log_score': -0.1}],
    ]
    for invalid in invalid_cases:
        try:
            score_candidate_tree(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('非法候选树拓扑应被拒绝')


test_normalize_fusion_weights()
test_fuse_layer_features()
test_score_candidate_tree()
test_candidate_tree_contract()
print('✅ EAGLE 表示融合与候选树测试通过')

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
# TODO 1：归一化多层表示权重。
def normalize_fusion_weights(weights: Sequence[float]) -> List[float]:
    """将非负层权重归一化为和为 1 的融合系数。"""
    if not weights or any(weight < 0 for weight in weights):
        raise ValueError('weights 必须非空且非负')
    total = float(sum(weights))
    if total <= 0:
        raise ValueError('权重总和必须为正')
    return [float(weight) / total for weight in weights]


# TODO 2：融合多层隐藏表示。
def fuse_layer_features(
    layer_features: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> List[float]:
    """按层权重逐维融合隐藏表示；所有层共享隐藏维度。"""
    if not layer_features or len(layer_features) != len(weights):
        raise ValueError('特征层数必须非零且与权重数相同')
    hidden_size = len(layer_features[0])
    if hidden_size == 0 or any(len(feature) != hidden_size for feature in layer_features):
        raise ValueError('所有层必须共享非零隐藏维度')
    normalized = normalize_fusion_weights(weights)
    return [
        sum(normalized[layer] * float(layer_features[layer][dimension]) for layer in range(len(layer_features)))
        for dimension in range(hidden_size)
    ]


# TODO 3：累计候选树路径分数。
def score_candidate_tree(nodes: Sequence[Dict[str, float]]) -> Tuple[List[float], List[int]]:
    """累计候选树路径分数，并返回最高分叶节点的 token 路径。"""
    if not nodes:
        raise ValueError('候选树不能为空')
    scores: List[float] = []
    for index, node in enumerate(nodes):
        parent = int(node['parent'])
        if parent >= index or parent < -1:
            raise ValueError('parent 必须是 -1 或更早的节点')
        parent_score = 0.0 if parent == -1 else scores[parent]
        scores.append(parent_score + float(node['log_score']))
    parent_nodes = {int(node['parent']) for node in nodes if int(node['parent']) >= 0}
    leaves = [index for index in range(len(nodes)) if index not in parent_nodes]
    best = max(leaves, key=lambda index: scores[index])
    path: List[int] = []
    while best != -1:
        path.append(int(nodes[best]['token']))
        best = int(nodes[best]['parent'])
    return scores, list(reversed(path))

```


### 解析

#### TODO 1：权重归一化

**思路：** 加起来，再逐项除。

**关键一行：**
```python
total = float(sum(weights))
return [float(weight) / total for weight in weights]
```

**为什么这样写：** 骨架已经保证了非空、非负，这里只需补"总和为正"这一条——全零权重没有意义，必须报错而不是返回一堆 NaN。转 `float` 是为了输入里混有 `int` 时结果类型一致。

**常见错法：** 用 softmax 归一化。softmax 会把 `[0, 0]` 变成 `[0.5, 0.5]`，掩盖掉"用户没给任何权重"这个错误；也会让 `[1, 3]` 变成 `[0.12, 0.88]` 而不是 `[0.25, 0.75]`，和测试对不上。这里要的是线性归一化。

#### TODO 2：逐维融合

**思路：** 对每个隐藏维度 $d$，把各层在 $d$ 上的值按归一化权重加起来。

**关键一行：**
```python
return [
    sum(normalized[layer] * float(layer_features[layer][dimension]) for layer in range(len(layer_features)))
    for dimension in range(hidden_size)
]
```

**为什么这样写：** 外层按维度、内层按层，输出长度等于 `hidden_size`，和任意一层的形状相同——drafter 的输入维度不因融合了几层而改变。用 Python 列表而不是张量，是为了让"逐维加权"这个动作一眼可见；真实实现当然是一次矩阵乘。

**常见错法：** 循环方向写反，输出长度变成层数。或者直接 `sum(layer_features)` 不乘权重——这忽略了指定的相对权重；权重归一化也不能替代特征归一化。

#### TODO 3：累计路径分数

**思路：** 根的父分数是 0，其他节点的父分数就是 `scores[parent]`。

**关键一行：**
```python
parent_score = 0.0 if parent == -1 else scores[parent]
```

**为什么这样写：** 因为节点按父在前排列，循环到第 `index` 个节点时 `scores` 里已经有 `index` 个值，`scores[parent]`（`parent < index`）一定存在。一行就完成了"沿路径累加"，不需要递归回溯——累加在构建时就做完了，回溯只在最后找最佳路径时做一次。

**常见错法：** 每个节点只记自己的 `log_score`，最后比较叶子时比的是单步分数。`test_score_candidate_tree` 里节点 3 单步 $-0.05$ 比节点 2 的 $-1.0$ 高，碰巧结论一样；但换一组数字——比如一条路前面很差后面很好——就会选错。路径的好坏是整条路的乘积，不是最后一步。

CPU 测试只验证融合算式和树上的分数与回溯。EAGLE 真正的收益取决于 drafter 和 target 是否严格配对、树宽树深怎么定、backend 是否真的用 tree attention 一次验证——这些要到 Step 5 的真机实验里看。

### Step 5：GPU 实验方案——EAGLE 配对草稿器

CPU 部分验证的是融合算式和树的打分逻辑，它完全不涉及真实的 drafter 权重。GPU 上要回答的是：拿一对公开声明配对的 target + EAGLE-3 speculator，在 vLLM 里跑起来，接受长度多少、drafter 和验证各花多少时间、TPOT 降了没有、输出质量有没有变。本节的真机条件比前几节苛刻——官方示例配对是 Qwen3-8B（BF16 约 16 GB）加它的 EAGLE-3 head，12 GB 显卡装不下；所以当前默认模式只做**条件检查**：确认配对资产、backend 和显存是否齐全，齐全了才进入真实测量。

#### 5.1 环境与 workload

首次运行先完成[推理优化实验 Quickstart](../topic_discussion/inference_optimization/quickstart.md#inf-cuda-mechanism-quickstart)。EAGLE 必须使用明确配对的目标模型与草稿器；名字相近不能证明隐藏表示、tokenizer 和权重版本兼容（见 Step 3 陷阱三）。

| 配置档 | 模型与候选 | 显存与执行条件 | 使用方式 |
| --- | --- | --- | --- |
| 单卡 12 GB 可尝试配置 | 与 `Qwen/Qwen3-8B` 同源的量化 verifier + `RedHatAI/Qwen3-8B-speculator.eagle3` | batch 1、上下文 1024–2048、候选 2–3；必须先确认量化 verifier 与 speculator 的隐藏层和 tokenizer 一致 | 可以试，不承诺装进 12 GB；BF16 8B 配对不是本档默认方案 |
| 单卡 24 GB 推荐配置 | 同源量化 Qwen3-8B verifier + 配对 EAGLE-3 speculator | batch 1、上下文 1024–2048、候选 2–4；为 KV Cache 与工作区留余量 | 先跑 target-only，再运行 EAGLE；通过后逐步放大上下文或候选树 |
| 扩展配置 | BF16 verifier、长上下文、高并发或更大配对模型 | 预检峰值显存；单卡不足时换更大显存或多卡 | 单独登记模型、量化格式、树宽树深和 backend，不与量化结果混合 |

当前 12 GB 本机没有完成官方 BF16 8B 配对的验证，所以默认只做条件检查。有 24 GB 显存、或已验证的同源量化目标模型时，填好配对模型与 backend 就能进入真实实验；仅目标模型路径、草稿器加载或隐藏层一致性任一失败时，保存失败阶段，不要用随机张量代替。

| 配置层 | 本实验的设定 | 修改规则 |
| --- | --- | --- |
| 配对资产 | 目标模型 + EAGLE 草稿器 + 不可变版本 | 两者必须是公开声明或自行训练确认的配对 |
| 候选树 | 写死树宽、树深和验证预算 | 修改树结构后递增 workload version |
| 对照 | 仅目标模型 → EAGLE | 写死输入、dtype、生成长度、量化格式和 backend |
| 结果 | 接受长度、草稿/验证成本、TPOT、显存、质量 | 正式运行使用唯一 `RUN_LABEL` |

官方可直接服务的 EAGLE-3 配对和长格式配置见 [Speculators 服务教程](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/serve_vllm/)。


```python
RUN_MODE = 'asset_check'  # 默认值 asset_check；资产齐全后执行真实 EAGLE benchmark。
RUN_LABEL = 'rtx5070ti_eagle_asset_v1'  # 默认资产审计标签；不表示已完成 GPU benchmark。
PROFILE_ID = 'INF-CUDA-MECHANISM'  # 默认单卡推理机制环境。
SEMANTIC_ID = 'INF-EAGLE'  # 稳定机制身份。
WORKLOAD_ID = 'inf-eagle-feature-drafter'  # 稳定工作负载身份。
WORKLOAD_VERSION = 'v1'  # 配对模型或树预算变化时递增。
RECOMMENDED_TARGET_MODEL_ID = 'Qwen/Qwen3-8B'  # 官方 Speculators 示例 verifier。
RECOMMENDED_EAGLE_MODEL_ID = 'RedHatAI/Qwen3-8B-speculator.eagle3'  # 官方示例配对权重。
TARGET_MODEL_ID = ''  # 默认空；确认运行硬件后再填写配对目标模型。
TARGET_MODEL_REVISION = ''  # 默认空；填写目标模型不可变 revision。
EAGLE_MODEL_ID = ''  # 默认空；填写配对 EAGLE speculator。
EAGLE_MODEL_REVISION = ''  # 默认空；填写 speculator 不可变 revision。
EAGLE_BACKEND = ''  # 默认空；推荐使用 vLLM + Speculators 的真实 EAGLE-3 路径。
LOCAL_GPU_MEMORY_GB = 11.48  # 当前参考设备显存；不足以承载推荐 BF16 8B 配对。
TREE_WIDTH = 4  # 默认候选树宽 4；修改后递增 workload version。
TREE_DEPTH = 4  # 默认候选树深 4；修改后递增 workload version。
OUTPUT_RELATIVE_PATH = f'02_PyTorch_Algorithms/benchmarks/results/inf_eagle_{RUN_LABEL}.json'

```


#### 5.2 检查实验条件并保存 JSON

本单元检查目标模型、EAGLE 草稿器和 backend 是否齐全。条件不满足时保存一份结构化的失败记录，不产生任何性能结论。


```python
import importlib
from tools.project_runtime import resolve_project_root
import tools.speculative_mechanism_runtime as speculative_runtime
speculative_runtime = importlib.reload(speculative_runtime)
RESULT_PATH = resolve_project_root() / OUTPUT_RELATIVE_PATH
GPU_REPORT = speculative_runtime.build_asset_gate_result(
    semantic_id=SEMANTIC_ID, profile_id=PROFILE_ID, run_mode=RUN_MODE, run_label=RUN_LABEL,
    workload_id=WORKLOAD_ID, workload_version=WORKLOAD_VERSION, mechanism='eagle',
    required_assets=(
        {'name': 'target_model_id', 'value': TARGET_MODEL_ID},
        {'name': 'target_model_revision', 'value': TARGET_MODEL_REVISION},
        {'name': 'eagle_model_id', 'value': EAGLE_MODEL_ID},
        {'name': 'eagle_model_revision', 'value': EAGLE_MODEL_REVISION},
        {'name': 'eagle_backend', 'value': EAGLE_BACKEND},
    ), required_packages=('vllm', 'speculators'), output_path=RESULT_PATH,
    not_ready_decision='defer_to_multi_gpu',
    not_ready_message='推荐 EAGLE 配对与后端尚未满足；当前 12 GB 单卡不执行替代性探针。',
)
GPU_REPORT

```


#### 5.3 读取结果

读取条件检查的结果。只有配对、backend 路径和显存三项都通过，并且跑完"仅目标模型"和"EAGLE"两条路径之后，才算有性能数据。


```python
from tools.speculative_mechanism_runtime import read_mechanism_result
report = read_mechanism_result(RESULT_PATH)
{key: report.get(key) for key in ('semantic_id', 'workload', 'asset_gate', 'failure', 'evidence_level', 'decision')}

```


#### 5.4 验证状态与后续判断

当前结果只说明实验条件尚未满足，不说明 EAGLE 无效。推荐的 Qwen3-8B / EAGLE-3 BF16 配对超出 12 GB 单卡的已验证范围；在 24 GB 单卡、更大显存，或经过验证的量化目标模型路径上，可以继续比较接受长度、候选树成本、TPOT、显存与输出质量。

| 观察项 | 读取字段 | 当前结果 | 解释与动作 |
| --- | --- | --- | --- |
| 配对资产 | 模型 ID 与不可变版本 | 推荐 Qwen3-8B + RedHatAI EAGLE-3 | 拿到配对资产后重新检查 |
| 环境与 backend | `vllm`、`speculators`、GPU 显存 | 当前 12 GB 单卡未验证 BF16 推荐组合 | 使用 24 GB 配置或验证量化路径 |
| 候选树 | 树宽、树深、验证预算 | 默认 4 × 4 | 写死后才能横向比较 |
| 性能数据 | 接受长度、草稿/验证成本、TPOT、显存 | 未测量 | 完成真实配对运行后填写 |
| 当前决策 | `decision` | `defer_to_multi_gpu` | 保留方案，等待满足执行条件 |

<a id="cheatsheet"></a>
## 速查卡

| | |
|:--|:--|
| **一句话机制** | 用与 target 配对的 drafter 复用隐藏特征和 token embedding，形成候选树；target 用 tree attention 验证，按验证规则提交连续路径。CPU 题目只演示特征加权与路径打分，不实现完整 EAGLE。 |
| **关键公式与数量级** | 教学融合：$\bar w_i=w_i/\sum_jw_j$，$h=\sum_i\bar w_i h^{(i)}$；路径分数是沿父链相加。权重 `[1,3]` 得 `[0.25,0.75]`，不会消除特征尺度差。真实 EAGLE-3 使用多层特征拼接与可训练投影。草稿分数不等于 target 接受概率；本节历史 GPU 记录仅为配对条件检查。 |
| **常见坑** | 1. 把草稿高分直接当成接受。2. 提交跨越拒绝位置的不连续节点。3. 未核对 target revision、层索引与隐藏维度就配 drafter。4. 混淆权重归一化与特征归一化。5. 子节点排在父节点之前。 |
| **延伸阅读** | 专题正文 [03 解码策略](../topic_discussion/inference_optimization/03_decoding_strategies.md) · 判断手册 [按现象选择机制](../topic_discussion/inference_optimization/casebook.md#按现象选择机制) · 并行出块候选 [INF-PARALLEL-DRAFTING](./INF-PARALLEL-DRAFTING_DFlash_and_DSpark.md) · 树该多大 [INF-PROPOSAL](./INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.md) · 真实后端 [68](./68_Speculative_Decoding_Benchmark.md) |

## 相关阅读

- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) —— 第一代；Step 1 的两条观察（feature 比 token 好预测、要把采样的 token 喂回去）出自这里
- [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) —— 动态候选树：按累计置信度展开和保留节点
- [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) —— 多层特征融合与直接预测 token
- [EAGLE 官方实现](https://github.com/SafeAILab/EAGLE) —— 树的构造和 tree attention mask 的参考代码
- [vLLM Speculators：算法选择](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/decision_guide/) —— 哪些 target 有现成的 EAGLE-3 配对
- [68. Speculative Decoding Benchmark | 投机解码基准](./68_Speculative_Decoding_Benchmark.md) —— 真实 backend 上的对照