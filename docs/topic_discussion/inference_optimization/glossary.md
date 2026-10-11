# 推理优化专题术语表

正文采用行业英文术语，每页首次出现加中文括注，之后沿用英文或常见缩写。中文列也用于搜索；API、模型名和 JSON 字段保持原文。全教程规则见[术语规范](../../terminology.md)。需要直接找练习时，使用[机制速查索引](./casebook.md#机制速查索引)。

| English / 缩写 | 中文括注 | 本专题含义 | 容易混淆之处 |
|---|---|---|---|
| TTFT | 首 Token 延迟 | 请求到达后获得首个输出 token 的时间 | 主要受排队与 Prefill 影响，不代表 Decode 速度 |
| TPOT | 每输出 token 时间 | Decode 阶段相邻输出 token 的平均时间 | 与端到端延迟、吞吐不是同一指标 |
| E2E latency | 端到端延迟 | 从请求到达到完成或失败的总时间 | 必须结合输入、输出长度和终止状态解释 |
| SLO | 服务等级目标 | 为延迟、吞吐、可用性、超时率或拒绝率设定的可测目标，必须注明指标、分位数和统计窗口 | 不是单个性能指标；也不等同于包含责任条款的对外服务等级协议（SLA） |
| Prefill | 预填充 | 一次处理输入 token 并建立初始 KV 状态 | 不等同于请求到达后的全部 TTFT |
| Decode | 解码 | 基于已有状态逐轮生成后续 Token | 与“有推理能力的模型”无关 |
| iteration token budget | 本轮 Token 预算 | 当前迭代允许处理的 Prefill 与 Decode Token 总量 | 不等同静态 batch size |
| batch slot | 执行槽位 | 当前迭代中可容纳一个活跃请求的位置 | 有空槽位不代表 KV 或 Token 预算足够 |
| iteration boundary | 迭代边界 | 一轮前向结束后释放终止请求、接纳新请求并更新批次的时点 | 不等同请求级优先级排序 |
| continuous batching | 连续批处理 | 在迭代边界移出完成请求并接纳新请求 | 请求优先级由请求级调度决定，批次成员由迭代级调度决定 |
| KV lifecycle | KV 生命周期 | KV block 的分配、追加、共享、引用、驱逐和回收 | 与请求公平性和优先级是不同问题 |
| prefix cache | 前缀缓存 | 复用相同前缀已计算的 KV 状态 | 命中率不能仅由 TTFT 变化反推 |
| PD disaggregation | 预填充—解码分离 | 将 Prefill 与 Decode 放到不同执行资源或服务层 | 分离后仍需处理 KV 交接和排队 |
| exposed transfer time | 暴露传输时间 | 无法被计算或准备工作隐藏的传输时间 | 传输总时间下降不代表暴露时间必然下降 |
| backend dispatch | 执行分派 | 根据能力门槛选择 eager、compile 或专用后端路径 | API 可调用不证明命中目标 kernel |
| fallback | 回退 | 候选路径不满足能力或稳定性条件时使用保守路径 | 回退必须记录原因，不能静默并入候选结果 |
| model artifact | 模型产物 | 权重、配置、分词器、量化元数据和文件哈希组成的可加载交付物 | 量化算法名称不等于可被目标后端加载的产物格式 |
| loader | 加载器 | 将模型产物解析为目标执行后端可运行状态的组件 | 成功加载不证明目标低精度计算内核已经执行 |
| kernel path | 计算内核路径 | 请求实际经过的浮点、低精度或融合计算实现 | 配置中的目标数据类型不等于运行时真实路径 |
| draft / assistant model | 草稿模型 | 在目标模型验证前生成候选 token 的轻量模型或附加模块 | 草稿输出不能绕过目标模型验证直接提交 |
| target / verifier model | 目标模型 | 验证草稿候选并决定接受、拒绝与修正的主模型 | 与奖励模型或评测模型不是同一角色 |
| accepted prefix length | 连续接受长度 | 从候选首位开始连续通过目标验证的 Token 数 | 后部命中不能跨过前部拒绝位置 |
| proposal budget / speculation lookahead | 候选预算 | 每轮允许草稿器提出并送入验证的最大候选数 | 越大不一定越快，还会增加草稿与修正成本 |
| token | 词元 | tokenizer 切分后的生成单位 | 不一定是一个字或一个词 |
| logits | 词表分数 | 归一化前的分数，可正可负 | 不是概率；最大值也不是置信度 |
| decoder-only | 仅解码器 | 本专题默认的因果自回归模型 | 不同于 encoder-decoder 的解码部分 |
| greedy / sampling | 贪心 / 采样 | 取 argmax / 从目标分布抽取 | 有限低温不等于显式 greedy；唯一最大值下趋零极限另论 |
| temperature | 温度 | logits 除以正数 T，再形成概率 | 不改变排序；极端缩放可能超出 dtype 范围 |
| top-k / top-p | 候选截断 / 核采样 | 按排名或累计概率质量过滤 | 并列、等号边界与最少保留数取决于实现，见 21 |
| draft / verify / accept / correct | 草稿 / 验证 / 接受 / 修正 | 先提出候选，再按目标分布验证；拒绝后修正 | 采样拒绝后不能简单从 p 重抽，23 使用 normalize(max(0,p−q)) |
| acceptance rate | 接受率 | 统计区间内接受候选数与提出数之比，须明确分母 | 不等于每次 target 前向推进量，也不是加速比 |
| verification budget | 验证预算 | 本轮允许验证的节点数或计算量 | 不一定等于草稿提出数；树节点数也不等于路径深度 |
| MTP | 多 token 预测 | 模型内置模块提供额外位置的预测 | 候选步数依赖 backend，不能总与层数画等号 |
| EAGLE | 特征草稿器 | 用与 target 配对的隐藏特征和 token embedding 生成候选 | 不同版本的特征构造与训练目标不同 |
| candidate tree / tree attention | 候选树 / 树注意力 | 每个节点看共享前缀、自身与祖先，不看兄弟分支 | 树分数不等于接受率；采样不能简单挑最长分支 |
| DFlash / DSpark | 并行草稿 / 半自回归草稿 | 块扩散草稿；DSpark 增加轻量顺序头与置信头 | CPU 逐行 argmax 练习不等于完整扩散模型 |
| online softmax | 在线 softmax | 新块到来时重标定行最大值、指数和与累计输出 | 不能把各块独立 softmax 的输出简单平均 |
| SDPA / attention backend | 注意力 API / 注意力后端 | 按设备、dtype、shape 等选择 flash、math 等实现 | API 成功不证明命中了特定 kernel，支持集合随版本变化 |
| fusion | 融合 | 合并算子以减少部分启动和中间读写 | 模式出现不代表当前输入可融合或一定更快 |
| eager / compiled | 逐算子执行 / 编译执行 | 直接执行算子 / 由编译后端组织计算 | 两者都可能使用高效 kernel；编译不保证融合或提速 |
| workload | 输入与运行条件 | 模型、输入长度、生成规则、并发及测量设置等 | 比较时改变了这些条件，就不能只归因于优化方法 |
| chunked prefill | 分块预填充 | 将长输入拆成调度片段，与其他请求交错 | 调度分块不同于 attention kernel 内部分块 |
| PagedAttention | 分页注意力 | 通过 block table 访问按块管理的 KV | 分页管理与减少分数矩阵写回解决不同问题 |
