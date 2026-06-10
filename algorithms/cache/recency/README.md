# recency/ — 时近性缓存族

按 **最近命中时间** 逐出（经典 cache replacement）。准入已对齐 ARC 论文范式：
**miss 驱动**——query 未命中 → escalation 去 L2 取 top-1 → 收入缓存（有缺必补）。
逐出最久未命中的；被踢出的文档计数清零（重新抓回当新人）。

| 文件 | 类 | 是什么 | 论文用？ |
|---|---|---|---|
| `lru.py` | `LRU` | miss 驱动准入 + 最久未命中逐出（ARC 范式下的真 LRU） | ✅ **主 baseline**，必留 |
| `fifo.py` | `FIFO` | miss 驱动准入 + 插入资历逐出（ARC 范式，盲目搬运工） | ✅ **主 baseline**（ARC 论文比的对象） |
| `temporal.py` | `TemporalAware` | 时间衰减加权变体（随机到货流，未对齐 ARC 范式） | ⚪ 变体，可忽略 |
| `temporal.py` | `RecencyTTL` | 带 TTL 过期（随机到货流） | ⚪ 变体，可忽略 |

> 📝 LRU/FIFO 均为 ARC 范式 miss 驱动。旧 `MissLRU` 并入 LRU；`RandomFIFO`（随机到货流）
> 保留为 motivation 范式参照，**不是**正文的 FIFO baseline。

→ 正文留 **LRU + FIFO** 两个（ARC 论文都比了）。

## 为什么会差（多跳 + 漂移）
- **LRU**：bridge 文档从不被 query 直接检索 → miss 时也取不到 → 进不了缓存，退化到接近 Static。
- **FIFO**：盲目搬运工——只看插入资历不看功劳。核心高频文档只因"来得早"就被新 miss 挤出队列。
  完全没利用语义局部性和查询偏差。

