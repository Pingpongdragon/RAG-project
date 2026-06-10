# recency/ — 时近性缓存族

按 **最近命中时间** 逐出（经典 cache replacement）。准入已对齐 ARC 论文范式：
**miss 驱动**——query 未命中 → escalation 去 L2 取 top-1 → 收入缓存（有缺必补）。
逐出最久未命中的；被踢出的文档计数清零（重新抓回当新人）。

| 文件 | 类 | 是什么 | 论文用？ |
|---|---|---|---|
| `lru.py` | `LRU` | miss 驱动准入 + 最久未命中逐出（ARC 范式下的真 LRU） | ✅ **主 baseline**，必留 |
| `temporal.py` | `TemporalAware` | 时间衰减加权变体（随机到货流，未对齐 ARC 范式） | ⚪ 变体，可忽略 |
| `temporal.py` | `RecencyTTL` | 带 TTL 过期（随机到货流） | ⚪ 变体，可忽略 |

> 📝 原 `LRU`（随机到货流）已扶正为 miss 驱动；旧的 `MissLRU` 已删除（逻辑并入 LRU）。

→ 正文留 **LRU** 一个。

## 为什么会差（多跳）
LRU 只看命中历史。bridge 文档从不被 query 直接检索 → 永远不会在 miss 时被取到 →
进不了缓存。多跳设定下退化到接近 Static。

