# recency/ — 时近性缓存族

按 **最近访问时间** 逐出（经典 cache replacement）。访问历史信号，无语义。

| 文件 | 类 | 是什么 | 论文用？ |
|---|---|---|---|
| `lru.py` | `LRU` | 最近最少使用逐出。最经典的 cache baseline | ✅ **主 baseline**，必留 |
| `temporal.py` | `TemporalAware` | 时间衰减加权的 LRU 变体 | ⚪ 变体，可忽略 |
| `temporal.py` | `RecencyTTL` | 带 TTL 过期的时近性 | ⚪ 变体，可忽略 |

→ 正文留 **LRU** 一个即可，另两个是探索性变体。

## 为什么会差（多跳）
LRU 只看访问历史。bridge 文档从不被直接检索 → 永远"最久未访问" → 第一个被逐出。
在多跳设定下退化到接近 Static。
