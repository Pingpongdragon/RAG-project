# frequency/ — 频率缓存族

按 **访问频率** 逐出（LFU 家族）。准入已对齐 ARC 论文范式：**miss 驱动** escalation 取 top-1。
只统计**缓存内**文档的命中频率；被踢出的文档计数清零。

| 文件 | 类 | 是什么 | 论文用？ |
|---|---|---|---|
| `tinylfu.py` | `TinyLFU` | miss 驱动准入 + 频率门控 + LFU 逐出 | ✅ **主 baseline**，留这个 |

> 📝 原 `TinyLFU`（随机到货流）已扶正为 miss 驱动；旧的 `MissLRU`/`MissTinyLFU` 已删除
> （MissLRU 并入 recency/LRU，MissTinyLFU 并入本 TinyLFU）。

## 为什么会差（多跳）
TinyLFU 只统计**缓存内被命中**的频率。bridge 文档从不被 query 直接检索 → miss 时也取不到 →
频率恒为 0 → 无法保护 → 不比 Static 好。（频率 vs 时近只改逐出顺序，不改准入触达范围。）

