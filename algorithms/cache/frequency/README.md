# frequency/ — 频率缓存族

按 **访问频率** 准入/逐出（LFU 家族）。访问历史信号，无语义。

| 文件 | 类 | 是什么 | 论文用？ |
|---|---|---|---|
| `tinylfu.py` | `TinyLFU` | 频率草图准入门控 + LFU 逐出（Caffeine 核心）。LFU 家代表 | ✅ **主 baseline**，留这个 |
| `miss_policies.py` | `MissLRU` | 基于 miss 信号 + 语义的 LRU 变体 | ⚪ 探索性变体，可忽略 |
| `miss_policies.py` | `MissTinyLFU` | miss + 语义信号的 TinyLFU 变体 | ⚪ 探索性变体，可忽略 |

→ 正文留 **TinyLFU** 一个。

## 为什么会差（多跳）
TinyLFU 估计的是**被观测到的成功访问**频率。bridge 文档从不被直接检索 → 频率恒为 0 →
无法保护 → 不比 Static 好。（代码 docstring 已写明这一点。）
