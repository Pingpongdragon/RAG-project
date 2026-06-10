# semantic/ — 语义缓存族

按 **query/文档的语义相似度** 决定缓存准入与逐出（而非访问历史）。
这是与你方法最该正面对比的一族（ARC 论文也是拿这族的两个方法当主 baseline）。

| 文件 | 类 | 是什么 | 论文用？ |
|---|---|---|---|
| `gptcache.py` | `GPTCacheStyle` | GPTCache 风格：语义去重门控准入 + 衰减相似度逐出 | ✅ **主 baseline**（ARC 论文比的对象之一） |
| `proximity.py` | `Proximity` | Bergman 2025：(过去 query→文档) FIFO 账本，相似 query 复用其文档 | ✅ **主 baseline**（ARC 论文比的对象之一） |

## 为什么这族在「多跳 + 漂移」下会差（机制固有，非人为设障）

1. **没有多跳/bridge**：只缓存与 query 相似的文档。bridge 文档（第二跳需要、与 query 不相似）
   永远不被取到、永远进不了缓存。
2. **没有漂移处理**：相似度门控 + FIFO/衰减逐出，没有"检测分布漂移并主动适应"的机制。
   漂移过渡期持续 miss。

→ 这两个 gap 正是你方法主攻的轴。**防稻草人**：务必给一组"单跳/无漂移"对照，证明它们的差
专来自漂移+多跳，而非实现劣质。详见 `docs/design/CODE_EXPLAINED.md §7`。
