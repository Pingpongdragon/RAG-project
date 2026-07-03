# REFERENCES — DRIP 相关文献（2026-06 调研）

> 两轮文献调研结果。⚠️ 标 *unverified-recent* 的是搜索引擎返回的 2026 arXiv ID 但未能开全文，引用前需核实。
> ≤2025 的较可靠。供 related work / baseline 选择用。

## A. 漂移检测（模块①）

| 文献 | 出处 | 核心 | 对 DRIP |
|---|---|---|---|
| DriftLens (full) | arXiv:2406.17813, 2024 | 文本 embedding 上 per-label Fréchet 漂移，offline 基线+在线窗口距离 | **基础**；我们改成对齐特征 FID 并用作缓存 gate |
| DriftLens (origin) | ICDMW 2021 | 原始短文 | 双引 |
| ADWIN-U | KAIS 2025 | 无监督自适应窗口 | 标量漂移 **baseline** |
| 经典 DDM/EDDM/PHT/ADWIN | comparative study | 误差率类 | baseline |
| covariate drift on text | arXiv:2309.10000, 2023 | doc embedding 上 KS+MMD | MMD **baseline** |
| concept drift in text streams (survey) | arXiv:2312.02901, 2023 | 综述 | related work |
| MS MARCO distribution shifts | arXiv:2205.02870, 2022 | query 语义/意图/长度 shift 伤检索 | **query shift 伤检索**引用 |
| query shift (cross-modal) | arXiv:2410.15624, 2024 | 在线 query 流分布不同 | query shift 定义 |
| workload drift benchmarking | arXiv:2510.10858, 2025 | 含 caching/indexing 的 workload drift 生成 | drift 实验设置依据 |
| Fréchet for IR | arXiv:2401.17543, 2024 | FID 用于 IR 离线评测 | 佐证 FID 选择 |

**空白**：未找到检测 query–KB *对齐* 漂移作触发信号的工作 → 这是 DRIP 贡献。

## B. RAG cache / 语义缓存 / 动态索引（模块③a）

| 文献 | 出处 | 缓存单位 | 触发 | 对 DRIP |
|---|---|---|---|---|
| GPTCache | github zilliztech | LLM response | 相似度命中 | **baseline** (=GPTCacheStyle) |
| RAGCache | arXiv:2404.12457, 2024 | doc KV states | 复用频率 | systems-side 语义缓存，单位不同 |
| LightRAG | arXiv:2410.05779, 2024 | graph node | 文档到达 | =DocArrival baseline |
| HippoRAG | arXiv:2405.14831, NeurIPS 2024 | KG 节点 | 文档到达 | **R3 最近邻**：PPR 实体游走解多跳；我们做缓存准入 |
| 学习型 admission/eviction | SLAP / Cold-RL(2508.12485) / Parrot(2301.11886) | CDN/KV | 学习 | "学习缓存存在但不用失败 query 的语义需求" |
| SPANN | arXiv:2111.08566, NeurIPS 2021 | vector | 静态 | hot-tier substrate |
| FreshDiskANN / SPFresh(2410.14452) / Quake(2506.03437) | — | vector | 插入/workload | substrate；DRIP 是其上 policy 层；Quake 最接近 workload-drift |

**空白**：无系统用 query–KB 对齐漂移驱动 bridge-aware hot-tier 准入（静态 corpus）。

## C. Agentic memory（非主线 related work）

DRIP v1 不维护独立 agent memory 层。以下文献只用于划清范式边界：它们管理对话、经验或长期记忆，而本文管理 L1 document hot-tier。

**背景文献**：
- MemGPT (2310.08560) — OS 式 context 分页，hot/cold 分层祖先
- MemOS (2505.22101) / MemoryOS (2506.06326) — memory 作 OS 资源 / 三层 heat 提升；**有 lifecycle 无漂移检测器**
- A-Mem (2502.12110) — Zettelkasten 自组织 memory，note evolution
- Generative Agents (2304.03442) — memory stream + reflection 巩固 + recency/importance 评分
- MemoryBank (2305.10250) — Ebbinghaus 遗忘曲线（≈缓存衰减/eviction）
- episodic/semantic 分层：2605.17625, Zep/Graphiti (2501.13956)

**共享 memory（不进入主实验）**：
- Collaborative Memory (2505.18279) — private+shared 双层 + access-control
- Memory Sharing INMS (2404.09982) / AGENT KB (2507.06229) — 跨 agent 共享池

**memory vs RAG corpus 分工（边界依据）**：
- Retrieval by Decoupling and Aggregation (2602.02007, *unverified-recent*) — 最清楚地区分 "agent memory(有界连贯) vs RAG corpus(大而异质)"
- Shared-Private Dual-Stream (2506.06240, EMNLP 2025)

**drift 驱动 memory 更新（future work / related work）**：
- SPRInG (2601.09974, *unverified-recent*) — novelty 驱动选择性更新，但**更新参数不是缓存**
- Adaptive Memory Admission Control (2603.04549, *unverified-recent*) — 标题最接近"memory 准入"，需核实
- Agentic Memory Management via RL (2604.01560, *unverified-recent*) — RL 学 memory 管理，可作对比

**memory 操作词汇**：survey 2505.00675（Consolidation/Update/Index/Forget/Retrieve/Compress）

**baseline 结论**：这些不是 v1 主表 baseline。主表只放 cache replacement / RAG cache / ARC-style document cache。

**多跳记忆复用**：HippoRAG, latent reasoning RAG (2605.06285, 缓存中间 thoughts), MA-RAG(2505.20096)

**安全（局限里提一句）**：cross-user contamination (2604.01350), long-term memory security survey (2604.16548)

---

## D. 自适应缓存 / cache replacement（主 baseline 族，2026-06 调研）

| 文献 | 出处 | 角色 |
|---|---|---|
| ARC (Adaptive Replacement Cache) | Megiddo&Modha FAST'03 | 经典自适应缓存，**必比**（随 workload 自调 recency/freq） |
| LeCaR | HotStorage'18 | LRU+LFU 专家 + regret 最小化，working-set≫cache SOTA |
| CACHEUS | FAST'21 | LeCaR 升级，把 workload 分 LFU/LRU/scan/churn 四 primitive |
| TinyLFU / W-TinyLFU | 1512.00727 | admission policy（Caffeine 核心） |
| Belady/MIN (OPT) | Belady 1966 | 离线最优上界（=现 Oracle，建议正名） |
| **ARC — Agent RAG Cache** ⚠️ | **arXiv:2511.02919 (2025.11)** | **DRIP 最近邻威胁**：query distribution + 缓存几何管理 per-agent 语料；必引并区分（漂移检测 vs 分布几何、实体桥接） |
| RAGCache | arXiv:2404.12457 | RAG 知识缓存（KV 状态 + 替换策略），systems-side 近邻 |
| Online semantic eviction | arXiv:2508.07675 | 语义缓存淘汰（带 mismatch cost）+ online adaptation，正面重叠 |
| GroundedCache | arXiv:2605.27494 | answer 缓存的 admission gate（相似度+证据重叠+版本有效性） |

## E. cache↔memory 等价（删除为主张，仅保留为背景）

- memory 六操作 survey（Consolidation/Updating/Indexing/Forgetting/Retrieval/Condensation） — arXiv:2505.00675 → 映射 cache 的 admit/merge/index/evict/lookup/compress
- xMemory「Beyond RAG for Agent Memory」（corpus shape 区分 RAG vs memory） — arXiv:2602.02007
- MemArchitect(decay/conflict/eviction governance) 2603.18330 / StageMem(promotion/eviction lifecycle) 2604.16774 / SleepGate(forgetting-as-eviction) 2603.14517 / FadeMem(decay forgetting) 2601.18642
- MemGPT(OS 虚拟内存/paging) 2310.08560 / MemoryOS(FIFO+page) 2506.06326 — memory 用 cache 词汇
- ⚠️ **无论文把 memory consolidation/forgetting ≡ cache admission/eviction 作为正式贡献证明** → DRIP v1 不使用该点作为贡献。

---

## DRIP v1 占据的白点

1. query–KB **alignment-feature FID** 作为 L1 document hot-tier 的更新触发信号。
2. drift signal 映射到显式写入预算 `lambda * B`，避免 miss-driven 无限换入。
3. **entity-chained bridge prefetch** 作为缓存准入信号，把 query 看不到的 second-hop document 主动换入 L1。

> 定位结论：主词 **cache** + drift-aware/adaptive + multi-hop bridge prefetch。必引并区分 ARC(2511.02919)。
