
**已尝试的最后一搏（2026-05-04）**：合并 2wiki 的 comparison + inference + compositional 三种 2-doc 类型作为新数据集 `2wiki_simple`：
- 全集 q/SF=1.67（vs comparison-only=1.12）有提升
- 但 subsample 到 n_source=15000 后实际 q/SF 降到 **1.19**（pool 太大稀释复用）
- 实验结果（sudden）：QDC H2=21.6% vs Static H2=17.9%，**仅 +3.7pp**（HotpotQA-comp 是 +25.6pp）
- 此外 H1 已饱和（97%+），证明 KB 相对 head_ctx 过大但仍解决不了 H2 问题
- **结论：2wiki 在任何子集组合下都达不到论文级证据强度**

**中期（如果 reviewer 要求）**：下载 TriviaQA-Wikipedia，运行复用率诊断脚本，确认满足上述四个条件后再跑正式实验。诊断脚本运行时间 < 5 分钟，不影响论文主线进度。

**叙事建议**：把 Mo1 定位为 "direct-evidence setting"（而非 single-hop），把 Mo2 定位为 "bridge-required setting"（而非 multi-hop）。这个区分与 HotpotQA-comparison（直接比较两个实体）和 multi-hop（A→B→答案）的结构完全一致，且不依赖"跳数"这个可争议的维度。

---

## 2WikiMultihopQA `bridge_comparison` — 失败 (2026-05-05)

**动机**：原 q/SF=2.90、>1 reuse=55.6% (n=37,382)，与 HotpotQA-comparison (2.75/73.6%) 接近，是本地数据集中**复用率最高**的近-direct 候选。

**实验**：n_source=8000, n_windows=100, window=50, KB=16k(sudden)/18.2k(gradual)。

| Drift | Strategy | Cov H1 | Cov H2 | Δ H2 vs Static |
|-------|----------|--------|--------|----------------|
| sudden  | Static | 98.4% | 51.9% | — |
| sudden  | **QDC** | 98.0% | **50.7%** | **−1.2pp** ❌ |
| sudden  | OnDemand | 99.2% | 72.7% | +20.8pp |
| sudden  | LogDriven | 98.1% | 62.1% | +10.2pp |
| gradual | Static | 98.5% | 77.7% | — |
| gradual | **QDC** | 98.3% | **77.6%** | **−0.1pp** ❌ |
| gradual | OnDemand | 99.2% | 86.6% | +8.9pp |

**结构原因（与 q/SF 数值无关）**：
- subsample 后 q/SF=1.66、reuse=26.9%，并不算低
- 但 bridge_comparison 的 SF 由 **2 个 query 命名实体（电影）+ 2 个桥接实体（导演/作曲家）** 组成
- **桥接实体在 query 文本中不出现**（query: "Are directors of film X and film Y the same nationality"）
- query-emb 与桥接节点 doc-emb 弱相关 → QDC 给桥接节点的优先级 ≈ 随机 → H2 无优势
- 反观 HotpotQA-comparison：两个 SF 实体**都直接出现在 query 文本中**，query-emb 高度对齐 SF doc-emb

**结论**：q/SF 高复用率是 *必要* 但非 *充分* 条件。还需要 **query-emb 与 SF doc-emb 强语义对齐**。本地剩余 multi-hop 数据集均不满足后一条件。

**最终 audit 结果**：HotpotQA-comparison 仍是唯一可观察到 query-driven KB curation 收益的本地数据集。Track A (Agent-memory framing, single dataset) 立场不变。
