# Next-Round Audit TODO （评审意见落地未尽事项）

> 来源：[评审意见.md](评审意见.md) + [会议记录_05.txt](会议记录_05.txt)
> 本轮（2026-05-26）已经完成：故事线 framing 升级到 L1/L2/L3、节标题改 "Why existing policies fail"、加 §Agentic shared-RAG access patterns、加 §Why this is a shared-cache problem（占位无假数字）、DRIP-Dense 段加承上启下句、bridge 段补 ~60% 覆盖率量化、与 per-user memory 互补表态、contributions 对齐。详见 [STORYLINE_v1.md](STORYLINE_v1.md) v4。
>
> 本文件记录尚未做的事，避免被下一次评审再次点出。

---

## 1. 必须补的实验（评审 §三.3 — 并发动机实验）

**目标**：用 100-agent batch 实验，把"bridge 是 *shared* 压力，不是单 session memory 能解决的"这一定性 claim 落成一个具体数字。

**指标（双指标，二选一或同时报告）**：
- **pairwise hidden-bridge sharing rate**：在一个 100-query batch 里，"至少有 ≥1 个 hidden bridge entity 与同 batch 另一 query 重合"的 query 占比
- **duplicate cold-read rate**：在一个 batch 的 cold-tier 桥接读取里，被同 batch 其它 agent 已经读过的比例

**数据源**：[motivation_2/data/results_*_2wiki_*.json](motivation_2/data/) 已有 gold sf_titles；保守起见只统计 gold，不计 distractor。

**脚本位置**（建议）：
- `motivation_2/batch_overlap.py` 新写
- 输出：`motivation_2/data/batch_overlap_2wiki_bridge.json`
- 画一个 appendix 柱状图（横轴 = batch size 50/100/200，纵轴 = 两个比率）

**话术回填**：跑出后回 [motivation.tex:Why this is a shared-cache problem](motivation.tex) 那一段，把"a non-trivial fraction"换成具体数字。

---

## 2. 必须补的图（评审 §三.1 — Fig 0 子图）

**目标**：在 Fig 0 上把"Agent 微观链式访问也是高频模式"这件事画出来，弥合"宏观 Fig 0 ↔ 微观 Fig 2"叙事断层。

**实施方案**：
- **首选**：用 AgentBench 或 WebShop 的真实 agent trace（评审推荐）。从 trace 里提取连续 query，统计"下一条 query 的关键实体不出现在当前 query 文本中、必须从上一条的检索结果推出"的比例
- **备选**：如果 AgentBench/WebShop trace 处理不了，用 2Wiki bridge / HotpotQA-bridge gold sf_titles 序列做合成多跳日志，统计同一个比例（评审也接受合成）

**预期形态**：
- Fig 0 改为三栏：(a) WildChat 月级 topic mixture | (b) Google Trends 多年权重重排 | (c) Agent trace 链式查询占比（新增）
- caption 把宏观→微观一句话串起来

**脚本位置**（建议）：
- `motivation_0/extract_chained_query_ratio.py`
- 重画器：`motivation_0/plot_mo0_drift.py` 升级到三栏

---

## 3. 缺失 BibTeX 项（与本轮 framing 改动无关，但 motivation.log 一直有 undefined citation 警告）

`references.bib` 里**没有**以下 cite key，但正文里被引用：

| cite key | 引用位置 | 应该指向 |
|---|---|---|
| `hu2025memory` | §Setting | 一篇关于 LLM agent memory 的近期综述 / 综合 paper |
| `bang2023gptcache` | §"Why existing policies fail" (ii) | Bang 等 GPTCache 论文（NeurIPS 2023 / arXiv 2307.10543） |
| `dynaquest2025` | §"Why existing policies fail" (i) | DynaQuest 论文（如果有具体引用） |
| `hoh2025` | §"Why existing policies fail" (i) | Temporal RAG / TTL-style 引用 |

**修复方式**：在 [references.bib](references.bib) 末尾追加这几个 entry。建议下一轮把整个 motivation.tex 跑一遍 `pdflatex && bibtex && pdflatex × 2`，把所有 `Citation X undefined` 警告清干净。

注：本轮我没有 *新增* 任何 undefined citation，已经把临时加进去的 `yao2023react` 撤掉了。

---

## 4. 后续会议记录潜在补强项

- **图表尺寸**（会议 §4）：Fig 0/1/2 当前 `width=0.98\textwidth` 横跨整页，会议建议"避免过大喧宾夺主"；可考虑 Fig 2 改 `figure` 单栏 + `width=\linewidth`，或 Fig 0 调小到 0.85
- **bridge 量化补充**（会议 §3）：除了 ~60% bridge-doc 覆盖率，还可以补一个具体 query 的失败实例（appendix case study），把"无法匹配桥接文档"实例化成一个 trace
- **Method 对应实验**（会议 §2）：当前 motivation.tex 不含 [METHOD] 自身的实验段；下一轮做完 [METHOD] 后，要回到 §"Deconstructing the multi-hop bottleneck" 末尾把"motivates [METHOD]"实例化成一个 +X pp 的预告数字

---

## 5. 优先级建议

```
P0  并发动机实验（§1）       → 最致命，评审明说 "极其关键的护城河"
P0  Fig 0(c) 链式查询子图（§2）→ 评审主诉求"宏观↔微观断层"的最直接修补
P1  references.bib 缺失项（§3）→ 编译警告会让 reviewer 第一眼就皱眉
P2  图表尺寸 / 案例补充（§4）  → 锦上添花，时间允许再做
```
