# Dataset Analysis: When Does DRIP-Dense (QDC) Work?

All experiments use **sudden distribution drift**, 100 windows (50 pre / 50 post), window size = 50 queries.
QDC delta = QDC H2 R@5 minus Static H2 R@5 (positive = better than frozen KB).
**Key reference**: OnDemandFetch is the per-query oracle-like fetch baseline.

---

## Three Conditions for QDC Effectiveness

| Condition | Criterion | What breaks when violated |
|---|---|---|
| **Cond-A** Demand reuse | q/SF ≥ 1.3 — multiple queries share the same SF documents | `demand[d]` never exceeds `serve[e]`; no replacement fires |
| **Cond-B** Query–SF alignment | cosine(query, SF doc) ≥ 0.6 — query text names or strongly implies the SF entity | Pool probe points at irrelevant docs; demand accumulates on wrong candidates |
| **Cond-C** Pool contains distractors | Pool is not dominated by gold-SF docs | Blind baselines (RandomFIFO) accidentally pick useful docs; QDC's selectivity has no value |

---

## Dataset Summary Table

| Dataset | Hops | KB/pool | q/SF | Align | A | B | C | QDC−Static | QDC−BestOther | Oracle−QDC | Verdict |
|---|:---:|---:|---:|---:|:---:|:---:|:---:|---:|---:|---:|---|
| **FEVER** (mo1) | 1 | 14.7% | 8.15 | 0.99 | ✓ | ✓ | ✓ | **+21.4 pp** | −11.2 pp | +15.8 pp | ✅ Works well |
| **HotpotQA-comp** (mo1) | 1 | 23.9% | 1.35 | 0.92 | ✓ | ✓ | ✓ | **+23.7 pp** | −51.0 pp | +57.6 pp | ✅ Works (large gap to Oracle remains) |
| **TriviaQA-Wiki** (mo1) | 1 | 32.9% | 1.36 | 0.88 | ✓ | ✓ | ✗ | +24.6 pp | **−47.5 pp** | +52.3 pp | ⚠️ Cond-C fails: RandomFIFO also works by accident |
| **SQuAD-kb1500** (mo1) | 1 | 7.2% | ~4–5/doc | ~0.5 | ~ | ✗ | ✓ | +4.2 pp | −57.7 pp | +75.8 pp | ❌ Paragraph-level granularity; probe misaligns |
| **SQuAD-probe-kb5000** (mo1) | 1 | 23.9% | ~4–5/doc | ~0.5 | ~ | ✗ | ✓ | +12.5 pp | **−43.2 pp** | +56.3 pp | ❌ Same root cause; bigger KB doesn't help |
| **2wiki-comparison** (mo1) | 2 | 37.6% | 1.02 | 0.85 | ✗ | ✓ | ✓ | +9.0 pp | −66.3 pp | +70.8 pp | ❌ Cond-A fails: low cross-query reuse |
| **2wiki-bridge-comp** (mo1) | 2 | 48.7% | 1.66 | 0.31 | ✓ | ✗ | ✓ | +1.8 pp | −14.9 pp | +25.2 pp | ❌ Cond-B fails: hop-2 entity never in query text |
| **2wiki-simple** (mo1) | 2 | 24.3% | 1.19 | 0.60 | ✗ | ~ | ✓ | +5.2 pp | −50.0 pp | +60.2 pp | ❌ Cond-A+B both marginal |
| **HotpotQA-bridge** (mo2) | 2 | 54.3% | — | — | — | — | — | +3.2 pp | −33.2 pp | +56.0 pp | ❌ KB near-full; no room to swap in tail docs |
| **2WikiMultihopQA** (mo2) | 2 | 72.0% | — | — | — | — | — | +2.9 pp | −29.5 pp | +41.0 pp | ❌ KB/pool=72%; retrieval bottleneck is multi-hop, not KB selection |
| **MuSiQue** (mo2) | 2 | 70.5% | — | — | — | — | — | +5.4 pp | −14.4 pp | +25.8 pp | ❌ KB/pool=71%; same as above |

---

## Per-Dataset Deep Dive

### ✅ FEVER (mo1) — QDC工作最好的场景

**配置**：pool=21,090, KB=3,100 (14.7%), sudden

```
Static   H1/H2  87.1 / 49.1%
QDC      H1/H2  86.2 / 70.5%   (+21.4 pp vs Static)
Oracle   H1/H2  89.0 / 86.3%
```

**为什么有效**：
- q/SF = 8.15：每个 evidence page 平均被 8 个 claim 引用，demand 信号极强
- alignment = 0.99：claim 文本几乎直接命名相关实体页
- pool 80% 是 distractor：KB 很小（14.7%），选哪些文档进 KB 非常关键
- QDC H1 稳定（86.2 vs Static 87.1），serve 信号保护高频 H1 文档不被驱逐
- RandomFIFO 在 H1 从 97.1% → 81.5% 自我损伤，QDC 保持 97.1% → 92.3%

**局限**：OnDemandFetch 仍然高出 QDC 11.2pp（81.7 vs 70.5），说明 per-query 即时抓取仍有明显优势

---

### ✅ HotpotQA-comparison (mo1) — QDC有效但Oracle gap大

**配置**：pool=41,796, KB=10,000 (23.9%), sudden

```
Static   H1/H2  56.0 / 6.1%
QDC      H1/H2  56.4 / 29.8%   (+23.7 pp vs Static)
Oracle   H1/H2  84.8 / 87.4%
```

**为什么有效**：A/B/C 三条件都满足（q/SF=1.35, align=0.92, pool有大量distractor）

**局限**：QDC H2=29.8%，Oracle H2=87.4%，仍差 57.6pp。原因是 drift 极剧烈（sudden），
KB 里 H1 文档完全失效后，每窗口最多换入 200 个文档（QD_REPLACE_CAP=200），
追赶速度跟不上查询需求。OnDemandFetch 能按需即时抓取所以 H2=80.8%。

---

### ⚠️ TriviaQA-Wikipedia (mo1) — Cond-C 失效

**配置**：pool=7,911, KB=2,600 (32.9%), sudden

```
Static       H1/H2  61.0 / 5.6%
RandomFIFO   H1/H2  49.5 / 26.5%
QDC          H1/H2  53.4 / 30.2%   (+24.6 pp vs Static, 但仅 +3.7 pp vs RandomFIFO)
Oracle       H1/H2  84.6 / 82.5%
```

**为什么有效看似很好，实际是假象**：
- q/SF=1.36, align=0.88，A/B 都满足，QDC 信号是准的
- 但是 pool 只有 7,911 个文档，每个都是某个 entity 的 Wikipedia 页，即某些 query 的 gold SF
- 所以 RandomFIFO 盲目换入文档时，也有高概率命中某个 query 的 gold doc
- QDC 只比 RandomFIFO 高 3.7pp，说明"选哪个文档"这件事本身已经不稀缺

---

### ❌ SQuAD-v1.1（两个变体）— Cond-B 失效（段落粒度太细）

**配置（probe-kb5000）**：pool=20,958, KB=5,000 (23.9%), sudden

```
Static       H1/H2  73.3 / 8.9%
QDC          H1/H2  67.5 / 21.4%   (+12.5 pp vs Static)
OnDemandFetch H1/H2  69.9 / 64.6%  (-43.2 pp below OnDemand)
Oracle       H1/H2  74.3 / 77.7%
```

**SQuAD 的特殊性**：每个 query 的 gold SF 是一个 paragraph（不是文章），每段平均对应 4–5 个问题（loaders.py 注释）。

**为什么 QDC 弱**：
- alignment 约 0.5：query 问的是段落内容，但 pool 里装的是 paragraph-level 文本，
  query 文本很难精准匹配到正确段落（"What year did X happen?" 对 "...X happened in 1985 when..."）
- 即使 demand 信号有一些，probe 阶段命中的也是话题相关但非 gold 的段落
- KB-kb1500 更极端：KB 只占 7.2%，容量极度受限，但 QDC 因对齐失败也只有 +4.2pp

**核心瓶颈**：SQuAD 的检索失败根源是精确段落匹配，不是 KB 选择问题。OnDemandFetch 
能直接按 query embedding 捞最相似段落，所以到 64.6%；QDC 的离线聚类换库无法解决这个问题。

---

### ❌ 2wiki-comparison (mo1) — Cond-A 失效（低复用）

**配置**：pool=37,214, KB=14,000 (37.6%), sudden

```
Static   H1/H2  77.8 / 3.5%
QDC      H1/H2  77.1 / 12.5%   (+9.0 pp)
Oracle   H2=83.3%
```

**为什么弱**：q/SF=1.02，每篇 SF 文档平均只被 1 个 query 问到。
demand 计数器几乎没有累积，任何文档的 demand 都不会超过 serve，换入逻辑从不触发。

---

### ❌ 2wiki-bridge-comparison (mo1) — Cond-B 失效（桥接对齐）

**配置**：pool=32,877, KB=16,000 (48.7%), sudden

```
Static   H1/H2  36.2 / 10.3%
QDC      H1/H2  36.0 / 12.1%   (+1.8 pp，几乎没有)
Oracle   H2=37.3%
```

**为什么弱**：alignment=0.31（NOTES.md 的 Failure Example）。
这类题的问法是"A is related to B; what is C's attribute?" —— query 文本提到的是 hop-1 实体，
但 hop-2 SF 文档写的是 hop-2 实体，两者 embedding 距离很远。
Pool probe 完全探测不到正确的 hop-2 候选，demand 根本积累不到目标文档上。

---

### ❌ HotpotQA-bridge / 2WikiMultihopQA / MuSiQue (mo2) — KB 充裕 + 多跳检索上限低

**配置**：KB/pool 比 54–72%（有意设计为覆盖所有 H1 文档），sudden，BGE-large 检索

```
                 Static H1/H2    QDC H1/H2     Oracle H2   QDC−Static
HotpotQA-bridge  76.1/23.9%     76.0/27.1%    83.1%        +3.2 pp
2WikiMultihopQA  66.9/27.9%     66.8/30.8%    71.8%        +2.9 pp
MuSiQue          49.6/25.3%     49.6/30.7%    56.5%        +5.4 pp
```

**为什么 QDC 基本无效**，两个相互独立的原因：

**原因1：KB 置换空间不足**。KB 已经装了 pool 的 54–72%，post-drift 在 KB 外的文档只剩
28–46%。即使 QDC 全部换对，Coverage 最多提升 3–6pp，直接导致 R@5 提升微乎其微。
这是实验设计的副作用：mo2 的 KB budget = head_ctx × 1.2，保证 H1 不受容量瓶颈，
但因此 H2 的换入空间消失了。

**原因2：多跳检索效率天花板**。单跳时 Coverage ≈ Recall（效率 ~0.85–0.99）；
桥接多跳需要两篇文档链式推理，任意一篇缺失 Recall=0。
- Oracle H1 检索效率（R@5/Coverage）：HotpotQA-bridge 84%，2wiki 69%，MuSiQue 52%
- 即使 KB 里有所有文档（Oracle），MuSiQue 仍有 48% 的查询检索失败
- 换言之，改善 KB 组成对 R@5 的边际价值已经很低，瓶颈在图检索/链式推理本身

---

## 结论：QDC 有效的充要图

```
                    KB/pool < ~35%?
                    ┌────YES────┐  ────NO────→ ❌ 换入空间不足 (mo2)
                    ↓
              q/SF ≥ 1.3?
         ┌───YES───┐ ───NO───→ ❌ demand 无法累积 (2wiki-comp)
         ↓
  query↔SF align ≥ 0.6?
  ┌──YES──┐ ──NO──→ ❌ probe 打偏 (2wiki-bridge-comp, SQuAD)
  ↓
pool 有足够 distractor?
┌─YES─┐ ─NO──→ ⚠️ 优于 Static 但不优于 RandomFIFO (TriviaQA)
↓
✅ QDC 明显有效 (FEVER, HotpotQA-comp)
```

**附注：OnDemandFetch 始终最强**。无论 QDC 是否有效，OnDemandFetch（per-query 即时从 pool 抓取最相似文档）在所有数据集上都显著领先。这说明 demand-aware 离线聚类换库的上限，始终低于 per-query 在线检索。QDC 的意义在于**无需额外检索成本**的批量更新，而非超越 OnDemandFetch。
