# Motivation 1 — Real Temporal Visible Drift and Recency Boundary

This directory is no longer the main proof that hidden evidence must be solved.
Its role in the current paper logic is narrower and cleaner:

```text
S1: real temporal visible evidence drift.
```

StreamingQA/TREC-COVID style streams test whether a hot RAG cache can follow a
real time-ordered evidence distribution. Because the required evidence is
query-visible and temporally local, LRU/FIFO can be very strong. That is the
expected boundary condition, not a failure of the setup.

Read this layer like the first experimental layer in a LogicRAG-style paper:

1. establish a real-world failure/boundary of vanilla RAG cache management;
2. show which simple baseline is already strong;
3. define the cost-aware objective for the next layer.

The relevant new method is `CostAwareDRIP`, not the old hidden-support branch.
Compare it on quality-cost tradeoff:

```text
has_answer_h2 / support_coverage_h2 / recall@5_h2
vs.
update_cost / evictions / churn_rate_mean / maint_retrieval_cost
```

Recommended main command:

```bash
cd /data/jyliu/RAG-project
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python experiments/direct/run.py \
  --datasets streamingqa_temporal \
  --drift temporal \
  --n-windows 50 \
  --window-size 100 \
  --strategies ARC LRU FIFO DRIP-QueryVisible CostAwareDRIP CostAwareDRIP-NoDrift OnDemandFetch Oracle \
  --output costaware_streamingqa_temporal_50w100.json
```

## Historical Notes: DRIP-Dense (QDC) Conditions for Effectiveness

QDC shows a **clear advantage over all offline baselines** when all three conditions below hold simultaneously.

These conditions are the real axis of the paper. The distinction is **not** simply single-hop vs multi-hop: a multi-hop dataset can work if the signal points to reusable evidence, and a single-hop dataset can fail or become weak if the pool has no distractors or demand reuse is too low.

---

## Condition A — Demand signal exists (q/SF ≥ 1.3)

Multiple queries must share the same supporting-fact (SF) documents.
QDC accumulates `demand[d]` via an exponentially-decayed similarity sum across failed queries probing the pool.
If each SF is referenced by ≤1 query on average, `demand` never exceeds `serve` for any KB doc, so no replacement fires.

**Threshold:** q/SF ≥ 1.3 in the post-subsample query stream.

---

## Condition B — Query text aligns with SF documents (alignment ≥ 0.6)

QDC probes the pool by embedding the query and finding pool docs with cosine sim > SF_HIT_THRESH (0.55).
The query text must **directly name or strongly imply** the SF entity.
If a query asks about a movie but the SF is the director's biography page, the embedding signal probes irrelevant docs.

**Failure example:** 2wiki-bridge-comparison — multi-hop bridge queries where the hop-2 entity never appears in the query text. Alignment = 0.31.

---

## Condition C — Pool contains many distractor documents

QDC's value is **identifying which pool docs are relevant amid noise**.
If nearly every pool document happens to be a gold SF for some query, then blind baselines (RandomFIFO) also accidentally pick useful docs, collapsing the QDC-vs-BestOther gap.

**Failure example:** TriviaQA-Wikipedia — every pool doc is an entity Wikipedia page (= a gold SF for some query). RandomFIFO randomly picks "useful" docs. QDC vs Static = +30.1 pp, but QDC vs BestOther = only +3.3 pp.

---

## Dataset summary

| Dataset | q/SF | Align | Cond-A | Cond-B | Cond-C | Sudden Δ | Gradual Δ |
|---|---:|---:|:---:|:---:|:---:|---:|---:|
| HotpotQA-comparison | 1.35 | 0.92 | ✓ | ✓ | ✓ | **+25.6 pp** | **+13.8 pp** |
| FEVER | 8.15 | 0.99 | ✓ | ✓ | ✓ | **+16.6 pp** | +2.7 pp |
| TriviaQA-Wikipedia | 1.36 | 0.88 | ✓ | ✓ | ✗ | +30.1 pp* | +3.7 pp* |
| 2wiki-bridge-comp | 1.66 | 0.31 | ✓ | ✗ | ✓ | −1.2 pp | −0.1 pp |
| 2wiki-comparison | 1.02 | 0.85 | ✗ | ✓ | ✓ | +9.3 pp | +7.3 pp |
| 2wiki-simple (mixed) | 1.19 | 0.60 | ✗ | ~ | ✓ | +3.7 pp | n/a |

\* TriviaQA: QDC vs **Static** looks good, but vs **BestOther** (RandomFIFO) only +3.3 pp because Cond-C fails.

---

## Why do some strategies degrade in H1 (before drift)?

> **Teacher question:** "Before drift, existing strategies should not update — why do RandomFIFO and KnowledgeEdit get worse even in H1?"

**Short answer:** They update *every* window by design. The H1 degradation is not a bug — it is the finding.

| Strategy | H1 design | H1 behavior |
|---|---|---|
| Static | Never updates | Stable (reflects natural per-window query variance) |
| **RandomFIFO** | Blindly replaces 0.4% of KB each window | Degrades: evicts useful H1 docs, injects distractors |
| KnowledgeEdit | Replaces docs whose embeddings shift | Over-reacts even when shift is noise |
| **QDC** | Replaces only when `demand[c] > serve[e] + demand[e]` | Stable: in H1, existing KB docs have high `serve`, so replacements rarely fire |

In FEVER sudden, RandomFIFO drops from W1=97.1% → W50=81.5% over 50 windows **purely because it keeps swapping good H1 docs out with distractors** (pool is 80% distractors). QDC maintains W1=97.1% → W50=92.3% because its serve signal protects docs that are being successfully queried.

**This H1 degradation is part of the paper's argument:** a KB maintenance system without demand-awareness is harmful even in the stable pre-drift phase, not just after drift.

---

## Why does sudden drift occur at window 50?

Experimental design: 100 windows total, split 50/50. H1 = first 50 windows (original topic), sudden drift = all queries switch to H2 topic at window 51. The 50/50 split gives equal evaluation time to pre- and post-drift behavior.

---

## QDC mechanism

Each window:
1. Embed window queries → probe pool → accumulate `demand[d]` (`DEMAND_DECAY = 0.85`)
2. Track `serve[d]` for KB docs that produce successful hits (`SERVE_DECAY = 0.92`)
3. Replace KB doc `e` with pool candidate `c` iff `demand[c] > serve[e] + demand[e]`
   — up to `QD_REPLACE_CAP = 200` swaps per window

---

## RandomFIFO fairness fix (2026-05-05)

`FIFO_BATCH` changed from a fixed 40 to **0.4% of |KB| per window**, so the
turnover rate is comparable across datasets with different KB sizes.

- HotpotQA (KB = 10 000): 0.4% × 10 000 = **40** — identical to before
- FEVER     (KB = 3 100):  0.4% × 3 100  = **12** — prevents H1 self-harm
