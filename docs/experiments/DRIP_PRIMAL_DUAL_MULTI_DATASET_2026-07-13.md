# DRIP primal-dual updater: multi-dataset audit (2026-07-13)

> **Superseded operating point.** This document records the initial strict
> updater ablation at `b_rep=0.25`.  The later global validation sweep selected
> `b_rep=0.50`, followed by held-out seeds and corrected span sampling.  Use
> `DRIP_V3_PROTOCOL_AND_RESULTS_2026-07-13.md` for current claims and commands.

## 1. What changed

This is a strict updater ablation. Both methods use the same query stream,
causal-prefix initialization, embeddings, evidence credit, MEF/HSU ledgers,
detector state, victim selection, write cap, and near-duplicate rule.

- `DRIP-DirectFull`: fixed replacement cost plus EMA churn pressure.
- `DRIP-PrimalDual`: online dual price for a long-run replacement constraint.

The comparison therefore answers one narrow question: does replacing the
hand-written churn penalty with a constrained online updater improve the
answerability/replacement trade-off?

## 2. Academic notation for the ledgers

The implementation keeps the short variable names `demand`, `serve`, and
`priority`, but the paper can define them more precisely.

### Miss-Conditioned Evidence Frequency (MEF)

For under-covered queries `U_t`, candidate evidence is accumulated as

```text
D_t(d) = beta_D,t(d) D_{t-1}(d) + sum_{q in U_t} E_t(q,d).
```

`D_t(d)` is a decayed weighted frequency, not a calibrated future-request
probability. Unlike ARC's cumulative Distance-Rank Frequency, MEF is updated
only after a hot-tier support failure and may selectively forget evidence that
does not match the current support working set.

### Hit-Conditioned Service Utility (HSU)

Let `H_t(q)` be the resident documents that actually served query `q`. One unit
of service credit is normalized across that set:

```text
A_t(d) = sum_{q in Q_t} I[d in H_t(q)] / max(1, |H_t(q)|)
S_t(d) = beta_S,t(d) S_{t-1}(d) + A_t(d).
```

HSU is measured online service attribution. It should not be renamed
"hubness": hubness is a structural centrality statistic, whereas HSU records
which resident evidence actually served the observed workload.

### Composite Residency Utility (CRU)

```text
P_t(v) = D_t(v) + S_t(v).
```

Admission is intentionally asymmetric: a non-resident candidate is valued by
its MEF, while evicting a resident pays its full CRU. The names make the
statistical roles clearer, but the novelty claim must rest on this mechanism and
the constrained updater, not on new acronyms alone.

## 3. Primal-dual replacement controller

Let `R_t` be replacements in window `t`, `B_write,t` the executable write cap,
and `b_rep` a deployment-level target load. The constrained objective is

```text
limsup_T (1/T) sum_t R_t / max(1, B_write,t) <= b_rep.
```

The online shadow price is

```text
lambda_{t+1}
= [lambda_t + eta_t (R_t / max(1, B_write,t) - b_rep)]_+

eta_t = 1 / sqrt(t + 1).
```

Candidate `c` replaces the weakest eligible resident `v` only when

```text
Delta_t(c,v) = D_t(c) - m P_t(v) - lambda_t > 0.
```

The current experiment fixes `b_rep=0.25` globally. It is not tuned separately
for each dataset.

## 4. Results

The exact 10% cache-ratio table and every source JSON are in:

```text
docs/experiments/tables/updater_ablation_cross_dataset_10pct.md
```

| Workload | Fixed HasAns / Repl. | Dual HasAns / Repl. | HasAns delta | Repl. reduction |
|---|---:|---:|---:|---:|
| StreamingQA official | 7.6 / 10,000 | 7.6 / 3,746 | -0.04 pp | 62.5% |
| HotpotQA comparison | 7.5 / 2,941 | 6.9 / 1,721 | -0.6 pp | 41.5% |
| 2Wiki comparison | 1.3 / 3,338 | 1.7 / 1,580 | +0.4 pp | 52.7% |
| Mind2Web controlled | 52.7 / 974 | 52.1 / 1,065 | -0.6 pp | -9.3% |
| MIND chronological | 95.1 / 3,631 | 93.3 / 1,771 | -1.8 pp | 51.2% |

Current cross-dataset evidence supports a cost-quality trade-off, not universal
dominance. The primal-dual updater substantially reduces replacements on
StreamingQA, HotpotQA, 2Wiki, and MIND; 2Wiki also gains answerability. It loses
some answerability on most other workloads, and on Mind2Web it makes slightly
more replacements. Therefore `b_rep=0.25` is a conservative operating point,
not a dataset-independent optimum.

StreamingQA official contains 25,000 strictly chronological evaluation queries,
20,554 unique supports, and only 17.784% queries answerable from previously seen
support. Its low absolute Has-Answer is therefore primarily a workload
cacheability boundary. The useful updater result is that the dual price preserves
essentially the same answerability while eliminating 62.5% of replacements.

MIND also exposes capacity sensitivity:

| Ratio | Fixed HasAns / Repl. | Dual HasAns / Repl. | HasAns delta | Repl. reduction |
|---:|---:|---:|---:|---:|
| 1% | 90.9 / 7,454 | 90.3 / 2,550 | -0.6 pp | 65.8% |
| 10% | 95.1 / 3,631 | 93.3 / 1,771 | -1.8 pp | 51.2% |

The paper should report a small global Pareto sweep over `b_rep` rather than
selecting a different value per dataset. A defensible next sweep is
`b_rep in {0.10, 0.25, 0.50, 0.75}` on a validation stream, followed by one
frozen value on all held-out workloads.

## 5. Reproduction commands

Run from the repository root:

```bash
cd /data/jyliu/RAG-project
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python

$PY motivation/run_cache_ratio_sweep.py streamingqa_official \
  --ratios 0.10 --strategies DRIP-DirectFull DRIP-PrimalDual \
  --drip-ablation --output-prefix updater_ablation --gpu 0
$PY motivation/run_cache_ratio_sweep.py hotpotqa_comparison \
  --ratios 0.10 --strategies DRIP-DirectFull DRIP-PrimalDual \
  --drip-ablation --output-prefix updater_ablation --gpu 1
$PY motivation/run_cache_ratio_sweep.py 2wiki_comparison \
  --ratios 0.10 --strategies DRIP-DirectFull DRIP-PrimalDual \
  --drip-ablation --output-prefix updater_ablation --gpu 2
$PY motivation/run_cache_ratio_sweep.py mind2web_agent \
  --ratios 0.10 --strategies DRIP-DirectFull DRIP-PrimalDual \
  --drip-ablation --output-prefix updater_ablation --gpu 3
$PY motivation/run_cache_ratio_sweep.py mind_news_access \
  --ratios 0.10 --strategies DRIP-DirectFull DRIP-PrimalDual \
  --drip-ablation --output-prefix updater_ablation --gpu 0
```

Raw JSON files are written to:

```text
experiments/direct/data/updater_ablation_<dataset>_r1000.json
experiments/hidden/data/updater_ablation_<dataset>_r1000.json
```

The wrapper prints the absolute `RESULT` path before each run and `DONE` after
successful completion. It skips a file only when the JSON contains the requested
dataset, ratio, causal protocol, and both requested strategies.
