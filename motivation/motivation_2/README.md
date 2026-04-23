# Motivation 2 — Multi-hop Drift at 100x50

Goal: diagnose the remaining gap after Motivation 1 already shows that immediate query-demand signal helps in simple direct-evidence regimes.

The right reading of mo2 is not "QDC is useless", but rather:

- the remaining failure mode is mainly complex bridge / bundle acquisition under low cross-query reuse
- the missing mechanism is bridge-aware persistent admission, not raw retrievability

## Artifact policy
- Active data:
  `data/results_100w_unified_sudden.json`,
  `data/results_100w_unified_gradual.json`
- Active figures:
  `figures/coverage_drift_100w_sudden.{pdf,png}`,
  `figures/coverage_drift_100w_gradual.{pdf,png}`
- The sudden and gradual figures are regenerated from their matching JSON files, and the plot title now names the drift mode explicitly.

## Scale
| Dataset | Pool | KB sudden | KB gradual | Stream |
|---|---:|---:|---:|---|
| HotpotQA-distractor | 54,862 | 28,300 | 32,650 | 100x50 |
| MuSiQue | 43,017 | 24,400 | 27,900 | 100x50 |
| 2WikiMultihopQA | 29,253 | 12,700 | 14,400 | 100x50 |

## 100w H2 results — Recall@5

### Sudden drift
| Dataset | Static | RandFIFO | DocArr | KEdit | LogDriven | **QueryDrivenCluster** | OnDemandFetch | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA      | 12.9 | 15.3 | 12.8 | 22.7 | 19.6 | 18.5 | 56.6 | 65.8 |
| MuSiQue       | 15.4 | 16.4 | 15.4 | 20.2 | 17.4 | **22.7** | 42.1 | 47.9 |
| 2WikiMultihop |  8.5 | 14.1 |  8.5 | 17.6 | 16.2 | 14.8 | 64.0 | 70.7 |

### Gradual drift
| Dataset | Static | RandFIFO | DocArr | KEdit | LogDriven | **QueryDrivenCluster** | OnDemandFetch | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA      | 25.8 | 27.7 | 25.7 | 33.3 | 32.1 | 29.4 | 57.1 | 65.6 |
| MuSiQue       | 23.9 | 24.6 | 23.9 | 26.5 | 26.1 | **28.2** | 41.6 | 46.0 |
| 2WikiMultihop | 17.6 | 22.1 | 17.6 | 23.4 | 23.7 | 22.9 | 61.5 | 66.3 |

## How to read mo2
- Read mo2 after mo1's `100x50` single-hop results. HotpotQA-comparison and sudden scale-matched SQuAD already show that the demand signal itself is real.
- The main remaining gap in mo2 is therefore a bridge / bundle gap: a failed multi-hop query rarely tells the persistent writer which reusable bridge documents should be admitted for future queries.
- MuSiQue is the only clean QueryDrivenCluster win at `100x50`: 22.7 sudden and 28.2 gradual, above the stronger lagged baselines.
- HotpotQA and 2WikiMultihopQA still favor KnowledgeEdit or LogDrivenArrival, so the missing capability is not generic demand collection but bridge-aware persistent acquisition.
- OnDemandFetch stays far above all persistent writers. The evidence is usually retrievable; the system just does not convert those bridge discoveries into the persistent KB.
- Concrete algorithm directions stay in `../DESIGN_DIRECTIONS.md`, not in the main README / tex narrative.

## Reproduce
```bash
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python
cd /home/jyliu/RAG-project/motivation/motivation_2

$PY run.py --n-windows 100 --window-size 50 --drift sudden  --expanded    --n-source 6000 --datasets hotpotqa 2wikimultihopqa musique    --output results_100w_unified_sudden.json

$PY run.py --n-windows 100 --window-size 50 --drift gradual --expanded    --n-source 6000 --datasets hotpotqa 2wikimultihopqa musique    --output results_100w_unified_gradual.json
```
