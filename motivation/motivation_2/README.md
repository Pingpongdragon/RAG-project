# Motivation 2 — Agent-Side Acquisition Gap at 100x50

Goal: diagnose what remains after Motivation 1 establishes the three conditions under which query-driven persistent KB maintenance works:

- **Cond-A**: repeated demand exists (`q/SF >= 1.3`)
- **Cond-B**: the query/trace signal aligns with the evidence to be written
- **Cond-C**: the candidate pool contains enough distractors that selection matters

The right reading of mo2 is not "single-hop vs multi-hop" and not "QDC is useless". It is:

- QueryDrivenCluster is the strongest **persistent writer** on all three datasets under both drift modes.
- OnDemandFetch is better understood as an agent-side **dynamic acquisition channel**: it can fetch evidence that is not currently in the persistent KB.
- In an agent system, those fetched documents can become QDC's next candidate pool. The open problem is not raw retrievability, but **persistent admission**: deciding which transient on-demand discoveries should be written into long-lived KB for future reuse.
- Mo2 exposes cases where Cond-B is weak for query-only QDC: a failed query points at the surface task, while the reusable evidence may be bridge/bundle documents surfaced only by on-demand retrieval traces.

## Artifact policy
- Active data:
  `data/results_100w_unified_sudden.json`,
  `data/results_100w_unified_gradual.json`
- Active figures:
  `figures/coverage_drift_100w_sudden.{pdf,png}`,
  `figures/coverage_drift_100w_gradual.{pdf,png}`
- The sudden and gradual figures are regenerated from their matching JSON files, and the plot title now names the drift mode explicitly.

## Scale
| Dataset | Pool | KB | KB% | Stream |
|---|---:|---:|---:|---|
| HotpotQA-distractor | 54,862 | 14,000 | 26% | 100×50 |
| MuSiQue | 43,017 | 11,000 | 26% | 100×50 |
| 2WikiMultihopQA | 29,253 | 7,000 | 24% | 100×50 |

## 100w H2 results — Recall@5

### Sudden drift
| Dataset | Static | RandFIFO | DocArr | KEdit | LogDriven | **QueryDrivenCluster** | OnDemandFetch | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA      |  2.8 |  6.1 |  2.9 |  9.4 |  9.2 | **11.0** | 60.7 | 70.0 |
| MuSiQue       |  4.9 |  7.2 |  5.1 |  7.7 |  6.9 | **17.6** | 43.7 | 52.7 |
| 2WikiMultihop |  1.9 |  8.7 |  2.3 |  8.3 |  8.1 |  **9.5** | 66.1 | 72.4 |

### Gradual drift
| Dataset | Static | RandFIFO | DocArr | KEdit | LogDriven | **QueryDrivenCluster** | OnDemandFetch | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA      | 23.5 | 23.4 | 23.3 | 22.1 | 24.2 | **30.9** | 60.5 | 71.9 |
| MuSiQue       | 17.2 | 17.1 | 17.1 | 14.9 | 15.4 | **24.7** | 42.1 | 50.2 |
| 2WikiMultihop | 18.2 | 17.9 | 18.1 | 15.6 | 15.8 | **24.1** | 55.9 | 61.8 |

## How to read mo2
- Read mo2 through the same three-condition lens as mo1, not through a single-hop/multi-hop split.
- QDC remains the best persistent writer, so the persistent-demand signal is still useful.
- OnDemandFetch is not merely a competing baseline. From an agent perspective, it is the online acquisition path that can populate the candidate pool with documents the static pool/QDC did not already expose.
- The main gap is therefore **fetch-to-write conversion**: after the agent discovers useful evidence on demand, which documents should be admitted into persistent KB for future queries?
- This reframes the missing capability as trace-aware / evidence-aware admission. Query-only QDC uses the query embedding as the demand signal; agent QDC should also use on-demand retrieval traces, clicked/read docs, successful answer contexts, and repeated failures.
- Concrete algorithm directions stay in `../DESIGN_DIRECTIONS.md`, not in the main README / tex narrative.

## Reproduce

```bash
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python
cd /home/jyliu/RAG-project/motivation/motivation_2

# Run each dataset separately with per-dataset KB budget, then merge

# --- Sudden ---
$PY run.py --n-windows 100 --window-size 50 --drift sudden --expanded \
  --n-source 6000 --datasets hotpotqa --kb-budget 14000 \
  --output data/results_100w_hotpot_sudden.json

$PY run.py --n-windows 100 --window-size 50 --drift sudden --expanded \
  --n-source 6000 --datasets 2wikimultihopqa --kb-budget 7000 \
  --output data/results_100w_2wiki_sudden.json

$PY run.py --n-windows 100 --window-size 50 --drift sudden --expanded \
  --n-source 6000 --datasets musique --kb-budget 11000 \
  --output data/results_100w_musique_sudden.json

# Merge
python3 -c "
import json
d = {}
for ds, fn in [('hotpotqa','results_100w_hotpot_sudden.json'),
               ('2wikimultihopqa','results_100w_2wiki_sudden.json'),
               ('musique','results_100w_musique_sudden.json')]:
    with open(f'data/{fn}') as f:
        d.update(json.load(f))
with open('data/results_100w_unified_sudden.json','w') as f:
    json.dump(d, f, indent=2)
"

# --- Gradual (same params, --drift gradual) ---
$PY run.py --n-windows 100 --window-size 50 --drift gradual --expanded \
  --n-source 6000 --datasets hotpotqa --kb-budget 14000 \
  --output data/results_100w_hotpot_gradual.json

$PY run.py --n-windows 100 --window-size 50 --drift gradual --expanded \
  --n-source 6000 --datasets 2wikimultihopqa --kb-budget 7000 \
  --output data/results_100w_2wiki_gradual.json

$PY run.py --n-windows 100 --window-size 50 --drift gradual --expanded \
  --n-source 6000 --datasets musique --kb-budget 11000 \
  --output data/results_100w_musique_gradual.json

# Merge
python3 -c "
import json
d = {}
for ds, fn in [('hotpotqa','results_100w_hotpot_gradual.json'),
               ('2wikimultihopqa','results_100w_2wiki_gradual.json'),
               ('musique','results_100w_musique_gradual.json')]:
    with open(f'data/{fn}') as f:
        d.update(json.load(f))
with open('data/results_100w_unified_gradual.json','w') as f:
    json.dump(d, f, indent=2)
"
```
