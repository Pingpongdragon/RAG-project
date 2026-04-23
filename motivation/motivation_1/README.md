# Motivation 1 - Single-Hop Drift at 100x50

This directory now keeps the main simple-case figure pair only.

- HotpotQA-comparison is the primary single-hop positive case and the only live figure pair.
- Scale-matched SQuAD remains a supplementary stress check in JSON/table form only.

## Artifact policy
- Active data:
  `data/results_probe_signal_hotpot_100w_sudden.json`,
  `data/results_probe_signal_hotpot_100w_gradual.json`,
  `data/results_probe_squad_ns30000_kb5000_100w_sudden.json`,
  `data/results_probe_squad_ns30000_kb5000_100w_gradual.json`
- Active figures:
  `figures/coverage_100w_sudden.{pdf,png}`,
  `figures/coverage_100w_gradual.{pdf,png}`

## Main simple-case setup
- HotpotQA-comparison: `n_source=6000`, realized pool = 41,796 paragraphs, KB = 8,000, stream = `100x50`
- H1/H2 use the same head-to-tail query-drift construction as before.
- Main question: does immediate failing-query demand signal help a persistent writer in a simple direct-evidence regime under both sudden and gradual drift?

## Main figure numbers - HotpotQA-comparison

### Sudden drift
| Strategy | Cov H1 / H2 | R@5 H2 |
|---|---:|---:|
| Static | 55.9 / 5.0 | 4.5 |
| RandomFIFO | 51.7 / 10.2 | 9.6 |
| DocArrival | 54.8 / 5.1 | 4.6 |
| KnowledgeEdit | 47.7 / 8.8 | 8.0 |
| LogDrivenArrival | 50.7 / 18.2 | 16.5 |
| **QueryDrivenCluster** | **56.3 / 30.5** | **28.3** |
| OnDemandFetch | 88.1 / 95.3 | 81.7 |
| Oracle | 100.0 / 100.0 | 88.3 |

### Gradual drift
| Strategy | Cov H1 / H2 | R@5 H2 |
|---|---:|---:|
| Static | 47.1 / 22.2 | 18.7 |
| RandomFIFO | 44.2 / 22.6 | 19.8 |
| DocArrival | 46.3 / 21.9 | 18.5 |
| KnowledgeEdit | 41.9 / 22.6 | 19.7 |
| LogDrivenArrival | 43.7 / 27.0 | 22.9 |
| **QueryDrivenCluster** | **48.1 / 34.8** | **31.0** |
| OnDemandFetch | 86.3 / 91.5 | 77.2 |
| Oracle | 100.0 / 100.0 | 86.7 |

## Supplementary scale-matched SQuAD stress check
- Target `n_source=30000`, realized pool = 20,958 paragraphs, KB = 5,000, stream = `100x50`.
- Sudden remains positive for QueryDrivenCluster: H2 Coverage / Recall@5 = `27.2 / 21.4`, above LogDrivenArrival `18.3 / 13.4` and RandomFIFO `19.0 / 14.5`.
- The long gradual SQuAD run is not the main figure because it mixes the demand-signal question with a separate capacity-allocation / eviction effect. Static and DocArrival preserve more H2 mass (`54.0 / 40.8`, `50.3 / 38.3`), while QueryDrivenCluster still stays above LogDrivenArrival (`33.4 / 25.9` vs `30.4 / 21.8`).

## How to read mo1
- The clean claim is now simple and stable: in the direct-evidence HotpotQA-comparison regime, QueryDrivenCluster is the strongest persistent writer under both drift modes: `30.5 / 28.3` sudden and `34.8 / 31.0` gradual.
- This is the single-hop evidence carried into the main paper narrative and the live figure pair.
- SQuAD remains useful as a stress check, but its long gradual run is not the main proof point for the simple-case claim.
- OnDemandFetch stays the non-persistent ceiling.

## Reproduction
```bash
cd /home/jyliu/RAG-project/motivation/motivation_1
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python

$PY run.py --datasets hotpotqa_comparison --n-windows 100 --window-size 50 --drift sudden  --n-source 6000  --kb-budget 8000 --output results_probe_signal_hotpot_100w_sudden.json

$PY run.py --datasets hotpotqa_comparison --n-windows 100 --window-size 50 --drift gradual --n-source 6000  --kb-budget 8000 --output results_probe_signal_hotpot_100w_gradual.json

$PY run.py --datasets squad --n-windows 100 --window-size 50 --drift sudden  --n-source 30000 --kb-budget 5000 --output results_probe_squad_ns30000_kb5000_100w_sudden.json

$PY run.py --datasets squad --n-windows 100 --window-size 50 --drift gradual --n-source 30000 --kb-budget 5000 --output results_probe_squad_ns30000_kb5000_100w_gradual.json
```

## Regenerate main figures
```bash
cd /home/jyliu/RAG-project/motivation/motivation_1
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python

$PY - <<'PY'
import json, sys
from pathlib import Path
base = Path('/home/jyliu/RAG-project/motivation/motivation_1')
sys.path.insert(0, str(base))
import run
for rel, suffix in [
    ('data/results_probe_signal_hotpot_100w_sudden.json', '_100w_sudden'),
    ('data/results_probe_signal_hotpot_100w_gradual.json', '_100w_gradual'),
]:
    data = json.loads((base / rel).read_text())
    run.generate_figures(data, run.STRATEGY_ORDER, suffix=suffix)
PY
```
