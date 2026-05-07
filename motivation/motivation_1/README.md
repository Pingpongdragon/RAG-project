# Motivation 1 — Knowledge-Drift Coverage Experiment

## Scripts

| Script | Role |
|---|---|
| `run.py` | Run experiments → `data/results_*.json` |
| `config.py` | Dataset configs (kb_head_mult, n_source, …) |
| `loaders.py` | Dataset loaders incl. FEVER |
| `strategies.py` | All 8 strategy implementations |
| `utils.py` | Embedding cache, retrieval helpers |
| `plot.py` | **All paper figures** → `mo1_{sudden,gradual}.{pdf,png}` + appendix variants |

## Figures (paper-ready)

- `figures/mo1_sudden.pdf` — 2×2 main figure (HotpotQA + FEVER, sudden drift, 6 methods)
- `figures/mo1_gradual.pdf` — 2×2 main figure (HotpotQA + FEVER, gradual drift, 6 methods)
- `figures/mo1_sudden_appendix.pdf` — full 8-method version (sudden)
- `figures/mo1_gradual_appendix.pdf` — full 8-method version (gradual)

## Key results

| Dataset | Drift | QDC vs Static |
|---|---|---|
| HotpotQA-comp | sudden | +25.6 pp |
| HotpotQA-comp | gradual | +13.8 pp |
| FEVER | sudden | +16.6 pp |
| FEVER | gradual | +2.7 pp |

See `NOTES.md` for the three conditions under which QDC has advantage.
