# Motivation 1 — Knowledge-Drift Coverage Experiment

## Scripts

| Script | Role |
|---|---|
| `run.py` | Run experiments → `data/results_*.json` |
| `config.py` | Dataset configs (kb_head_mult, n_source, …) |
| `loaders.py` | Dataset loaders incl. FEVER |
| `strategies.py` | All 8 strategy implementations |
| `utils.py` | Embedding cache, retrieval helpers |
| `plot_mo1_combined.py` | **All paper figures** → `mo1_combined`, `dataset_delta_bar`, `dataset_condition_space` + `dataset_analysis_table.tex` |
| `gen_mo1_table.py` | **Method table** → `mo1_method_comparison_table.tex` |

## Figures (paper-ready)

- `figures/mo1_combined.pdf` — 2×2 main figure (HotpotQA + FEVER × sudden/gradual, 8 methods)
- `figures/dataset_delta_bar.pdf` — QDC Δ over Static per dataset
- `figures/dataset_condition_space.pdf` — q/SF vs alignment scatter (success/fail)

## Key results

| Dataset | Drift | QDC vs Static | QDC vs BestOther |
|---|---|---|---|
| HotpotQA-comp | sudden | +25.6pp | +11.8pp |
| HotpotQA-comp | gradual | +13.8pp | +11.5pp |
| FEVER | sudden | +16.6pp | +13.6pp |
| FEVER | gradual | +2.7pp | +5.1pp |

See `QDC_CONDITIONS.txt` for the three conditions under which QDC has advantage.
