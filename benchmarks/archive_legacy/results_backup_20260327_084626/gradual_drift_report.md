# gradual_drift — Experiment Report

Generated: 2026-03-26 21:26:27

Settings: Pool Size = 5000, KB Budget = 200

## Retrieval Metrics

| Method | Recall@10 | Prec@10 | MRR | Gold_KB | Turnover | Updates | Time |
|--------|-----------|---------|-----|---------|----------|---------|------|
| QARC | 0.2300 | 0.0230 | 0.1504 | 0.2900 | 0.0027 | 4 | 135.6s |
| ComRAG | 0.1800 | 0.0199 | 0.1683 | 0.3400 | 0.0094 | 67 | 129.7s |
| ERASE | 0.2000 | 0.0200 | 0.1467 | 0.3500 | 0.0106 | 46 | 138.4s |
| Static | 0.0200 | 0.0020 | 0.0200 | 0.0200 | 0.0000 | 0 | 125.3s |
| Random | 0.0300 | 0.0030 | 0.0220 | 0.0300 | 0.0100 | 1 | 127.4s |

## Generation Metrics (LLM End-to-End)

| Method | EM | Token F1 | Recall@10 | Updates |
|--------|-----|----------|-----------|---------|
| QARC | 0.0000 | 0.1395 | 0.2300 | 4 |
| ComRAG | 0.0000 | 0.1263 | 0.1800 | 67 |
| ERASE | 0.0000 | 0.1341 | 0.2000 | 46 |
| Static | 0.0000 | 0.1162 | 0.0200 | 0 |
| Random | 0.0000 | 0.1074 | 0.0300 | 1 |

## Ranking (by Recall@10)

1. **QARC** — recall=0.2300 ⭐
2. **ERASE** — recall=0.2000
3. **ComRAG** — recall=0.1800
4. **Random** — recall=0.0300
5. **Static** — recall=0.0200

