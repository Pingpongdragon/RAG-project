# sudden_shift — Experiment Report

Generated: 2026-03-26 21:37:08

Settings: Pool Size = 5000, KB Budget = 200

## Retrieval Metrics

| Method | Recall@10 | Prec@10 | MRR | Gold_KB | Turnover | Updates | Time |
|--------|-----------|---------|-----|---------|----------|---------|------|
| QARC | 0.1300 | 0.0130 | 0.0951 | 0.1600 | 0.0024 | 4 | 136.7s |
| ComRAG | 0.1500 | 0.0162 | 0.1292 | 0.2600 | 0.0103 | 68 | 125.6s |
| ERASE | 0.1300 | 0.0130 | 0.0953 | 0.3000 | 0.0103 | 45 | 130.5s |
| Static | 0.0200 | 0.0020 | 0.0027 | 0.0300 | 0.0000 | 0 | 114.4s |
| Random | 0.0100 | 0.0010 | 0.0020 | 0.0200 | 0.0100 | 1 | 127.9s |

## Generation Metrics (LLM End-to-End)

| Method | EM | Token F1 | Recall@10 | Updates |
|--------|-----|----------|-----------|---------|
| QARC | 0.0000 | 0.1232 | 0.1300 | 4 |
| ComRAG | 0.0000 | 0.1246 | 0.1500 | 68 |
| ERASE | 0.0000 | 0.1296 | 0.1300 | 45 |
| Static | 0.0000 | 0.1019 | 0.0200 | 0 |
| Random | 0.0000 | 0.0959 | 0.0100 | 1 |

## Ranking (by Recall@10)

1. **ComRAG** — recall=0.1500 ⭐
2. **QARC** — recall=0.1300
3. **ERASE** — recall=0.1300
4. **Static** — recall=0.0200
5. **Random** — recall=0.0100

