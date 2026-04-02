# cyclic_return — Experiment Report

Generated: 2026-03-26 21:47:50

Settings: Pool Size = 5000, KB Budget = 200

## Retrieval Metrics

| Method | Recall@10 | Prec@10 | MRR | Gold_KB | Turnover | Updates | Time |
|--------|-----------|---------|-----|---------|----------|---------|------|
| QARC | 0.2400 | 0.0240 | 0.1822 | 0.2800 | 0.0020 | 3 | 129.0s |
| ComRAG | 0.2000 | 0.0245 | 0.1733 | 0.2700 | 0.0085 | 53 | 115.8s |
| ERASE | 0.1800 | 0.0180 | 0.1237 | 0.2600 | 0.0061 | 29 | 126.6s |
| Static | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 119.4s |
| Random | 0.0100 | 0.0010 | 0.0050 | 0.0100 | 0.0100 | 1 | 138.6s |

## Generation Metrics (LLM End-to-End)

| Method | EM | Token F1 | Recall@10 | Updates |
|--------|-----|----------|-----------|---------|
| QARC | 0.0000 | 0.1355 | 0.2400 | 3 |
| ComRAG | 0.0000 | 0.1393 | 0.2000 | 53 |
| ERASE | 0.0000 | 0.1220 | 0.1800 | 29 |
| Static | 0.0000 | 0.0929 | 0.0000 | 0 |
| Random | 0.0000 | 0.0958 | 0.0100 | 1 |

## Ranking (by Recall@10)

1. **QARC** — recall=0.2400 ⭐
2. **ComRAG** — recall=0.2000
3. **ERASE** — recall=0.1800
4. **Random** — recall=0.0100
5. **Static** — recall=0.0000

