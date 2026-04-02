# hotpotqa_walk — Experiment Report

Generated: 2026-03-26 21:59:59

Settings: Pool Size = 888, KB Budget = 200

## Retrieval Metrics

| Method | Recall@10 | Prec@10 | MRR | Gold_KB | Turnover | Updates | Time |
|--------|-----------|---------|-----|---------|----------|---------|------|
| QARC | 0.3600 | 0.0720 | 0.6000 | 0.3800 | 0.0034 | 7 | 139.8s |
| ComRAG | 0.2400 | 0.0534 | 0.3768 | 0.5600 | 0.0124 | 86 | 143.8s |
| ERASE | 0.3100 | 0.0620 | 0.4717 | 0.6250 | 0.0075 | 36 | 150.7s |
| Static | 0.1150 | 0.0230 | 0.2250 | 0.1150 | 0.0000 | 0 | 144.6s |
| Random | 0.2900 | 0.0580 | 0.4417 | 0.2950 | 0.0087 | 1 | 146.7s |

## Generation Metrics (LLM End-to-End)

| Method | EM | Token F1 | Recall@10 | Updates |
|--------|-----|----------|-----------|---------|
| QARC | 0.0000 | 0.0517 | 0.3600 | 7 |
| ComRAG | 0.0000 | 0.0373 | 0.2400 | 86 |
| ERASE | 0.0000 | 0.0436 | 0.3100 | 36 |
| Static | 0.0000 | 0.0338 | 0.1150 | 0 |
| Random | 0.0000 | 0.0412 | 0.2900 | 1 |

## Ranking (by Recall@10)

1. **QARC** — recall=0.3600 ⭐
2. **ERASE** — recall=0.3100
3. **Random** — recall=0.2900
4. **ComRAG** — recall=0.2400
5. **Static** — recall=0.1150

