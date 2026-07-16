# MIND natural exact-residency result (2026-07-16)

## Protocol

- MIND-small positive clicks in official `event_ts` order.
- 1,500-event chronological warm-up and 25,000-event evaluation.
- 500 events per window; 51,282 news objects.
- Calibration-only cache sizing gives capacity 140.
- The official impression slate is observable current-query context; the
  clicked news ID is revealed only after hit scoring.
- Candidate credit excludes the current clicked support to avoid double
  rewarding gold evidence.

## Write-budget sweep

Entries are exact click hit percentage / physical cache writes.

| Cap | LRU | TinyLFU | DRIP-Reactive | DRIP-DomainAdapt |
|---:|---:|---:|---:|---:|
| 1 | 15.824 / 50 | 15.824 / 50 | **37.884 / 50** | 37.316 / 50 |
| 2 | 32.144 / 100 | 32.144 / 100 | **50.552 / 100** | 49.616 / 100 |
| 3 | 39.144 / 150 | 39.112 / 150 | **57.664 / 131** | 57.168 / 131 |
| 4 | 48.708 / 200 | 43.528 / 200 | **53.372 / 128** | 52.640 / 128 |
| 5 | **54.988 / 250** | 48.836 / 250 | 49.052 / 125 | 48.816 / 125 |
| 6 | **59.604 / 300** | 47.244 / 300 | 45.604 / 122 | 45.708 / 122 |
| 7 | **62.084 / 350** | 48.124 / 350 | 45.160 / 119 | 45.092 / 119 |
| 8 | **64.560 / 400** | 49.392 / 400 | 44.132 / 116 | 44.004 / 116 |
| 9 | **66.212 / 450** | 48.700 / 450 | 42.960 / 113 | 42.936 / 113 |

## Interpretation

- The Pareto point at cap 3 is strong: DomainAdapt reaches 57.168% with 131
  writes, while LRU needs 250 writes to reach 54.988% and 300 writes to exceed
  it at 59.604%.
- The best document-only Reactive point is 0.496 percentage points higher than
  DomainAdapt at the same writes. Thus MIND validates switching-cost-aware
  placement but not extra impression-slate candidate credit.
- This is an intended natural-time appendix result. It must not be used to
  claim that every candidate source improves placement.

Reproduction:

```bash
python experiments/agent/run_mind_exact_trace.py \
  --write-budgets 1 2 3 4 5 6 7 8 9 \
  --output experiments/agent/data/domain_adapt_mind_exact_natural.json
```

