# Unified baseline matrix (2026-07-15)

Protocol: causal disjoint warm-up, SQuAD 20x25 with 5% hot tier, HotpotQA
comparison 8x25 with 10% hot tier, seeds 42--46, MiniLM serving embeddings,
`DRIP_REPLACEMENT_TARGET=0.25`, and `DRIP_DIRECT_TOPK=4`.

`TotalAccess = AMAT + maintenance logical reads / number of queries`.
Values are mean +/- sample standard deviation across five seeds.

## SQuAD recurring

| Method | HasAns | R@5-H2 | AMAT | Replacements | Maintenance | TotalAccess |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 6.48 +/- 0.54 | 2.32 +/- 0.52 | 10.352 | 455.8 | 479.8 | 11.312 |
| TinyLFU | 23.12 +/- 1.95 | 20.48 +/- 2.18 | 8.688 | 383.4 | 432.0 | 9.552 |
| ARC / AgentRAGCache | 36.40 +/- 2.84 | **37.76 +/- 3.76** | 7.360 | **73.0** | **400.6** | **8.161** |
| Proximity | 25.96 +/- 3.40 | 21.44 +/- 3.76 | 8.404 | 378.0 | 493.6 | 9.391 |
| GPTCache-style | 18.68 +/- 2.70 | 14.72 +/- 3.00 | 9.132 | 414.2 | 1600.0 | 12.332 |
| DRIP w/o MEF | 25.96 +/- 3.40 | 21.44 +/- 3.76 | 8.404 | 0.0 | 1849.6 | 12.103 |
| DRIP fixed experts | **36.84 +/- 1.98** | 37.60 +/- 2.93 | **7.316** | 157.2 | 1779.2 | 10.874 |
| DRIP-Reactive | 36.52 +/- 1.56 | 37.04 +/- 1.93 | 7.348 | 142.4 | 1780.0 | 10.908 |

Paired HasAns difference, DRIP minus ARC: +0.12 points (approximate 95% CI
half-width 2.61). The methods are tied on quality, while ARC uses 49% fewer
replacements and 77% fewer maintenance reads.

## SQuAD shuffled

| Method | HasAns | R@5-H2 | AMAT | Replacements | Maintenance | TotalAccess |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 23.44 +/- 3.54 | 24.08 +/- 9.59 | 8.656 | 392.2 | 433.2 | 9.522 |
| TinyLFU | 26.76 +/- 2.29 | 26.48 +/- 2.54 | 8.324 | 374.6 | 425.0 | 9.174 |
| ARC / AgentRAGCache | **36.52 +/- 3.28** | **38.88 +/- 3.71** | **7.348** | **62.0** | **398.6** | **8.145** |
| Proximity | 26.36 +/- 3.53 | 23.04 +/- 4.90 | 8.364 | 378.2 | 492.0 | 9.348 |
| GPTCache-style | 16.48 +/- 3.32 | 11.20 +/- 3.14 | 9.352 | 417.6 | 1600.0 | 12.552 |
| DRIP w/o MEF | 26.36 +/- 3.53 | 23.04 +/- 4.90 | 8.364 | 0.0 | 1856.0 | 12.076 |
| DRIP fixed experts | 35.40 +/- 3.76 | 36.48 +/- 4.83 | 7.460 | 156.4 | 1784.8 | 11.030 |
| DRIP-Reactive | 35.24 +/- 1.97 | 36.24 +/- 2.66 | 7.476 | 145.4 | 1781.6 | 11.039 |

Paired HasAns difference, DRIP minus ARC: -1.28 points (approximate 95% CI
half-width 2.54). ARC is better on every reported mean metric.

## HotpotQA recurring reusable-evidence slice

| Method | HasAns | R@5-H2 | AMAT | Replacements | Maintenance | TotalAccess |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 14.40 +/- 5.15 | 33.20 +/- 4.47 | 9.560 | 109.0 | 124.6 | 10.183 |
| TinyLFU | 14.40 +/- 5.15 | 33.40 +/- 4.35 | 9.560 | 108.6 | 124.2 | 10.181 |
| ARC / AgentRAGCache | **20.30 +/- 4.51** | **49.00 +/- 3.06** | **8.970** | 1467.6 | **110.2** | **9.521** |
| Proximity | 13.70 +/- 5.39 | 20.00 +/- 6.43 | 9.630 | 148.2 | 199.4 | 10.627 |
| GPTCache-style | 13.60 +/- 5.38 | 19.70 +/- 6.42 | 9.640 | 552.4 | 640.0 | 12.840 |
| DRIP w/o MEF | 13.70 +/- 5.39 | 20.00 +/- 6.43 | 9.630 | **0.0** | 703.2 | 13.146 |
| DRIP fixed experts | 14.70 +/- 5.20 | 33.90 +/- 5.34 | 9.530 | 101.4 | 685.6 | 12.958 |
| DRIP-Reactive | 14.70 +/- 5.20 | 33.90 +/- 5.34 | 9.530 | 101.4 | 685.6 | 12.958 |

DRIP gains only 0.30 HasAns points over LRU/TinyLFU and loses 5.60 points to
ARC. ARC pays roughly 14.5x more writes than DRIP, so HotpotQA exposes a
quality--write trade-off rather than a DRIP win.

## Ablation conclusions

- MEF is necessary on SQuAD: removing it lowers HasAns by 10.56 points on the
  recurring stream and 8.88 points on shuffled. On HotpotQA the gain is only
  1.00 point.
- Online Fixed-Share learning is not supported as a quality contributor. Fixed
  0.5/0.5 experts differ from learned experts by +0.32 points on SQuAD recurring,
  +0.16 on shuffled, and exactly 0 on HotpotQA. Learned weights reduce SQuAD
  replacements by about 9%, but this is not enough to reduce maintenance cost.
- Current DRIP does not beat ARC. On SQuAD it approximately ties ARC in quality
  but is materially more expensive; on shuffled SQuAD and HotpotQA ARC has
  higher quality.

The strict no-MEF ablation intentionally keeps the maintenance probe while
zeroing only MEF credit. Its maintenance number isolates mechanism rather than
representing an optimized no-MEF deployment, which would skip the probe.
