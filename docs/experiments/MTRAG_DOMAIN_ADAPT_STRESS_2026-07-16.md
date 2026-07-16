# MT-RAG low-reuse stress result (2026-07-16)

Official MT-RAG qrel passage IDs are scored before post-service feedback. Turn
order is preserved inside all 110 conversations. The corpus contains 366,479
passages and 777 qrel-bearing queries across four named domains. Capacity is
calibrated from 150 warm-up turns; 627 turns are evaluated with one write per
25-query window. Five scheduler seeds are reported.

## Five-seed means

Percentages are mean +/- sample standard deviation.

| Protocol | Method | Strict all-support | Any-support | Evidence coverage | Writes |
|---|---|---:|---:|---:|---:|
| Session round-robin | LRU/TinyLFU | 0.096 +/- 0.087 | 1.467 +/- 0.612 | 0.536 +/- 0.231 | 26 |
| Session round-robin | DRIP-Reactive | 0.255 +/- 0.143 | 2.711 +/- 0.252 | 1.127 +/- 0.060 | 26 |
| Session round-robin | DRIP-DomainAdapt | 0.255 +/- 0.143 | 2.711 +/- 0.252 | 1.127 +/- 0.060 | 26 |
| Controlled four-domain | LRU/TinyLFU | 0.000 +/- 0.000 | 1.085 +/- 0.484 | 0.387 +/- 0.173 | 26 |
| Controlled four-domain | DRIP-Reactive | 0.319 +/- 0.000 | 2.360 +/- 0.133 | 1.082 +/- 0.108 | 26 |
| Controlled four-domain | DRIP-DomainAdapt | 0.319 +/- 0.000 | 2.360 +/- 0.133 | 1.082 +/- 0.108 | 26 |

## Interpretation

- Absolute hit rates remain extremely low, confirming the intended low-reuse
  boundary rather than a positive main result.
- MT-RAG provides no observable query candidate list in this exact trace.
  DomainAdapt therefore equals Reactive bit-for-bit and performs no extra
  candidate fetch or write. This is the required no-harm/self-disable result.
- The small advantage of priced document placement over LRU is secondary and
  should not be presented as meaningful end-to-end RAG quality.

