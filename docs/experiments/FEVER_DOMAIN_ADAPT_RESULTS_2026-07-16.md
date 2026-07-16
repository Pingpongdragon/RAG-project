# FEVER query-adaptive cache result (2026-07-16)

## Protocol

- Dataset: BEIR FEVER, 8,000 claims sampled before the fixed calibration/test split.
- Cache object: gold Wikipedia evidence page.
- Stream: 20 evaluation windows of 25 requests after four warm-up windows;
  controlled recurring evidence regimes, with constructor labels hidden online.
- Cold store: 21,340 pages partitioned offline into 1,334 balanced semantic
  regions (target size 16). The partition uses document text only.
- Cache size: 24 pages; candidate budget: 24; random seed: 42.
- Feedback: gold evidence IDs are revealed only after serving the request.

This is an oracle evidence-residency trace, not an end-to-end answer-quality
evaluation. Its purpose is to isolate whether query-conditioned cold-store
routing improves which evidence pages reside in the hot tier.

## Result at write cap 5

| Policy | Evidence hit (%) | Writes | Routed candidate reads |
|---|---:|---:|---:|
| LRU | **17.93** | 100 | 0 |
| Classical ARC | 13.45 | 100 | 0 |
| AgentRAGCache | 12.87 | 87 | 0 |
| DRIP-Reactive | 15.59 | 29 | 0 |
| DRIP-DomainAdapt | 15.98 | **27** | 480 |

## Interpretation

- DomainAdapt improves over Reactive by only 0.39 percentage points while
  making two fewer writes, and adds 480 routed candidate reads.
- LRU remains 1.95 points higher in raw hit rate. The result does not establish
  a full-cost Pareto improvement; it is a small placement gain over Reactive.
- ARC and AgentRAGCache do not dominate the proposed policy on this recurring
  evidence-regime stream.
- The result uses a single fixed seed. It should be reported as a controlled
  mechanism test and paired with the five-seed WoW result and natural-time MIND
  appendix, rather than presented as standalone evidence of broad superiority.

The previous 16.96% result used post-service gold evidence embeddings as the
route input and is invalid as evidence of current-query routing. The corrected
runner routes original query embeddings before scoring.

Raw output:
`experiments/direct/data/domain_adapt_fever_recurring_causal_v2_s42.json`.

The earlier tight-budget JSON predates the causal-v2 route-input fix and is not
used in the paper.
