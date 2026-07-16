# Query-adaptive domain routing audit (2026-07-16)

## Decision

The defensible main mechanism is **query-adaptive domain routing followed by
document-level, switching-cost-aware placement**.  It is not next-topic
prediction.  Semantic/metadata regions are retrieval indices, not cache
objects: a request selects a small number of regions, concrete documents are
ranked inside them, and only document evidence can change the persistent hot
tier.

The audited history-prior and window-topic-placement variants are not promoted
to the main method.  They failed to generalize from the SQuAD calibration
source-family pool to the held-out test pool.

## Protocol

- Dataset: SQuAD controlled factorized evidence stream.
- Cold pool: 1,200 documents; semantic pages contain at most 16 documents.
- Evaluation: 20 windows x 25 queries, following four warm-up windows.
- Hot tier: 24 documents; shared maximum of five writes per window.
- Main DRIP settings selected on the calibration source pool only:
  demand/serve decay 0.25, initial switching price 0.25, replacement target
  0.25, region candidate budget 24.
- Metrics freeze the hot tier for the entire current window. Original query
  embeddings route before scoring; post-service gold evidence IDs update only
  the next-window cache.  This is an oracle
  evidence-demand residency trace, not yet an end-to-end RAG answer result.

## Held-out test-pool result

| Stream | Method | Evidence hit | Writes | Route reads |
|---|---|---:|---:|---:|
| Recurring | LRU | 22.0% | 97 | 0 |
| Recurring | Classical ARC | 19.2% | 98 | 0 |
| Recurring | AgentRAGCache | 18.4% | 74 | 0 |
| Recurring | DRIP-Reactive | 20.6% | 30 | 0 |
| Recurring | Query-adaptive routing | **23.4%** | 32 | 480 |
| Shuffled | LRU | 15.8% | 99 | 0 |
| Shuffled | Classical ARC | 17.2% | 99 | 0 |
| Shuffled | AgentRAGCache | 16.6% | 82 | 0 |
| Shuffled | DRIP-Reactive | **19.8%** | 30 | 0 |
| Shuffled | Query-adaptive routing | **19.8%** | **28** | 480 |
| Stationary | LRU | **76.2%** | 87 | 0 |
| Stationary | Classical ARC | 74.8% | 87 | 0 |
| Stationary | AgentRAGCache | 43.4% | **11** | 0 |
| Stationary | DRIP-Reactive | 69.4% | 31 | 0 |
| Stationary | Query-adaptive routing | 64.0% | 27 | 480 |

Across five recurring held-out seeds, DomainAdapt averages 21.84% hit and 27.4
writes versus Reactive at 20.72%/30.0 and LRU at 22.44%/98.4. It therefore
improves over Reactive but does not beat LRU, and its 480 route reads must be
reported. On shuffled traffic it adds no hit gain; under stationary traffic it
hurts Reactive by 5.4 points.

## Failed variants

1. **Contextual similar-query residual.** It improves route recall but produces
   20.56% calibration hit versus 21.16% for static centroid routing.
2. **Downstream-validated route credit.** It obtains only 19.80% calibration
   hit and performs more writes.

Both failed branches were removed from the core implementation. Full details
are in `MRAG_PARTITION_ROUTING_AUDIT_2026-07-16.md`.

## Next required validation

- Compare region-routed retrieval with full-corpus dense retrieval at matched
  recall and document-read cost.
- Repeat the fixed configuration on WoW controlled multi-turn sessions and on
  chronological MIND.
- Treat MT-RAG as a low-reuse no-harm stress test, not a gain dataset.
- Report the hit--write Pareto frontier, per-window recovery after shifts, and
  region scan/fetch counts.  Do not claim pure hit-rate SOTA from this audit.
