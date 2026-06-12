# DRIP Algorithm Explained

> Current implementation-facing explanation. DRIP is a proactive document
> hot-tier cache for Agent RAG. The current runnable method is DRIPCore.

Source of truth:

- Final policy: `algorithms/drip/support_flow/__init__.py`
- Query router: `algorithms/drip/support_flow/query_router.py`
- Graph bridge evidence: `algorithms/drip/support_flow/graph_index.py`
- Query-demand baseline: `algorithms/cache/ours/query_driven.py`
- Strategy registry: `algorithms/cache/registry.py`

## Claim

DRIP routes each query to the right evidence channel, credits dense or graph
evidence into a support ledger, and admits documents whose evidence beats the
weakest resident cache item.

The paper should frame DRIP as solving:

> Given a full cold corpus and a small L1 document hot tier, which cold
> documents should be proactively admitted so future RAG queries need fewer
> cold-corpus fetches?

## What DRIP Is Not

| Not this | Why |
|---|---|
| A prompt-tuning method | DRIP changes cached evidence, not prompts. |
| A document-ingestion pipeline | If the document is absent from the cold corpus, DRIP cannot create it. |
| A pure eviction algorithm | The main contribution is route-aware admission evidence. |
| A conversational-memory system | No episodic or semantic memory tier is implemented or evaluated in this track. |

## System Objects

| Object | Meaning | Budgeted? |
|---|---|---|
| Query stream | User or agent requests arriving in windows. | No |
| Cold corpus | Full document pool. | No |
| L1 hot tier / cache | Small ANN-ready subset used for low-latency RAG. | Yes |

## Module Overview

```text
query window
    |
    v
ROUTE: SINGLE / MULTI_DIRECT / BRIDGE
    |
    v
EVIDENCE: dense direct evidence or GraphIndex bridge evidence
    |
    v
ADMIT/EVICT: demand + serve ledger with support-priority replacement
```

## Graph Bridge Evidence

For bridge queries, DRIPCore uses:

```text
q -> A -> entity e -> B
```

`A` is a dense first-hop document. `B` is a hidden second-hop candidate found by
shared entities. In code, GraphIndex scores:

```text
score(B) = sum_{A,e} s(q,A) * IDF(e) / g(e)^rho * novelty(B|C_t) * complementarity(A,B)
```

The score is then clipped to top bridge candidates and normalized per query.

## Admission

All evidence goes into `demand[d]`. Resident cache usefulness goes into
`serve[d]`. The writer ranks non-resident candidates by demand and resident
victims by support priority:

```text
priority(d) = serve[d] + demand[d] - redundancy_penalty(d)
```

Candidate `c` replaces victim `v` only when:

```text
demand[c] > gain_margin * priority[v]
```

Near-duplicates of the current cache are skipped.

## Runnable Variants

| Variant | Status | Meaning |
|---|---|---|
| `QueryDriven` | retained | Minimal direct-demand SemFlow baseline. |
| `DRIP` | current method | DRIPCore with route-aware dense and graph evidence. |

Retired `RoutedCache` and detector-wrapped `algorithms/cache/ours/drip.py`
should not be used as current strategies.
