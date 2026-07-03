# DRIP Algorithm Explained

> Current implementation-facing explanation. DRIP is a proactive document
> hot-tier cache for Agent RAG. The current runnable method is DRIPCore.

Source of truth:

- Current policy: `algorithms/drip/cache_manager/__init__.py`
- Paper-facing variants: `algorithms/drip/cache_manager/drip.py`
- Multi-agent drift controller: `algorithms/drip/detection/multi_agent_drift.py`
- Query router: `algorithms/drip/cache_manager/query_router.py`
- Dense/direct channel: `algorithms/drip/cache_manager/embedding_index.py`
- Query-hidden support channel: `algorithms/drip/cache_manager/support_completion.py`
- Graph metadata: `algorithms/drip/cache_manager/graph_index.py`
- Strategy registry: `algorithms/cache/registry.py`

## Claim

DRIP first decides whether the missing evidence is query-visible or
query-hidden. It then credits dense/direct or evidence-conditioned hidden
support into a support ledger, and admits documents whose evidence beats the
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
DRIFT CONTROL: rho_t from multi-agent query-cache alignment
    |
    v
ROUTE: QUERY_VISIBLE / QUERY_HIDDEN
    |
    v
EVIDENCE: dense direct evidence or evidence-conditioned hidden support
    |
    v
ADMIT/EVICT: demand + serve + pair-lease ledger with support-priority replacement
```

## Evidence Channels

The drift controller runs before evidence routing. It uses query-cache
alignment, not query type, to tune update aggressiveness:

```text
write_cap_t = write_cap_0 * (1 + alpha * rho_t)
demand_decay_t = demand_decay_0 * (1 - beta * rho_t)
margin_t = margin_0 * (1 - gamma * rho_t)
```

For query-visible evidence, `DRIP-QueryVisible` writes candidates from the
dense/direct channel:

```text
D_dir(d | q) = max_{r <= K} sim(q, d_r) * rank_decay(r)
```

For query-hidden evidence, `DRIP-QueryHidden` adds evidence-conditioned
hidden-support completion:

```text
q -> A -> entity e -> B
```

`A` is a dense first-hop document. `B` is a hidden second-hop candidate found by
conditioning on the evidence in `A`. In code, GraphIndex metadata and the
hidden-support scorer produce:

```text
D_hid(B | q, A) = E_hidden(q, B | A)
```

The hidden mode then protects completed support pairs:

```text
P(d) = S(d) + D_dir(d) + D_hid(d) + lambda_pair * L(d)
```

where `S(d)` is resident serve evidence and `L(d)` is the pair lease assigned
to both documents after an A+B bridge support pair is completed.

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

| Variant | Meaning |
|---|---|
| `DRIP-QueryVisible` | Embedding/direct channel only. |
| `DRIP-QueryHidden` | Query-visible A plus evidence-conditioned hidden support B and pair lease. |
| `DRIP` | Current main alias for `DRIP-QueryHidden`. |
