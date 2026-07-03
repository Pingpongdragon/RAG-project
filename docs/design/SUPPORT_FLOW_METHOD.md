# DRIP Cache Manager Method

> This note keeps the old filename for continuity, but the method name is now
> DRIP. The current implementation lives in `algorithms/drip/cache_manager/`.

## Paper-Facing Variants

| Strategy | Meaning | Code |
|---|---|---|
| `DRIP-QueryVisible` | Embedding/direct channel only. | `DRIPQueryVisible` in `drip.py` |
| `DRIP-QueryHidden` | Query-visible A plus evidence-conditioned hidden support B and pair lease. | `DRIPQueryHidden` in `drip.py` |
| `DRIP` | Current main alias for `DRIP-QueryHidden`. | `DRIP` in `drip.py` |

## Problem

Given a large cold corpus `D`, a bounded hot evidence cache `C_t`, and a query
window `W_t`, keep the hot tier useful under demand shift:

```text
C_t subset D, |C_t| <= B
```

The objective is not to write aggressively. The cache should admit documents
only when their expected support utility beats the resident evidence they would
evict.

## Evidence Channels

`DRIP-QueryVisible` handles evidence visible from the query embedding:

```text
D_dir(d | q) = max_{r <= K} sim(q, d_r) * rank_decay(r)
```

`DRIP-QueryHidden` adds hidden support completion. If the query can reach
first-hop evidence `A`, DRIP conditions on `A` to find missing support `B`:

```text
q -> A -> B
D_hid(B | q, A) = E_hidden(q, B | A)
```

The hidden variant adds a lease on completed support pairs so that admitting
`B` does not immediately evict the already useful `A`:

```text
P(d) = S(d) + D_dir(d) + D_hid(d) + lambda_pair * L(d)
```

where `S(d)` is serve evidence from resident documents and `L(d)` is pair-lease
evidence assigned to documents that form a completed A+B support pair.

## Admission Rule

The writer ranks non-resident candidates by accumulated evidence and resident
victims by support priority. A candidate replaces a resident only when:

```text
score(candidate) > gain_margin * priority(victim)
```

This keeps the dense/direct, hidden-support, and pair-lease signals in one
support-priority framework instead of treating them as separate algorithms.

## Current Code Path

| Component | File |
|---|---|
| Cache manager | `algorithms/drip/cache_manager/__init__.py` |
| Paper variants | `algorithms/drip/cache_manager/drip.py` |
| Query router | `algorithms/drip/cache_manager/query_router.py` |
| Dense/direct evidence | `algorithms/drip/cache_manager/embedding_index.py` |
| Hidden support and pair lease | `algorithms/drip/cache_manager/support_completion.py` |
| Graph metadata | `algorithms/drip/cache_manager/graph_index.py` |
| Registry keys | `algorithms/cache/registry.py` |
