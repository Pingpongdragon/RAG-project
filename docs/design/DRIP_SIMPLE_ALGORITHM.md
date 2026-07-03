# DRIP Algorithm, Short Version

## Goal

DRIP manages a small RAG hot tier under query demand shift. The cold corpus is assumed to exist in full, but the online hot tier is capacity-limited, so the system must decide which document chunks should be resident before future queries need them.

The key distinction from a pure retriever is that DRIP optimizes cache admission, not only current-query ranking.

## State

DRIP maintains three lightweight states:

```text
C_t              current hot-tier cache
Serve_t(d)       recent serving utility of resident document d
Demand_t(d)      accumulated future-demand signal for candidate document d
GraphIndex       doc -> entities, entity -> docs, IDF(entity)
```

The graph is not a full knowledge graph. It is a small entity posting index used only to find possible bridge evidence.

## Route-Aware Evidence

Each query is routed into one of three evidence regimes:

```text
r(q) in {Single, MultiDirect, Bridge}
```

DRIP then credits demand using the evidence channel matched to the route:

```text
E_r(q,d) =
  E_D(q,d),  if r(q) in {Single, MultiDirect}
  E_G(q,d),  if r(q) = Bridge
```

For visible evidence, the dense channel is simply:

```text
E_D(q,d) = [sim(q,d)]_+
```

For bridge evidence, the target path is:

```text
q -> A -> e -> B
```

where `A` is a dense first-hop document, `e` is a shared entity, and `B` is the hidden second-hop document.

## GraphIndex Evidence

GraphIndex first applies hard gates:

```text
1 < deg(e) <= d_max
Rel(q,A,e,B) > 0
B not already in C_t
B not one of the first-hop A documents
```

Then it scores valid bridge paths:

```text
phi(q,A,e,B)
= sim(q,A)^alpha
  * Link(e)^beta
  * Rel(q,A,e,B)^gamma
```

where:

```text
Link(e) = normalized_IDF(e) / deg(e)^rho
```

The bridge evidence for `B` aggregates all valid paths:

```text
z(q,B) = sum_{A,e} phi(q,A,e,B)
E_G(q,B) = bridge_alpha * (1 - exp(-z(q,B) / s))
```

Finally, candidates are selected with a small MMR-style redundancy penalty:

```text
SelectScore(B) = E_G(q,B) - mu * max_{u in C_t union S} sim(B,u)
```

This keeps GraphIndex focused on evidence quality. Cache replacement is handled separately by the writer.

## Demand Update

For each under-covered query, DRIP adds route-aware evidence to the demand ledger:

```text
Demand_t(d)
= lambda * Demand_{t-1}(d) + eta_r * E_r(q_t,d)
```

Bridge evidence can use a larger `eta_r` because bridge documents are often invisible to the original query embedding and need proactive warming.

## Cache Priority

Residents are protected by serving utility and accumulated demand:

```text
Priority_t(d)
= Serve_t(d) + Demand_t(d) - RedundancyPenalty_t(d)
```

Serving utility decays over time:

```text
Serve_t(d) = serve_decay * Serve_{t-1}(d) + recent_hits(d)
```

Demand also decays:

```text
Demand_t(d) = demand_decay * Demand_{t-1}(d) + new_evidence(d)
```

## Admission Rule

DRIP admits a candidate only if it is clearly better than the weakest resident:

```text
admit c replacing v iff
Demand_t(c) > margin * Priority_t(v)
```

where:

```text
v = argmin_{d in C_t} Priority_t(d)
```

The margin is the simple anti-churn control. It avoids replacing cache entries for tiny score differences. We do not model low-level write cost explicitly because hot-tier replacement is assumed cheap compared with repeated cold-tier retrieval; the important cost is wrong replacement causing future misses.

## Relation to ARC

ARC uses one priority formula:

```text
Priority_ARC(p)
= [beta * log(h_k(p)+1) + (1-beta) * DRF(p)] / log(w(p)+1)
```

Its logic is:

```text
history-weighted retrieval frequency + embedding hubness - memory footprint
```

DRIP keeps the same cache-management spirit but changes the demand source:

```text
ARC:  demand comes from retrieved top-K history.
DRIP: demand comes from route-aware evidence.
```

The main difference is bridge routing:

```text
ARC can only credit documents that are retrieved for the current query.
DRIP can credit hidden bridge documents B through q -> A -> e -> B.
```

## One-Sentence Summary

DRIP is a route-aware hot-tier cache manager: dense evidence handles query-visible documents, GraphIndex exposes hidden bridge documents, and a margin-based writer admits only candidates whose accumulated demand clearly exceeds the weakest resident.
