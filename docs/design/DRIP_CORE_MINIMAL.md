# DRIP Minimal Design: Query-Visible vs Query-Hidden

> Goal: keep the method explanation compact. DRIP is one cache manager with two
> evidence visibility regimes, not a collection of separately named methods.

## Naming

```text
DRIP-QueryVisible = embedding/direct channel only
DRIP-QueryHidden  = query-visible A + hidden-support completion B
DRIP              = current main alias for DRIP-QueryHidden
```

Pair lease remains part of `DRIP-QueryHidden`'s retention policy. It is no
longer exposed as a third public query class.

## Components

```text
QueryRouter
  <- MultiAgentDriftDetector sets update aggressiveness rho_t
  -> EmbeddingIndex for dense/direct evidence
  -> SupportCompletion for query-hidden support
  -> support-priority admission with pair lease inside hidden mode
```

### 1. QueryRouter

Before routing, `MultiAgentDriftDetector` estimates the current shared-cache
misalignment:

```text
rho_t = Drift({q_t^a}_a, C_t)
```

`rho_t` adjusts demand decay, write cap, and admission margin. It does not
choose `QUERY_VISIBLE` vs `QUERY_HIDDEN`.

### 2. QueryRouter

The router assigns each under-covered query to:

```text
QUERY_VISIBLE  target_slots = 1 or 2, evidence reachable from query embedding
QUERY_HIDDEN   target_slots = 2, query exposes A but B requires conditioning on A
```

`qtype` and `route_hint` are diagnostic labels by default. They are used for
routing only when `use_oracle_route_hint=True` in an oracle-router ablation.

### 3. EmbeddingIndex

EmbeddingIndex produces dense/direct candidates for `DRIP-QueryVisible`:

```text
D_dir(d | q) = max_{r <= K} sim(q, d_r) * rank_decay(r)
```

This channel covers single-hop drift and direct multi-hop evidence.

### 4. SupportCompletion

SupportCompletion produces evidence-conditioned hidden support for
`DRIP-QueryHidden`:

```text
q -> A -> B
D_hid(B | q, A) = E_hidden(q, B | A)
```

Pair lease protects completed A+B support pairs after hidden support is admitted.

## Two Formulas

### Formula 1: Route-Aware Evidence

```text
E_t(d | q) =
    I[r(q) = QUERY_VISIBLE] * D_dir(d | q)
  + I[r(q) = QUERY_HIDDEN]  * (D_anchor(d | q) + D_hid(d | q, A))
```

### Formula 2: Support Priority

```text
P_t(d) = S_t(d) + D_dir,t(d) + D_hid,t(d) + lambda_pair * L_t(d) - R_t(d)
```

where:

```text
S_t(d)   = resident serve evidence
D_dir,t  = dense/direct demand evidence
D_hid,t  = hidden-support evidence
L_t(d)   = pair lease evidence
R_t(d)   = redundancy penalty
```

Admission:

```text
admit candidate c over resident v iff
    score(c) > gain_margin * P_t(v)
```

## ARC Contrast

ARC caches documents that were historically central under query-distribution
geometry. DRIP caches documents that current under-covered queries need next,
including hidden support that becomes visible only after conditioning on
first-hop evidence.
