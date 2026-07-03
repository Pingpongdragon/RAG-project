# DRIP — Current Method

> This is the current paper-facing method definition. DRIP is a proactive L1
> document hot-tier cache for Agent RAG. The implementation source of truth is
> `algorithms/drip/cache_manager/`.

## One-Line Positioning

DRIP maintains a bounded shared RAG hot tier under query demand shift and
multi-hop evidence visibility gaps. The method is presented as two evidence
visibility regimes:

```text
DRIP-QueryVisible = embedding/direct channel only
DRIP-QueryHidden  = query-visible A + hidden-support completion B
DRIP              = current main alias for DRIP-QueryHidden
```

Pair lease is part of `DRIP-QueryHidden`'s retention policy, not a third public
query class.

## System Model

```text
query window W_t
    |
    v
DETECT: multi-agent drift severity rho_t
    |
    v
ROUTE: QUERY_VISIBLE / QUERY_HIDDEN
    |
    v
EVIDENCE: dense/direct candidates or evidence-conditioned hidden candidates
    |
    v
ADMIT/EVICT: support-priority replacement with optional pair lease
    |
    v
L1 document hot tier C_t over full L2 corpus D
```

The cache has capacity `B`:

```text
C_t subset D, |C_t| <= B
```

The online question is:

> Which cold-corpus documents should be admitted so future Agent RAG queries
> need fewer cold-tier fetches while the hot tier remains small?

## Drift Controller

The detector is a controller over update aggressiveness, not a source of
evidence and not a query router.  For each window it computes a multi-agent
drift severity:

```text
rho_t = Drift({q_t^a}_a, C_t)
```

using only query embeddings, agent ids, and query-cache alignment signals:
unsupported-query rate, top-1/top-k cache similarity, and per-agent query
centroid shift. It does not use `qtype`, `route_hint`, support labels, or
answers.

The severity controls cache maintenance:

```text
write_cap_t = write_cap_0 * (1 + alpha * rho_t)
demand_decay_t = demand_decay_0 * (1 - beta * rho_t)
margin_t = margin_0 * (1 - gamma * rho_t)
```

So detector answers "when should the shared cache move, and how strongly?";
DRIP's evidence channels answer "which documents should move in?"

## Evidence Channels

### DRIP-QueryVisible

For single-hop and direct multi-hop queries, the needed evidence is visible from
the query embedding. DRIP-QueryVisible credits direct candidates:

```text
D_dir(d | q) = max_{r <= K} sim(q, d_r) * rank_decay(r)
```

This is the direct semantic admission channel. It is useful for topic drift and
direct-evidence multi-hop, but it cannot reliably reach a hidden second-hop
document that is not close to the initial query.

### DRIP-QueryHidden

For hidden-support queries, the query can reveal or retrieve first-hop evidence
`A`, while the reusable second-hop evidence `B` is hidden:

```text
q -> A -> B
```

DRIP-QueryHidden adds evidence-conditioned hidden-support completion:

```text
D_hid(B | q, A) = E_hidden(q, B | A)
```

The key difference from direct semantic admission is that `B` is scored after
conditioning on `A`, not only by `sim(q, B)`.

After DRIP-QueryHidden completes an A+B support pair, the pair lease helps the
cache avoid immediately evicting the first-hop document that made the hidden
support useful:

```text
P(d) = S(d) + D_dir(d) + D_hid(d) + lambda_pair * L(d)
```

`S(d)` is serve evidence from resident documents. `L(d)` is lease evidence
assigned to documents in a completed A+B pair.

## Admission Rule

Candidates and residents compete in one support-priority framework:

```text
admit candidate c over victim v iff
    score(c) > gain_margin * priority(v)
```

This makes the ablations clean:

| Variant | Isolated question | Registry key |
|---|---|---|
| DRIP-QueryVisible | Is direct embedding evidence enough? | `DRIP-QueryVisible` |
| DRIP-QueryHidden | Does hidden-support completion plus pair retention improve hidden-support tasks? | `DRIP-QueryHidden` |

## Comparison Set

Use these as the main paper-facing strategies:

```text
LRU
TinyLFU
GPTCacheStyle
AgentRAGCache
DRIP-QueryVisible
DRIP-QueryHidden
Oracle
```

`AgentRAGCache` should be labeled `ARC` in paper tables.

## Current Code

| Component | File |
|---|---|
| Registry keys | `algorithms/cache/registry.py` |
| Paper variants | `algorithms/drip/cache_manager/drip.py` |
| Core manager | `algorithms/drip/cache_manager/__init__.py` |
| Query router | `algorithms/drip/cache_manager/query_router.py` |
| Multi-agent drift detector | `algorithms/drip/detection/multi_agent_drift.py` |
| Dense/direct index | `algorithms/drip/cache_manager/embedding_index.py` |
| Hidden support and pair lease | `algorithms/drip/cache_manager/support_completion.py` |
| Graph metadata | `algorithms/drip/cache_manager/graph_index.py` |

## Limitations

- DRIP targets one-hop bridge completion `q -> A -> B`, not arbitrary-length
  reasoning chains.
- Hidden-support scoring depends on whether the observed first-hop evidence
  exposes useful entities or text cues for `B`.
- Pair lease should be evaluated with reuse-aware metrics; otherwise a one-shot
  bridge benchmark may understate the cache-management benefit.
