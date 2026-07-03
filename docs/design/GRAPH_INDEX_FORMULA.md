# GraphIndex Bridge Formula (canonical)

Date: 2026-06-19
Scope: `algorithms/drip/cache_manager/graph_index.py` (`GraphIndex.graph_evidence`)
Status: implemented and tested

This document is the authoritative statement of the bridge-evidence formula
DRIP actually computes today, plus the extractor-agnostic framing for the paper.
It supersedes the shorthand in `GRAPH_INDEX_V2_EXECUTION_PLAN.md` §3-§4, which
described the target before implementation.

---

## 0. Role in DRIP

GraphIndex is the **content-side admission signal** for the BRIDGE route. For a
query `q` whose first hop only exposes document `A`, it scores hidden
second-hop documents `B` reachable through a shared entity `e`, and emits
`E_graph(q, B)` as demand credit. It does **not** call an LLM in the hot path
and it is **not** a global KG — it is a per-strategy postings index plus
query-local text evidence.

```text
q --dense--> A --shared entity e--> B        (the Q->A->B bridge)
```

---

## 1. The graph

Built once per KB state from `pool_ents` (offline-extracted entities) in
`GraphIndex.build()`:

```text
doc_to_ents : pi    -> [normalised entity]     (title entity always injected)
ent_to_docs : entity -> {pi}                    (postings)
ent_idf     : entity -> log((1+N) / (1+df(e))) + 1
```

Generic-nationality entities (`_GENERIC_ENTITIES`) and sub-`min_entity_len`
strings are dropped at build time. `N` = number of indexed docs,
`df(e) = |ent_to_docs[e]|`.

---

## 2. The formula (as implemented)

For one query `q`, evidence mass accumulates over first hops `A` and shared
entities `e`:

```text
raw(B) = sum_{A in TopK1(q)}  sum_{e in seed(A)}  1[e links A,B]
           * PathEvidence( s(q,A), Link(e), Rel(q,A,e,B) )

E_graph(q,B) = bridge_alpha * clip_tau( raw(B) )       if raw(B) > 0
```

with the **geometric-mean path kernel** (this is the `sim^α · Link^β · Rel^γ`
form, weights renormalised to sum to 1):

```text
PathEvidence(s, l, r) = s^a * l^b * r^c ,   a,b,c = (α,β,γ)/(α+β+γ)
```

implemented in log-space as
`exp( a·ln s + b·ln l + c·ln r )` (`_path_evidence`), with
`(α,β,γ) = (bridge_evidence_alpha, _beta, _gamma)` defaulting to
`(0.45, 0.30, 0.25)`.

**No per-query max normalisation.** Suppression uses an absolute threshold +
saturation instead, so a lone weak candidate cannot be rescaled up to
`bridge_alpha`:

```text
clip_tau(x) = 1 - exp( -x / bridge_score_saturation )       (soft saturation)
E_graph kept only if  bridge_alpha * clip_tau(raw) >= bridge_abs_threshold
```

### 2.1 Term definitions

| Term | Symbol | Code | Meaning |
|---|---|---|---|
| First-hop confidence | `s(q,A)` | `a_sim` | query→A cosine (gated by `bridge_min_firsthop_sim`). |
| Link strength | `Link(e)` | `_link_strength` | `clip( IDF(e)/IDF_max / degree(e)^ρ , 1e-6, 1)`, `ρ = entity_degree_power`. |
| Relation alignment | `Rel(q,A,e,B)` | `_relation_score` | query terms ∩ (entity-context terms ∪ B-title terms); `+bridge_title_relation_bonus` if `e` is B's title; capped at 1.5, clipped to 1 inside the kernel. |

### 2.2 Gates (applied in order, all logged in `last_stats`)

```text
bridge_min_firsthop_sim   s(q,A) must exceed this or A is skipped
degree > 1                an entity in only one doc cannot bridge
max_entity_degree         drop global hub entities
Rel > 0                   require query/relation alignment (no blind entity overlap)
graph_novelty_floor       B must be far enough from current cache C_t
bridge_abs_threshold      final E_graph must clear an absolute bar
```

Surviving candidates are diversified by **MMR** (`_select_mmr`,
`bridge_mmr_mu`) and capped at `bridge_max_docs`.

---

## 3. Two instantiations (paper framing)

The framework is **extractor-agnostic**. The kernel in §2 is unchanged; only
how `ent_to_docs` / `Rel` are populated differs.

### 3.1 Light GraphIndex — current implementation (no LLM)

```text
Ent(d)   = spaCy_NER(title + first 512 chars of d)   # graph_retrieval.py
Edges    = d <-> e   (bipartite, entity postings)
Rel(q,p) = lexical overlap between q and entity-context / B-title windows
```

Pros: cheap, reproducible, zero LLM cost, fully offline-cached
(`pool_ents_*.json`). Cons: recall-limited (explicit named entities only, no
typed relations, may miss bridges past the 512-char window).

### 3.2 LLM GraphIndex — paper extension (offline only)

Built once during cold-tier preprocessing, **never in the hot path**:

```text
LLM(d) -> { entities: [canonical + aliases],
            relations: [(head, rel, tail)],
            bridge_targets: [candidate linked pages] }
```

This upgrades the graph to typed edges:

```text
doc -> entity ;  entity -> doc ;  entity --rel--> entity
```

and the path generalises from "shared entity" to "typed relation hint" `h`:

```text
q -> A -> h -> B         where h is an entity OR an LLM-extracted typed relation
```

`Rel(q,A,h,B)` then becomes a typed-relation match (e.g. the query asks
`director_of` and the path predicate is `director_of`) rather than lexical
overlap. The kernel `s^a · Link^b · Rel^c`, the gates, the absolute threshold,
and the MMR selection all carry over unchanged.

### 3.3 Recommended paper sentence

> We instantiate GraphIndex with a lightweight NER-based extractor to isolate
> the cache-management mechanism. The framework is extractor-agnostic and can
> use LLM/KG extractors, as in GraphRAG / LightRAG / HippoRAG, to improve
> bridge recall. Online cache management performs only cheap postings lookup
> and scoring; LLM extraction, when used, builds the index offline.

Related work anchors: GraphRAG (LLM entity/relation extraction + graph
communities), LightRAG (entity-relation graph + vector retrieval, incremental
update), HippoRAG (KG-style memory + PPR for multi-hop).

---

## 4. Config knobs

All knobs live in `DRIPCoreConfig` (no silent `getattr` defaults remain).

| Knob | Default | Controls |
|---|---|---|
| `bridge_alpha` | 0.6 | overall bridge-evidence scale |
| `bridge_evidence_alpha/beta/gamma` | 0.45/0.30/0.25 | path-kernel exponents `(a,b,c)` |
| `bridge_score_saturation` | 1.0 | `clip_tau` soft-saturation scale |
| `bridge_abs_threshold` | 0.08 | absolute keep bar (replaces max-norm) |
| `bridge_min_firsthop_sim` | 0.0 | gate weak first hops `A` |
| `entity_degree_power` (ρ) | 0.5 | hub down-weighting in `Link(e)` |
| `max_entity_degree` | 200 | hard hub cutoff |
| `bridge_relation_floor` | 0.05 | base relation score |
| `bridge_relation_overlap_weight` | 0.35 | per-overlap-term relation weight |
| `bridge_min_relation_overlap` | 1 | min query/context overlap to keep a path |
| `bridge_title_relation_bonus` | 0.25 | bonus when `e` is B's title (Wikipedia bridges) |
| `graph_novelty_floor` | 0.05 | min distance of B from cache |
| `bridge_mmr_mu` | 0.02 | MMR redundancy penalty |
| `bridge_max_docs` | 20 | max bridge candidates per query |
| `bridge_max_seed_entities` | 12 | seed entities per first hop |

---

## 5. Tests

| File | Proves |
|---|---|
| `tests/test_graph_index_v2.py` | relation path beats generic shared entity. |
| `tests/test_graph_index_hostile.py` | hub entity does not win; weak first hop gated; sub-threshold lone candidate dropped (no max-norm); degree-1 entity cannot bridge. |

Run:

```bash
PYTHONPATH=. python algorithms/drip/tests/test_graph_index_v2.py
PYTHONPATH=. python algorithms/drip/tests/test_graph_index_hostile.py
```
