# DRIP Engineering Review: Graph, Routing, and Cache Management

> `/plan-eng-review` output for the current DRIP redesign. This is a planning document, not an implementation spec to blindly code. The goal is to decide the architecture before we keep writing knobs into the algorithm.

## Verdict

DRIP should be the system name. `DRIP cache manager` should be the core cache engine inside DRIP.

Recommended architecture:

```text
DRIP
  ├── DRIP cache manager cache engine
  │     ├── CoverageObserver
  │     ├── TypedEvidenceLedger
  │     ├── GraphBridgeRouter
  │     └── BudgetedWriter
  └── optional EpochController / detector later
```

The current entity chain in [support_completion.py](../../algorithms/drip/cache_manager/support_completion.py) is too weak. It finds many bridge candidates, but it does not prove they are the missing second-hop evidence. That is why H2 coverage did not improve in the 20-window bridge run.

We should add a lightweight graph-backed bridge module. Not a big LLM-built global knowledge graph. Not yet.

## Step 0: Scope Challenge

### What already exists

| Sub-problem | Existing code | Reuse? | Notes |
|---|---|---|---|
| Entity extraction | [graph_retrieval.py](../../experiments/hidden/graph_retrieval.py) `extract_pool_entities`, `extract_query_entities` | Yes | Already cached, spaCy-based, good enough for first version. |
| Entity graph retrieval | [LightGraphRAG](../../experiments/hidden/graph_retrieval.py) | Partially | Has passage-entity graph, IDF, co-occurrence edges, PPR. Useful reference, but retrieval-time graph is not the same as write-time admission. |
| Direct demand ledger | [cache_manager/__init__.py](../../algorithms/drip/cache_manager/__init__.py) | Yes | Keep, but split evidence by channel. |
| Bridge candidate generation | [support_completion.py](../../algorithms/drip/cache_manager/support_completion.py) | Replace internals | Current bridge chain has recall but poor precision. |
| Budgeted admission | [cache_manager/__init__.py](../../algorithms/drip/cache_manager/__init__.py) | Modify | Needs typed budgets and typed evidence, not one scalar demand. |
| ARC baseline | [agent_rag_cache.py](../../algorithms/cache/paradigm_ref/agent_rag_cache.py) | Keep | Good contrast: historical DRF + hubness vs under-coverage repair. |

### Minimum complete change

Do not build a full KG system first. Build a write-side **evidence graph index** with three boring pieces:

1. `GraphIndex`: pool-wide doc/entity adjacency and IDF.
2. `GraphBridgeRouter`: candidate generation with path + novelty + complementarity scoring.
3. `TypedEvidenceLedger`: separate direct, bridge, temporal evidence channels.

This is enough to fix the current failure mode without spending an innovation token on a heavy KG.

### Complexity check

The plan naturally touches more than two classes. That is acceptable here because the code is already split into components and the complexity is real:

```text
entity extraction       existing
graph candidate search  new or refactored
typed evidence ledger   new behavior
writer arbitration      new behavior
```

Do not add an LLM agent to the online path. That would be a scope expansion with high variance and slow iteration.

## Architecture Review

### 1. Do we need a knowledge graph?

Yes, but use a lightweight evidence graph, not a full knowledge graph.

What we need:

```text
doc --mentions--> entity
entity --mentioned_by--> docs
entity --co_occurs_with--> entity
doc --similar_to--> doc        optional, small top-k only
```

What we do not need yet:

```text
LLM-extracted relation triples
global graph reasoning service
multi-hop PPR for every write decision
online LLM relation classification
```

Recommended data structure:

```text
GraphIndex
  doc_ents:       doc_idx -> [entity]
  ent_docs:       entity -> compact doc_idx list
  ent_idf:        entity -> float
  ent_cooc:       entity -> top related entities
  doc_neighbors:  doc_idx -> top semantic neighbors, optional
```

Why this is enough:

- It keeps bridge candidate generation fast.
- It reuses existing spaCy extraction and cached entities.
- It gives us graph structure for multi-hop without turning the project into a graph database project.
- It is deterministic, reproducible, and easy to test.

Layer judgment: **Layer 1 / boring technology**. Entity inverted index + co-occurrence graph. No fancy infra.

### 2. Current entity chain, exactly

Current [support_completion.py](../../algorithms/drip/cache_manager/support_completion.py) does this:

```text
under-covered query q
  -> dense top step1_k docs A from full pool
  -> collect entities from A
  -> weight entity by sim(q,A) * idf(entity)
  -> retrieve all docs B sharing those entities
  -> score B by rare-entity overlap
  -> add B into demand ledger
```

ASCII flow:

```text
q
│
├─ dense(q, D) ──> A1, A2, A3
│                  │
│                  ├─ ents(A1) = {e1, e2}
│                  ├─ ents(A2) = {e2, e3}
│                  └─ ents(A3) = {e4}
│
└─ entity index ──> e2 -> {B1, B2, B3, ...}
                   e3 -> {B4, B5, ...}

B score = sum sim(q,A) * idf(shared_entity)
```

Why it failed in the 20w run:

```text
bridge_updates high
bridge_mass high
Coverage H2 flat
```

That means the router fires, but it fires at the wrong targets. Shared entity is necessary but not sufficient.

### 3. What the bridge score should be

Replace scalar entity overlap with path-quality scoring:

```text
score(q, A, B) =
    rel(q, A)
  * link(A, B)
  * novelty(B | cache)
  * complement(B | A, q)
  * nonhub(B)
```

Concrete definitions:

| Term | Meaning | Cheap implementation |
|---|---|---|
| `rel(q,A)` | first-hop relevance | cosine(q, A) |
| `link(A,B)` | rare shared entity path | sum IDF(shared entities) |
| `novelty(B|cache)` | B is not already covered | `1 - max_sim(B, cache)` clipped |
| `complement(B|A,q)` | B adds evidence not identical to A | entity Jaccard distance + semantic distance from A |
| `nonhub(B)` | penalize generic entity hubs | inverse average degree of shared entities |

New candidate logic:

```text
for q in under_covered:
  A = dense_topk(q, pool)
  for a in A:
    for entity in rare_entities(a):
      for b in ent_docs[entity]:
        if b == a: continue
        if redundant(b, cache): continue
        score[b] += rel(q,a) * link(a,b) * novelty(b) * complement(a,b)
```

This is still not a full KG. It is a path-scored candidate generator.

### 4. How to maintain the graph

There are two maintenance layers.

#### Offline / corpus-level index

Built once per experiment corpus:

```text
doc_pool
  -> entity extraction
  -> doc_ents
  -> ent_docs inverted index
  -> entity IDF
  -> entity co-occurrence top-k
```

This should be cached exactly like `extract_pool_entities()` already does.

#### Online / cache-level state

Updated every window after writes:

```text
cache membership C_t
  -> cache entity coverage
  -> resident redundancy scores
  -> optional cache-local entity degrees
```

Do not rebuild the pool graph each window. Only recompute cache-local summaries after writes.

Data flow:

```text
        offline once                         online per window
┌──────────────────────────┐        ┌─────────────────────────────┐
│ doc_pool                 │        │ query window W_t             │
│  -> doc_ents             │        │  -> under-coverage observer  │
│  -> ent_docs             │───────▶│  -> graph bridge candidates  │
│  -> ent_idf              │        │  -> typed evidence ledger    │
└──────────────────────────┘        │  -> budgeted writer          │
                                    └─────────────────────────────┘
```

### 5. How to avoid hurting single-hop temporal

This is the biggest architecture issue.

Current `CoverageObserver.target_slots()` uses 2 slots whenever entity metadata exists. That is too blunt. It can turn single-hop queries into fake under-covered queries and cause unnecessary maintenance.

Recommended design:

```text
Route per query, not per dataset.

single-hop-like query:
  target_slots = 1
  direct channel budget only

multi-hop/comparison-like query:
  target_slots = 2
  direct + bridge channel budget
```

But do not put an online LLM agent in the hot path.

Use a deterministic router first:

```text
QueryRouter
  Inputs:
    top1 cache sim
    top2 cache coverage gap
    number of named entities in query
    comparison cues: compare, difference, same, both, which, between
    bridge cues: relation between two entities, entity count >= 2
    retrieval disagreement: dense top docs mention different entity sets

  Output:
    route = SINGLE | MULTI_DIRECT | BRIDGE
```

Routing policy:

```text
SINGLE:
  target_slots = 1
  bridge off
  write budget = #under_covered

MULTI_DIRECT:
  target_slots = 2
  direct top-k demand only
  bridge optional low budget

BRIDGE:
  target_slots = 2
  graph bridge on
  reserve part of budget for bridge candidates
```

This protects temporal single-hop because entity metadata existing globally no longer forces every query to behave like multi-hop.

### 6. Do we need an LLM agent to classify queries?

Not in the online path.

Recommendation:

```text
Phase 1: deterministic QueryRouter
Phase 2: offline LLM labels for analysis / ablation only
Phase 3: optional cached LLM classifier if deterministic routing fails
```

Why no online LLM agent now:

- Slow and expensive.
- Adds nondeterminism to cache maintenance.
- Makes experiments harder to reproduce.
- Can hide algorithm weaknesses behind prompting.
- Single-hop temporal should not need an LLM.

Where LLM is acceptable:

- Offline query type labels for error analysis.
- Offline relation extraction to build a richer graph cache.
- Optional cached query decomposition, never live per-window decisioning.

This matters because the paper needs an algorithm, not a prompt pipeline wearing a lab coat.

## Proposed DRIP cache manager v2

New modules under `algorithms/drip/cache_manager/`:

```text
query_router.py      new
graph_index.py       new
bridge.py            replace internals
ledger.py            split direct/bridge/temporal channels
writer.py            typed budget arbitration
observer.py          route-aware target slots
```

Pipeline:

```text
W_t
 │
 ▼
QueryRouter
 │        ┌──────── SINGLE ──────── direct demand only
 │        ├──────── MULTI_DIRECT ── top-2 support demand
 │        └──────── BRIDGE ──────── graph path candidates
 ▼
CoverageObserver
 ▼
TypedEvidenceLedger
 │
 ├── direct_demand[d]
 ├── bridge_demand[d]
 └── serve[d]
 ▼
BudgetedWriter
 │
 ├── reserve direct writes for single-hop repair
 ├── reserve bridge writes for second-hop repair
 └── evict low serve / redundant residents
 ▼
C_{t+1}
```

Typed writer scoring:

```text
candidate_score(d) =
    w_direct  * direct_demand[d]
  + w_bridge  * bridge_demand[d]
  + w_temporal * temporal_demand[d]

resident_value(d) =
    serve[d]
  + resident_demand[d]
  - redundancy_penalty[d]
```

Budget arbitration:

```text
budget_t = min(WRITE_CAP, #under_covered_queries * target_slots)

direct_budget = queries_routed_SINGLE_OR_MULTI_DIRECT
bridge_budget = queries_routed_BRIDGE

unused budget can spill over, but bridge cannot consume all direct budget.
```

This directly addresses the single-hop/multi-hop balance.

## Test Review

Current test coverage for `algorithms/drip/cache_manager/` is effectively missing.

Coverage diagram:

```text
CODE PATH COVERAGE
==================
[+] CoverageObserver
    ├── [GAP] no entity metadata -> target_slots = 1
    ├── [GAP] entity metadata + bridge route -> target_slots = 2
    └── [GAP] empty KB behavior

[+] EvidenceLedger
    ├── [GAP] serve credit only for hit docs
    ├── [GAP] direct demand for under-covered queries
    └── [GAP] decay prunes low evidence

[+] EntityBridgeRouter
    ├── [GAP] builds doc/entity inverted index
    ├── [GAP] ignores generic/high-degree entities
    ├── [GAP] excludes first-hop A from second-hop B
    ├── [GAP] scores rare shared entity path
    └── [GAP] no metadata -> no bridge writes

[+] BudgetedWriter
    ├── [GAP] respects WRITE_CAP
    ├── [GAP] skips duplicate candidates
    ├── [GAP] evicts lowest resident value
    └── [GAP] typed budgets preserve direct writes

COVERAGE: 0/15 paths tested
```

Minimum tests to add before trusting results:

| Test file | What it should prove |
|---|---|
| `algorithms/drip/tests/test_cache_manager_router.py` | single vs bridge routing does not make all entity datasets multi-hop |
| `algorithms/drip/tests/test_graph_index.py` | entity index, IDF, high-degree filtering |
| `algorithms/drip/tests/test_bridge_router.py` | path-scored bridge candidate beats generic shared-entity candidate |
| `algorithms/drip/tests/test_budgeted_writer.py` | budget cap, duplicate skip, typed budget reservation |
| `algorithms/drip/tests/test_cache_manager_integration.py` | synthetic single-hop temporal and synthetic bridge both improve coverage |

## Performance Review

Current risk:

```text
bridge.credit()
  under queries * dense full pool matmul
  plus entity expansion over potentially high-degree entities
```

This is okay for experiments, but the graph router must cap high-degree entities.

Recommended limits:

```text
max_entity_degree_for_bridge
max_bridge_candidates_per_query
top_entities_per_first_hop_doc
cache novelty computed with vectorized max_sim
```

Without degree caps, one entity like a country/person/org can flood demand. That is exactly the current failure mode in a quieter outfit.

## Failure Modes

| Failure | Current handling | Needed handling |
|---|---|---|
| Entity extraction misses the true bridge | Falls back to direct demand | Track no-entity route and do not spend bridge budget |
| Generic entity floods candidates | Weak IDF only | Hard degree cap + nonhub penalty |
| Single-hop query in entity dataset gets target_slots=2 | Current observer can do this | QueryRouter route-aware target slots |
| Bridge candidates are redundant with cache | Duplicate threshold only | novelty term before admission |
| Bridge consumes direct budget | Current scalar demand can do this | typed budget reservation |

## NOT in scope

- Full LLM-built KG. Too much machinery before the deterministic graph router is proven.
- Online LLM agent per query. Too slow, too nondeterministic, bad for reproducible experiments.
- Drift detector. It returns later as an epoch controller after DRIP cache manager works.
- Production graph database. In-memory indexed dicts are enough for current corpora.

## Worktree Parallelization

Potential split:

| Step | Modules touched | Depends on |
|---|---|---|
| QueryRouter | `algorithms/drip/cache_manager/` | none |
| GraphIndex + BridgeRouter | `algorithms/drip/cache_manager/` | none |
| TypedLedger + Writer | `algorithms/drip/cache_manager/` | QueryRouter contracts |
| Tests | `algorithms/drip/tests/` | all above |
| Experiment docs | `docs/design/`, `motivation/` | results |

Because the core work all touches the same module, implementation should be mostly sequential. Parallelization is useful only for docs/experiments after the contracts are stable.

## Recommendation

Build **DRIP cache manager v2** before running more result-chasing experiments:

1. Add deterministic `QueryRouter`.
2. Add lightweight `GraphIndex`.
3. Replace `EntityBridgeRouter` overlap scoring with path + novelty + complementarity scoring.
4. Split ledger channels into direct and bridge evidence.
5. Make writer reserve budget by route/channel.
6. Add synthetic tests before running 50w experiments.

This is not over-engineering. This is the minimum structure needed to explain why the cache writes the right document, not just that it wrote something.
