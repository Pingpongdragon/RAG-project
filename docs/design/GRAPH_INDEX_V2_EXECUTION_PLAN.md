# GraphIndex v2 Execution Plan: Relation-Aware Bridge Evidence

Date: 2026-06-15
Branch: arc-baseline-and-detectors
Status: execution draft

## 0. Goal

Fix the bridge channel before making the full DRIP claim.

Current DRIP cache manager already has the right high-level split:

```text
QueryRouter -> EmbeddingIndex or GraphIndex -> evidence ledger -> writer
```

The failure is narrower:

```text
2Wiki bridge route accuracy:       5000 / 5000 correct
first-hop direct gold signal:      about 55.0%
bridge candidate gold hit rate:    1543 / 88491 = 1.7%
```

So the next plan is not "better routing" and not "more dense top-k". The work is:

```text
entity-doc GraphIndex
  -> relation-aware path GraphIndex
  -> safer bridge evidence credit
  -> benchmark bridge precision before claiming recall gains
```

## 1. CEO Premise Challenge

The right problem is not "build a memory system like Mem0".

Mem0 manages conversational memory: it extracts, consolidates, updates, and retrieves compact memories from long user-agent conversations. It also has a graph-memory variant and evaluates on LOCOMO, which measures long-term conversational memory across single-hop, temporal, multi-hop, and open-domain questions.

Our system manages a bounded RAG hot tier: the cold corpus is fixed, and the cache unit is an original evidence document or chunk. We do not synthesize user memories. We decide which existing chunks become resident in a small low-latency index.

Borrow the lifecycle, not the object model.

```text
Mem0:
  conversation -> extract memory -> consolidate/update -> retrieve memory

DRIP:
  query miss/retrieval trace -> attribute evidence demand -> admit/prefetch chunks -> retrieve from hot tier
```

Source anchors:

- Mem0 paper: https://arxiv.org/abs/2504.19413
- LOCOMO paper: https://arxiv.org/abs/2402.17753

## 1.5 Reviewer Objection: Why Not Let the Agent Rewrite the Query?

A reader will ask:

```text
If q only mentions A, shouldn't an agent reason over q, rewrite/decompose it,
then retrieve B directly from the cold corpus? Why call this prefetch?
```

The answer is: yes, an agent can do that for the current query. That is reactive retrieval, not cache management.

DRIP is solving the next layer down:

```text
reactive agent retrieval:
  q -> think/rewrite/decompose -> cold-tier retrieve B -> answer this query

proactive hot-tier management:
  q/retrieval trace reveals demand for A->B
  -> admit B into shared hot tier
  -> future similar queries and later agent-loop steps hit RAM instead of cold tier
```

These are complementary. Query rewriting decides what to fetch now. DRIP decides what should stay warm after that signal appears.

This distinction matters because the hot tier has a fixed budget. Even if the agent rewrites perfectly, the system still needs to answer:

```text
Which fetched/revealed chunks become resident?
Which resident chunks should be evicted?
How much bridge evidence should be trusted?
How do we avoid writing thousands of plausible but useless B candidates?
```

The prefetch is not magic before any signal exists. It is triggered by observed demand:

```text
1. Current query q is under-covered by C_t.
2. Dense retrieval or agent trace exposes first-hop A.
3. A's content exposes possible second-hop B through entities/relations.
4. GraphIndex scores B as reusable bridge evidence.
5. BudgetedWriter admits B only if it beats resident value.
```

So for the first occurrence of a bridge question, the system may still pay cold-tier cost. The win is amortized over a multi-user stream, repeated tasks, follow-up queries, and agent loops where the same bridge evidence becomes useful again.

Recommended paper framing:

```text
LLM query decomposition is a reactive serving strategy: it can recover missing
evidence for the current query, but it repeatedly pays cold-tier retrieval and
does not prescribe which evidence should occupy the bounded hot tier. DRIP uses
the retrieval/decomposition trace as an admission signal, converting one query's
miss into reusable hot-tier evidence for subsequent related demand.
```

Recommended baseline:

```text
OnDemandRewrite:
  For every miss, decompose/rewrite and retrieve from the full corpus.
  Count this as cold-tier retrieval cost.

DRIP:
  Use the same exposed A/B trace only as cache-admission evidence.
  Measure whether later queries hit the hot tier with lower cold-tier pressure.
```

Do not overclaim:

```text
DRIP does not replace agent reasoning.
DRIP reduces repeated cold-tier work after reasoning/retrieval exposes reusable evidence.
```

## 2. What Already Exists

| Sub-problem | Existing code | Reuse? | Notes |
|---|---|---|---|
| Canonical cache manager | `algorithms/drip/cache_manager/__init__.py` | Yes | `DRIPCore.step()` is the main integration point. |
| Dense direct evidence | `algorithms/drip/cache_manager/embedding_index.py` | Yes | Keep as direct channel. |
| Bridge evidence | `algorithms/drip/cache_manager/graph_index.py` | Replace internals | Current entity overlap is too weak. |
| Routing | `algorithms/drip/cache_manager/query_router.py` | Keep | Router is not the bottleneck in 2Wiki rerun. |
| Config knobs | `algorithms/drip/cache_manager/config.py` | Extend | Add relation/path thresholds and caps. |
| Logs | `bridge_log`, `route_log`, `last_admission` | Extend | Add candidate precision diagnostics. |
| Bench runner | `motivation/motivation_2/run.py` | Extend | Already records bridge and route logs. |
| Multi-hop loaders | `motivation/motivation_2/loaders.py` | Extend | Existing HotpotQA, 2Wiki, MuSiQue loaders are enough for primary eval. |

## 3. Current Formula and Failure

Current paper shorthand:

```text
E_G(q, B) = max_A s(q, A) * l(A, B) * n(B | C_t) * c(A, B)
```

Current code behavior in `GraphIndex.graph_evidence()` is closer to:

```text
raw(B) =
  sum over A,e:
      s(q, A)
    * IDF(e) / g(e)^rho
    * n(B | C_t)
    * c(A, B)

E_G(q, B) = bridge_alpha * raw(B) / max_B raw(B)
```

Variables:

| Symbol | Code variable | Meaning |
|---|---|---|
| `q` | `nqe[qi]`, `query` | Incoming query. |
| `A` | `first_hops` | Dense top documents from full corpus. |
| `B` | `linked_docs` candidate | Candidate hidden second-hop document. |
| `e` | `ent` | Entity shared by `A` and `B`. |
| `s(q,A)` | `a_sim` | Cosine similarity from query to first-hop document. |
| `IDF(e)` | `ent_idf[ent]` | Rare entity weight. |
| `g(e)` | `len(ent_to_docs[e])` | Entity degree, number of docs containing entity. |
| `rho` | `entity_degree_power` | Degree penalty exponent. |
| `n(B|C_t)` | `_novelty()` | Larger when B is far from current cache. |
| `c(A,B)` | `_complementarity()` | Larger when B is far from A. |

Why this fails:

```text
shared rare entity => related
related != relation needed by query
```

`novelty` and `complementarity` can accidentally reward noise. The final per-query max normalization then amplifies the strongest noisy candidate into a high evidence value.

## 4. New Formula: Relation-Aware Graph Evidence

Move from entity overlap to path evidence.

Define a bridge path:

```text
p = (A, a, r, b, B)
```

where:

| Symbol | Meaning |
|---|---|
| `A` | first-hop document visible to query |
| `a` | query-visible anchor entity in A |
| `r` | relation or predicate cue extracted near the bridge mention |
| `b` | candidate bridge entity or title |
| `B` | hidden second-hop document |

The bridge score becomes:

```text
E_G(q, B) =
  alpha_G * clip_tau(
    max_A s(q, A)
      * max_{p in P(A,B)} [
          L(p) * R(q,p) * O(p) * N(B | C_t) * C(A,B | q) * H(B)
        ]
  )
```

Do not normalize by the best candidate inside the query. Use an absolute threshold plus clipping:

```text
clip_tau(x) = 0                         if x < tau_bridge_abs
clip_tau(x) = min(1, x / tau_bridge_sat) otherwise
```

Term definitions:

| Term | Meaning | Implementation |
|---|---|---|
| `s(q,A)` | first-hop confidence | Existing dense cosine. |
| `P(A,B)` | valid bridge paths from A to B | Built from entity mentions, titles, and relation cues. |
| `L(p)` | path link strength | `IDF(b) / degree(b)^rho * relation_confidence`. |
| `R(q,p)` | query-path relation alignment | Lexical/dependency cue match between query and path predicate. |
| `O(p)` | orientation match | Does the path direction match query wording, e.g. "spouse of X" vs "X's spouse". |
| `N(B|C_t)` | cache novelty | Keep, but cap it. Novelty is a gate, not a reason by itself. |
| `C(A,B|q)` | evidence complementarity | Require B to add entities or relation types not already covered by A. |
| `H(B)` | anti-hub prior | Penalize generic documents/entities even after entity degree filtering. |

Implementation form can still be a sum for stability:

```text
raw(B) =
  sum over A in TopK_1(q):
    s(q,A) * top_m_path_score(q,A,B)

top_m_path_score(q,A,B) =
  sum of top m valid path scores from P(A,B)

E_G(q,B) = alpha_G * clip_tau(raw(B))
```

Rules:

- Cap paths per `(A,B)` to prevent long generic pages from winning by count.
- Require at least one relation-compatible path unless the dataset has only title/entity metadata.
- Remove per-query max normalization.
- Log the reason each bridge candidate survived.

## 5. New Components

### 5.1 `RelationGraphIndex`

Replace the internals of `GraphIndex`, keep the public name first to avoid a wide diff.

```text
GraphIndex
  doc_to_ents:       doc_idx -> [entity]
  ent_to_docs:       entity -> compact doc_idx list
  ent_idf:           entity -> float
  doc_mentions:      doc_idx -> [Mention]
  path_postings:     anchor/title/entity -> [Path]
  relation_stats:    relation cue -> degree/confidence stats
```

Suggested dataclasses:

```text
Mention:
  entity: str
  start: int
  end: int
  sentence_id: int
  is_title: bool

RelationPath:
  src_doc: int
  dst_doc: int
  anchor_entity: str
  bridge_entity: str
  relation_cue: str
  confidence: float
  degree: int
```

Phase 1 relation extraction should be deterministic:

- title aliases
- same-sentence entity co-occurrence
- dependency-light lexical windows
- cue words from query and sentence: `born`, `spouse`, `director`, `capital`, `located`, `member`, `founded`, `parent`, `child`, `author`, `film`, `album`

Offline LLM extraction is allowed only as an ablation or cache-building variant, not in the online hot path.

### 5.2 `BridgeCandidateGenerator`

Inside `GraphIndex.graph_evidence()`:

```text
for q:
  A = dense_topk(q)
  query_cues = extract_query_relation_cues(q)
  for A_i:
    anchors = entities_in_query_or_title(A_i)
    paths = path_postings[anchors]
    score B by relation-aware formula
  return candidates that pass absolute threshold
```

Candidate gates:

| Gate | Purpose |
|---|---|
| `max_entity_degree` | Drop global hubs. |
| `max_paths_per_pair` | Prevent count-based spam. |
| `min_relation_score` | Keep only paths that match query relation. |
| `min_firsthop_sim` | Avoid building bridges from weak A. |
| `min_bridge_abs_score` | Avoid max-normalized noise. |
| `max_candidates_per_query` | Bound runtime. |

### 5.3 Typed Evidence Ledger

Current `self.demand[p]` merges dense and graph evidence. Split it:

```text
direct_demand[p]
bridge_demand[p]
serve[p]
```

Admission score:

```text
candidate_value(d) =
    w_D * direct_demand[d]
  + w_G * bridge_demand[d]
  + w_S * serve[d]

resident_value(d) =
    serve[d]
  + direct_demand[d]
  + bridge_resident_bonus * bridge_demand[d]
  - redundancy_penalty[d]
```

Why: a noisy bridge batch should not drown direct evidence, and direct single-hop repair should not evict hard-won second-hop support without a margin.

### 5.4 Budgeted Writer

Budget rule:

```text
budget_t = min(WRITE_CAP, sum target_slots of under-covered queries)

direct_budget = count(SINGLE + MULTI_DIRECT under-covered)
bridge_budget = count(BRIDGE under-covered)
```

Spillover is allowed only after the native channel has no qualifying candidate:

```text
direct unused -> bridge allowed
bridge unused -> direct allowed
bridge cannot consume direct budget while direct candidates pass threshold
```

### 5.5 Diagnostics

Add to `bridge_log`:

```text
bridge_candidates_total
bridge_candidates_after_degree_gate
bridge_candidates_after_relation_gate
bridge_candidates_after_abs_threshold
bridge_gold_candidates
bridge_gold_rate
firsthop_gold_rate
no_path_queries
top_noise_entities
top_relation_cues
normalization_mode
```

The PR is not ready unless this log can explain why a bridge write happened.

## 6. System Diagram

```text
query window W_t
  │
  ▼
QueryRouter
  │
  ├── SINGLE ───────▶ EmbeddingIndex ───────▶ direct_demand
  │
  ├── MULTI_DIRECT ─▶ EmbeddingIndex ───────▶ direct_demand
  │
  └── BRIDGE ───────▶ EmbeddingIndex top A
                       │
                       ▼
                    GraphIndex v2
                       │
                       ├── doc_to_ents
                       ├── ent_to_docs
                       ├── doc_mentions
                       └── relation paths
                       │
                       ▼
                    bridge_demand

direct_demand + bridge_demand + serve
  │
  ▼
BudgetedWriter
  │
  ▼
C_{t+1}
```

## 7. Data Flow and Shadow Paths

```text
doc_pool
  -> entity extraction
  -> mention extraction
  -> relation path extraction
  -> path index
  -> bridge scoring
  -> typed demand
  -> write
```

| Path | Expected behavior |
|---|---|
| Nil entity metadata | GraphIndex reports `has_metadata=False`; router falls back to dense/direct. |
| Empty entity list for A | Bridge query logs `no_path_queries += 1`; no bridge credit. |
| Entity extraction error | Skip graph build for that document, log doc id and exception type. |
| High-degree entity | Filter before candidate scoring. |
| No relation cue | Use entity-only fallback with lower `alpha_G_fallback`, logged separately. |
| All candidates below threshold | No bridge writes. This is better than writing normalized noise. |
| Candidate already in cache | Count as covered/serve signal, not as a write candidate. |

## 8. Benchmarks

### 8.1 Primary Benchmarks

| Benchmark | Purpose | Required outcome |
|---|---|---|
| 2Wiki bridge/compositional | Main hidden second-hop stress test | Bridge candidate gold rate must rise before claiming Recall@5 gain. |
| HotpotQA comparison/direct | Direct multi-hop regression guard | No loss from typed budgets or bridge gates. |
| StreamingQA | Single-hop drift guard | Bridge channel should remain off or harmless. |
| MuSiQue | Longer compositional stress | Report separately; do not make it the first success criterion. |

Core metrics:

```text
Recall@5 H1/H2
KB coverage H1/H2
bridge candidate gold rate
bridge write gold rate
first-hop gold rate
update_cost
maint_retrieval_cost
serve_retrieval_cost
route match when labels exist
```

A bridge improvement is credible only if:

```text
bridge_candidate_gold_rate increases
bridge_write_gold_rate increases
H2 KB coverage increases
Recall@5 does not trade off by burning too many writes
```

### 8.2 LOCOMO as Supplemental Benchmark

Use LOCOMO to connect to agent-memory literature, not as the main result.

Adaptation:

```text
LOCOMO conversation sessions
  -> fixed cold corpus of evidence chunks
      option A: raw turns
      option B: session chunks
      option C: event/fact chunks if annotations expose them
  -> query stream ordered by session/time
  -> cache unit = original chunk, not synthesized memory
```

Evaluate by question type:

```text
single-hop
temporal
multi-hop
open-domain
```

Baselines:

| Baseline | Why |
|---|---|
| Full context | Upper-bound cost/latency reference. |
| RAG top-k over all chunks | Retrieval baseline. |
| DRIP-Dense | Query-visible semantic demand admission baseline. |
| DRIP v1 GraphIndex | Current entity-doc bridge baseline. |
| DRIP v2 RelationGraphIndex | Proposed method. |
| Mem0 or Mem0-style memory retrieval | Optional related-work comparison, only if integration is cheap and reproducible. |

Report:

```text
evidence recall@k
answer accuracy if generation is added
p95 retrieval latency
logical cold-tier fetches/query
token cost proxy
cache writes
```

Important framing:

```text
LOCOMO tests long-term conversational memory.
DRIP tests bounded evidence-chunk hot-tier management.
```

If LOCOMO is used, call it a supplemental agent-memory transfer setting.

## 9. Implementation Phases

### Phase 0: Instrument Current Failure

Files:

- `algorithms/drip/cache_manager/graph_index.py`
- `algorithms/drip/cache_manager/__init__.py`
- `motivation/motivation_2/run.py`

Tasks:

- Add bridge candidate precision counters.
- Add reason codes for degree gate, novelty gate, complementarity gate, threshold gate.
- Save top noisy entities and top noisy candidate titles.

Exit criteria:

```text
A 2Wiki bridge run can explain where 88491 candidates came from.
```

### Phase 1: RelationGraphIndex Data Model

Files:

- `algorithms/drip/cache_manager/graph_index.py`
- `algorithms/drip/cache_manager/config.py`

Tasks:

- Add mention/path data structures.
- Build relation paths offline from doc text and titles.
- Keep old `doc_to_ents`, `ent_to_docs`, and `ent_idf` for fallback.

Exit criteria:

```text
Synthetic test: A contains relation cue to B; noisy same-entity C exists; B ranks above C.
```

### Phase 2: Relation-Aware Bridge Scoring

Files:

- `algorithms/drip/cache_manager/graph_index.py`

Tasks:

- Implement `R(q,p)`, `O(p)`, `H(B)`.
- Replace per-query max normalization with absolute threshold and saturation.
- Add candidate caps.

Exit criteria:

```text
2Wiki bridge candidate gold rate improves materially over 1.7%.
```

### Phase 3: Typed Ledger and Writer

Files:

- `algorithms/drip/cache_manager/__init__.py`
- optional `algorithms/drip/cache_manager/__init__.py`
- optional `algorithms/drip/cache_manager/__init__.py`

Tasks:

- Split direct and bridge demand.
- Add channel budgets.
- Protect single-hop and direct multi-hop writes from bridge noise.

Exit criteria:

```text
StreamingQA and HotpotQA direct do not regress beyond a small tolerance.
```

### Phase 4: Benchmark Harness

Files:

- `motivation/motivation_2/loaders.py`
- `motivation/motivation_2/run.py`
- optional `motivation/motivation_3_locom_cache/`

Tasks:

- Add bridge precision table output.
- Add LOCOMO-cache loader if dataset is available locally.
- Keep LOCOMO supplemental.

Exit criteria:

```text
One command reproduces primary 2Wiki/Hotpot/StreamingQA evidence.
Optional command runs LOCOMO-cache adaptation.
```

### Phase 5: Paper Integration

Files:

- `motivation/motivation.tex`
- `docs/design/DRIP_ALGORITHM_EXPLAINED.md`
- `docs/design/EXPERIMENT_PROTOCOL.md`

Tasks:

- Replace `[Method]` with final name only after bridge precision improves.
- Add Mem0/LOCOMO related-work paragraph.
- Keep clear distinction between memory management and evidence hot-tier cache.

## 10. Tests

Minimum tests:

| Test file | What it proves |
|---|---|
| `algorithms/drip/tests/test_graph_index_paths.py` | Relation path beats generic shared entity. |
| `algorithms/drip/tests/test_graph_index_thresholds.py` | No per-query max normalization writes noise. |
| `algorithms/drip/tests/test_cache_manager_typed_ledger.py` | Bridge demand cannot consume all direct write budget. |
| `algorithms/drip/tests/test_cache_manager_router.py` | Single-hop stays single even when entity metadata exists. |
| `algorithms/drip/tests/test_cache_manager_integration.py` | Synthetic bridge improves H2 without hurting synthetic single-hop. |

Hostile QA tests:

- A very high-degree entity appears in all docs.
- A candidate is novel but relation-mismatched.
- A candidate is complementary but not answer-bearing.
- Entity metadata exists but every query is single-hop.
- First-hop A is wrong; GraphIndex should not fan out aggressively.

## 11. Error and Rescue Registry

| Codepath | What can go wrong | Rescue action | User/researcher sees |
|---|---|---|---|
| Graph build | Missing `pool_ents` | Disable graph route | `route.reason=no_graph_metadata`. |
| Entity normalization | Non-string entity | Convert to string, skip empty | Count in build diagnostics. |
| Mention extraction | Regex/dependency parser fails | Fall back to title/entity index | Warning with doc id. |
| Relation path build | Too many paths | Apply caps | `paths_truncated` metric. |
| Bridge scoring | No valid path | Return no candidates | `no_path_queries` metric. |
| Candidate scoring | All scores below threshold | Return no candidates | No write, no silent max normalization. |
| LOCOMO loader | Dataset missing | Skip supplemental benchmark | Clear message, primary eval unaffected. |

## 12. Failure Modes Registry

| Codepath | Failure mode | Rescued? | Test? | Logged? |
|---|---|---|---|---|
| GraphIndex v2 | Relation extractor misses true bridge | Yes, dense fallback | Yes | Yes |
| GraphIndex v2 | Generic hub entity dominates | Yes, degree and anti-hub gates | Yes | Yes |
| Bridge scoring | Novel noise wins | Yes, relation gate and abs threshold | Yes | Yes |
| Writer | Bridge evicts direct support | Yes, typed budgets | Yes | Yes |
| Router | Single-hop routed bridge | Partial, typed budgets reduce harm | Yes | Yes |
| LOCOMO adapter | Measures memory synthesis instead of chunk caching | Avoided by design | Doc test/manual audit | Yes |

## 13. Not In Scope

- Online LLM relation classification in the hot path.
- Production graph database.
- Arbitrary-length graph reasoning or PPR for every query.
- Synthesizing memories from conversations.
- Claiming LOCOMO as the main benchmark.
- Solving all multi-hop types before fixing one-hop bridge precision.

## 14. Recommended Approach

Use selective expansion:

```text
baseline scope:
  fix GraphIndex bridge precision for 2Wiki

accepted expansion:
  add Mem0/LOCOMO as related-work framing and optional supplemental benchmark

not accepted by default:
  full memory system
  online LLM graph construction
```

This is the clean path:

```text
smallest useful diff:
  GraphIndex internals + diagnostics

best 12-month direction:
  evidence graph hot-tier manager that can absorb relation paths, temporal cues,
  and agent traces without becoming a conversational memory product
```

## 15. Execution Commands

Primary smoke:

```bash
python -m compileall -q algorithms/drip/cache_manager algorithms/cache/registry.py
```

Primary bridge run:

```bash
python motivation/motivation_2/run.py \
  --expanded \
  --datasets 2wikimultihopqa \
  --q-type bridge_comparison \
  --n-source 5000 \
  --n-windows 20 \
  --window-size 25 \
  --strategies DRIP-Dense DRIP Oracle \
  --output graph_index_v2_2wiki_bridge.json
```

Regression run:

```bash
python motivation/motivation_2/run.py \
  --expanded \
  --datasets hotpotqa \
  --q-type comparison \
  --n-source 5000 \
  --n-windows 20 \
  --window-size 25 \
  --strategies DRIP-Dense DRIP Oracle \
  --output graph_index_v2_hotpot_comparison.json
```

Single-hop guard remains in Motivation 1 / StreamingQA.

## 16. Completion Bar

Do not update the paper claim until all are true:

```text
[ ] bridge candidate gold rate materially above 1.7%
[ ] bridge write gold rate materially above current baseline
[ ] 2Wiki H2 KB coverage improves
[ ] 2Wiki Recall@5 improves or cold-tier pressure drops at equal recall
[ ] Hotpot direct/comparison does not regress
[ ] StreamingQA single-hop does not regress
[ ] logs explain every bridge write channel
[ ] LOCOMO is clearly labeled supplemental if used
```
