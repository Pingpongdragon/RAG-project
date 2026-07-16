# DRIP Topic-State Results (2026-07-16)

## Frozen conclusion

The cold-corpus topic layer is now integrated as a **demand-state encoder**.
It maps completed evidence accesses to a topic histogram, learns recurrent
histogram states, and proposes concrete documents from causal
`state -> historically accessed document` tables. It does **not** load an
entire topic partition and does not predict documents that have never appeared
in the prefix.

The evidence supports a conditional result:

- DRIP-Reactive is strong on the natural, timestamp-ordered MIND access trace.
- DRIP-TopicState improves residency on predictable recurring drift whose
  within-state documents recur.
- Its empirical Hedge gate shuts the predictive branch off on shuffled,
  one-shot, gradual, stationary, and MIND traces under the frozen default.
- Falling back to Reactive is not a guarantee of beating every conventional
  cache: LRU is much stronger on the easy one-shot, gradual, and stationary
  controls.

## Protocol integrity

### MIND natural trace

- Fixed corpus: 51,282 news objects.
- Natural timestamp order; no topic-based stream reordering.
- Warm-up: 3 x 500 events. Evaluation: 100 x 500 events.
- Capacity `B=144` comes from the calibration working-set statistic rounded to
  a 16-object systems multiple; `W=9`, `M=72`.
- The clicked-news ID is revealed only after current-request scoring and may
  update the next window.
- The first 20 evaluation windows select Reactive write controls; the last 80
  are the temporal holdout.

### Controlled SQuAD evidence-demand trace

- Fixed corpus: 1,000 paragraphs.
- 4,640 source queries become 4,630 after removing 10 content-identical
  duplicates.
- A SHA-256 content split assigns connected support-family components to a
  calibration pool (2,206 queries / 481 supports) or test pool (2,424 / 519).
- Exact query overlap = 0, support overlap = 0, family overlap = 0.
- Five calibration streams use seeds 11--15; five reported test streams use
  seeds 42--46. Each stream has one 25-query warm-up window and 20 x 25 test
  queries, with no repeated exact query.
- The five calibration warm-ups each contain 25 unique evidence objects. We
  round upward to `B=32`, use `W=4`, and fix proposal budget `M=32`.
- Calibration selects `replacement_target=0.25`, `initial_price=0` from the
  predeclared grid. The objective is hit-first, writes as tie-break; it is not a
  weighted total-read objective.
- Semantic pages are built independently from dense document embeddings. The
  online policy never sees workload/regime labels. Gold support IDs are exposed
  only after current-window scoring, so this is an **oracle evidence-demand
  residency trace**, not end-to-end RAG retrieval.

## Results

### MIND temporal holdout

| Method | Last-80 hit rate | Last-80 writes |
|---|---:|---:|
| DRIP-Reactive | **76.14%** | 720 |
| DRIP-TopicState | **76.14%** | 720 |
| LRU | 67.70% | 720 |
| Classical ARC | 66.56% | 720 |
| FIFO | 60.81% | 720 |
| AgentRAGCache | 58.74% | 134 |
| TinyLFU | 34.89% | 720 |

DRIP-Reactive improves over LRU by 8.44 percentage points under equal capacity
and writes. TopicState is exactly equal to Reactive because the gate activates
in zero windows at the frozen document-score decay of 0.5.

The diagnostic explains why. Previous-window subcategory prediction obtains
80% top-1 accuracy versus 10% for a static predictor, but the resulting
topic-conditioned document proposal recall is 67.24%, below global-frequency
recall of 68.32%. Accurate topics do not imply accurate documents.

### Held-out SQuAD recurring drift (five seeds)

All methods share `B=32` and per-window write cap `W=4`. Values are means across
the five held-out construction seeds.

| Method | Evidence / strict hit | Writes | Sync cold misses | Proactive doc fetches | Total doc reads |
|---|---:|---:|---:|---:|---:|
| **DRIP-TopicState** | **32.76%** | 50.6 | 336.2 | 40.0 | 376.2 |
| DRIP-Reactive | 29.28% | 23.2 | 353.6 | 0 | 353.6 |
| FIFO | 28.72% | 76.0 | 356.4 | 0 | 356.4 |
| LRU | 28.12% | 76.0 | 359.4 | 0 | 359.4 |
| AgentRAGCache | 25.08% | 60.0 | 374.6 | 0 | 374.6 |
| Classical ARC | 23.80% | 67.4 | 381.0 | 0 | 381.0 |
| TinyLFU | 22.52% | 64.0 | 387.4 | 0 | 387.4 |

TopicState exceeds Reactive by 3.48 points (paired 95% t interval
`[2.765, 4.195]`, `p=0.000174`) and the strongest traditional policy, FIFO, by
4.04 points (`[3.562, 4.518]`, `p=0.0000195`). These intervals quantify
construction-seed variation within one fixed held-out source pool.

The gain is not free. Relative to FIFO, TopicState performs 33.4% fewer writes
but 5.6% more total document reads. Relative to Reactive, it uses both more
writes and more reads in exchange for higher residency. Across seeds the state
predictor is correct on 9/10 scored predictions, activates in 10 windows, and
40 speculative fetches yield 32.8 unique later promotions on average (82%).

### Schedule controls with the same frozen configuration

| Schedule | TopicState | Reactive | Best conventional | Topic active windows |
|---|---:|---:|---:|---:|
| recurring 4-state cycle | **32.76%** | 29.28% | FIFO 28.72% | 10/20 |
| shuffled marginals | **26.76%** | **26.76%** | FIFO 25.44% | 0/20 |
| one-shot shift | 43.72% | 43.72% | **LRU 70.28%** | 0/20 |
| gradual shift | 46.72% | 46.72% | **LRU 81.04%** | 0/20 |
| stationary | 52.28% | 52.28% | **LRU 79.76%** | 0/20 |

This supports a recurring-drift specialization, not universal non-stationary
SOTA. The predictive branch is a safe empirical fallback relative to its own
Reactive backbone, not a formal no-harm guarantee against every baseline.

### Capacity and proposal sensitivity

The capacity sweep fixes `M=32`, preserves `W/B=1/8`, and does not retune the
policy on test data.

| Capacity | TopicState | Best other method |
|---:|---:|---:|
| 8 | **8.08%** | FIFO 7.72% |
| 16 | **16.28%** | FIFO 15.40% |
| 32 | **32.76%** | DRIP-Reactive 29.28% |
| 64 | **53.52%** | LRU 49.64% |

At `B=32`, proposal budgets `M={8,16,32,64}` yield
`{32.00, 32.76, 32.76, 32.84}%`; the main result is not a single exact-M spike.
Changing the semantic-partition seed over `{7,19,42,73,101}` at workload seed
42 gives 33.0% for four seeds and 33.2% for one.

### MTRAG external-validity diagnostic

MTRAG contains 366,479 passages and 777 qrel-bearing ordered queries. Only
about 5.15% of full query support sets have been completely seen earlier, so it
is a low-reuse stress test.

- Natural conversation round-robin: TinyLFU any-support hit is 1.47%, while the
  trace-level CausalDomainState realization is 0.57%. Its topic gate closes and
  it falls back to LRU, which is weaker than TinyLFU.
- Controlled recurring-domain scheduling: CausalDomainState reaches 7.24%
  any-support hit versus 4.15% for TinyLFU, but adds 362.4 writes on average and
  only 10.91% of its proactive fetches are later useful.

The MTRAG controller is an embedding-free trace-level realization of the same
state-to-document idea, not a direct invocation of `algorithms/drip/policy.py`,
so it is a diagnostic rather than the main matched-budget result.

## Exact implementation semantics

1. `SemanticTopicPartition` / `MetadataTopicPartition` encode the full cold
   corpus and expose topic membership.
2. `RecurrentStateDocumentForecaster` clusters completed-window histograms,
   learns causal state transitions, and maintains decayed global and
   state-conditioned scores for previously observed document IDs.
3. Delayed-feedback Hedge compares virtual global/state candidate occurrence
   recall. Activation combines accumulated state-expert weight and transition
   confidence.
4. Speculative writes share total capacity and `W` with Reactive, occupy at
   most a fixed fraction of the cache, are promoted on use, and prefer stale
   speculation as victims for later speculative admissions.
5. Speculative writes are hard-budgeted but not shadow-price gated. The dual
   price directly gates only residual Reactive swaps; this is not one unified
   switching-cost optimizer.
6. Topic proposals are in-memory metadata/history lookups. They are not counted
   as cold-index probes or document reads. Physical document reads are
   synchronous misses plus actual proactive fetches.

## Reproducibility artifacts

- Active code: `algorithms/drip/{config,controller,index,topic_partition,topic_state,policy}.py`
- Controlled runner/tuner: `benchmarks/run_controlled_topic_trace.py` and
  `benchmarks/tune_controlled_topic.py`
- Calibration:
  `experiments/hidden/data/tune_squad_topic_disjoint_b32_w4_cal11_15_test42_46.json`
- SQuAD main seeds:
  `experiments/hidden/data/formal_squad_disjoint_b32_w4_m32_test_s42.json`
  through `s46.json`
- SQuAD controls: `control_squad_disjoint_factorized_*_b32_w4_m32_test_s*.json`
- Capacity: `capacity_squad_disjoint_b{8,16,64}_w*_m32_test_s*.json`
- Candidate sensitivity: `candidate_squad_disjoint_b32_w4_m*_test_s*.json`
- MIND:
  `experiments/agent/data/formal_mind_prefix_b144_w9_m72_decay05_s42.json`
- MTRAG: `mtrag_exact_round_robin_s*.json` and
  `mtrag_exact_recurring_domain_s*.json`

The old B=24/pre-disjoint SQuAD artifacts were removed from the active data
directory because they predate source-family isolation and must not be used in
a paper table.
