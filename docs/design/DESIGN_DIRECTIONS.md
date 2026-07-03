# Design Directions After Motivation 1/2

This file is the place for algorithm design directions. The paper / README text should stay empirical and high-level; concrete mechanism proposals live here.

## 1. What the current 100w motivation runs establish

### 1.1 Single-hop demand signal is real, but simple-case gaps split in two
- **mo1 / HotpotQA-comparison (`100x50`)** is the clean single-hop positive example.
  - Sudden H2: DRIP-Dense = **32.4 / 29.8** vs LogDriven 20.6 / 18.6.
  - Gradual H2: DRIP-Dense = **49.2 / 43.1** vs LogDriven 37.7 / 31.8.
- **Scale-matched SQuAD (`100x50`)** is positive under sudden drift but not a full win under gradual drift.
  - Target setting: `n_source=30000`, realized pool = **20,958**, KB = **5,000**.
  - Sudden H2: DRIP-Dense = **27.2 / 21.4** vs LogDriven 18.3 / 13.4.
  - Gradual H2: DRIP-Dense = 33.4 / 25.9 vs Static 54.0 / 40.8, DocArrival 50.3 / 38.3, LogDriven 30.4 / 21.8.
- Conclusion: direct-evidence regimes still confirm that query-demand signal is useful, but long gradual streams expose a separate **capacity-allocation / eviction** problem even without multi-hop bridge complexity.

### 1.2 mo2 — QDC wins but bridge gap vs OnDemandFetch persists
- **DRIP-Dense is the strongest persistent writer on all three datasets** under both drift modes.
- **HotpotQA-expanded** (`pool=54,862`, `KB=14,000`, 26%):
  - DRIP-Dense = **11.0** sudden / **30.9** gradual.
  - KnowledgeEdit = 9.4 / 22.1; LogDrivenArrival = 9.2 / 24.2.
  - OnDemandFetch = 60.7 / 60.5 → large persistent gap.
- **2Wiki-expanded** (`pool=29,253`, `KB=7,000`, 24%):
  - DRIP-Dense = **9.5** sudden / **24.1** gradual.
  - KnowledgeEdit = 8.3 / 15.6; LogDrivenArrival = 8.1 / 15.8.
  - OnDemandFetch = 66.1 / 55.9 → large persistent gap.
- **MuSiQue-expanded** (`pool=43,017`, `KB=11,000`, 26%):
  - DRIP-Dense = **17.6** sudden / **24.7** gradual.
  - KnowledgeEdit = 7.7 / 14.9; LogDrivenArrival = 6.9 / 15.4.
  - OnDemandFetch = 43.7 / 42.1 → large persistent gap.
- Conclusion: after the single-hop positives are established, the demand signal helps even in multi-hop settings, but the main remaining failure mode is **bridge-aware persistent acquisition under low cross-query reuse** (gap vs OnDemandFetch, not gap vs baselines).

## 2. Gap taxonomy

### G-1 Bridge identification gap
A failed multi-hop query usually does not reveal which reusable bridge document or bridge-document bundle should be written into the KB for future queries. The current cluster centroid is too coarse for that decision.

### G-2 Same-window causality gap
Persistent writers update only after the window has been scored, so they cannot repair the same failing window. OnDemandFetch's advantage is largely this same-window access to bridge evidence.

### G-3 Fetch-to-persist handoff gap
OnDemandFetch often finds useful bridge documents, but that information is not handed to the persistent KB writer. Retrieval and persistence are solving the same problem with disconnected signals.

### G-4 Regime-routing gap
High-reuse direct-evidence regimes (Hotpot-comparison, sudden scale-matched SQuAD) and sparse-bridge multi-hop regimes should not use the same policy. Current QDC is too monolithic.

### G-5 Capacity-allocation / eviction gap
Even when the signal is useful, the writer still needs to protect sticky head content while reserving budget for new tail evidence. In the current runs, this is clearest in the `100x50` gradual SQuAD probe with realized pool 20,958 and KB 5,000, where DRIP-Dense beats LogDrivenArrival but still trails Static / DocArrival.

## 3. Algorithm design directions

### D-1 Bridge-aware bundle memory
Maintain a per-cluster memory of document bundles that historically satisfied queries from that cluster, instead of relying only on centroid-to-doc nearest neighbors. New queries first look up this bundle memory, then fall back to embedding retrieval.

Addresses: G-1.

### D-2 Fetch-to-persist handoff
When OnDemandFetch retrieves a document that verifies a failed query, expose that document to the persistent writer with a special high-confidence admission path, possibly TTL-gated. This shrinks the gap between same-window recovery and future-window persistence.

Addresses: G-2, G-3.

### D-3 Reuse-aware routing
Learn or estimate whether a cluster / query is in a high-reuse direct-evidence regime or a sparse-bridge regime. Route high-reuse queries through persistent QDC-style maintenance and low-reuse ones through fetch-heavy behavior.

Addresses: G-4.

### D-4 Two-stage bridge prediction
Replace single-shot cluster centroid retrieval with a two-stage policy:
1. predict likely bridge entity / bridge title candidates
2. condition second-hop bundle prediction on that bridge

Addresses: G-1.

### D-5 Budget partitioning
Split the KB budget into at least three logical compartments:
- sticky reserve for proven high-reuse evidence
- exploratory / bridge cache for low-reuse fetch discoveries
- drift buffer for newly emerging clusters

Addresses: G-5, and partially G-3.

### D-6 Drift-aware eviction throttle
Protect clusters that are still actively queried and only relax eviction when a cluster's mass is clearly decaying. This is the most direct stability fix for long gradual streams like the current scale-matched SQuAD run.

Addresses: G-5.

## 4. What should stay out of the main motivation text
- Fine-grained mechanism proposals
- Routing policies and threshold details
- Fetch-to-persist TTL rules
- Bundle-memory data structures
- Budget partition heuristics

Those belong here, not in the main README / tex story.

## 5. Current recommended narrative split
- **mo1 README / tex**: use HotpotQA-comparison as the main positive single-hop example, use sudden scale-matched SQuAD as corroborating evidence, and describe gradual scale-matched SQuAD as a capacity-allocation caution rather than a bridge gap.
- **mo2 README / tex**: say the main remaining gap is complex bridge / bundle acquisition under low reuse, not a generic failure of demand signal.
- **DESIGN_DIRECTIONS.md**: keep all concrete algorithmic follow-ups here.
