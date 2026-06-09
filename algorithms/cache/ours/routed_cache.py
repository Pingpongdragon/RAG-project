"""RoutedCache: SemFlow + entity-chained bridge prefetch (ours)."""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.ours.query_driven import QueryDriven

class RoutedCache(QueryDriven):
    """DRYAD admission backend (R1 + R3), WITHOUT the explicit drift
    detection / decision layer. This is the ablation "DRYAD w/o detect".

    Builds on SemFlow (QueryDriven) and adds an **entity-chained prefetch**
    repair routine (R3) for bridge multi-hop misses, the regime where the
    second-hop evidence B is unreachable from the query embedding because
    the query only mentions the first-hop entity A.

    Per failing query, in addition to SemFlow's query-neighborhood demand:
      R3 (bridge): take the query's top step-1 documents A_i (dense), read
        their entities, look up pool docs B_j that share those entities, and
        credit B_j with demand proportional to entity-overlap count. This is
        the *write-side* analogue of recall_at_k_entity_expand: instead of
        only ranking at retrieval time, it warms the hidden bridge doc into
        the persistent KB so future queries can retrieve it for free.

    All R1/R3 candidates flow into the SAME demand/serve ledger and compete
    under SemFlow's unified admission gate, so the KB budget stays bounded
    and bridge prefetch only wins when an entity is repeatedly needed.

    pool_ents ({doc_id: [entity str]}) is injected by run.py via the
    `_pool_ents` attribute. If it is absent, R3 is a no-op and RoutedCache
    degrades exactly to SemFlow — making the ablation trivial.
    """
    # R3 hyperparameters
    R3_STEP1_K     = 3      # how many first-hop docs A_i to read per miss
    R3_ALPHA       = 0.6    # bridge demand weight per unit entity overlap
    R3_MAX_BRIDGE  = 20     # cap bridge candidates per miss (cost control)
    R3_MIN_ENT_LEN = 3      # ignore very short / noisy entity strings

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self._pool_ents = None     # injected by run.py: {doc_id: [ent,...]}
        self._ent_index = None     # entity -> set(pool_idx), built lazily
        self._doc_ents_pi = None   # pool_idx -> [ent,...]

    def _build_entity_index(self):
        """Build entity -> pool-idx inverted index once (pool-wide)."""
        if self._ent_index is not None or not self._pool_ents:
            return
        from collections import defaultdict
        idx = defaultdict(set)
        doc_ents_pi = {}
        for did, ents in self._pool_ents.items():
            pi = self.d2p.get(did)
            if pi is None:
                continue
            norm = [e.lower().strip() for e in ents
                    if len(e.strip()) >= self.R3_MIN_ENT_LEN]
            doc_ents_pi[pi] = norm
            for e in norm:
                idx[e].add(pi)
        self._ent_index = idx
        self._doc_ents_pi = doc_ents_pi

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        # --- R3 entity-chained prefetch: credit hidden bridge docs B_j ---
        # Done BEFORE the SemFlow gate so bridge demand competes in the same
        # ledger / same window. Pure additive: SemFlow's own demand updates
        # still happen inside super().step().
        self._build_entity_index()
        if self._ent_index:
            kb_list = sorted(self.kb)
            kb_idx = np.array([self.d2p[d] for d in kb_list])
            kb_emb = self.doc_embs[kb_idx]
            norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
            nqe = window_query_embs / np.clip(norms, 1e-10, None)
            q_kb = nqe @ kb_emb.T
            max_s = np.max(q_kb, axis=1)
            fail = max_s < _P.SF_HIT_THRESH
            n_fail = int(fail.sum())
            if n_fail > 0:
                fqe = nqe[fail]
                # First hop: top step-1 docs over the full pool (A_i).
                pool_sims = fqe @ self.doc_embs.T
                s1k = self.R3_STEP1_K
                self.maint_retrieval_cost += n_fail * s1k
                from collections import defaultdict
                for qi in range(n_fail):
                    row = pool_sims[qi]
                    a_idx = np.argpartition(row, -s1k)[-s1k:]
                    # Collect entities from the first-hop docs A_i.
                    a_ents = set()
                    for ai in a_idx:
                        a_ents.update(self._doc_ents_pi.get(int(ai), []))
                    if not a_ents:
                        continue
                    # Second hop: pool docs B_j sharing those entities,
                    # scored by entity-overlap count (exclude the A_i set).
                    a_set = set(int(x) for x in a_idx)
                    overlap = defaultdict(int)
                    for e in a_ents:
                        for bj in self._ent_index.get(e, ()):  # pool idx
                            if bj not in a_set:
                                overlap[bj] += 1
                    if not overlap:
                        continue
                    # Credit top bridge candidates into the shared demand ledger.
                    ranked = sorted(overlap.items(), key=lambda x: -x[1])
                    for bj, ov in ranked[:self.R3_MAX_BRIDGE]:
                        self.demand[bj] = self.demand.get(bj, 0.0) + ov * self.R3_ALPHA
        # --- R1 + unified admission gate (SemFlow) ---
        super().step(window_queries, window_query_embs, window_idx)


