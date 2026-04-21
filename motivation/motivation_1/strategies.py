"""
KB update strategies for the query-drift motivation experiment.

All strategies share the same interface:
  - __init__(name, doc_pool, doc_embs, title_to_idx)
  - set_kb(ids: set[str])   -- initialise the KB document ID set
  - step(window_queries, window_query_embs, window_idx)  -- observe one window and update KB
  - .kb: set[str]           -- current KB document IDs
  - .cost: int              -- cumulative number of KB replacements

Strategies model different KB maintenance paradigms from the literature:

1. Static       -- No updates (baseline). Shows pure drift degradation.

2. DocArrival   -- Supply-side: new documents arrive randomly and replace
                   KB entries based on similarity.
                   Models HippoRAG2 (Gutiérrez et al., 2025) and
                   LightRAG (Guo et al., 2024) document-arrival pipelines.
                   Each window: sample DOC_ARRIVE docs from pool, replace up
                   to DOC_ADD_CAP KB entries via similarity thresholds.

3. KnowledgeEdit -- Supply-side: existing KB entries are edited/replaced with
                    semantically similar alternatives.
                    Models RECIPE (Luo et al., 2024) knowledge-edit pipeline.
                    Each window: select EDIT_BATCH KB docs, find the most
                    similar non-KB doc (0.4 < sim < 0.8) and swap.

4. QueryDriven  -- Demand-side (ours): detect poorly-served queries via
                   alignment scoring, retrieve better candidates from pool,
                   evict least-useful KB entries.
                   Key features:
                   - Auto-adaptive scoring: MEAN vs MAX chosen by baseline
                     signal quality (threshold 0.46)
                   - Drift-proportional replacement cap: small drift -> few
                     replacements, large drift -> full QD_REPLACE_CAP
                   - Candidate retrieval: top-QD_TOP_K per failing query
                   - Eviction: lowest mean usefulness across current queries

5. Oracle       -- Upper bound: at H2 transition, rebuilds KB with all gold
                   supporting-fact documents for H2 queries (uses future
                   knowledge). Remaining budget filled by highest-scoring
                   non-SF docs. Represents the theoretical ceiling.
"""
import numpy as np
from config import (SEED, SF_HIT_THRESH, DOC_ARRIVE, DOC_ADD_CAP,
                    EDIT_BATCH, QD_TOP_K, QD_REPLACE_CAP,
                    FIFO_BATCH, FETCH_TOP_K, LOG_FIX_TOP_K, LOG_FIX_CAP,
                    LOG_LAG_WINDOWS,
                    log)


class BaseStrategy:
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        self.name = name
        self.doc_pool = doc_pool
        self.doc_embs = doc_embs
        self.title_to_idx = title_to_idx
        self.d2p = {d['doc_id']: i for i, d in enumerate(doc_pool)}
        self.p2d = {i: d['doc_id'] for i, d in enumerate(doc_pool)}
        self.kb = set()
        self.update_cost = 0           # KB writes (insertions/replacements)
        self.maint_retrieval_cost = 0  # background pool scans (offline batch)
        self.serve_retrieval_cost = 0  # per-query pool fetches (online latency)

    @property
    def retrieval_cost(self):
        """Total pool retrievals (maintenance + serve-time)."""
        return self.maint_retrieval_cost + self.serve_retrieval_cost

    @property
    def cost(self):
        """Backward-compat alias: total operations (writes + retrievals)."""
        return self.update_cost + self.retrieval_cost

    def set_kb(self, ids):
        self.kb = set(ids)

    def step(self, window_queries, window_query_embs, window_idx):
        raise NotImplementedError


class Static(BaseStrategy):
    """No-update baseline.  KB is frozen after initialisation."""
    def step(self, window_queries, window_query_embs, window_idx):
        pass


class DocArrival(BaseStrategy):
    """Document-arrival-driven KB update (HippoRAG2 / LightRAG style).

    Each window, DOC_ARRIVE documents are randomly sampled from the full pool
    (simulating new documents arriving in a real system). For each arrival:
      - If similarity to some KB doc > 0.7: replace that KB doc (update).
      - If similarity to all KB docs < 0.3: evict the stalest KB doc (insert).
      - Otherwise: skip (not novel enough and not redundant enough).
    At most DOC_ADD_CAP replacements per window.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.rng = np.random.default_rng(SEED + 100)
        self.all_ids = [d['doc_id'] for d in doc_pool]
        self.ts = {}

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]
        arrivals = self.rng.choice(self.all_ids,
                                   min(DOC_ARRIVE, len(self.all_ids)),
                                   replace=False)
        self.maint_retrieval_cost += len(arrivals)
        n = 0
        for did in arrivals:
            if n >= DOC_ADD_CAP:
                break
            if did in self.kb:
                self.ts[did] = window_idx
                continue
            ni = self.d2p[did]
            ne = self.doc_embs[ni]
            sims = kb_emb @ ne
            best = float(np.max(sims))
            if best > 0.7:
                pos = int(np.argmax(sims))
                old = kb_list[pos]
                self.kb.discard(old)
                self.kb.add(did)
                self.ts[did] = window_idx
                kb_list[pos] = did
                kb_idx[pos] = ni
                kb_emb[pos] = ne
                n += 1
            elif best < 0.3:
                stale = min(self.kb, key=lambda d: self.ts.get(d, -1))
                self.kb.discard(stale)
                self.kb.add(did)
                self.ts[did] = window_idx
                n += 1
        self.update_cost += n


class KnowledgeEdit(BaseStrategy):
    """Knowledge-edit-driven KB update (RECIPE style).

    Each window, EDIT_BATCH KB documents are randomly selected for "editing".
    For each, find the most similar non-KB document (0.4 < sim < 0.8) and
    swap it in.  This models the RECIPE paradigm where a knowledge graph is
    continuously revised via local edits rather than wholesale replacement.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.rng = np.random.default_rng(SEED + 200)

    def step(self, window_queries, window_query_embs, window_idx):
        kb_list = sorted(self.kb)
        if not kb_list:
            return
        n_ed = min(EDIT_BATCH, len(kb_list))
        targets = self.rng.choice(kb_list, n_ed, replace=False)
        self.maint_retrieval_cost += n_ed  # n_ed full-pool NN scans
        n = 0
        for tid in targets:
            tpi = self.d2p[tid]
            te = self.doc_embs[tpi]
            sims = self.doc_embs @ te
            for d in self.kb:
                if d in self.d2p:
                    sims[self.d2p[d]] = -1
            cands = np.where((sims > 0.4) & (sims < 0.8))[0]
            if len(cands) == 0:
                continue
            best = cands[np.argmax(sims[cands])]
            self.kb.discard(tid)
            self.kb.add(self.p2d[best])
            n += 1
        self.update_cost += n


class QueryDriven(BaseStrategy):
    """Query-demand-driven KB update (ours).

    Core idea: use current-window query signals to detect KB-query misalignment,
    then surgically replace the least-useful KB documents with better candidates.

    Algorithm per window:
      1. Compute alignment: max cosine similarity of each query to KB.
      2. First window sets baseline_mean; subsequent windows compute drift.
      3. Drift-proportional cap: mild drift -> few swaps, severe -> full cap.
      4. Identify "failing" queries (max_sim < SF_HIT_THRESH).
      5. For each failing query, retrieve top-QD_TOP_K candidates from pool.
      6. Score candidates by MEAN (or MAX if baseline < 0.46) similarity to
         all failing queries.
      7. Evict KB docs with lowest mean usefulness across current queries.
      8. Swap in candidates that outscore evicted docs.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.baseline_mean = None
        self.use_max = False

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)
        q_kb = nqe @ kb_emb.T
        max_s = np.max(q_kb, axis=1)
        cur_mean = float(max_s.mean())
        if self.baseline_mean is None:
            self.baseline_mean = cur_mean
            self.use_max = cur_mean < 0.46
            mode = "MAX" if self.use_max else "MEAN"
            log.info(f"[{self.name}] baseline={self.baseline_mean:.3f} scoring={mode}")
        drift = max(0.0, self.baseline_mean - cur_mean)
        if drift < 0.01:
            cap = max(5, QD_REPLACE_CAP // 10)
        elif drift < 0.05:
            cap = max(10, int(QD_REPLACE_CAP * drift / 0.05))
        else:
            cap = QD_REPLACE_CAP
        fail = max_s < SF_HIT_THRESH
        if not fail.any():
            return
        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T
        cand_set = set()
        tk = min(QD_TOP_K, len(self.doc_embs))
        self.maint_retrieval_cost += len(fqe) * tk  # n_failing × top-K
        for qi in range(len(fqe)):
            top = np.argpartition(pool_sims[qi], -tk)[-tk:]
            cand_set.update(top.tolist())
        cand_set -= {self.d2p[d] for d in self.kb}
        if not cand_set:
            return
        if self.use_max:
            cands = [(float(np.max(fqe @ self.doc_embs[pi])), pi)
                     for pi in sorted(cand_set)]
        else:
            cands = [(float(np.mean(fqe @ self.doc_embs[pi])), pi)
                     for pi in sorted(cand_set)]
        cands.sort(reverse=True)
        usefulness = np.mean(q_kb, axis=0)
        evict = np.argsort(usefulness)
        n, ei = 0, 0
        for cs, cpi in cands[:cap]:
            if ei >= len(evict):
                break
            epos = evict[ei]
            if cs <= float(usefulness[epos]):
                break
            self.kb.discard(kb_list[epos])
            self.kb.add(self.p2d[cpi])
            n += 1
            ei += 1
        self.update_cost += n


class Oracle(BaseStrategy):
    """Oracle (per-window upper bound).

    At every window, the KB is reconstructed from the gold supporting-fact
    documents of THAT window's queries; remaining capacity is filled with
    documents most similar to that window's queries. This is the tightest
    achievable upper bound on Recall@K (any retriever cannot exceed it
    given the KB capacity), shown as a *constant* upper envelope before
    AND after drift.

    Note: this is strictly a non-causal reference; it consumes ground-truth
    SF labels not available at deployment.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.t2i = title_to_idx

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        budget = len(self.kb)
        # gold SF pool indices from current window
        sf_pool = set()
        for q in window_queries:
            for t in q.get('sf_titles', []):
                if t in self.t2i:
                    sf_pool.add(self.t2i[t])
        # fill remainder by similarity to this window's queries
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        norm_qe = window_query_embs / np.clip(norms, 1e-10, None)
        doc_scores = np.mean(norm_qe @ self.doc_embs.T, axis=0)
        new_kb = set()
        for pi in sorted(sf_pool, key=lambda i: -doc_scores[i])[:budget]:
            new_kb.add(self.p2d[pi])
        if len(new_kb) < budget:
            sorted_docs = np.argsort(-doc_scores)
            for pi in sorted_docs:
                if len(new_kb) >= budget:
                    break
                if pi in sf_pool:
                    continue
                new_kb.add(self.p2d[int(pi)])
        added = len(new_kb - self.kb)
        self.update_cost += added
        self.kb = new_kb



# ── Factory registry ──────────────────────────────
STRATEGY_FACTORIES = {
    'Static':           lambda doc_pool, doc_embs, title_to_idx: Static('Static', doc_pool, doc_embs, title_to_idx),
    'RandomFIFO':       lambda doc_pool, doc_embs, title_to_idx: RandomFIFO('RandomFIFO', doc_pool, doc_embs, title_to_idx),
    'DocArrival':       lambda doc_pool, doc_embs, title_to_idx: DocArrival('DocArrival', doc_pool, doc_embs, title_to_idx),
    'KnowledgeEdit':    lambda doc_pool, doc_embs, title_to_idx: KnowledgeEdit('KnowledgeEdit', doc_pool, doc_embs, title_to_idx),
    'OnDemandFetch':    lambda doc_pool, doc_embs, title_to_idx: OnDemandFetch('OnDemandFetch', doc_pool, doc_embs, title_to_idx),
    'LogDrivenArrival': lambda doc_pool, doc_embs, title_to_idx: LogDrivenArrival('LogDrivenArrival', doc_pool, doc_embs, title_to_idx),
    'QueryDriven':      lambda doc_pool, doc_embs, title_to_idx: QueryDriven('QueryDriven', doc_pool, doc_embs, title_to_idx),
    'Oracle':           lambda doc_pool, doc_embs, title_to_idx: Oracle('Oracle', doc_pool, doc_embs, title_to_idx),
}


class RandomFIFO(BaseStrategy):
    """Blind supply-side: random new docs replace oldest KB entries (FIFO).

    Models a naive scheduled ingest pipeline that periodically refreshes
    KB content without any relevance signal.  Each window, FIFO_BATCH docs
    are randomly drawn from pool and replace the oldest-inserted KB entries.
    Demonstrates that blind supply-side updates inject noise faster than
    useful content, especially under drift.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.rng = np.random.default_rng(SEED + 300)
        self.all_ids = [d['doc_id'] for d in doc_pool]
        self.insert_order = []  # track insertion order for FIFO eviction

    def set_kb(self, ids):
        super().set_kb(ids)
        self.insert_order = list(ids)

    def step(self, window_queries, window_query_embs, window_idx):
        batch = min(FIFO_BATCH, len(self.all_ids))
        arrivals = self.rng.choice(self.all_ids, batch, replace=False)
        self.maint_retrieval_cost += batch
        n = 0
        for did in arrivals:
            if did in self.kb:
                continue
            if not self.insert_order:
                break
            old = self.insert_order.pop(0)
            self.kb.discard(old)
            self.kb.add(did)
            self.insert_order.append(did)
            n += 1
        self.update_cost += n


class OnDemandFetch(BaseStrategy):
    """Passive on-demand search: fetch from pool per-query, don't store.

    Models CRAG / Agent-style RAG: when a query's best KB hit is below
    SF_HIT_THRESH, search the external pool for top-K and use those results
    directly.  The KB itself is NEVER updated (static), but the query gets
    augmented results.  Cost counts external search calls.

    This demonstrates that even perfect external search cannot replace
    KB consolidation: (a) every query incurs search latency, (b) repeated
    queries re-search the same docs, (c) no learning across windows.

    For evaluation, we temporarily add fetched docs to a "virtual KB" for
    that window only, then remove them.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.fetch_k = FETCH_TOP_K
        self._fetched_this_window = set()

    def get_effective_kb(self, window_queries, window_query_embs):
        """Return KB + fetched docs for this window (for recall eval)."""
        return self.kb | self._fetched_this_window

    def step(self, window_queries, window_query_embs, window_idx):
        self._fetched_this_window = set()
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]

        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)

        for qi in range(len(window_queries)):
            q_kb_sim = nqe[qi] @ kb_emb.T
            if float(np.max(q_kb_sim)) >= SF_HIT_THRESH:
                continue
            # Fail query: search external pool
            pool_sim = nqe[qi] @ self.doc_embs.T
            top_idx = np.argpartition(pool_sim, -self.fetch_k)[-self.fetch_k:]
            for pi in top_idx:
                self._fetched_this_window.add(self.p2d[pi])
            self.serve_retrieval_cost += len(top_idx)  # per-query, paid online


class LogDrivenArrival(BaseStrategy):
    """Lagging log-driven update: analyse previous window's failures, fix next.

    Models a human-in-the-loop or scheduled-batch pipeline:
    after window T, inspect which queries failed, find their best pool
    candidates, and add those to KB for window T+1.

    Always one window behind — demonstrates the "lagging effect" under drift:
    by the time the fix arrives, the query distribution may have shifted again.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self._pending_adds = []  # docs to add at next review cycle
        self._fail_buffer_qe = []  # accumulated fail query embeddings

    def step(self, window_queries, window_query_embs, window_idx):
        # Apply pending adds at review-cycle boundaries
        if self._pending_adds and window_idx % LOG_LAG_WINDOWS == 0:
            kb_list = sorted(self.kb)
            # Evict least-useful current docs to make room
            if len(kb_list) > 0:
                kb_idx = np.array([self.d2p[d] for d in kb_list])
                kb_emb = self.doc_embs[kb_idx]
                norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
                nqe = window_query_embs / np.clip(norms, 1e-10, None)
                usefulness = np.mean(nqe @ kb_emb.T, axis=0)
                evict_order = np.argsort(usefulness)
                n = 0
                ei = 0
                for did in self._pending_adds:
                    if did in self.kb:
                        continue
                    if ei >= len(evict_order):
                        break
                    old = kb_list[evict_order[ei]]
                    self.kb.discard(old)
                    self.kb.add(did)
                    n += 1
                    ei += 1
                self.update_cost += n
            self._pending_adds = []
            self._fail_buffer_qe = []

        # Accumulate fail queries every window
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)
        q_kb = nqe @ kb_emb.T
        max_s = np.max(q_kb, axis=1)
        fail = max_s < SF_HIT_THRESH
        if fail.any():
            self._fail_buffer_qe.append(nqe[fail])

        # At end of each lag cycle, analyse accumulated failures for next cycle
        if (window_idx + 1) % LOG_LAG_WINDOWS == 0 and self._fail_buffer_qe:
            fqe = np.concatenate(self._fail_buffer_qe, axis=0)
            pool_sims = fqe @ self.doc_embs.T
            cand_set = set()
            tk = min(LOG_FIX_TOP_K, len(self.doc_embs))
            self.maint_retrieval_cost += len(fqe) * tk
            for qi in range(len(fqe)):
                top = np.argpartition(pool_sims[qi], -tk)[-tk:]
                cand_set.update(top.tolist())
            cand_set -= {self.d2p[d] for d in self.kb}
            scored = [(float(np.mean(fqe @ self.doc_embs[pi])), self.p2d[pi])
                      for pi in sorted(cand_set)]
            scored.sort(reverse=True)
            self._pending_adds = [did for _, did in scored[:LOG_FIX_CAP]]
