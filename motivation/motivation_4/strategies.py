"""
KB update strategies for the query-drift motivation experiment.

All strategies share the same interface:
  - __init__(name, doc_pool, doc_embs, title_to_idx)
  - set_kb(ids: set[str])   -- initialise the KB document ID set
  - step(wq, wqe, wi)       -- observe one window's queries and update KB
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
        self.update_cost = 0     # KB writes (insertions/replacements)
        self.retrieval_cost = 0  # external pool retrievals (no KB write)

    @property
    def cost(self):
        """Backward-compat alias: total operations (writes + retrievals)."""
        return self.update_cost + self.retrieval_cost

    def set_kb(self, ids):
        self.kb = set(ids)

    def step(self, wq, wqe, wi):
        raise NotImplementedError


class Static(BaseStrategy):
    """No-update baseline.  KB is frozen after initialisation."""
    def step(self, wq, wqe, wi):
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
    def __init__(self, name, dp, de, ti):
        super().__init__(name, dp, de, ti)
        self.rng = np.random.default_rng(SEED + 100)
        self.all_ids = [d['doc_id'] for d in dp]
        self.ts = {}

    def step(self, wq, wqe, wi):
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]
        arrivals = self.rng.choice(self.all_ids,
                                   min(DOC_ARRIVE, len(self.all_ids)),
                                   replace=False)
        n = 0
        for did in arrivals:
            if n >= DOC_ADD_CAP:
                break
            if did in self.kb:
                self.ts[did] = wi
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
                self.ts[did] = wi
                kb_list[pos] = did
                kb_idx[pos] = ni
                kb_emb[pos] = ne
                n += 1
            elif best < 0.3:
                stale = min(self.kb, key=lambda d: self.ts.get(d, -1))
                self.kb.discard(stale)
                self.kb.add(did)
                self.ts[did] = wi
                n += 1
        self.update_cost += n


class KnowledgeEdit(BaseStrategy):
    """Knowledge-edit-driven KB update (RECIPE style).

    Each window, EDIT_BATCH KB documents are randomly selected for "editing".
    For each, find the most similar non-KB document (0.4 < sim < 0.8) and
    swap it in.  This models the RECIPE paradigm where a knowledge graph is
    continuously revised via local edits rather than wholesale replacement.
    """
    def __init__(self, name, dp, de, ti):
        super().__init__(name, dp, de, ti)
        self.rng = np.random.default_rng(SEED + 200)

    def step(self, wq, wqe, wi):
        kb_list = sorted(self.kb)
        if not kb_list:
            return
        n_ed = min(EDIT_BATCH, len(kb_list))
        targets = self.rng.choice(kb_list, n_ed, replace=False)
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
    def __init__(self, name, dp, de, ti):
        super().__init__(name, dp, de, ti)
        self.baseline_mean = None
        self.use_max = False

    def step(self, wq, wqe, wi):
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]
        norms = np.linalg.norm(wqe, axis=1, keepdims=True)
        nqe = wqe / np.clip(norms, 1e-10, None)
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
        for qi in range(len(fqe)):
            tk = min(QD_TOP_K, len(self.doc_embs))
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
    """Oracle upper bound: perfect future-knowledge KB rebuild at H2 start.

    Behaviour:
      - H1 (windows 0..half-1): no updates (same as Static).
      - At window == half: one-time rebuild using ALL gold supporting-fact
        documents for H2 queries.  Remaining budget filled with docs most
        similar to H2 query distribution.

    This represents the absolute ceiling: what recall would be if the KB
    always contained the right documents.  The gap between Oracle and
    QueryDriven quantifies the room for improvement.
    """
    def __init__(self, name, dp, de, ti):
        super().__init__(name, dp, de, ti)
        self.t2i = ti
        self._half = 10
        self._stream = None
        self._query_embs = None
        self._rebuilt = False

    def step(self, wq, wqe, wi):
        if not self.kb:
            return
        if wi < self._half or self._rebuilt:
            return
        if self._stream is None:
            return
        self._rebuilt = True
        ws = len(wq)
        h2_queries = self._stream[self._half * ws:]
        sf_pool_indices = set()
        for q in h2_queries:
            for t in q.get('sf_titles', []):
                if t in self.t2i:
                    sf_pool_indices.add(self.t2i[t])
        h2_qe = np.array([self._query_embs[q['qidx']] for q in h2_queries])
        h2_norms = np.linalg.norm(h2_qe, axis=1, keepdims=True)
        h2_nqe = h2_qe / np.clip(h2_norms, 1e-10, None)
        doc_scores = np.mean(h2_nqe @ self.doc_embs.T, axis=0)
        budget = len(self.kb)
        new_kb = set()
        sf_list = sorted(sf_pool_indices, key=lambda pi: -doc_scores[pi])
        for pi in sf_list[:budget]:
            new_kb.add(self.p2d[pi])
        if len(new_kb) < budget:
            non_sf = [(doc_scores[i], i) for i in range(len(self.doc_pool))
                      if i not in sf_pool_indices]
            non_sf.sort(reverse=True)
            for score, pi in non_sf:
                if len(new_kb) >= budget:
                    break
                new_kb.add(self.p2d[pi])
        self.update_cost = len(new_kb - self.kb)
        self.kb = new_kb
        log.info(f"[Oracle] Rebuilt KB: {len(sf_pool_indices)} SFs available, "
                 f"{min(len(sf_pool_indices), budget)} SFs in KB, "
                 f"cost={self.cost}")


# ── Factory registry ──────────────────────────────
STRATEGY_FACTORIES = {
    'Static':           lambda dp, de, ti: Static('Static', dp, de, ti),
    'RandomFIFO':       lambda dp, de, ti: RandomFIFO('RandomFIFO', dp, de, ti),
    'DocArrival':       lambda dp, de, ti: DocArrival('DocArrival', dp, de, ti),
    'KnowledgeEdit':    lambda dp, de, ti: KnowledgeEdit('KnowledgeEdit', dp, de, ti),
    'OnDemandFetch':    lambda dp, de, ti: OnDemandFetch('OnDemandFetch', dp, de, ti),
    'LogDrivenArrival': lambda dp, de, ti: LogDrivenArrival('LogDrivenArrival', dp, de, ti),
    'QueryDriven':      lambda dp, de, ti: QueryDriven('QueryDriven', dp, de, ti),
    'Oracle':           lambda dp, de, ti: Oracle('Oracle', dp, de, ti),
}


class RandomFIFO(BaseStrategy):
    """Blind supply-side: random new docs replace oldest KB entries (FIFO).

    Models a naive scheduled ingest pipeline that periodically refreshes
    KB content without any relevance signal.  Each window, FIFO_BATCH docs
    are randomly drawn from pool and replace the oldest-inserted KB entries.
    Demonstrates that blind supply-side updates inject noise faster than
    useful content, especially under drift.
    """
    def __init__(self, name, dp, de, ti):
        super().__init__(name, dp, de, ti)
        self.rng = np.random.default_rng(SEED + 300)
        self.all_ids = [d['doc_id'] for d in dp]
        self.insert_order = []  # track insertion order for FIFO eviction

    def set_kb(self, ids):
        super().set_kb(ids)
        self.insert_order = list(ids)

    def step(self, wq, wqe, wi):
        batch = min(FIFO_BATCH, len(self.all_ids))
        arrivals = self.rng.choice(self.all_ids, batch, replace=False)
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
    def __init__(self, name, dp, de, ti):
        super().__init__(name, dp, de, ti)
        self.fetch_k = FETCH_TOP_K
        self._fetched_this_window = set()

    def get_effective_kb(self, wq, wqe):
        """Return KB + fetched docs for this window (for recall eval)."""
        return self.kb | self._fetched_this_window

    def step(self, wq, wqe, wi):
        self._fetched_this_window = set()
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]

        norms = np.linalg.norm(wqe, axis=1, keepdims=True)
        nqe = wqe / np.clip(norms, 1e-10, None)

        for qi in range(len(wq)):
            q_kb_sim = nqe[qi] @ kb_emb.T
            if float(np.max(q_kb_sim)) >= SF_HIT_THRESH:
                continue
            # Fail query: search external pool
            pool_sim = nqe[qi] @ self.doc_embs.T
            top_idx = np.argpartition(pool_sim, -self.fetch_k)[-self.fetch_k:]
            for pi in top_idx:
                self._fetched_this_window.add(self.p2d[pi])
            self.retrieval_cost += len(top_idx)  # external retrieval, no KB write


class LogDrivenArrival(BaseStrategy):
    """Lagging log-driven update: analyse previous window's failures, fix next.

    Models a human-in-the-loop or scheduled-batch pipeline:
    after window T, inspect which queries failed, find their best pool
    candidates, and add those to KB for window T+1.

    Always one window behind — demonstrates the "lagging effect" under drift:
    by the time the fix arrives, the query distribution may have shifted again.
    """
    def __init__(self, name, dp, de, ti):
        super().__init__(name, dp, de, ti)
        self._pending_adds = []  # docs to add at next review cycle
        self._fail_buffer_qe = []  # accumulated fail query embeddings

    def step(self, wq, wqe, wi):
        # Apply pending adds at review-cycle boundaries
        if self._pending_adds and wi % LOG_LAG_WINDOWS == 0:
            kb_list = sorted(self.kb)
            # Evict least-useful current docs to make room
            if len(kb_list) > 0:
                kb_idx = np.array([self.d2p[d] for d in kb_list])
                kb_emb = self.doc_embs[kb_idx]
                norms = np.linalg.norm(wqe, axis=1, keepdims=True)
                nqe = wqe / np.clip(norms, 1e-10, None)
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
        norms = np.linalg.norm(wqe, axis=1, keepdims=True)
        nqe = wqe / np.clip(norms, 1e-10, None)
        q_kb = nqe @ kb_emb.T
        max_s = np.max(q_kb, axis=1)
        fail = max_s < SF_HIT_THRESH
        if fail.any():
            self._fail_buffer_qe.append(nqe[fail])

        # At end of each lag cycle, analyse accumulated failures for next cycle
        if (wi + 1) % LOG_LAG_WINDOWS == 0 and self._fail_buffer_qe:
            fqe = np.concatenate(self._fail_buffer_qe, axis=0)
            pool_sims = fqe @ self.doc_embs.T
            cand_set = set()
            for qi in range(len(fqe)):
                tk = min(LOG_FIX_TOP_K, len(self.doc_embs))
                top = np.argpartition(pool_sims[qi], -tk)[-tk:]
                cand_set.update(top.tolist())
            cand_set -= {self.d2p[d] for d in self.kb}
            scored = [(float(np.mean(fqe @ self.doc_embs[pi])), self.p2d[pi])
                      for pi in sorted(cand_set)]
            scored.sort(reverse=True)
            self._pending_adds = [did for _, did in scored[:LOG_FIX_CAP]]
