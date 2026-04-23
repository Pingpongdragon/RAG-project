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

4. QueryDrivenCluster -- Demand-side (ours): detect poorly-served
                   queries via top-2 KB-coverage gating, turn each failure
                   into a small candidate-doc bundle, accumulate those bundles
                   across windows with decay, then write the docs that cover
                   the most weighted failure bundles under a fixed KB budget.
                   Updates run in `step` (post-eval), so the comparison with
                   blind persistent baselines is fair.
                   Key features:
                   - cover_s (top-2) failure detection for multi-support
                     queries
                   - Drift-proportional replacement cap
                   - Candidate sources: failing-query top-K bundles
                   - Selection: weighted bundle coverage + persistent doc demand
                   - Eviction: current fail-usefulness + recent demand utility

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

    def prepare_window(self, window_queries, window_query_embs, window_idx):
        pass

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





class QueryDrivenCluster(BaseStrategy):
    """Query-demand-driven KB writer (single mechanism, no regime branching).

    Two long-lived per-doc statistics drive every decision:
      - demand[d]: exponentially-decayed sum of sim(q, d) over windows
        where d was a top-K pool candidate for a *failing* query. Big
        demand means many recent queries lacked KB coverage and would
        have benefited from d.
      - serve[d]: exponentially-decayed count of (window, query) pairs
        where d was the best KB hit for that query above the SF
        threshold. Big serve means d is currently doing useful work.

    Per window:
      1. Score every query against KB by top-2 coverage (cover_s).
      2. Decay both stats (demand/serve) by their respective rates.
      3. For each query that *succeeds* (max_s >= SF_HIT_THRESH), credit
         the best KB doc with +1 serve.
      4. For each query that *fails* (cover_s < SF_HIT_THRESH), retrieve
         top-K pool docs and add their similarity to demand[d].
      5. Sort non-KB candidates by demand desc; sort KB docs by
         (serve + demand_inside_kb) asc.
      6. Replace KB doc e by candidate c iff demand[c] > serve[e] + demand[e].
         This is the only admission test: a candidate must carry more
         accumulated evidence than the weakest currently-resident doc.
      7. Write cap per window = number of failing queries. The cap is
         therefore drift-proportional with no explicit drift detection.

    Why this should beat the baselines:
      - vs Static: ever updates at all.
      - vs RandomFIFO / DocArrival: uses query-failure signal, so writes
        target documents the workload actually needs.
      - vs KnowledgeEdit: uses query-side signal instead of pure KB-internal
        similarity, which is wrong-signal under workload drift.
      - vs LogDrivenArrival: no per-cycle lag, no fixed write quota, and
        accumulates evidence across windows so a doc relevant to many
        recurring failures wins over a doc relevant to a single query.
      - vs OnDemandFetch: persistent (free at serve time), but admits
        gracefully (only when demand outweighs current serving value).
    """
    DEMAND_DECAY = 0.85
    SERVE_DECAY  = 0.92
    PROBE_TOPK   = 8
    MIN_STAT     = 0.01

    SERVE_PRIOR  = 1.0  # initial trust per resident KB doc

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.demand = {}   # pool_idx -> float
        self.serve  = {}   # pool_idx -> float
    def set_kb(self, ids):
        super().set_kb(ids)
        # Bayesian prior: trust the initial KB. Every resident doc starts
        # with one unit of serve evidence, equivalent to "served once".
        # This protects the initial state until live failures prove a doc
        # should be replaced. The prior decays with the same rate as
        # serve, so its influence vanishes over a few windows if the doc
        # is never actually used.
        for did in self.kb:
            pi = self.d2p[did]
            self.serve[pi] = self.SERVE_PRIOR

    def _decay(self):
        d = self.DEMAND_DECAY; s = self.SERVE_DECAY; m = self.MIN_STAT
        self.demand = {p: v*d for p, v in self.demand.items() if v*d >= m}
        self.serve  = {p: v*s for p, v in self.serve.items()  if v*s >= m}

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

        self._decay()

        # 1) Credit serve for every succeeding query (best KB doc gets +1).
        succ = max_s >= SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                pi = int(kb_idx[pos])
                self.serve[pi] = self.serve.get(pi, 0.0) + 1.0

        # 2) Credit demand for every failing query.
        # Unified failure definition: a query fails iff its best KB hit is
        # below the SF threshold. Same definition as serve, applies to
        # single-hop and multi-hop without dataset-specific switches.
        fail = max_s < SF_HIT_THRESH
        n_fail = int(fail.sum())
        if n_fail == 0:
            self.update_cost += 0
            return
        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T
        probe = min(self.PROBE_TOPK, pool_sims.shape[1])
        self.maint_retrieval_cost += n_fail * probe
        kb_pos = set(int(i) for i in kb_idx)
        for qi in range(n_fail):
            top = np.argpartition(pool_sims[qi], -probe)[-probe:]
            top = top[np.argsort(-pool_sims[qi, top])]
            # Unit-aligned with serve: each failing query contributes
            # exactly 1.0 demand mass, distributed by similarity weight.
            sims = pool_sims[qi, top].astype(float)
            sims = np.maximum(sims, 0.0)
            tot = sims.sum()
            if tot <= 0:
                continue
            weights = sims / tot
            for w, pi in zip(weights, top):
                pi = int(pi)
                self.demand[pi] = self.demand.get(pi, 0.0) + float(w)

        # 3) Candidate ordering: high accumulated demand wins.
        cands = sorted(
            ((v, p) for p, v in self.demand.items() if p not in kb_pos),
            reverse=True,
        )
        if not cands:
            return

        # 4) Eviction ordering: low (serve + demand) loses first.
        evict_val = {int(p): self.serve.get(int(p), 0.0) + self.demand.get(int(p), 0.0)
                     for p in kb_idx}
        evictable = sorted(kb_pos, key=lambda p: evict_val[p])

        # 5) Universal admission gate: replace e by c iff demand[c] > evict_val[e].
        # No write cap: the gate is self-limiting because remaining
        # candidates are sorted by demand desc and evict_val[e] is sorted asc.
        n = 0
        ei = 0
        for cval, cp in cands:
            if ei >= len(evictable):
                break
            ep = evictable[ei]
            if cval <= evict_val[ep]:
                break  # remaining candidates are even smaller
            self.kb.discard(self.p2d[ep])
            self.kb.add(self.p2d[cp])
            self.serve.pop(ep, None)   # fresh doc has no serve history
            ei += 1
            n += 1
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
    'Static':              lambda doc_pool, doc_embs, title_to_idx: Static('Static', doc_pool, doc_embs, title_to_idx),
    'RandomFIFO':          lambda doc_pool, doc_embs, title_to_idx: RandomFIFO('RandomFIFO', doc_pool, doc_embs, title_to_idx),
    'DocArrival':          lambda doc_pool, doc_embs, title_to_idx: DocArrival('DocArrival', doc_pool, doc_embs, title_to_idx),
    'KnowledgeEdit':       lambda doc_pool, doc_embs, title_to_idx: KnowledgeEdit('KnowledgeEdit', doc_pool, doc_embs, title_to_idx),
    'OnDemandFetch':       lambda doc_pool, doc_embs, title_to_idx: OnDemandFetch('OnDemandFetch', doc_pool, doc_embs, title_to_idx),
    'LogDrivenArrival':    lambda doc_pool, doc_embs, title_to_idx: LogDrivenArrival('LogDrivenArrival', doc_pool, doc_embs, title_to_idx),
    'QueryDrivenCluster':  lambda doc_pool, doc_embs, title_to_idx: QueryDrivenCluster('QueryDrivenCluster', doc_pool, doc_embs, title_to_idx),
    'Oracle':              lambda doc_pool, doc_embs, title_to_idx: Oracle('Oracle', doc_pool, doc_embs, title_to_idx),
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
    directly. The KB itself is NEVER updated (static), but the query gets
    augmented results in the same window. Cost counts external search calls.

    This demonstrates that even perfect external search cannot replace
    KB consolidation: (a) every query incurs search latency, (b) repeated
    queries re-search the same docs, (c) no learning across windows.

    For evaluation, we temporarily add fetched docs to a "virtual KB" for
    that window only, then remove them before the next window.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.fetch_k = FETCH_TOP_K
        self._fetched_this_window = set()

    def prepare_window(self, window_queries, window_query_embs, window_idx):
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
            pool_sim = nqe[qi] @ self.doc_embs.T
            top_idx = np.argpartition(pool_sim, -self.fetch_k)[-self.fetch_k:]
            for pi in top_idx:
                self._fetched_this_window.add(self.p2d[pi])
            self.serve_retrieval_cost += len(top_idx)

    def get_effective_kb(self, window_queries, window_query_embs):
        """Return KB + fetched docs for this window (for recall eval)."""
        return self.kb | self._fetched_this_window

    def step(self, window_queries, window_query_embs, window_idx):
        pass


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
