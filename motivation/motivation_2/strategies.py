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

4. QueryDriven -- Demand-side (ours): detect poorly-served
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





class QueryDriven(BaseStrategy):
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
    DEMAND_DECAY = 0.92
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

    PREFETCH_TOPK = 5
    TAU_ADMIT     = 0.95
    LAMBDA_RED    = 1.5
    RED_THRESH    = 0.85
    NEIGH_GAMMA   = 0.4
    WRITE_CAP_T1  = 200

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

        succ = max_s >= SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                pi = int(kb_idx[pos])
                self.serve[pi] = self.serve.get(pi, 0.0) + 1.0

        fail = max_s < SF_HIT_THRESH
        n_fail = int(fail.sum())
        if n_fail == 0:
            self.update_cost += 0
            return

        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T

        # Top-1 miss demand
        for qi in range(n_fail):
            t1 = int(np.argmax(pool_sims[qi]))
            self.demand[t1] = self.demand.get(t1, 0.0) + 1.0

        # NEIGHBORHOOD DEMAND: warm semantic neighbors of each missed query
        topk = min(self.PREFETCH_TOPK, pool_sims.shape[1])
        self.maint_retrieval_cost += n_fail * topk
        for qi in range(n_fail):
            row = pool_sims[qi]
            top = np.argpartition(row, -topk)[-topk:]
            sims = np.maximum(row[top].astype(float), 0.0)
            tot = sims.sum()
            if tot <= 0:
                continue
            w = sims / tot
            for wj, pi in zip(w, top):
                pi = int(pi)
                self.demand[pi] = self.demand.get(pi, 0.0) + float(wj) * self.NEIGH_GAMMA

        # Redundancy field
        kb_self = kb_emb @ kb_emb.T
        np.fill_diagonal(kb_self, -1.0)
        red_vec = kb_self.max(axis=1)
        red_map = {int(p): float(red_vec[i]) for i, p in enumerate(kb_idx)}
        kb_pos = set(int(i) for i in kb_idx)

        def _is_dup(cp, kb_emb_now):
            return float((self.doc_embs[cp] @ kb_emb_now.T).max()) > self.TAU_ADMIT

        # Build KB embeddings for tier-2
        kb_arr = np.array(sorted(kb_pos), dtype=int)
        kb_emb_now = self.doc_embs[kb_arr]
        kb_self2 = kb_emb_now @ kb_emb_now.T
        np.fill_diagonal(kb_self2, -1.0)
        red_vec2 = kb_self2.max(axis=1)
        red_map2 = {int(p): float(red_vec2[i]) for i, p in enumerate(kb_arr)}

        # === TIER 2: demand-gated admission with redundancy-aware eviction ===
        cands = sorted(
            ((v, p) for p, v in self.demand.items() if p not in kb_pos),
            reverse=True,
        )
        if not cands:
            return

        def vscore(p):
            base = self.serve.get(p, 0.0) + self.demand.get(p, 0.0)
            pen = self.LAMBDA_RED * max(0.0, red_map2.get(p, 0.0) - self.RED_THRESH)
            return base - pen
        evict_val = {p: vscore(p) for p in kb_pos}
        evictable = sorted(kb_pos, key=lambda p: evict_val[p])

        n = 0
        ei = 0
        # Optional per-window write budget (DRYAD module ② sets this to λ·B;
        # None = SemFlow's original behaviour where the gate self-limits).
        wbudget = getattr(self, '_write_budget', None)
        for cval, cp in cands:
            if ei >= len(evictable):
                break
            if wbudget is not None and n >= wbudget:
                break
            ep = evictable[ei]
            if cval <= evict_val[ep]:
                break
            if _is_dup(cp, kb_emb_now):
                continue
            self.kb.discard(self.p2d[ep])
            self.kb.add(self.p2d[cp])
            self.serve.pop(ep, None)
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
    'QueryDriven':  lambda doc_pool, doc_embs, title_to_idx: QueryDriven('QueryDriven', doc_pool, doc_embs, title_to_idx),
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


# ═══════════════════════════════════════════════════════════════════
# NEW BASELINES — cache-style and agent-memory-style baselines.
# IMPORTANT: these baselines do NOT receive QDC's query-failure-driven
# top-K candidate retrieval. Their admission stream is the same passive
# DocArrival stream used by HippoRAG/LightRAG; only the eviction policy
# differs. This isolates the *signal source* (failure-targeted demand)
# as QDC's contribution, rather than letting baselines piggyback on it.
# ═══════════════════════════════════════════════════════════════════


class _ArrivalCacheBase(BaseStrategy):
    """Shared scaffolding: random doc arrivals + per-window query touch.

    Each window:
      1. Sample DOC_ARRIVE pool docs as "arrivals" (no failure probing).
      2. Update per-doc bookkeeping based on which KB docs got hit by queries.
      3. Subclass decides which KB docs to evict and whether to admit each
         arrival, then admits up to DOC_ADD_CAP entries.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.rng = np.random.default_rng(SEED + 200 + hash(name) % 1000)
        self.all_ids = [d['doc_id'] for d in doc_pool]

    # subclass hooks
    def _on_init_kb(self, did, pi):
        pass
    def _on_query_hit(self, kb_pi, window_idx):
        pass
    def _on_query_seen(self, kb_pi, sim, window_idx):
        pass
    def _evict_score(self, pi, window_idx):
        # lower = more evictable
        return 0.0
    def _admit_score(self, pi, sim_to_kb_max, window_idx):
        # higher = more worth admitting; return None to skip
        return 1.0

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self._on_init_kb(did, self.d2p[did])

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return

        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)
        q_kb = nqe @ kb_emb.T  # (n_q, |KB|)

        # per-KB-doc passive observation: max sim from any query in window
        max_sim_per_kb = np.max(q_kb, axis=0) if q_kb.size else np.zeros(len(kb_list))
        for i, pi in enumerate(kb_idx):
            self._on_query_seen(int(pi), float(max_sim_per_kb[i]), window_idx)

        # successful hits credit recency/frequency
        max_s = np.max(q_kb, axis=1) if q_kb.size else np.zeros(len(window_queries))
        succ = max_s >= SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                self._on_query_hit(int(kb_idx[pos]), window_idx)

        # passive arrival stream (no failure-driven probing)
        arrivals = self.rng.choice(
            self.all_ids,
            min(DOC_ARRIVE, len(self.all_ids)),
            replace=False,
        )
        self.maint_retrieval_cost += len(arrivals)

        kb_pos_set = set(int(i) for i in kb_idx)
        # admission candidates with scores
        cand_scored = []
        for did in arrivals:
            ni = self.d2p[did]
            if ni in kb_pos_set:
                continue
            ne = self.doc_embs[ni]
            sims = kb_emb @ ne
            best_sim = float(np.max(sims)) if sims.size else 0.0
            score = self._admit_score(ni, best_sim, window_idx)
            if score is None:
                continue
            cand_scored.append((score, ni))

        if not cand_scored:
            return

        # most attractive arrivals first
        cand_scored.sort(reverse=True)
        # least valuable KB docs first (eviction order)
        evict_order = sorted(
            kb_pos_set,
            key=lambda p: self._evict_score(p, window_idx),
        )

        n = 0
        for score, ni in cand_scored:
            if n >= min(DOC_ADD_CAP, len(evict_order)):
                break
            ep = evict_order[n]
            old = self.p2d[ep]
            new = self.p2d[ni]
            self.kb.discard(old)
            self.kb.add(new)
            self._on_query_hit(ni, window_idx)
            n += 1
        self.update_cost += n


class LRU(_ArrivalCacheBase):
    """Pure least-recently-used cache.

    Admission source: random doc-arrival stream (no failure probing).
    Eviction: oldest last-touch wins (lowest last_hit window).
    Passive recency: any KB doc that gets touched by a query updates last_hit.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.last_hit = {}

    def _on_init_kb(self, did, pi):
        self.last_hit[pi] = -1
    def _on_query_hit(self, kb_pi, window_idx):
        self.last_hit[kb_pi] = window_idx
    def _evict_score(self, pi, window_idx):
        return self.last_hit.get(pi, -1)


class GPTCacheStyle(_ArrivalCacheBase):
    """Semantic cache (GPTCache-style).

    Admission source: random arrivals with semantic-dedup gate.
      - skip arrival if sim to existing KB > DEDUP_HIGH (already covered)
      - skip arrival if sim to existing KB < DEDUP_LOW  (off-topic noise)
    Eviction: lowest decayed cache score (max sim seen from queries lately).
    """
    DEDUP_HIGH = 0.85
    DEDUP_LOW  = 0.30
    DECAY      = 0.80

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.cache_score = {}

    def _on_init_kb(self, did, pi):
        self.cache_score[pi] = 1.0
    def _on_query_seen(self, kb_pi, sim, window_idx):
        # decayed running max
        prev = self.cache_score.get(kb_pi, 0.0) * self.DECAY
        self.cache_score[kb_pi] = max(prev, sim)
    def _evict_score(self, pi, window_idx):
        return self.cache_score.get(pi, 0.0)
    def _admit_score(self, pi, sim_to_kb_max, window_idx):
        if sim_to_kb_max > self.DEDUP_HIGH:
            return None
        if sim_to_kb_max < self.DEDUP_LOW:
            return None
        # admit with score = relevance to current KB topic distribution
        return sim_to_kb_max


class MemGPTStyle(_ArrivalCacheBase):
    """MemGPT-style importance memory.

    Admission source: random arrivals.
    Importance(d) = decayed access frequency.
    Eviction: lowest importance.
    Passive freq: any query hit on a KB doc increments freq.
    """
    DECAY = 0.88
    INIT_FREQ = 1.0

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.freq = {}
        self.last_hit = {}

    def _importance(self, pi, window_idx):
        f = self.freq.get(pi, 0.0)
        lh = self.last_hit.get(pi, -1)
        return f * (self.DECAY ** max(0, window_idx - lh))

    def _on_init_kb(self, did, pi):
        self.freq[pi] = self.INIT_FREQ
        self.last_hit[pi] = 0
    def _on_query_hit(self, kb_pi, window_idx):
        self.freq[kb_pi] = self.freq.get(kb_pi, 0.0) + 1.0
        self.last_hit[kb_pi] = window_idx
    def _evict_score(self, pi, window_idx):
        return self._importance(pi, window_idx)
    def _admit_score(self, pi, sim_to_kb_max, window_idx):
        # new arrivals start at INIT_FREQ; return that as ranking score
        return self.INIT_FREQ * sim_to_kb_max


# ── Updated registry: add new baselines ──
STRATEGY_FACTORIES.update({
    'LRU':           lambda dp, de, ti: LRU('LRU', dp, de, ti),
    'GPTCacheStyle': lambda dp, de, ti: GPTCacheStyle('GPTCacheStyle', dp, de, ti),
    'MemGPTStyle':   lambda dp, de, ti: MemGPTStyle('MemGPTStyle', dp, de, ti),
})


class QueryDrivenLoose(QueryDriven):
    """QDC with relaxed probe width and admission gate (sensitivity check).

    Differences from QueryDriven:
      - PROBE_TOPK 8 -> 50 (each failing query credits demand to many more
        pool candidates, dramatically increasing eviction pressure).
      - Admission gate: cval > 0.7 * evict_val (vs strict cval > evict_val),
        so candidates can dislodge resident docs even at parity.
    Used to test whether QDC's losses against LRU/GPTCache come from being
    too conservative on writes vs from a fundamentally weaker signal.
    """
    PROBE_TOPK = 50
    GATE_RATIO = 0.7

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
        succ = max_s >= SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                pi = int(kb_idx[pos])
                self.serve[pi] = self.serve.get(pi, 0.0) + 1.0
        fail = max_s < SF_HIT_THRESH
        n_fail = int(fail.sum())
        if n_fail == 0:
            return
        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T
        probe = min(self.PROBE_TOPK, pool_sims.shape[1])
        self.maint_retrieval_cost += n_fail * probe
        kb_pos = set(int(i) for i in kb_idx)
        for qi in range(n_fail):
            top = np.argpartition(pool_sims[qi], -probe)[-probe:]
            sims = pool_sims[qi, top].astype(float)
            sims = np.maximum(sims, 0.0)
            tot = sims.sum()
            if tot <= 0:
                continue
            weights = sims / tot
            for w, pi in zip(weights, top):
                pi = int(pi)
                self.demand[pi] = self.demand.get(pi, 0.0) + float(w)
        cands = sorted(
            ((v, p) for p, v in self.demand.items() if p not in kb_pos),
            reverse=True,
        )
        if not cands:
            return
        evict_val = {int(p): self.serve.get(int(p), 0.0) + self.demand.get(int(p), 0.0)
                     for p in kb_idx}
        evictable = sorted(kb_pos, key=lambda p: evict_val[p])
        n = 0
        ei = 0
        for cval, cp in cands:
            if ei >= len(evictable):
                break
            ep = evictable[ei]
            if cval <= self.GATE_RATIO * evict_val[ep]:
                break
            self.kb.discard(self.p2d[ep])
            self.kb.add(self.p2d[cp])
            self.serve.pop(ep, None)
            ei += 1
            n += 1
        self.update_cost += n



class TinyLFU(BaseStrategy):
    """TinyLFU: frequency-sketch admission gate with LFU eviction.

    Core mechanism (Einziger et al., 2017 / Caffeine cache):
      - Access frequency is tracked per document (exact counts; in production
        a Count-Min Sketch approximates this over a bounded sliding window).
      - Admission gate: a candidate document d_new replaces victim d_old
        only if freq(d_new) >= freq(d_old).
      - New arriving documents always start at freq=0; once KB documents
        accumulate frequency they become progressively harder to evict.

    Admission source: random doc-arrival stream (identical to LRU/DocArrival).
    Eviction order: lowest frequency first (LFU).

    Key contrast with NQM:
      TinyLFU estimates frequency of OBSERVED (successful) accesses.
      NQM estimates unmet semantic demand from FAILED queries and projects
      it into document embedding space—a signal unavailable to any
      access-history policy.

    In the multi-hop setting, bridge documents are never directly retrieved
    (the query omits the bridge entity), so their frequency stays at 0.
    TinyLFU cannot protect them and performs no better than Static.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.rng = np.random.default_rng(SEED + 700)
        self.all_ids = [d['doc_id'] for d in doc_pool]
        self.freq = {}   # pool_idx -> int (access count)

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self.freq[self.d2p[did]] = 0

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx  = np.array([self.d2p[d] for d in kb_list])
        kb_emb  = self.doc_embs[kb_idx]
        norms   = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe     = window_query_embs / np.clip(norms, 1e-10, None)
        q_kb    = nqe @ kb_emb.T
        max_s   = np.max(q_kb, axis=1)

        # 1) Increment frequency for docs that successfully serve queries.
        succ = max_s >= SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                pi = int(kb_idx[pos])
                self.freq[pi] = self.freq.get(pi, 0) + 1

        # 2) Process random doc arrivals through the TinyLFU admission gate.
        arrivals = self.rng.choice(
            self.all_ids, min(DOC_ARRIVE, len(self.all_ids)), replace=False)
        self.maint_retrieval_cost += len(arrivals)

        # Eviction queue: ascending frequency (least-frequently-used first).
        kb_pos_set  = set(int(p) for p in kb_idx)
        evict_queue = sorted(kb_pos_set, key=lambda p: self.freq.get(p, 0))

        n  = 0
        ei = 0
        for did in arrivals:
            if n >= DOC_ADD_CAP or ei >= len(evict_queue):
                break
            if did in self.kb:
                continue
            ni         = self.d2p[did]
            cand_freq  = self.freq.get(ni, 0)        # new arrivals: 0
            ep         = evict_queue[ei]
            victim_freq = self.freq.get(ep, 0)
            # TinyLFU gate: admit only if candidate has >= frequency as victim.
            if cand_freq >= victim_freq:
                self.kb.discard(self.p2d[ep])
                self.kb.add(did)
                self.freq.pop(ep, None)
                self.freq[ni] = 0
                ei += 1
                n  += 1
        self.update_cost += n


STRATEGY_FACTORIES.update({
    'QueryDrivenLoose': lambda dp, de, ti: QueryDrivenLoose('QueryDrivenLoose', dp, de, ti),
})

STRATEGY_FACTORIES.update({
    'TinyLFU': lambda dp, de, ti: TinyLFU('TinyLFU', dp, de, ti),
})


# ═══════════════════════════════════════════════════════════════════
# Miss-driven baselines: cache miss → fetch top-1 from cold pool by
# query embedding → admit → evict per policy.
# This is the textbook cache pattern that uses query signal directly,
# but without NQM's cross-query evidence aggregation and admission gate.
# ═══════════════════════════════════════════════════════════════════
class MissLRU(BaseStrategy):
    """Miss-driven LRU: textbook cache.

    Per window, for each query: if best KB hit < SF_HIT_THRESH (miss),
    fetch the pool's top-1 doc for that query embedding and admit it,
    evicting the KB doc with smallest last_hit. Successful queries refresh
    last_hit on the best-hit KB doc. No admission threshold, no aggregation.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.last_hit = {}

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self.last_hit[self.d2p[did]] = -1

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
        # refresh last_hit on successful queries
        succ = max_s >= SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                self.last_hit[int(kb_idx[pos])] = window_idx
        # miss-driven admit
        fail = ~succ
        if not fail.any():
            return
        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T
        self.maint_retrieval_cost += int(fail.sum())
        n = 0
        for qi in range(pool_sims.shape[0]):
            top1 = int(np.argmax(pool_sims[qi]))
            cand_did = self.p2d[top1]
            if cand_did in self.kb:
                self.last_hit[top1] = window_idx
                continue
            # evict the LRU KB doc
            victim = min(self.kb, key=lambda d: self.last_hit.get(self.d2p[d], -1))
            vp = self.d2p[victim]
            self.kb.discard(victim)
            self.last_hit.pop(vp, None)
            self.kb.add(cand_did)
            self.last_hit[top1] = window_idx
            n += 1
        self.update_cost += n


class MissTinyLFU(BaseStrategy):
    """Miss-driven TinyLFU: textbook cache + frequency-sketch admission.

    Like MissLRU, but the admission gate uses access frequency:
      - access_freq[d]: incremented when d serves a successful query (in KB)
      - fetch_freq[d]: incremented each time d is fetched as top-1 for a
        failing query (across all windows, even when not admitted)
      - admit candidate c (replacing victim e) iff fetch_freq[c] >= access_freq[e]
      - eviction order: lowest access_freq first
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.access_freq = {}
        self.fetch_freq = {}

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self.access_freq[self.d2p[did]] = 0

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
        succ = max_s >= SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                self.access_freq[int(kb_idx[pos])] = self.access_freq.get(int(kb_idx[pos]), 0) + 1
        fail = ~succ
        if not fail.any():
            return
        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T
        self.maint_retrieval_cost += int(fail.sum())
        n = 0
        for qi in range(pool_sims.shape[0]):
            top1 = int(np.argmax(pool_sims[qi]))
            self.fetch_freq[top1] = self.fetch_freq.get(top1, 0) + 1
            cand_did = self.p2d[top1]
            if cand_did in self.kb:
                self.access_freq[top1] = self.access_freq.get(top1, 0) + 1
                continue
            victim = min(self.kb, key=lambda d: self.access_freq.get(self.d2p[d], 0))
            vp = self.d2p[victim]
            cand_f = self.fetch_freq.get(top1, 0)
            vict_f = self.access_freq.get(vp, 0)
            if cand_f < vict_f:
                continue  # TinyLFU gate rejects
            self.kb.discard(victim)
            self.access_freq.pop(vp, None)
            self.kb.add(cand_did)
            self.access_freq[top1] = 0
            n += 1
        self.update_cost += n


STRATEGY_FACTORIES.update({
    'MissLRU':     lambda dp, de, ti: MissLRU('MissLRU', dp, de, ti),
    'MissTinyLFU': lambda dp, de, ti: MissTinyLFU('MissTinyLFU', dp, de, ti),
})


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
            fail = max_s < SF_HIT_THRESH
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


STRATEGY_FACTORIES.update({
    'RoutedCache': lambda dp, de, ti: RoutedCache('RoutedCache', dp, de, ti),
})


class DRYAD(RoutedCache):
    """DRYAD — Drift-aware Demand-driven Admission with entity-chained prefetch.

    The full final method = three modules on one pipeline:

      ① DETECT  — explicit drift signal from the query-KB AlignmentGap
                  G(t) = 1 - mean_q max_{d in KB} sim(q, d), with an
                  EMA/MAD adaptive threshold (a lightweight, KB-rebuild-free
                  instantiation of the QARC DriftLens idea; the full
                  alignment-feature FID lives in updator/qarc and is the
                  drop-in upgrade).
      ② DECIDE  — rule agent maps the drift signal to an action + replacement
                  ratio λ, which becomes the per-window write budget λ·B:
                    NoOp        (gap normal, no drift)      -> budget 0
                    Mild        (gap above EMA+k·MAD)       -> λ_mild·B
                    Aggressive  (drift: gap >> baseline)    -> λ_aggr·B
                    warmup windows                          -> Aggressive
      ③ ADMIT   — inherited from RoutedCache: R1 (SemFlow neighborhood demand)
                  + R3 (entity-chained bridge prefetch) competing in one
                  demand/serve ledger under the unified gate, now capped by
                  the budget λ·B from module ②.

    This replaces SemFlow's implicit "write cap = #failures" with an explicit
    detect→decide→admit loop, so the system updates *when alignment drifts*
    and *as hard as the drift warrants*, not on every miss.
    """
    # Module ② decision hyperparameters (mirror updator/qarc kb_agent defaults)
    WARMUP_WINDOWS = 3
    GAP_EMA_BETA   = 0.85   # EMA smoothing for the gap baseline
    GAP_K          = 1.0    # MAD multiplier for the "gap high" threshold
    LAMBDA_MILD    = 0.05   # mild update: 5% of budget B
    LAMBDA_AGGR    = 0.15   # aggressive update: 15% of budget B
    DRIFT_GAP_MULT = 1.5    # gap > DRIFT_GAP_MULT * EMA  => treat as drift

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self._budget = None          # B = KB capacity, set on first step
        self._gap_ema = None         # EMA of AlignmentGap
        self._gap_mad = 0.0          # MAD-like deviation tracker
        self._win = 0                # window counter (for warmup)
        self.drift_log = []          # per-window (gap, action) for analysis

    def _alignment_gap(self, nqe, kb_emb):
        """G(t) = 1 - mean_q max_{d in KB} sim(q, d)."""
        q_kb = nqe @ kb_emb.T
        return float(1.0 - np.mean(np.max(q_kb, axis=1)))

    def _decide(self, gap):
        """Module ②: map gap -> (action, lambda) with EMA/MAD adaptive thresh."""
        # Warmup: aggressive while the baseline is still being established.
        if self._win < self.WARMUP_WINDOWS:
            action, lam = 'Aggressive', self.LAMBDA_AGGR
        else:
            ema = self._gap_ema if self._gap_ema is not None else gap
            thresh = ema + self.GAP_K * self._gap_mad
            if gap > self.DRIFT_GAP_MULT * ema:
                action, lam = 'Aggressive', self.LAMBDA_AGGR   # drift
            elif gap > thresh:
                action, lam = 'Mild', self.LAMBDA_MILD          # gap high
            else:
                action, lam = 'NoOp', 0.0                       # stable
        # Update EMA / MAD trackers.
        if self._gap_ema is None:
            self._gap_ema = gap
        else:
            dev = abs(gap - self._gap_ema)
            self._gap_mad = self.GAP_EMA_BETA * self._gap_mad + (1 - self.GAP_EMA_BETA) * dev
            self._gap_ema = self.GAP_EMA_BETA * self._gap_ema + (1 - self.GAP_EMA_BETA) * gap
        return action, lam

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        if self._budget is None:
            self._budget = len(self.kb)   # B = KB capacity
        # ── Module ① DETECT ──
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)
        gap = self._alignment_gap(nqe, kb_emb)
        # ── Module ② DECIDE ──
        action, lam = self._decide(gap)
        self._win += 1
        self.drift_log.append((round(gap, 4), action))
        if action == 'NoOp':
            # Still credit serve for hits so useful docs are protected, but
            # write nothing this window.
            self._write_budget = 0
            super().step(window_queries, window_query_embs, window_idx)
            self._write_budget = None
            return
        # ── Module ③ ADMIT/EVICT under budget λ·B (R1 + R3 via RoutedCache) ──
        self._write_budget = max(1, int(lam * self._budget))
        super().step(window_queries, window_query_embs, window_idx)
        self._write_budget = None


STRATEGY_FACTORIES.update({
    'DRYAD': lambda dp, de, ti: DRYAD('DRYAD', dp, de, ti),
})

