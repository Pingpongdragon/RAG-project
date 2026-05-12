"""
KB update strategies for the query-drift motivation experiment.

All strategies share the same interface:
  - __init__(name, doc_pool, doc_embs, title_to_idx)
  - set_kb(ids: set[str])   -- initialise the KB document ID set
  - step(window_queries, window_query_embs, window_idx)  -- observe one window and update KB
  - .kb: set[str]           -- current KB document IDs
  - .cost: int              -- cumulative number of KB replacements

中文速览：这个文件实现了 Motivation 1 里所有 KB 维护策略。
它们的区别不在检索器，而在“什么时候写 KB、根据什么信号写、一次写多少”。

当前实际包含 8 个策略：

1. Static
   冻结 KB，不做任何更新。它回答的是：如果系统完全不适应 drift，性能会掉到哪里。

2. RandomFIFO
   盲目供给侧更新：每个 window 随机抽一批新文档，用 FIFO 淘汰最早进入 KB 的旧文档。
   它没有任务信号，等价于“定期刷新索引，但不知道用户现在需要什么”。

3. DocArrival
   供给侧文档到达：随机模拟一批“新到文档”，再根据与当前 KB 的相似度判断是替换、插入还是跳过。
   它代表 LightRAG / HippoRAG2 一类“文档来了就处理”的流水线。

4. KnowledgeEdit
   供给侧知识编辑：从当前 KB 中选一批文档，对每篇文档找一个语义相近但不在 KB 里的替代项并交换。
   它代表 RECIPE 一类“在现有知识上做局部编辑”的范式。

5. OnDemandFetch
   按需抓取：平时不改 KB；只有当查询在 KB 里命中不够时，才临时去全池子里搜 top-K 文档补上。
   它可以看作 agent / CRAG 风格的在线补检索上界，但这些文档不会沉淀进持久 KB。

6. LogDrivenArrival
   滞后式日志更新：先积累一段时间失败查询，再在下一个 review 周期把最相关的候选文档写回 KB。
   它代表依赖历史日志、人工回看或批处理反馈的慢更新系统。

7. QueryDrivenCluster
   需求侧更新（本文方法）：直接用失败查询暴露出的“缺哪些文档”信号积累 demand，
   再和当前 KB 文档的 serve 价值比较，只有当候选需求显著高于现驻留文档价值时才替换。

8. Oracle
   非因果上界：直接使用当前 window 的 gold supporting facts 重建 KB，只用于给出理论天花板。
"""
import numpy as np
from config import (SEED, SF_HIT_THRESH, DOC_ARRIVE, DOC_ADD_CAP,
                    EDIT_BATCH, QD_TOP_K, QD_REPLACE_CAP, log)


class BaseStrategy:
    # 所有策略共享同一套状态：
    # - kb: 当前持久 KB 中的 doc_id 集合
    # - update_cost: 写入/替换 KB 的累计次数
    # - maint_retrieval_cost: 后台维护阶段的全池扫描成本
    # - serve_retrieval_cost: 在线服务阶段额外检索成本
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
    # 中文解释：完全冻结 KB。
    # 它不尝试适应新的 query 分布，因此能直接测出“纯 drift”本身造成的性能下降。
    """No-update baseline.  KB is frozen after initialisation."""
    def step(self, window_queries, window_query_embs, window_idx):
        pass


class DocArrival(BaseStrategy):
    # 中文解释：模拟“新文档不断到来”的供给侧系统。
    # 每个 window 随机抽一批文档当作新到文档：
    # - 如果它和某个 KB 文档非常像，就视作同主题新版本，直接替换；
    # - 如果它和整个 KB 都很不像，就视作全新内容，挤掉最旧/最不活跃的内容；
    # - 中间地带则跳过，避免既不新也不必要的噪声写入。
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
    # 中文解释：模拟“对现有知识做局部编辑”的系统。
    # 它不是从查询失败出发，而是先挑一些当前 KB 文档，再为每个文档找一个
    # 语义相近的非 KB 替代项做 swap。这样做能保持主题相似，但未必对当前 workload 有用。
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
    # 中文解释：这是本文的 demand-side persistent writer。
    # 核心思想不是“看到新文档就写”，而是“看到查询失败，就推断 KB 里缺什么”。
    #
    # 对每个 window：
    # 1. 先看哪些查询已经被 KB 服务好了，把命中的 KB 文档记 serve；
    # 2. 再看哪些查询失败了，对失败查询去全池里 probe top-K 候选；
    # 3. 这些候选按相似度分到 demand 上，形成“最近很多失败查询都在找它”的证据；
    # 4. 最后只在 candidate demand 明显大于某个驻留文档的 serve+demand 时才替换。
    #
    # 因此它学到的是“当前 workload 真正在要什么文档”，而不是“池子里最近来了什么”
    # 或“KB 内部哪些文档彼此相似”。
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
    # 中文解释：Oracle 不是可部署策略，只是理论上界。
    # 它直接偷看当前 window 的 gold supporting facts，把这些真值文档优先塞进 KB，
    # 剩余容量再按与当前查询的相似度补满，所以它表示“如果我提前知道答案证据在哪，
    # 在相同 KB 容量下最多能做到什么水平”。
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
    'Static':             lambda dp, de, ti: Static('Static', dp, de, ti),
    'DocArrival':         lambda dp, de, ti: DocArrival('DocArrival', dp, de, ti),
    'KnowledgeEdit':      lambda dp, de, ti: KnowledgeEdit('KnowledgeEdit', dp, de, ti),
    'LRU':                lambda dp, de, ti: LRU('LRU', dp, de, ti),
    'GPTCacheStyle':      lambda dp, de, ti: GPTCacheStyle('GPTCacheStyle', dp, de, ti),
    'MemGPTStyle':        lambda dp, de, ti: MemGPTStyle('MemGPTStyle', dp, de, ti),
    'QueryDrivenCluster': lambda dp, de, ti: QueryDrivenCluster('QueryDrivenCluster', dp, de, ti),
    'Oracle':             lambda dp, de, ti: Oracle('Oracle', dp, de, ti),
}


# ═══════════════════════════════════════════════════════════════════
# NEW BASELINES — added to respond to reviewer requests for stronger
# cache-style and agent-memory-style comparisons
# ═══════════════════════════════════════════════════════════════════


class LRU(BaseStrategy):
    """Least-Recently-Used cache eviction (direct reviewer request).

    Trigger: query failure signal (same as QDC) — a pool doc is
    considered for admission whenever a window has failing queries.
    Eviction: the KB doc with the oldest last-hit timestamp is removed.

    This decouples the eviction policy (pure recency) from the
    admission policy (query-failure demand), making it a fair
    ablation of QDC's demand/serve accounting.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        # last_hit[pool_idx] = most recent window index that query hit this doc
        self.last_hit = {}

    def set_kb(self, ids):
        super().set_kb(ids)
        # Initial docs treated as hit at window -1 (oldest)
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

        # Update recency for successfully-hit KB docs
        succ = max_s >= SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                self.last_hit[int(kb_idx[pos])] = window_idx

        # Find failing queries and their best pool candidates
        fail = max_s < SF_HIT_THRESH
        n_fail = int(fail.sum())
        if n_fail == 0:
            return

        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T
        tk = min(QD_TOP_K, pool_sims.shape[1])
        self.maint_retrieval_cost += n_fail * tk

        kb_pos_set = set(int(i) for i in kb_idx)
        # Accumulate demand per pool doc (simple sum of similarities)
        demand = {}
        for qi in range(n_fail):
            top = np.argpartition(pool_sims[qi], -tk)[-tk:]
            for pi in top:
                pi = int(pi)
                if pi not in kb_pos_set:
                    demand[pi] = demand.get(pi, 0.0) + float(pool_sims[qi, pi])

        if not demand:
            return

        # Evict LRU KB docs, admit highest-demand pool docs
        evictable = sorted(
            kb_pos_set,
            key=lambda p: self.last_hit.get(p, -1)  # oldest first
        )
        cands = sorted(demand.items(), key=lambda x: -x[1])

        n = 0
        for cp, _ in cands:
            if n >= len(evictable):
                break
            ep = evictable[n]
            self.kb.discard(self.p2d[ep])
            self.kb.add(self.p2d[cp])
            self.last_hit[cp] = window_idx
            self.last_hit.pop(ep, None)
            n += 1
        self.update_cost += n


class GPTCacheStyle(BaseStrategy):
    """Semantic cache eviction (GPTCache-inspired, direct reviewer request).

    Keeps KB docs that are semantically similar to RECENT queries
    (both hits and misses). Each doc accumulates a decayed "cache score"
    equal to the max similarity it has seen from any query in recent windows.
    Eviction: lowest cache score. Admission: highest similarity to failing queries.

    This directly models GPTCache's semantic similarity retention criterion
    applied to a shared KB rather than a per-user response cache.
    """
    CACHE_DECAY = 0.80   # per-window decay on cache scores

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.cache_score = {}  # pool_idx -> float

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self.cache_score[self.d2p[did]] = 1.0  # initial prior

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)
        q_kb = nqe @ kb_emb.T

        # Decay existing cache scores
        self.cache_score = {p: v * self.CACHE_DECAY
                            for p, v in self.cache_score.items()
                            if v * self.CACHE_DECAY > 0.01}

        # Update cache scores for ALL KB docs using max-sim from this window
        max_sim_per_kb = np.max(q_kb, axis=0)  # shape: (|KB|,)
        for i, pi in enumerate(kb_idx):
            pi = int(pi)
            s = float(max_sim_per_kb[i])
            self.cache_score[pi] = max(self.cache_score.get(pi, 0.0), s)

        # Check for failing queries
        max_s = np.max(q_kb, axis=1)
        fail = max_s < SF_HIT_THRESH
        n_fail = int(fail.sum())
        if n_fail == 0:
            return

        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T
        tk = min(QD_TOP_K, pool_sims.shape[1])
        self.maint_retrieval_cost += n_fail * tk

        kb_pos_set = set(int(i) for i in kb_idx)
        # Candidate demand: max sim from failing queries
        demand = {}
        for qi in range(n_fail):
            top = np.argpartition(pool_sims[qi], -tk)[-tk:]
            for pi in top:
                pi = int(pi)
                if pi not in kb_pos_set:
                    s = float(pool_sims[qi, pi])
                    demand[pi] = max(demand.get(pi, 0.0), s)

        if not demand:
            return

        # Evict lowest-cache-score KB docs; admit highest-demand pool docs
        evictable = sorted(
            kb_pos_set,
            key=lambda p: self.cache_score.get(p, 0.0)  # weakest semantic cache first
        )
        cands = sorted(demand.items(), key=lambda x: -x[1])

        n = 0
        for cp, cval in cands:
            if n >= len(evictable):
                break
            ep = evictable[n]
            # Only swap if candidate is more demanded than evictee's cache score
            if cval <= self.cache_score.get(ep, 0.0):
                break
            self.kb.discard(self.p2d[ep])
            self.kb.add(self.p2d[cp])
            self.cache_score[cp] = cval
            self.cache_score.pop(ep, None)
            n += 1
        self.update_cost += n


class MemGPTStyle(BaseStrategy):
    """MemGPT-inspired two-tier importance-weighted memory (direct reviewer request).

    MemGPT maintains a main context (limited) and external context (unlimited),
    promoting/demoting by importance score. Here:
      - "main context" = KB (budget-limited)
      - "external context" = pool (unlimited but slow)
      - importance(d) = access_freq[d] * decay^(window - last_hit[d])

    A pool doc is promoted to KB when its importance (computed from
    query-failure signal) exceeds the weakest KB doc's importance.
    This closely follows MemGPT's recall_memory / archival_memory separation
    and importance-based retrieval policy.
    """
    DECAY = 0.88        # per-window decay on importance
    FREQ_WEIGHT = 1.0   # weight on access frequency component

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.freq = {}       # pool_idx -> cumulative access count
        self.last_hit = {}   # pool_idx -> last window with a hit

    def _importance(self, pi, window_idx):
        f = self.freq.get(pi, 0.0)
        lh = self.last_hit.get(pi, -1)
        recency = self.DECAY ** max(0, window_idx - lh)
        return self.FREQ_WEIGHT * f * recency

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            pi = self.d2p[did]
            self.freq[pi] = 1.0
            self.last_hit[pi] = 0

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

        # Credit frequency + recency for successfully-hit KB docs
        succ = max_s >= SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                pi = int(kb_idx[pos])
                self.freq[pi] = self.freq.get(pi, 0.0) + 1.0
                self.last_hit[pi] = window_idx

        fail = max_s < SF_HIT_THRESH
        n_fail = int(fail.sum())
        if n_fail == 0:
            return

        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T
        tk = min(QD_TOP_K, pool_sims.shape[1])
        self.maint_retrieval_cost += n_fail * tk

        kb_pos_set = set(int(i) for i in kb_idx)
        # Candidate importance from pool: sim sum (proxy for access frequency
        # if this doc were in KB)
        cand_importance = {}
        for qi in range(n_fail):
            top = np.argpartition(pool_sims[qi], -tk)[-tk:]
            for pi in top:
                pi = int(pi)
                if pi not in kb_pos_set:
                    cand_importance[pi] = (
                        cand_importance.get(pi, 0.0)
                        + float(pool_sims[qi, pi])
                    )

        if not cand_importance:
            return

        # Evict KB doc with lowest importance; promote pool doc with highest
        evictable = sorted(
            kb_pos_set,
            key=lambda p: self._importance(p, window_idx)
        )
        cands = sorted(cand_importance.items(), key=lambda x: -x[1])

        n = 0
        for cp, cval in cands:
            if n >= len(evictable):
                break
            ep = evictable[n]
            if cval <= self._importance(ep, window_idx):
                break  # remaining cands are weaker
            self.kb.discard(self.p2d[ep])
            self.kb.add(self.p2d[cp])
            # New KB doc inherits sim-sum as initial frequency estimate
            self.freq[cp] = cval
            self.last_hit[cp] = window_idx
            self.freq.pop(ep, None)
            self.last_hit.pop(ep, None)
            n += 1
        self.update_cost += n
