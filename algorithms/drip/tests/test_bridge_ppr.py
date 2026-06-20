"""
局部 PPR bridge 实验: 真正实现 LOCAL_PPR_SUPPORT_FLOW.md 的组件 0-3。
对比 stock DRIPCore (单边 relation 门, 杀光第二跳候选)
 vs  PPRDRIPCore  (局部子图 + 转移矩阵 + 截断幂迭代, E_graph = pi^(L))

代码里每段都标注对应设计文档的【组件 N】, 方便对照公式。
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.drip.cache_manager import DRIPCore
from algorithms.drip.cache_manager.graph_index import GraphIndex

np.random.seed(0)
DIM = 48

def _unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)

# ----------------------------------------------------------------------
# 合成多跳语料: q_k 直接相似于 A_k; gold 第二跳 B_k 与 q_k 正交(够不着),
# 但 A_k 与 B_k 共享稀有实体 shared_ent_k (桥)。大量噪声占 KB 名额。
# ----------------------------------------------------------------------
N_BRIDGE = 12
N_NOISE = 120
KB_BUDGET = 24

docs = []
def add(did, title, emb, ents):
    docs.append({"doc_id": did, "title": title, "text": title,
                 "_emb": emb, "_ents": ents})

basis = _unit(np.random.randn(N_BRIDGE, DIM))
for k in range(N_BRIDGE):
    a_emb = _unit(basis[k] + 0.02 * np.random.randn(DIM))
    b_dir = np.random.randn(DIM); b_dir -= (b_dir @ basis[k]) * basis[k]
    b_emb = _unit(b_dir)
    add(f"A_{k}", f"AnchorPerson_{k}", a_emb, {f"shared_ent_{k}", f"anchor_{k}"})
    add(f"B_{k}", f"BridgeFact_{k}",  b_emb, {f"shared_ent_{k}", f"factoid_{k}"})

noise = _unit(np.random.randn(N_NOISE, DIM))
for j in range(N_NOISE):
    add(f"N_{j}", f"Noise_{j}", noise[j], {f"noise_ent_{j}"})

doc_embs = _unit(np.array([d["_emb"] for d in docs], dtype=float))
title_to_idx = {d["title"]: i for i, d in enumerate(docs)}
pool_ents = {d["doc_id"]: d["_ents"] for d in docs}

queries = []
for _round in range(6):
    for k in range(N_BRIDGE):
        a_idx = title_to_idx[f"AnchorPerson_{k}"]
        q_emb = _unit(doc_embs[a_idx] + 0.05 * np.random.randn(DIM))
        queries.append({
            "question": f"bridge_q_{k}_r{_round}",
            "qtype": "bridge",
            "sf_titles": [f"AnchorPerson_{k}", f"BridgeFact_{k}"],
            "_emb": q_emb,
        })
np.random.shuffle(queries)

GOLD_B = {title_to_idx[f"BridgeFact_{k}"] for k in range(N_BRIDGE)}
WINDOW = 6


def initial_kb():
    return set([f"N_{j}" for j in range(KB_BUDGET - 4)] + [f"A_{k}" for k in range(4)])


# ======================================================================
# 局部 PPR 引擎: LOCAL_PPR_SUPPORT_FLOW.md 组件 0-3
# ======================================================================
class LocalPPR:
    """复用 GraphIndex 已建好的 ent_to_docs / doc_to_ents / ent_idf。
    不依赖 relation 词重叠门 —— 这正是要解决 stock 的瓶颈。
    """

    def __init__(self, graph_index, doc_embs, c=0.5, L=3, R=2, K0=5, d_cap=50):
        self.gi = graph_index          # 已 build() 的 GraphIndex
        self.doc_embs = doc_embs
        self.c = c                     # 重启概率
        self.L = L                     # 截断步数 = 有效最大跳数
        self.R = R                     # BFS 半径
        self.K0 = K0                   # 种子数 (dense top-K0)
        self.d_cap = d_cap             # 实体度数硬上限 (截断 hub)

    # --- 组件 1 的边权: w(A,B)=sum_e (idf(e)/idf_max)/degree(e)^p ---
    def _doc_neighbors(self, pi, idf_max, p):
        """返回 {邻居文档: 累计边权}, 复用 graph_index 的实体倒排。"""
        out = {}
        for ent in self.gi.doc_entities(pi):
            docs_e = self.gi.ent_to_docs.get(ent, ())
            deg = len(docs_e)
            if deg == 0 or deg > self.d_cap:      # 组件0: 度数硬上限截断 hub
                continue
            idf = self.gi.ent_idf.get(ent, 1.0)
            w_e = (idf / max(1e-9, idf_max)) / (deg ** p)
            for q in docs_e:
                if q == pi:
                    continue
                out[q] = out.get(q, 0.0) + w_e
        return out

    # --- 组件 0: 以 seeds 为中心做有界 BFS 取邻域 V_q ---
    def _build_subgraph(self, seeds, idf_max, p):
        V = set(seeds)
        frontier = set(seeds)
        adj = {}
        for _ in range(self.R):
            nxt = set()
            for u in frontier:
                if u not in adj:
                    adj[u] = self._doc_neighbors(u, idf_max, p)
                for v in adj[u]:
                    if v not in V:
                        nxt.add(v)
            V |= nxt
            frontier = nxt
            if not frontier:
                break
        for u in V:                               # 补全 V 内节点的邻接
            if u not in adj:
                adj[u] = self._doc_neighbors(u, idf_max, p)
        return V, adj

    # --- 组件 1-3: 转移矩阵 W + 种子 s + 截断幂迭代 -> E_graph = pi^(L) ---
    def evidence(self, first_hops):
        """first_hops: [(pi, sim)]. 返回 {doc: pi^(L)(doc)}。"""
        p = float(getattr(self.gi.config, "entity_degree_power", 0.5))
        idf_max = max(self.gi.ent_idf.values()) if self.gi.ent_idf else 1.0

        # 组件 2: 种子向量 s(d) = max(0,sim)/sum, 仅 seeds 非零
        seeds = [(int(pi), max(0.0, float(sim)))
                 for pi, sim in first_hops[: self.K0] if sim > 0.0]
        if not seeds:
            return {}
        ssum = sum(s for _, s in seeds) or 1.0
        s_vec = {pi: s / ssum for pi, s in seeds}

        V, adj = self._build_subgraph(list(s_vec), idf_max, p)
        idx = {node: i for i, node in enumerate(sorted(V))}
        n = len(idx)
        if n == 0:
            return {}

        # 组件 1: 行归一化转移矩阵 W
        W = np.zeros((n, n), dtype=float)
        for u, nbrs in adj.items():
            iu = idx[u]
            for v, w in nbrs.items():
                if v in idx:
                    W[iu, idx[v]] = w
        rs = W.sum(axis=1, keepdims=True)
        W = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)

        s = np.zeros(n)
        for pi, val in s_vec.items():
            s[idx[pi]] = val

        # 组件 3: pi^(l) = c*s + (1-c)*W^T pi^(l-1), 迭代 L 步
        pi_vec = s.copy()
        Wt = W.T
        for _ in range(self.L):
            pi_vec = self.c * s + (1.0 - self.c) * (Wt @ pi_vec)

        # E_graph(d) = pi^(L)(d); 排除种子自身(已是 first-hop, 走 dense 腿)
        seed_set = set(s_vec)
        return {node: float(pi_vec[i]) for node, i in idx.items()
                if node not in seed_set and pi_vec[i] > 0.0}


class PPRDRIPCore(DRIPCore):
    """用局部 PPR 的 pi^(L) 替换 graph_evidence 的单边 relation 门评分。
    其余 (demand 账本/admission) 沿用父类, 以隔离 "E_graph 来源" 这一个变量。
    """

    def __init__(self, *a, ppr_kwargs=None, **kw):
        super().__init__(*a, **kw)
        self._ppr_kwargs = ppr_kwargs or {}
        self._ppr = None

    def _ensure_ppr(self):
        if self._ppr is None:
            self.graph_index.build()
            self._ppr = LocalPPR(self.graph_index, self.doc_embs, **self._ppr_kwargs)
        return self._ppr

    def _credit_graph(self, candidates, gold_pos=None):
        # candidates 此处已是 PPR 产出的 [(B, pi^(L))]
        return super()._credit_graph(candidates, gold_pos)

    def set_kb(self, ids):
        super().set_kb(ids)
        # 安装 PPR 版 graph_evidence (替换单边 relation 门评分)
        ppr = self._ensure_ppr()
        gi = self.graph_index

        def ppr_graph_evidence(query, first_hops, kb_pos, kb_emb, doc_embs):
            gi.last_stats = {"bridge_selected": 0, "bridge_top_entities": []}
            ev = ppr.evidence(first_hops)
            gi.last_stats["bridge_selected"] = len(ev)
            return sorted(ev.items(), key=lambda kv: -kv[1])

        gi.graph_evidence = ppr_graph_evidence


# ======================================================================
# 运行 + 对比
# ======================================================================
def run(core, label, diagnose=False):
    for w0 in range(0, len(queries), WINDOW):
        win = queries[w0:w0 + WINDOW]
        qe = _unit(np.array([q["_emb"] for q in win], dtype=float))
        core.step(win, qe, w0 // WINDOW)

    kb_pos = {core.d2p[d] for d in core.kb}
    b_in_kb = len(GOLD_B & kb_pos)

    if diagnose:
        present = {p: core.demand.get(p, 0.0) for p in GOLD_B if core.demand.get(p, 0.0) > 0}
        print(f"    [诊断] gold-B 拿到 demand: {len(present)}/{len(GOLD_B)}", end="")
        if present:
            vals = list(present.values())
            print(f"  (demand {min(vals):.4f}~{max(vals):.4f})")
        else:
            print()
    print(f"[{label:18s}] B(第二跳) in KB: {b_in_kb:2d}/{len(GOLD_B)}  "
          f"KB={len(core.kb)}  writes={core.update_cost}")
    return b_in_kb


def make_core(cls, **kw):
    core = cls("DRIP", docs, doc_embs, title_to_idx, **kw)
    core.graph_index.set_pool_entities(pool_ents)
    core.graph_index.build()
    core.set_kb(initial_kb())
    return core


if __name__ == "__main__":
    print("=" * 64)
    print(f"局部 PPR bridge 实验: {N_BRIDGE} 三元组, {len(queries)} 查询, "
          f"KB={KB_BUDGET}, 窗口={WINDOW}")
    print("=" * 64)
    stock = run(make_core(DRIPCore), "stock(单边门)", diagnose=True)
    print()
    ppr = run(make_core(PPRDRIPCore, ppr_kwargs=dict(c=0.5, L=3, R=2, K0=5)),
              "local-PPR", diagnose=True)
    print("\n" + "=" * 64)
    print(f"  stock(单边relation门) : {stock:2d}/{len(GOLD_B)}")
    print(f"  local-PPR(组件0-3)    : {ppr:2d}/{len(GOLD_B)}")
    print(f"  增量                   : {ppr - stock:+d}")
    print("=" * 64)


