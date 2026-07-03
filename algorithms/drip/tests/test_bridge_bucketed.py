"""
受控合成 bridge 实验: 验证「分桶预算 + S_brg」是否改善 H2(第二跳) 命中。
对比 stock DRIPCore (合并预算, bridge 赚不到 serve)
 vs  BucketedDRIPCore (按 route 分桶预算 + bridge serve 腿)

只验证机制(协议第1步), 不改变 bridge evidence。
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.drip.cache_manager import DRIPCore


class BucketedDRIPCore(DRIPCore):
    """协议第1步变体: 按 route 分桶预算 + 记录 bridge 候选来源。

    只覆写 _credit_graph (打标 bridge 候选) 和 _write (分桶 admission)。
    不改任何生产代码。
    """

    def __init__(self, *a, bridge_reserve=0.5, **kw):
        super().__init__(*a, **kw)
        self._bridge_keys = set()
        self._bridge_reserve = bridge_reserve

    def _credit_graph(self, candidates, gold_pos=None):
        for pi, score in candidates:
            if float(score) > 0.0:
                self._bridge_keys.add(int(pi))
        return super()._credit_graph(candidates, gold_pos)

    def _write(self, kb_idx, kb_emb, budget, gold_pos=None):
        if budget <= 0:
            return super()._write(kb_idx, kb_emb, budget, gold_pos)
        kb_pos = set(int(p) for p in kb_idx)
        priority = self._resident_priority(kb_idx, kb_emb)
        victims = sorted(kb_pos, key=lambda p: priority[p])

        # 候选按来源分桶
        all_cand = sorted(((v, p) for p, v in self.demand.items()
                           if p not in kb_pos), reverse=True)
        brg_cand = [(v, p) for v, p in all_cand if p in self._bridge_keys]
        dir_cand = [(v, p) for v, p in all_cand if p not in self._bridge_keys]

        budget_brg = int(np.ceil(self._bridge_reserve * budget)) if brg_cand else 0
        budget_dir = budget - budget_brg
        if os.environ.get("DRIP_DEBUG"):
            nb = sum(1 for _, p in brg_cand if p in GOLD_B)
            print(f"      _write: budget={budget} brg_cand={len(brg_cand)}"
                  f"(含gold-B {nb}) dir_cand={len(dir_cand)} "
                  f"budget_brg={budget_brg} bridge_keys={len(self._bridge_keys)}")

        writes = 0
        gold_writes = 0
        victim_i = 0

        def admit_from(cands, cap):
            nonlocal writes, gold_writes, victim_i
            local = 0
            for cand_value, cp in cands:
                if local >= cap or victim_i >= len(victims):
                    break
                victim = victims[victim_i]
                gain = cand_value - self.config.gain_margin * priority[victim]
                if gain <= 0.0:
                    break
                cur = self.doc_embs[np.array(sorted(kb_pos), dtype=np.int64)]
                dup = float((self.doc_embs[cp] @ cur.T).max())
                if dup > self.config.tau_duplicate:
                    continue
                self.kb.discard(self.p2d[victim])
                self.kb.add(self.p2d[cp])
                self.serve.pop(victim, None)
                kb_pos.discard(victim)
                kb_pos.add(cp)
                victim_i += 1
                writes += 1
                local += 1
                gold_writes += int(int(cp) in (gold_pos or set()))

        admit_from(brg_cand, budget_brg)   # bridge 桶先用 (预留额度)
        admit_from(dir_cand, budget_dir)

        cand_n = len(all_cand)
        gold_cand = sum(1 for _, p in all_cand if int(p) in (gold_pos or set()))
        return {
            "writes": int(writes),
            "candidates": int(cand_n),
            "gold_candidates": int(gold_cand),
            "gold_writes": int(gold_writes),
            "gold_rate": float(gold_writes / writes) if writes else 0.0,
        }

np.random.seed(0)
DIM = 48

def _unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)

# ----------------------------------------------------------------------
# 构造合成多跳语料:
#   每个 bridge query q_k 直接相似于 first-hop 文档 A_k,
#   真正的 gold 第二跳是 B_k, 它和 q_k 在 embedding 上"够不着",
#   但 A_k 与 B_k 共享一个稀有实体 e_k (桥)。
#   另有大量噪声文档占据 KB 名额。
# ----------------------------------------------------------------------
N_BRIDGE = 12          # bridge 三元组数量
N_NOISE = 120          # 噪声文档
KB_BUDGET = 24

docs = []              # {doc_id, title, text, _emb, _ents}
def add(did, title, emb, ents):
    docs.append({"doc_id": did, "title": title, "text": title,
                 "_emb": emb, "_ents": ents})

# A/B 三元组
basis = _unit(np.random.randn(N_BRIDGE, DIM))
for k in range(N_BRIDGE):
    a_emb = _unit(basis[k] + 0.02 * np.random.randn(DIM))
    # B 与 A 近似正交 (query 够不着), 自己一个子空间
    b_dir = np.random.randn(DIM); b_dir -= (b_dir @ basis[k]) * basis[k]
    b_emb = _unit(b_dir)
    add(f"A_{k}", f"AnchorPerson_{k}", a_emb, {f"shared_ent_{k}", f"anchor_{k}"})
    add(f"B_{k}", f"BridgeFact_{k}",  b_emb, {f"shared_ent_{k}", f"factoid_{k}"})

# 噪声文档
noise = _unit(np.random.randn(N_NOISE, DIM))
for j in range(N_NOISE):
    add(f"N_{j}", f"Noise_{j}", noise[j], {f"noise_ent_{j}"})

doc_embs = _unit(np.array([d["_emb"] for d in docs], dtype=float))
title_to_idx = {d["title"]: i for i, d in enumerate(docs)}
pool_ents = {d["doc_id"]: d["_ents"] for d in docs}

# bridge 查询流: query 贴近 A_k, gold = B_k (第二跳), 重复多轮制造历史
queries = []
for _round in range(6):
    for k in range(N_BRIDGE):
        a_idx = title_to_idx[f"AnchorPerson_{k}"]
        q_emb = _unit(doc_embs[a_idx] + 0.05 * np.random.randn(DIM))
        queries.append({
            "question": (
                f"Whose missing fact is connected to AnchorPerson_{k} "
                f"in round {_round}?"
            ),
            "qtype": "bridge",
            "sf_titles": [f"AnchorPerson_{k}", f"BridgeFact_{k}"],
            "_emb": q_emb,
            "_goldB": title_to_idx[f"BridgeFact_{k}"],
        })
np.random.shuffle(queries)

GOLD_B = {title_to_idx[f"BridgeFact_{k}"] for k in range(N_BRIDGE)}
WINDOW = 6


def initial_kb():
    # 初始 KB: 全噪声 + 少量 A, 故意不含任何 B
    ids = [f"N_{j}" for j in range(KB_BUDGET - 4)] + [f"A_{k}" for k in range(4)]
    return set(ids)


def run(core_cls, label, diagnose=False):
    core = core_cls("DRIP", docs, doc_embs, title_to_idx)
    core.graph_index.set_pool_entities(pool_ents)
    core.graph_index.build()
    core.set_kb(initial_kb())

    for w0 in range(0, len(queries), WINDOW):
        win = queries[w0:w0 + WINDOW]
        qe = _unit(np.array([q["_emb"] for q in win], dtype=float))
        core.step(win, qe, w0 // WINDOW)

    kb_pos = {core.d2p[d] for d in core.kb}
    b_in_kb = len(GOLD_B & kb_pos)

    if diagnose:
        # B 是否进入了 demand 账本? demand 多大? victim priority 多大?
        b_demand = {p: core.demand.get(p, 0.0) for p in GOLD_B}
        present = {p: v for p, v in b_demand.items() if v > 0}
        print(f"    [诊断] gold-B 拿到 demand 的数量: {len(present)}/{len(GOLD_B)}")
        if present:
            vals = list(present.values())
            print(f"    [诊断] gold-B demand 范围: "
                  f"{min(vals):.4f} ~ {max(vals):.4f}")
        # 当前 KB 内 priority 最低者 (victim 候选)
        kb_idx = np.array(sorted(kb_pos), dtype=np.int64)
        prio = core._resident_priority(kb_idx, doc_embs[kb_idx])
        lo = sorted(prio.values())[:5]
        print(f"    [诊断] KB 最低 priority 前5: {[round(x,4) for x in lo]}")
        # 非 KB 候选里 demand 最高的几个 (admission 实际会选谁)
        cand = sorted(((v, p) for p, v in core.demand.items()
                       if p not in kb_pos), reverse=True)[:5]
        print(f"    [诊断] 非KB候选 demand top5: "
              f"{[(round(v,4), 'B' if p in GOLD_B else 'other') for v,p in cand]}")

    print(f"[{label:22s}] B(第二跳) in KB: {b_in_kb}/{len(GOLD_B)}  "
          f"KB大小={len(core.kb)}  total_writes={core.update_cost}")
    return b_in_kb


if __name__ == "__main__":
    print("=" * 64)
    print(f"合成 bridge 实验: {N_BRIDGE} 三元组, {len(queries)} 查询, "
          f"KB预算={KB_BUDGET}, 窗口={WINDOW}")
    print("目标: 让第二跳 gold 文档 B (query够不着) 被admit进KB")
    print("=" * 64)
    stock = run(DRIPCore, "stock (合并预算)", diagnose=True)
    print()
    bucketed = run(BucketedDRIPCore, "bucketed+S_brg", diagnose=True)
    print("\n" + "=" * 64)
    print(f"  stock     : {stock}/{len(GOLD_B)} 第二跳命中")
    print(f"  bucketed  : {bucketed}/{len(GOLD_B)} 第二跳命中")
    delta = bucketed - stock
    print(f"  增量       : {delta:+d}")
    print("=" * 64)
    # 区分两个病: 上限 = 实际拿到 demand 的 B 数量 (匹配能力上限)
    print("注: bucketed 只能救回'已被 bridge router 找到'的 B;")
    print("    找不到的 B 是匹配病(graph_index judgment), 需更强 bridge evidence 才能解。")
