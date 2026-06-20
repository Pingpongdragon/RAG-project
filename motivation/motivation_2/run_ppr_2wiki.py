"""
真实嵌入 (bge-large-en-v1.5) + 真实 2wiki + spaCy 实体, 跑 stock vs local-PPR。
复用 motivation_2 的 loaders / 嵌入缓存 / 实体抽取基础设施。

运行 (在 ljy_rag_ft env, motivation_2 目录下):
  python run_ppr_2wiki.py --dataset 2wikimultihopqa --n-source 1500
"""
import os, sys, argparse
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(THIS, "..", "..")))  # project root (lower priority)
sys.path.insert(0, THIS)                                   # motivation_2 (highest priority)

import loaders
import utils
from graph_retrieval import extract_pool_entities
from algorithms.drip.cache_manager import DRIPCore
from algorithms.drip.tests.test_bridge_ppr import LocalPPR, PPRDRIPCore

WINDOW = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wikimultihopqa")
    ap.add_argument("--n-source", type=int, default=1500)
    ap.add_argument("--kb-budget", type=int, default=300)
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--K0", type=int, default=5)
    ap.add_argument("--d-cap", type=int, default=30)
    ap.add_argument("--workload", default="temporal_bridge_reuse")
    ap.add_argument("--drift-mode", default="sudden")
    args = ap.parse_args()

    from config import DATASET_CONFIGS
    base_key = "2wikimultihopqa" if "2wiki" in args.dataset else args.dataset.split("_")[0]
    cfg = dict(DATASET_CONFIGS.get(base_key, {}))
    cfg.setdefault("n_windows", 50)
    cfg.setdefault("window_size", 50)
    nw, ws = cfg["n_windows"], cfg["window_size"]

    print(f"加载 {args.dataset} (n_source={args.n_source}) ...")
    loader = loaders.LOADERS[args.dataset]
    out = loader(n_source=args.n_source)
    doc_pool, queries = (out[0], out[1])

    tag = f"{args.dataset}_{nw}w_{ws}s"
    print(f"  pool={len(doc_pool)}  queries={len(queries)}")

    # 真实 bge 嵌入 (带缓存, 与 mo2 同 tag 复用)
    print("编码 (bge-large-en-v1.5, 带缓存) ...")
    doc_embs, query_embs = utils.compute_embeddings(doc_pool, queries, tag=tag)
    doc_embs = doc_embs.astype("f")
    query_embs = query_embs.astype("f")

    # spaCy 实体
    print("抽取实体 (spaCy NER, 带缓存) ...")
    pool_ents = extract_pool_entities(doc_pool, tag=tag)

    title_to_idx = {d["title"]: i for i, d in enumerate(doc_pool)}

    # ── mo2 标准查询流 + KB 预算公式 (复用 utils) ──
    stream, centroids, head_set = utils.build_query_stream(
        queries, query_embs, cfg, workload=args.workload, drift_mode=args.drift_mode)
    t2l = {d["title"]: i for i, d in enumerate(doc_pool)}
    head_ctx = set()
    for q in stream:
        if not q.get("is_tail", True):
            for t in q.get("ctx_titles", []):
                if t in t2l:
                    head_ctx.add(t2l[t])
    kb_head_mult = cfg.get("kb_head_mult", 1.2)
    kb_budget = max(300, int(round(len(head_ctx) * kb_head_mult / 50)) * 50)
    init_ids = utils.head_biased_init_kb(
        doc_pool, doc_embs, centroids, head_set, kb_budget, stream)
    print(f"  [标准规模] stream={len(stream)} ({nw}x{ws})  "
          f"head_ctx={len(head_ctx)}  KB预算={kb_budget}")

    # bridge gold 第二跳位置 (取流中实际出现的 query 的 sf_titles)
    gold_b = set()
    for q in stream:
        for t in q.get("sf_titles", []):
            pi = title_to_idx.get(t)
            if pi is not None:
                gold_b.add(pi)
    init_pos = {title_to_idx[d["title"]] for d in doc_pool if d["doc_id"] in init_ids}
    print(f"  bridge gold docs={len(gold_b)}, 初始已在KB={len(gold_b & init_pos)}")

    # 流中每个 query 用 qidx 映射回嵌入
    qstream = [{"question": q["question"], "qtype": "bridge",
                "sf_titles": q.get("sf_titles", []), "_qidx": q["qidx"]}
               for q in stream]

    def run(core, label):
        for w0 in range(0, len(qstream), WINDOW):
            win = qstream[w0:w0 + WINDOW]
            qe = np.array([query_embs[q["_qidx"]] for q in win], dtype="f")
            n = np.clip(np.linalg.norm(qe, axis=1, keepdims=True), 1e-9, None)
            core.step(win, qe / n, w0 // WINDOW)
        kb_pos = {core.d2p[d] for d in core.kb}
        hit = len(gold_b & kb_pos)
        print(f"[{label:14s}] bridge gold in KB: {hit:3d}/{len(gold_b)} "
              f"({hit/len(gold_b)*100:.1f}%)  KB={len(core.kb)}  writes={core.update_cost}")
        return hit

    def make(cls, **kw):
        core = cls("DRIP", doc_pool, doc_embs, title_to_idx, **kw)
        core.graph_index.set_pool_entities(pool_ents)
        core.graph_index.build()
        core.set_kb(init_ids)
        return core

    print("=" * 64)
    stock = run(make(DRIPCore), "stock(单边门)")
    ppr = run(make(PPRDRIPCore, ppr_kwargs=dict(
        c=args.c, L=args.L, R=args.R, K0=args.K0, d_cap=args.d_cap)), "local-PPR")
    print("=" * 64)
    print(f"  stock     : {stock}/{len(gold_b)}")
    print(f"  local-PPR : {ppr}/{len(gold_b)}")
    d = ppr - stock
    print(f"  增量       : {d:+d}  ({d/len(gold_b)*100:+.1f}pp)")
    print("=" * 64)


if __name__ == "__main__":
    main()
