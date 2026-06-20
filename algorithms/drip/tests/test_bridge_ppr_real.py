"""
真实数据 (hotpotqa_entity_walk, 264 bridge queries) 上的 PPR vs stock 对比。
修复 benchmark 接线 bug: 注入抽取实体 + 透传 sf_titles/qtype。
annotation-free 实体抽取: 大写专有名词短语。
"""
import os, sys, json, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.drip.cache_manager import DRIPCore
from algorithms.drip.tests.test_bridge_ppr import LocalPPR, PPRDRIPCore

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "benchmark", "datasets", "hotpotqa_entity_walk.json")
WINDOW = 8
KB_BUDGET = 200


def extract_ents(text):
    cands = re.findall(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}", text or "")
    return {c.lower() for c in cands if len(c) >= 3}


def load():
    d = json.load(open(os.path.abspath(DATA)))
    docs_raw = d["document_pool"]
    docs = [{"doc_id": x["doc_id"], "title": x.get("title", x["doc_id"]),
             "text": x["text"]} for x in docs_raw]
    pool_ents = {x["doc_id"]: extract_ents(x["text"]) for x in docs_raw}
    title_to_idx = {x["title"]: i for i, x in enumerate(docs)}
    queries = []
    for q in d["query_stream"]:
        m = q.get("metadata", {})
        queries.append({
            "question": q["question"],
            "qtype": m.get("type", ""),
            "sf_titles": m.get("sf_titles", []),
            "gold_doc_ids": q.get("gold_doc_ids", []),
        })
    return docs, pool_ents, title_to_idx, queries


def embed(docs, queries):
    """无 LLM: 用 hashing-based 伪嵌入(确定性), 仅供检索相似度。
    真实实验应换 sentence-transformer; 此处验证机制故用轻量稳定嵌入。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    corpus = [d["text"] for d in docs] + [q["question"] for q in queries]
    tfidf = TfidfVectorizer(max_features=4096, stop_words="english").fit(corpus)
    X = tfidf.transform(corpus)
    svd = TruncatedSVD(n_components=128, random_state=0).fit(X)
    Z = svd.transform(X)
    Z = Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9, None)
    return Z[: len(docs)], Z[len(docs):]


def initial_kb(docs, doc_embs, queries, query_embs, title_to_idx):
    """初始 KB: 用前若干 query 的直接 top 命中填充(模拟冷启动直接缓存),
    故意不含难够的第二跳。"""
    seen = []
    for qe in query_embs[:50]:
        sims = doc_embs @ qe
        seen.extend(np.argsort(-sims)[:4].tolist())
    uniq = list(dict.fromkeys(seen))[:KB_BUDGET]
    return {docs[i]["doc_id"] for i in uniq}


def bridge_gold_positions(queries, title_to_idx):
    """每个 bridge query 的 gold 第二跳位置(sf_titles 映射到 idx)。"""
    gold = set()
    for q in queries:
        if q["qtype"] != "bridge":
            continue
        for t in q["sf_titles"]:
            pi = title_to_idx.get(t)
            if pi is not None:
                gold.add(pi)
    return gold


def run(core, label, queries, query_embs, gold_b):
    for w0 in range(0, len(queries), WINDOW):
        win = queries[w0:w0 + WINDOW]
        qe = query_embs[w0:w0 + WINDOW]
        qe = qe / np.clip(np.linalg.norm(qe, axis=1, keepdims=True), 1e-9, None)
        core.step(win, qe, w0 // WINDOW)
    kb_pos = {core.d2p[d] for d in core.kb}
    hit = len(gold_b & kb_pos)
    print(f"[{label:14s}] bridge gold in KB: {hit:3d}/{len(gold_b)}  "
          f"({hit/len(gold_b)*100:.1f}%)  KB={len(core.kb)}  writes={core.update_cost}")
    return hit


def make_core(cls, docs, doc_embs, title_to_idx, pool_ents, init_ids, **kw):
    core = cls("DRIP", docs, doc_embs, title_to_idx, **kw)
    core.graph_index.set_pool_entities(pool_ents)
    core.graph_index.build()
    core.set_kb(init_ids)
    return core


if __name__ == "__main__":
    print("加载真实数据 hotpotqa_entity_walk ...")
    docs, pool_ents, title_to_idx, queries = load()
    n_bridge = sum(1 for q in queries if q["qtype"] == "bridge")
    print(f"  docs={len(docs)}  queries={len(queries)}  bridge={n_bridge}")
    print("编码 (TF-IDF + SVD) ...")
    doc_embs, query_embs = embed(docs, queries)
    gold_b = bridge_gold_positions(queries, title_to_idx)
    init_ids = initial_kb(docs, doc_embs, queries, query_embs, title_to_idx)
    init_pos = {title_to_idx[d["title"]] for d in docs
                if d["doc_id"] in init_ids}
    print(f"  bridge gold docs={len(gold_b)}, 初始已在KB={len(gold_b & init_pos)}")
    print("=" * 64)

    stock = run(make_core(DRIPCore, docs, doc_embs, title_to_idx, pool_ents, init_ids),
                "stock(单边门)", queries, query_embs, gold_b)
    ppr = run(make_core(PPRDRIPCore, docs, doc_embs, title_to_idx, pool_ents, init_ids,
                        ppr_kwargs=dict(c=0.5, L=3, R=2, K0=5, d_cap=80)),
              "local-PPR", queries, query_embs, gold_b)
    print("=" * 64)
    print(f"  stock     : {stock}/{len(gold_b)}")
    print(f"  local-PPR : {ppr}/{len(gold_b)}")
    print(f"  增量       : {ppr - stock:+d}  ({(ppr-stock)/len(gold_b)*100:+.1f}pp)")
    print("=" * 64)

