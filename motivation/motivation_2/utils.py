"""
Shared utilities: embeddings, stream construction, KB initialisation, Recall@K.

Functions in this module are "pure" helpers -- they don't depend on which
strategies are being run.  They implement the experimental infrastructure
described in the motivation section of our paper.
"""
import hashlib
import numpy as np
from collections import Counter
from config import (SEED, EMBED_MODEL, CACHE_DIR, SF_HIT_THRESH, K_LIST, log, BGE_QUERY_PREFIX)


# ═══════════════════════════════════════════════════
#  Embedding
# ═══════════════════════════════════════════════════

def compute_embeddings(doc_pool, queries, tag):
    """Encode documents and queries with Sentence-BERT, with disk caching.

    Uses BGE-large-en-v1.5 (1024-dim, L2-normalised).  Cache key is derived
    from pool size + query count + tag to avoid stale reads.

    Returns:
        doc_embs:   np.ndarray (N_docs, 384)  float32
        query_embs: np.ndarray (N_queries, 384) float32
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(
        f"{len(doc_pool)}_{len(queries)}_{tag}_{EMBED_MODEL}_v12".encode()
    ).hexdigest()[:12]
    dc = CACHE_DIR / f'de_{key}.npy'
    qc = CACHE_DIR / f'qe_{key}.npy'
    if dc.exists() and qc.exists():
        log.info("Loading cached embeddings")
        return np.load(dc).astype('f'), np.load(qc).astype('f')
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    log.info(f"Encoding {len(doc_pool)} docs...")
    de = model.encode(
        [f"{d['title']}: {d['text'][:256]}" for d in doc_pool],
        batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    log.info(f"Encoding {len(queries)} queries...")
    _qprefix = BGE_QUERY_PREFIX if 'bge' in EMBED_MODEL.lower() else ''
    qe = model.encode(
        [_qprefix + q['question'] for q in queries],
        batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    np.save(dc, de)
    np.save(qc, qe)
    return de.astype('f'), qe.astype('f')


# ═══════════════════════════════════════════════════
#  Query stream construction
# ═══════════════════════════════════════════════════


def cluster_and_build_stream(queries, query_embs, cfg, drift_mode='sudden'):
    """Build a query stream with controlled topic drift.

    Step 1 -- Topic clustering:
      KMeans on raw query embeddings (sentence-transformers all-MiniLM-L6-v2).
      Top-N largest clusters are "head" topics; remaining are "tail" topics.
      Follows the query-semantic shift methodology of Lupart et al. (MS-Shift, ECIR 2023).

    Step 2 -- Stream construction (two halves):
      - H1 (first half):  97% head, 3% tail  (head-dominant)
      - H2 (second half): 3% head, 97% tail  (tail-dominant, simulates drift)

    drift_mode:
      - 'sudden':  instant flip at midpoint
      - 'gradual': linear interpolation from 97/3 to 3/97 across H2 windows

    Returns:
        stream:    list of query dicts (annotated with cluster, is_tail, qidx)
        centroids: KMeans cluster centres (n_clusters, 384)
        head_set:  set of cluster IDs treated as head topics
    """
    from sklearn.cluster import KMeans
    nc = cfg['n_clusters']
    th = cfg['top_head']
    n_q = cfg['n_windows'] * cfg['window_size']
    rng = np.random.default_rng(SEED + 10)
    km = KMeans(nc, n_init=5, random_state=SEED).fit(query_embs)
    labels = km.labels_
    sizes = Counter(labels)
    ranked = sorted(sizes, key=lambda c: -sizes[c])
    head_set = set(ranked[:th])
    for i, q in enumerate(queries):
        q['cluster'] = int(labels[i])
        q['is_tail'] = labels[i] not in head_set
        q['qidx'] = i
    heads = [q for q in queries if not q['is_tail']]
    tails = [q for q in queries if q['is_tail']]
    rng.shuffle(heads)
    rng.shuffle(tails)
    log.info(f"Clusters (query_emb): {[sizes[c] for c in ranked]}, "
             f"head={len(heads)} tail={len(tails)}")
    half = n_q // 2
    ws = cfg['window_size']

    def pick(src, n):
        take = min(n, len(src))
        out = src[:take]
        del src[:take]
        return out

    if drift_mode == 'sudden':
        n_hf = int(half * 0.97)
        n_tf = half - n_hf
        n_hb = int(half * 0.03)
        n_tb = half - n_hb
        h_front = pick(heads, n_hf)
        t_front = pick(tails, n_tf)
        front = h_front + t_front
        # Guard: if head pool < n_hf, pad front to exactly 'half' with extra tail
        # queries so the drift transition always falls at window half/ws (= n_windows//2)
        if len(front) < half:
            extra = pick(tails, half - len(front))
            front += extra
            log.warning(f"Head pool exhausted ({len(h_front)} < {n_hf}); "
                        f"padded front with {len(extra)} extra tail → front={len(front)}")
        rng.shuffle(front)
        back = pick(heads, n_hb) + pick(tails, n_tb)
        rng.shuffle(back)
        stream = list(front) + list(back)
    elif drift_mode == 'gradual':
        # Following Gama et al. (2014) Sec. 2.2 definition of gradual drift:
        # "As time passes, the probability of sampling from source S_I decreases,
        #  probability of sampling from source S_II increases."
        # We implement this as a linear ramp: at H2 window wi,
        #   P(head) = 0.97*(1-α) + 0.03*α  where α = wi/(n_windows-1)
        # Full-pool with-replacement sampling avoids pool exhaustion.
        n_hf = int(half * 0.97)
        n_tf = half - n_hf
        front = pick(heads, n_hf) + pick(tails, n_tf)
        rng.shuffle(front)
        stream = list(front)
        # Save full head/tail pools (before H1 depletion) for H2 sampling
        all_heads = [q for q in queries if not q['is_tail']]
        all_tails = [q for q in queries if q['is_tail']]
        n_h2_windows = half // ws
        for wi in range(n_h2_windows):
            head_pct = 0.97 - wi * 0.94 / max(n_h2_windows - 1, 1)
            n_h_w = int(ws * head_pct)
            n_t_w = ws - n_h_w
            h_idx = rng.integers(0, len(all_heads), n_h_w)
            t_idx = rng.integers(0, len(all_tails), n_t_w)
            window = [all_heads[i] for i in h_idx] + [all_tails[i] for i in t_idx]
            rng.shuffle(window)
            stream.extend(window)
    else:
        raise ValueError(f"Unknown drift_mode: {drift_mode}")

    remaining = heads + tails
    while len(stream) < n_q and remaining:
        stream.append(remaining.pop())
    stream = stream[:n_q]
    nh = sum(1 for q in stream if not q['is_tail'])
    nt = sum(1 for q in stream if q['is_tail'])
    log.info(f"Stream: {len(stream)} (head={nh} tail={nt})")
    return stream, km.cluster_centers_, head_set



# ═══════════════════════════════════════════════════
#  Pool focusing (optional)
# ═══════════════════════════════════════════════════

def focus_pool(doc_pool, title_to_idx, doc_embs, stream):
    """Shrink doc_pool to only titles referenced by stream queries.

    Used for HotpotQA where validation_distractor.json provides 10 context
    paragraphs per question, making the full pool manageable.  For expanded
    datasets (2Wiki 385K, MuSiQue 84K) we skip focusing to test at scale.
    """
    used = set()
    for q in stream:
        for t in q.get('ctx_titles', []):
            used.add(t)
    old_to_new = {}
    new_pool, new_idx = [], {}
    for t in sorted(used):
        if t in title_to_idx:
            oi = title_to_idx[t]
            ni = len(new_pool)
            old_to_new[oi] = ni
            new_pool.append(dict(doc_pool[oi]))
            new_pool[-1]['doc_id'] = f'f{ni:05d}'
            new_idx[t] = ni
    sorted_old = sorted(old_to_new.keys())
    new_embs = doc_embs[sorted_old]
    log.info(f"Focused pool: {len(new_pool)} (from {len(doc_pool)})")
    for q in stream:
        q['sf_titles'] = [t for t in q['sf_titles'] if t in new_idx]
        q['ctx_titles'] = [t for t in q['ctx_titles'] if t in new_idx]
    n_sf = sum(len(q['sf_titles']) for q in stream)
    log.info(f"Stream SF in pool: {n_sf}")
    return new_pool, new_idx, new_embs


# ═══════════════════════════════════════════════════
#  KB initialisation
# ═══════════════════════════════════════════════════

def head_biased_init_kb(doc_pool, doc_embs, centroids, head_set, budget, stream):
    """Initialise KB biased towards head-topic documents.

    Strategy:
      1. Collect all doc-pool indices referenced by head-cluster queries.
      2. If these exceed budget, keep the ones most aligned with head centroids.
      3. If fewer, fill remaining slots with pool docs closest to head centroids.

    This simulates a real system whose KB was built for the originally-dominant
    query distribution (head topics).  The experiment then measures how well
    each strategy adapts when queries shift to tail topics.
    """
    norm_c = centroids / np.clip(
        np.linalg.norm(centroids, axis=1, keepdims=True), 1e-10, None)
    t2l = {d['title']: i for i, d in enumerate(doc_pool)}
    head_idx = set()
    for q in stream:
        if not q.get('is_tail', True):
            for t in q.get('ctx_titles', []):
                if t in t2l:
                    head_idx.add(t2l[t])
    head_idx = sorted(head_idx)
    log.info(f"Head-context docs: {len(head_idx)}")
    hc = norm_c[list(head_set)]
    if len(head_idx) >= budget:
        aff = doc_embs[head_idx] @ hc.T
        scores = aff.max(axis=1)
        top = np.argsort(scores)[-budget:]
        selected = [head_idx[i] for i in top]
    else:
        selected = list(head_idx)
        rest = [i for i in range(len(doc_pool)) if i not in set(selected)]
        if rest:
            aff = doc_embs[rest] @ hc.T
            scores = aff.max(axis=1)
            need = budget - len(selected)
            top = np.argsort(scores)[-need:]
            selected.extend([rest[i] for i in top])
    init_kb = {doc_pool[i]['doc_id'] for i in sorted(selected)}
    # Diagnostic: how many stream SFs are reachable from init KB
    kb_embs = doc_embs[sorted(selected)]
    h_h = t_h = h_t = t_t = 0
    for q in stream:
        sf_idx = [t2l[t] for t in q['sf_titles'] if t in t2l]
        if not sf_idx:
            continue
        best = float(np.max(doc_embs[sf_idx] @ kb_embs.T))
        if q['is_tail']:
            t_t += 1
            t_h += best >= SF_HIT_THRESH
        else:
            h_t += 1
            h_h += best >= SF_HIT_THRESH
    log.info(f"Init KB ({len(init_kb)}): "
             f"SF head={h_h}/{h_t} ({h_h/max(h_t,1):.1%}), "
             f"tail={t_h}/{t_t} ({t_h/max(t_t,1):.1%})")
    return init_kb


# ═══════════════════════════════════════════════════
#  Recall@K evaluation
# ═══════════════════════════════════════════════════

def recall_at_k(kb_doc_ids, queries, d2p, doc_embs, title_to_idx,
                query_embs, k_list=K_LIST):
    """Compute Recall@K for a set of queries against the current KB.

    For each query, retrieve top-K documents from KB by cosine similarity,
    then measure what fraction of gold supporting-fact titles appear in
    the retrieved set.  This is the standard multi-hop QA retrieval metric
    used in HippoRAG (Gutiérrez et al., 2024) and IRCoT (Trivedi et al., 2023).

    Returns:
        dict mapping K -> mean recall across all queries with gold SFs.
    """
    if not kb_doc_ids:
        return {k: 0.0 for k in k_list}
    kb_list = sorted(kb_doc_ids)
    kb_indices = [d2p[d] for d in kb_list]
    kb_emb = doc_embs[kb_indices]
    p2t = {v: k for k, v in title_to_idx.items()}
    kb_i2t = {i: p2t[pi] for i, pi in enumerate(kb_indices) if pi in p2t}
    max_k = max(k_list)
    recalls = {k: [] for k in k_list}
    for q in queries:
        gold = set(q['sf_titles'])
        if not gold:
            continue
        qe = query_embs[q['qidx']]
        sims = kb_emb @ qe
        if len(sims) <= max_k:
            top = np.argsort(sims)[::-1]
        else:
            top = np.argpartition(sims, -max_k)[-max_k:]
            top = top[np.argsort(sims[top])[::-1]]
        for k in k_list:
            retrieved = {kb_i2t[i] for i in top[:k] if i in kb_i2t}
            r = len(gold & retrieved) / len(gold)
            recalls[k].append(r)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in recalls.items()}
