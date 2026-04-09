"""
Motivation 4  Real-Data Diagnostic: Existing KB Update Paradigms

Dataset : HotpotQA (focused subset)
Pool    : stream queries context docs (~3.9k unique)
KB      : 300 docs, head-biased init
Stream  : 400 queries, 20 windows, head -> tail drift

Key: QARC uses per-query retrieval to discover NEW docs for KB.
     ERASE receives head-biased doc push (supply-side).
     ComRAG/Static: KB frozen.

Metric: SF-Based Coverage
  Binary  : frac of queries with >= 1 SF doc in KB (sim >= threshold)
  Continuous : avg over queries of max SF-KB doc similarity
"""

import json, os, sys, time, hashlib, logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR    = BASE_DIR / 'data'
FIG_DIR     = BASE_DIR / 'figures'
CACHE_DIR   = BASE_DIR / 'cache'

sys.path.insert(0, str(BASE_DIR.parent))
sys.path.insert(0, str(PROJECT_DIR))
from plot_config import setup_style, COLORS, save_fig

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ---- Config ----
SEED          = 42
EMBED_MODEL   = 'all-MiniLM-L6-v2'
EMBED_DIM     = 384
KB_BUDGET     = 300
WINDOW_SIZE   = 20
N_WINDOWS     = 20
N_QUERIES     = WINDOW_SIZE * N_WINDOWS   # 400
N_SOURCE      = 800
N_CLUSTERS    = 8
DOC_ARRIVE    = 30          # docs per window for ERASE

SF_HIT_THRESH    = 0.70     # binary SF coverage
OOS_SF_THRESH    = 0.55     # query is OOS if best SF-KB sim < this
ERASE_SIM_THRESH = 0.50     # ERASE replace-if-similar
QARC_TOP_K       = 30       # per-query retrieval depth for QARC
ABSTAIN = {'Static': 0.55, 'ERASE': 0.50, 'ComRAG': 0.40, 'QARC': 0.52}


def load_hotpotqa():
    p = PROJECT_DIR / 'datasets' / 'hotpotqa' / 'validation_distractor.json'
    log.info(f"Loading {p}")
    with open(p) as f:
        return json.load(f)


def build_pool_and_queries(data):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(data), N_SOURCE, replace=False)
    src = [data[i] for i in idx]
    dmap = {}
    for it in src:
        for t, ss in zip(it['context']['title'], it['context']['sentences']):
            if t not in dmap:
                dmap[t] = ' '.join(ss).strip()
    doc_pool = []
    t2i = {}
    for i, (t, txt) in enumerate(sorted(dmap.items())):
        doc_pool.append({'doc_id': f'd{i:05d}', 'title': t, 'text': txt})
        t2i[t] = i
    qs = []
    for it in src:
        sf = list({t for t in it['supporting_facts']['title'] if t in t2i})
        if len(sf) < 2: continue
        ctx = [t for t in it['context']['title'] if t in t2i]
        qs.append({'question': it['question'], 'answer': it['answer'],
                   'sf_titles': sf, 'ctx_titles': ctx})
    log.info(f"Full doc pool: {len(doc_pool)}, Valid queries: {len(qs)}")
    return doc_pool, qs, t2i


def compute_embs(doc_pool, queries, tag='full'):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{len(doc_pool)}_{len(queries)}_{tag}_v9".encode()).hexdigest()[:12]
    dc, qc = CACHE_DIR / f'de_{key}.npy', CACHE_DIR / f'qe_{key}.npy'
    if dc.exists() and qc.exists():
        log.info("Loading cached embeddings")
        return np.load(dc).astype('f'), np.load(qc).astype('f')
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBED_MODEL)
    log.info(f"Encoding {len(doc_pool)} docs...")
    de = m.encode([f"{d['title']}: {d['text'][:256]}" for d in doc_pool],
                  batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    log.info(f"Encoding {len(queries)} queries...")
    qe = m.encode([q['question'] for q in queries],
                  batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    np.save(dc, de); np.save(qc, qe)
    return de.astype('f'), qe.astype('f')


def cluster_and_stream(queries, qembs):
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(SEED + 10)
    km = KMeans(N_CLUSTERS, n_init=5, random_state=SEED).fit(qembs)
    labels = km.labels_
    sizes = Counter(labels)
    ranked = sorted(sizes, key=lambda c: -sizes[c])
    head_set = set(ranked[:3])
    centroids = km.cluster_centers_
    for i, q in enumerate(queries):
        q['cluster'] = int(labels[i])
        q['is_tail'] = labels[i] not in head_set
        q['qidx'] = i
    hq = [q for q in queries if not q['is_tail']]
    tq = [q for q in queries if q['is_tail']]
    rng.shuffle(hq); rng.shuffle(tq)
    log.info(f"Cluster sizes: {[sizes[c] for c in ranked]}")
    log.info(f"Head queries: {len(hq)}, Tail queries: {len(tq)}")
    # Sharper drift: 90/10 -> 10/90
    half = N_QUERIES // 2
    nh1 = int(half * 0.90); nt1 = half - nh1
    nh2 = int(half * 0.10); nt2 = half - nh2
    def pick(src, n):
        out = src[:n]; del src[:n]; return out
    front = pick(hq, nh1) + pick(tq, nt1); rng.shuffle(front)
    back  = pick(hq, nh2) + pick(tq, nt2); rng.shuffle(back)
    stream = list(front) + list(back)
    while len(stream) < N_QUERIES:
        pool = hq + tq
        if pool: stream.append(pool[rng.integers(len(pool))])
        else: stream.append(stream[rng.integers(len(stream))])
    stream = stream[:N_QUERIES]
    nh = sum(1 for q in stream if not q['is_tail'])
    nt = sum(1 for q in stream if q['is_tail'])
    log.info(f"Stream: {len(stream)} (head={nh} tail={nt})")
    return stream, centroids, head_set


def focus_pool(doc_pool_full, t2i_full, dembs_full, stream):
    """Only keep docs from stream queries' contexts."""
    used = set()
    for q in stream:
        for t in q.get('ctx_titles', []):
            used.add(t)
    old2new = {}
    doc_pool = []; t2i = {}
    for t in sorted(used):
        if t in t2i_full:
            oi = t2i_full[t]; ni = len(doc_pool)
            old2new[oi] = ni
            doc_pool.append(dict(doc_pool_full[oi]))
            doc_pool[-1]['doc_id'] = f'f{ni:05d}'
            t2i[t] = ni
    sk = sorted(old2new.keys())
    dembs = dembs_full[sk]
    log.info(f"Focused pool: {len(doc_pool)} docs (from {len(doc_pool_full)})")
    for q in stream:
        q['sf_titles'] = [t for t in q['sf_titles'] if t in t2i]
        q['ctx_titles'] = [t for t in q['ctx_titles'] if t in t2i]
    n_sf = sum(len(q['sf_titles']) for q in stream)
    log.info(f"Stream SF titles in pool: {n_sf} ({n_sf/(2*len(stream)):.0%} of expected)")
    return doc_pool, t2i, dembs


def head_biased_init_kb(doc_pool, dembs, centroids, head_set, budget, stream):
    cn = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-10, None)
    t2i_loc = {d['title']: i for i, d in enumerate(doc_pool)}
    head_idx = set()
    for q in stream:
        if not q.get('is_tail', True):
            for t in q.get('ctx_titles', []):
                if t in t2i_loc: head_idx.add(t2i_loc[t])
    head_idx = sorted(head_idx)
    log.info(f"Head-context docs: {len(head_idx)}")
    hc = cn[list(head_set)]
    if len(head_idx) >= budget:
        aff = dembs[head_idx] @ hc.T
        scores = aff.max(axis=1)
        top_k = np.argsort(scores)[-budget:]
        sel = [head_idx[i] for i in top_k]
    else:
        sel = list(head_idx)
        rem = [i for i in range(len(doc_pool)) if i not in set(sel)]
        aff = dembs[rem] @ hc.T
        scores = aff.max(axis=1)
        need = budget - len(sel)
        top_k = np.argsort(scores)[-need:]
        sel.extend([rem[i] for i in top_k])
    sel = sorted(sel)
    kb = {doc_pool[i]['doc_id'] for i in sel}
    tc = cn[[c for c in range(N_CLUSTERS) if c not in head_set]]
    h, t = float(np.mean(np.max(dembs[sel] @ hc.T, axis=0))), \
           float(np.mean(np.max(dembs[sel] @ tc.T, axis=0)))
    log.info(f"Init KB ({len(kb)}): head-cent-cov={h:.3f}, tail-cent-cov={t:.3f}")
    ke = dembs[sel]
    hh = th = ht_total = tt_total = 0
    for q in stream:
        si = [t2i_loc[t] for t in q['sf_titles'] if t in t2i_loc]
        if not si: continue
        se = dembs[si]; best = float(np.max(se @ ke.T))
        if q['is_tail']: tt_total += 1; th += best >= SF_HIT_THRESH
        else: ht_total += 1; hh += best >= SF_HIT_THRESH
    log.info(f"Init SF hits: head={hh}/{ht_total} ({hh/max(ht_total,1):.1%}), "
             f"tail={th}/{tt_total} ({th/max(tt_total,1):.1%})")
    return kb



# ---- Metrics ----

def _kb_emb(kb_ids, id2pi, dembs):
    idx = sorted(id2pi[d] for d in kb_ids if d in id2pi)
    return dembs[idx] if idx else np.zeros((0, EMBED_DIM), dtype='f')

def sf_cov_binary(kb_ids, wq, id2pi, dembs, t2i):
    ke = _kb_emb(kb_ids, id2pi, dembs)
    if ke.shape[0] == 0: return 0.0
    hit = 0
    for q in wq:
        si = [t2i[t] for t in q['sf_titles'] if t in t2i]
        if not si: continue
        best = float(np.max(dembs[si] @ ke.T))
        if best >= SF_HIT_THRESH: hit += 1
    return hit / len(wq) if wq else 0.0

def sf_cov_cont(kb_ids, wq, id2pi, dembs, t2i):
    ke = _kb_emb(kb_ids, id2pi, dembs)
    if ke.shape[0] == 0: return 0.0
    s = []
    for q in wq:
        si = [t2i[t] for t in q['sf_titles'] if t in t2i]
        if not si: s.append(0.0); continue
        s.append(float(np.max(dembs[si] @ ke.T)))
    return float(np.mean(s))

def oos_far(kb_ids, wq, id2pi, dembs, t2i, qembs, a_thr):
    ke = _kb_emb(kb_ids, id2pi, dembs)
    if ke.shape[0] == 0: return 0.0
    no = nf = 0
    for q in wq:
        si = [t2i[t] for t in q['sf_titles'] if t in t2i]
        if not si: continue
        best = float(np.max(dembs[si] @ ke.T))
        if best < OOS_SF_THRESH:
            no += 1
            qe = qembs[q['qidx']]
            if float(np.max(ke @ qe)) >= a_thr: nf += 1
    return nf / no if no else 0.0


# ---- Strategies ----

class Base:
    def __init__(self, name, dp, de, t2i):
        self.name = name; self.pool = dp; self.dembs = de; self.t2i = t2i
        self.id2pi = {d['doc_id']: i for i, d in enumerate(dp)}
        self.t2did = {d['title']: d['doc_id'] for d in dp}
        self.kb = set(); self.cost = 0
    def set_kb(self, kb): self.kb = set(kb)
    def step(self, wq, we, widx): pass
    def kb_emb(self): return _kb_emb(self.kb, self.id2pi, self.dembs)

class Static(Base):
    def __init__(self, *a): super().__init__('Static', *a)

class ERASE(Base):
    """Supply-side push: head-biased doc arrivals, similarity replacement."""
    def __init__(self, *a):
        super().__init__('ERASE', *a)
        self.rng = np.random.default_rng(SEED + 1)
        self._head_pool = []
    def set_head_docs(self, stream):
        ids = set()
        for q in stream:
            if not q.get('is_tail', True):
                for t in q.get('ctx_titles', []):
                    if t in self.t2did: ids.add(self.t2did[t])
        self._head_pool = sorted(ids)
    def step(self, wq, we, widx):
        if not self._head_pool: return
        ops = 0
        n = min(DOC_ARRIVE, len(self._head_pool))
        arr = self.rng.choice(self._head_pool, n, replace=False)
        kbl = list(self.kb); kbi = np.array([self.id2pi[d] for d in kbl])
        kbe = self.dembs[kbi]
        for did in arr:
            if did in self.kb: continue
            di = self.id2pi[did]; ne = self.dembs[di]
            if len(self.kb) < KB_BUDGET:
                self.kb.add(did); ops += 1; continue
            sims = kbe @ ne; best = int(np.argmax(sims))
            if sims[best] > ERASE_SIM_THRESH:
                old = kbl[best]
                self.kb.discard(old); self.kb.add(did)
                kbl[best] = did; kbi[best] = di; kbe[best] = ne
                ops += 2
        self.cost += ops

class ComRAG(Base):
    """KB frozen, QA memory grows."""
    def __init__(self, *a): super().__init__('ComRAG', *a)
    def step(self, wq, we, widx): self.cost += len(wq)

class QARC(Base):
    """Query-driven KB curation via per-query retrieval + submodular selection.

    Each window:
    1. Detect alignment gap between queries and KB
    2. If gap exceeds threshold -> trigger curation
    3. Retrieve top-K docs per query from pool as candidates
    4. Submodular greedy: select best KB_BUDGET docs from {KB ∪ candidates}
    5. Replace up to lambda*budget docs
    """
    def __init__(self, *a):
        super().__init__('QARC', *a)
        self.rng = np.random.default_rng(SEED + 3)
        self.w = 0
        self.ema = None; self.mad = None
        self._roll_C = None; self._roll_w = None

    def step(self, wq, we, widx):
        self.w += 1
        ke = self.kb_emb()
        if ke.shape[0] == 0: return
        X = we / np.clip(np.linalg.norm(we, axis=1, keepdims=True), 1e-10, None)

        # Alignment gap
        gap = 1.0 - float(np.mean(np.max(X @ ke.T, axis=1)))
        if self.ema is None:
            self.ema = gap; self.mad = 0.02
        else:
            b = 0.75
            self.mad = b * self.mad + (1 - b) * abs(gap - self.ema)
            self.ema = b * self.ema + (1 - b) * gap

        # Always curate but modulate intensity
        th = self.ema + 0.8 * self.mad
        if gap > th + 0.05:
            lam = 0.50                     # aggressive
        elif gap > th:
            lam = 0.30                     # moderate
        else:
            lam = 0.10                     # maintenance

        # Update interest model
        self._update_interest(X)

        # Per-query retrieval from pool
        sims = X @ self.dembs.T   # (n_q, n_pool)
        cand = set()
        for i in range(len(X)):
            top = np.argpartition(sims[i], -QARC_TOP_K)[-QARC_TOP_K:]
            cand.update(top.tolist())
        # Add current KB
        for d in self.kb:
            if d in self.id2pi: cand.add(self.id2pi[d])

        self.cost += self._recurate(lam, sorted(cand))

    def _update_interest(self, X):
        from sklearn.cluster import KMeans
        nc = max(2, min(5, len(X)))
        km = KMeans(nc, n_init=3, random_state=SEED).fit(X)
        C = km.cluster_centers_
        C = C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-10, None)
        wt = np.bincount(km.labels_, minlength=nc).astype('f')
        wt /= wt.sum()
        if self._roll_C is None:
            self._roll_C = C; self._roll_w = wt
        else:
            self._roll_C = np.vstack([self._roll_C * 0.5, C])
            self._roll_w = np.concatenate([self._roll_w * 0.5, wt])
            self._roll_w /= self._roll_w.sum()
            if len(self._roll_w) > 15:
                topk = np.argsort(self._roll_w)[-15:]
                self._roll_C = self._roll_C[topk]
                self._roll_w = self._roll_w[topk]
                self._roll_w /= self._roll_w.sum()

    def _recurate(self, lam, cl):
        C = self._roll_C; wt = self._roll_w; nc = len(wt)
        ce = self.dembs[cl]; cs = ce @ C.T
        sel = set(); mx = np.full(nc, -1.0); order = []
        for _ in range(min(KB_BUDGET, len(cl))):
            gains = (wt[None, :] * np.maximum(cs - mx[None, :], 0)).sum(axis=1)
            for j in sel: gains[j] = -np.inf
            b = int(np.argmax(gains))
            if gains[b] <= 0: break
            sel.add(b); order.append(b)
            mx = np.maximum(mx, cs[b])
        ideal = {self.pool[cl[j]]['doc_id'] for j in order}
        to_add = ideal - self.kb; to_rem = self.kb - ideal
        cap = max(1, int(lam * KB_BUDGET))
        if len(to_add) > cap:
            scored = sorted(to_add,
                key=lambda d: float(wt @ (C @ self.dembs[self.id2pi[d]])),
                reverse=True)
            to_add = set(scored[:cap])
        if len(to_rem) > cap:
            scored = sorted(to_rem,
                key=lambda d: float(wt @ (C @ self.dembs[self.id2pi[d]])))
            to_rem = set(scored[:cap])
        n = min(len(to_add), len(to_rem))
        ai, ri = iter(to_add), iter(to_rem)
        for _ in range(n):
            self.kb.discard(next(ri)); self.kb.add(next(ai))
        return 2 * n


# ---- Run ----

ORDER  = ['Static', 'ERASE', 'ComRAG', 'QARC']
LABELS = {'Static': 'Static / Reject-only', 'ERASE': 'ERASE  (Edit/Prune)',
          'ComRAG': 'ComRAG (Cache/Memory)', 'QARC': 'QARC  (Ours)'}
STYLES = {'Static': {'c': '#DC2626', 'm': 'D', 'ls': '--'},
          'ERASE':  {'c': '#059669', 'm': '^', 'ls': '-'},
          'ComRAG': {'c': '#D97706', 'm': 's', 'ls': '-'},
          'QARC':   {'c': '#2563EB', 'm': 'o', 'ls': '-'}}


def run(doc_pool, dembs, stream, se, t2i, qembs, centroids, head_set):
    strats = {
        'Static': Static(doc_pool, dembs, t2i),
        'ERASE':  ERASE(doc_pool, dembs, t2i),
        'ComRAG': ComRAG(doc_pool, dembs, t2i),
        'QARC':   QARC(doc_pool, dembs, t2i),
    }
    id2pi = {d['doc_id']: i for i, d in enumerate(doc_pool)}
    init = head_biased_init_kb(doc_pool, dembs, centroids, head_set, KB_BUDGET, stream)
    strats['ERASE'].set_head_docs(stream)

    results = {}
    for name, st in strats.items():
        st.set_kb(init)
        log.info(f"\n{'='*30} {name} {'='*30}")
        metrics = []
        for w in range(N_WINDOWS):
            a, b = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
            wq = stream[a:b]; we = se[a:b]
            cb = sf_cov_binary(st.kb, wq, id2pi, dembs, t2i)
            cc = sf_cov_cont(st.kb, wq, id2pi, dembs, t2i)
            far = oos_far(st.kb, wq, id2pi, dembs, t2i, qembs, ABSTAIN[name])
            hq = [q for q in wq if not q['is_tail']]
            tq = [q for q in wq if q['is_tail']]
            hcb = sf_cov_binary(st.kb, hq, id2pi, dembs, t2i) if hq else 0.0
            tcb = sf_cov_binary(st.kb, tq, id2pi, dembs, t2i) if tq else 0.0
            hcc = sf_cov_cont(st.kb, hq, id2pi, dembs, t2i) if hq else 0.0
            tcc = sf_cov_cont(st.kb, tq, id2pi, dembs, t2i) if tq else 0.0
            metrics.append({
                'window': w, 'sf_cov_bin': cb, 'sf_cov_cont': cc,
                'oos_far': far, 'cost': st.cost, 'kb_size': len(st.kb),
                'head_cov_bin': hcb, 'tail_cov_bin': tcb,
                'head_cov_cont': hcc, 'tail_cov_cont': tcc,
                'n_head': len(hq), 'n_tail': len(tq),
            })
            st.step(wq, we, w)
            log.info(f"  W{w:02d}: Bin={cb:.3f} Cont={cc:.3f} "
                     f"(h_b={hcb:.3f} t_b={tcb:.3f} h_c={hcc:.3f} t_c={tcc:.3f}) "
                     f"FAR={far:.2f} cost={st.cost}")
        results[name] = metrics
    return results


# ---- Figure ----

def make_figure(results):
    setup_style()
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11.5, 'axes.labelsize': 10.5,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
        'axes.linewidth': 0.8, 'grid.linewidth': 0.4, 'grid.alpha': 0.2,
        'lines.linewidth': 2.2, 'lines.markersize': 6,
    })
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30,
                          left=0.07, right=0.96, top=0.93, bottom=0.07)
    w = np.arange(N_WINDOWS)
    mid = N_WINDOWS // 2

    def shade(ax):
        ax.axvspan(-0.5, mid-0.5, alpha=0.05, color='#D1FAE5', zorder=0)
        ax.axvspan(mid-0.5, N_WINDOWS-0.5, alpha=0.05, color='#FEE2E2', zorder=0)

    # Use continuous metric for smoother curves
    # (a) SF Coverage (continuous)
    ax = fig.add_subplot(gs[0, 0]); shade(ax)
    for n in ORDER:
        s = STYLES[n]
        v = [m['sf_cov_cont'] * 100 for m in results[n]]
        ax.plot(w, v, color=s['c'], marker=s['m'], ls=s['ls'],
                markerfacecolor='white', markeredgewidth=1.0, label=LABELS[n], zorder=3)
    ax.set_xlabel('Time Window'); ax.set_ylabel('SF Coverage Score (%)')
    ax.set_xlim(-0.5, N_WINDOWS-0.5)
    ax.set_title('(a)  Supporting-Fact Coverage Over Time', fontweight='bold', pad=8)
    ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='#ccc')
    ax.annotate('Head-dominant', xy=(0.02, 0.95), fontsize=7, color='#059669',
                fontstyle='italic', xycoords='axes fraction')
    ax.annotate('Tail-dominant \u2192', xy=(0.68, 0.95), fontsize=7, color='#DC2626',
                fontstyle='italic', xycoords='axes fraction')

    # (b) OOS FAR
    ax = fig.add_subplot(gs[0, 1]); shade(ax)
    for n in ORDER:
        s = STYLES[n]
        v = [m['oos_far'] * 100 for m in results[n]]
        ax.plot(w, v, color=s['c'], marker=s['m'], ls=s['ls'],
                markerfacecolor='white', markeredgewidth=1.0, label=LABELS[n], zorder=3)
    ax.set_xlabel('Time Window'); ax.set_ylabel('OOS False-Answer Rate (%)')
    ax.set_xlim(-0.5, N_WINDOWS-0.5); ax.set_ylim(-2, 102)
    ax.axhline(50, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.set_title('(b)  OOS False Answer Rate Over Time', fontweight='bold', pad=8)
    ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='#ccc')
    ax.annotate('Head-dominant', xy=(0.02, 0.95), fontsize=7, color='#059669',
                fontstyle='italic', xycoords='axes fraction')
    ax.annotate('Tail-dominant \u2192', xy=(0.68, 0.95), fontsize=7, color='#DC2626',
                fontstyle='italic', xycoords='axes fraction')

    # (c) Cost
    ax = fig.add_subplot(gs[1, 0])
    for n in ORDER:
        s = STYLES[n]
        v = [m['cost'] for m in results[n]]
        ax.plot(w, v, color=s['c'], marker=s['m'], ls=s['ls'],
                markerfacecolor='white', markeredgewidth=1.0, label=LABELS[n], zorder=3)
    ax.set_xlabel('Time Window'); ax.set_ylabel('Cumulative Update Ops')
    ax.set_xlim(-0.5, N_WINDOWS-0.5); ax.set_ylim(bottom=-10)
    ax.set_title('(c)  Cumulative Update Cost Over Time', fontweight='bold', pad=8)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#ccc')

    # (d) Scatter: avg SF cov vs cost for second (tail) half
    ax = fig.add_subplot(gs[1, 1])
    for n in ORDER:
        s = STYLES[n]; m = results[n]
        # Use second-half (tail-dominant) coverage
        tail_cov = np.mean([i['sf_cov_cont'] for i in m[mid:]]) * 100
        afar = np.mean([i['oos_far'] for i in m]) * 100
        fc = m[-1]['cost']
        sz = 120 + afar * 1.5
        ax.scatter(fc, tail_cov, color=s['c'], marker=s['m'],
                   s=sz, edgecolors='white', linewidths=1.2, zorder=5)
        ox, oy = 12, 0
        if n == 'Static': oy = -2.5
        elif n == 'QARC': oy = 2.5
        elif n == 'ComRAG': ox = -80; oy = -3
        ax.annotate(f"{LABELS[n]}\n(FAR={afar:.0f}%)",
                    xy=(fc, tail_cov), xytext=(ox, oy), textcoords='offset points',
                    fontsize=7.5, color=s['c'], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=s['c'], alpha=0.85))
    ax.annotate('\u2190 Ideal\n(high cov, low cost)',
                xy=(0.15, 0.88), fontsize=8, fontstyle='italic',
                color='#059669', ha='center', xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', fc='#D1FAE5', ec='#059669', alpha=0.6))
    ax.set_xlabel('Final Cumulative Update Cost')
    ax.set_ylabel('Avg Tail-Phase SF Coverage (%)')
    ax.set_title('(d)  Coverage\u2013Cost Trade-off (Tail Phase)', fontweight='bold', pad=8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_fig(fig, str(FIG_DIR / 'paradigm_diagnostic.png'))
    plt.close()


def print_summary(results):
    mid = N_WINDOWS // 2
    print(f"\n{'='*90}")
    print("  Motivation 4: HotpotQA - SF-Based Coverage Diagnostic")
    print(f"{'='*90}")
    print(f"  {N_QUERIES} queries ({N_WINDOWS}x{WINDOW_SIZE}), KB={KB_BUDGET}, "
          f"90/10->10/90 drift, SF_HIT={SF_HIT_THRESH}")
    h = (f"  {'Method':<26s} {'Cont':>6s} {'Bin':>6s} "
         f"{'FAR':>6s} {'Cost':>6s}  {'H-C':>5s} {'T-C':>5s}  "
         f"{'H-B':>5s} {'T-B':>5s}  {'TailCont':>8s}")
    print(h); print("  " + "-" * (len(h) - 2))
    for n in ORDER:
        m = results[n]
        ac = np.mean([i['sf_cov_cont'] for i in m])
        ab = np.mean([i['sf_cov_bin'] for i in m])
        af = np.mean([i['oos_far'] for i in m])
        co = m[-1]['cost']
        hc = np.mean([i['head_cov_cont'] for i in m])
        tc = np.mean([i['tail_cov_cont'] for i in m])
        hb = np.mean([i['head_cov_bin'] for i in m])
        tb = np.mean([i['tail_cov_bin'] for i in m])
        tc2 = np.mean([i['sf_cov_cont'] for i in m[mid:]])
        print(f"  {LABELS[n]:<26s} {ac:>5.1%} {ab:>5.1%} "
              f"{af:>5.1%} {co:>6d}  {hc:>4.1%} {tc:>4.1%}  "
              f"{hb:>4.1%} {tb:>4.1%}  {tc2:>7.1%}")
    print()
    print("  Coverage trend (first 5 vs last 5 windows, continuous):")
    for n in ORDER:
        m = results[n]
        f5 = np.mean([i['sf_cov_cont'] for i in m[:5]])
        l5 = np.mean([i['sf_cov_cont'] for i in m[-5:]])
        tf = np.mean([i['tail_cov_cont'] for i in m[:5]])
        tl = np.mean([i['tail_cov_cont'] for i in m[-5:]])
        print(f"  {LABELS[n]:<26s}  All: {f5:.1%}->{l5:.1%} ({l5-f5:+.1%})  "
              f"Tail: {tf:.1%}->{tl:.1%} ({tl-tf:+.1%})")
    print()


def save_data(results):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = DATA_DIR / 'paradigm_diagnostic.json'
    with open(p, 'w') as f:
        json.dump({'config': dict(n_windows=N_WINDOWS, window_size=WINDOW_SIZE,
                                  kb_budget=KB_BUDGET, embed_model=EMBED_MODEL,
                                  n_clusters=N_CLUSTERS, sf_hit_thresh=SF_HIT_THRESH),
                   'results': results}, f, indent=2)
    log.info(f"Saved: {p}")


def main():
    t0 = time.time()
    data = load_hotpotqa()
    dp_full, queries, t2i_full = build_pool_and_queries(data)
    dembs_full, qembs = compute_embs(dp_full, queries)
    stream, centroids, head_set = cluster_and_stream(queries, qembs)
    doc_pool, t2i, dembs = focus_pool(dp_full, t2i_full, dembs_full, stream)
    se = qembs[[q['qidx'] for q in stream]]
    results = run(doc_pool, dembs, stream, se, t2i, qembs, centroids, head_set)
    make_figure(results)
    print_summary(results)
    save_data(results)
    log.info(f"Done in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()

