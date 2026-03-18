"""对比原版 DriftLens 和我们的 GMM 检测器"""
import numpy as np, time, warnings, logging, sys
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

from test.experiment_datasets import build_gradual_drift
ds = build_gradual_drift(100)
queries = ds.query_stream
pool = ds.document_pool
print(f"Dataset: {len(queries)} queries, {len(pool)} pool docs")

from core.data_processor import embedding_service
embed = lambda t: np.array(embedding_service.embed_query(t))

print("Embedding queries...")
q_embs = np.array([embed(q.question) for q in queries])
print(f"  q: {q_embs.shape}")

pool_texts = [d.text for d in pool[:500]]
print("Embedding pool (500)...")
pool_embs = np.array([embed(t) for t in pool_texts])
print(f"  pool: {pool_embs.shape}")

def l2n(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-10, None)
q_n, p_n = l2n(q_embs), l2n(pool_embs)

ws = 10
nw = len(q_n) // ws

# == 原版 DriftLens ==
print("\n" + "=" * 60)
print("原版 DriftLens (FID + PCA + Offline Threshold)")
print("=" * 60)
from driftlens.driftlens import DriftLens
from sklearn.cluster import KMeans

# 用3个聚类，减少窗口内标签缺失问题
n_cls = 3
km = KMeans(n_clusters=n_cls, random_state=42, n_init=3)
pool_labels = km.fit_predict(p_n)
print(f"Pool 标签分布: {np.bincount(pool_labels)}")

dl = DriftLens()
npc = min(50, p_n.shape[1] // 2, p_n.shape[0] // 3)
plpc = min(30, npc)

t0 = time.time()
dl.estimate_baseline(E=p_n, Y=pool_labels, label_list=list(range(n_cls)),
                     batch_n_pc=npc, per_label_n_pc=plpc)
print(f"Offline Baseline: {time.time()-t0:.2f}s (batch_n_pc={npc}, per_label_n_pc={plpc})")

t0 = time.time()
pbs, _ = dl.random_sampling_threshold_estimation(
    label_list=list(range(n_cls)), E=p_n, Y=pool_labels,
    batch_n_pc=npc, per_label_n_pc=plpc,
    window_size=ws, n_samples=500, flag_shuffle=True, flag_replacement=True)
print(f"Offline Threshold: {time.time()-t0:.2f}s, n_samples=500")
tp95 = float(np.real(pbs[int(len(pbs)*0.95)]))
tp99 = float(np.real(pbs[int(len(pbs)*0.99)]))
print(f"  P95={tp95:.4f}, P99={tp99:.4f}")

ql = km.predict(q_n)
print(f"\nOnline: {nw} windows x {ws}")
dl_scores = []
dl_drifts = []
for i in range(nw):
    s, e = i * ws, (i + 1) * ws
    w_q = q_n[s:e]
    w_l = ql[s:e]
    # 确保所有标签都有样本，缺失的标签用最近的poolemb补
    present = set(w_l)
    missing = set(range(n_cls)) - present
    if missing:
        aug_q = list(w_q)
        aug_l = list(w_l)
        for ml in missing:
            mask = pool_labels == ml
            if mask.any():
                aug_q.append(p_n[mask][0])
                aug_l.append(ml)
        w_q = np.array(aug_q)
        w_l = np.array(aug_l)
    try:
        dist = dl.compute_window_distribution_distances(w_q, w_l)
        bd = float(np.real(dist['per-batch']))
    except Exception as ex:
        bd = float('nan')
    d = bd > tp95 if not np.isnan(bd) else False
    dl_scores.append(bd)
    dl_drifts.append(d)
    mt = queries[s].topic[:25]
    flag = '*** DRIFT ***' if d else ''
    print(f"  W{i+1:02d} [{mt:>25}]: FID={bd:8.4f} {flag}")
print(f"\nDriftLens: Drift {sum(dl_drifts)}/{nw} windows (P95={tp95:.4f})")

# == 我们的 GMM ==
print("\n" + "=" * 60)
print("我们的 GMM 检测器 (GMM KL + EMA+MAD)")
print("=" * 60)
logging.disable(logging.NOTSET)
import logging as _lg
_lg.getLogger('updator.qarc').setLevel(_lg.WARNING)
from updator.qarc.interest_model import GMMDriftDetector, auto_kmeans, compute_alignment_gap

kb_e = p_n[:200]
gd = GMMDriftDetector(n_components_range=(1, 5), beta=0.85, k_drift=2.5)
c0, _, _ = auto_kmeans(q_n[:ws])
gd.set_reference(kb_e, c0)
print(f"Baseline: {kb_e.shape[0]} KB docs, {len(c0)} centroids")

gmm_scores = []
gmm_drifts = []
gaps = []
for i in range(nw):
    s, e = i * ws, (i + 1) * ws
    cw, _, _ = auto_kmeans(q_n[s:e])
    gap = compute_alignment_gap(q_n[s:e], kb_e)
    gr = gd.compute_drift_score(q_n[s:e], cw)
    t = gr.get('triggered', False)
    gmm_scores.append(gr['drift_score'])
    gmm_drifts.append(t)
    gaps.append(gap.gap)
    mt = queries[s].topic[:25]
    flag = '*** DRIFT ***' if t else ''
    print(f"  W{i+1:02d} [{mt:>25}]: Gap={gap.gap:.4f}, KL={gr['drift_score']:.4f} {flag}")
print(f"\nGMM: Drift {sum(gmm_drifts)}/{nw} windows")

# == 对比分析 ==
print("\n" + "=" * 60)
print("对比分析")
print("=" * 60)

# 真实主题变化
topics = [queries[i * ws].topic for i in range(nw)]
true_changes = []
for i in range(nw):
    if i == 0:
        true_changes.append(False)
    else:
        true_changes.append(topics[i] != topics[i-1])
n_true = sum(true_changes)
print(f"\n真实主题变化: {n_true}/{nw} windows")

print(f"\n{'Win':>4} {'Topic':>25} {'Changed':>7} {'DL_FID':>8} {'DL_drift':>8} {'GMM_KL':>8} {'GMM_dr':>6} {'Gap':>6}")
print("-" * 84)
for i in range(nw):
    print(f"W{i+1:02d}  {topics[i][:25]:>25} {'YES' if true_changes[i] else '':>7} "
          f"{dl_scores[i]:8.4f} {'DRIFT' if dl_drifts[i] else '':>8} "
          f"{gmm_scores[i]:8.4f} {'DRIFT' if gmm_drifts[i] else '':>6} "
          f"{gaps[i]:6.4f}")

# 计算检测质量指标
def detection_metrics(detected, actual):
    tp = sum(1 for d, a in zip(detected, actual) if d and a)
    fp = sum(1 for d, a in zip(detected, actual) if d and not a)
    fn = sum(1 for d, a in zip(detected, actual) if not d and a)
    tn = sum(1 for d, a in zip(detected, actual) if not d and not a)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return tp, fp, fn, tn, prec, rec, f1

dl_tp, dl_fp, dl_fn, dl_tn, dl_prec, dl_rec, dl_f1 = detection_metrics(dl_drifts, true_changes)
gm_tp, gm_fp, gm_fn, gm_tn, gm_prec, gm_rec, gm_f1 = detection_metrics(gmm_drifts, true_changes)

print(f"\n{'指标':>12} {'DriftLens':>10} {'GMM':>10}")
print("-" * 34)
print(f"{'TP':>12} {dl_tp:>10} {gm_tp:>10}")
print(f"{'FP':>12} {dl_fp:>10} {gm_fp:>10}")
print(f"{'FN':>12} {dl_fn:>10} {gm_fn:>10}")
print(f"{'TN':>12} {dl_tn:>10} {gm_tn:>10}")
print(f"{'Precision':>12} {dl_prec:>10.3f} {gm_prec:>10.3f}")
print(f"{'Recall':>12} {dl_rec:>10.3f} {gm_rec:>10.3f}")
print(f"{'F1':>12} {dl_f1:>10.3f} {gm_f1:>10.3f}")

agree = sum(1 for a, b in zip(dl_drifts, gmm_drifts) if a == b)
print(f"\n两检测器一致性: {agree}/{nw} ({100*agree/nw:.1f}%)")

print("\n主题变化序列:")
for i in range(nw):
    c = "→" if true_changes[i] else " "
    print(f"  W{i+1:02d} {c} {topics[i]}")
