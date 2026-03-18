"""调试 GMM 检测器为什么 KL 全部为 0"""
import numpy as np, warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

from test.experiment_datasets import build_gradual_drift
ds = build_gradual_drift(100)
queries = ds.query_stream
pool = ds.document_pool

from core.data_processor import embedding_service
embed = lambda t: np.array(embedding_service.embed_query(t))
q_embs = np.array([embed(q.question) for q in queries])
pool_embs = np.array([embed(d.text) for d in pool[:200]])

def l2n(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-10, None)
q_n, p_n = l2n(q_embs), l2n(pool_embs)

from updator.qarc.interest_model import GMMDriftDetector, auto_kmeans

ws = 10
c0, _, _ = auto_kmeans(q_n[:ws])
print(f"初始聚类数: {len(c0)}")

gd = GMMDriftDetector(n_components_range=(1, 5), beta=0.85, k_drift=2.5)
gd.set_reference(p_n, c0)

print(f"reference_features shape: {gd.reference_features.shape if gd.reference_features is not None else None}")
print(f"reference_gmm: {gd.reference_gmm}")
print(f"drift_ema: {gd.drift_ema}")

w_q = q_n[:ws]
c1, _, _ = auto_kmeans(w_q)
print(f"\n窗口1 聚类数: {len(c1)}")

feat = gd._compute_distance_features(w_q, c1)
print(f"距离特征 shape: {feat.shape}")
print(f"距离特征 min={feat.min():.4f}, max={feat.max():.4f}")
print(f"距离特征样本:\n{feat[:3]}")

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=1, random_state=42).fit(feat)
print(f"\nGMM means: {gmm.means_}")
print(f"GMM covariances shape: {gmm.covariances_.shape}")

score = gd._symmetric_kl(gd.reference_gmm, gmm, feat.shape[1])
print(f"KL score: {score}")

print(f"\nref GMM means: {gd.reference_gmm.means_}")
print(f"ref GMM n_components: {gd.reference_gmm.n_components}")

# 窗口5 (Pizza)
w_q5 = q_n[40:50]
c5, _, _ = auto_kmeans(w_q5)
print(f"\n窗口5 (Pizza) 聚类数: {len(c5)}")
feat5 = gd._compute_distance_features(w_q5, c5)
print(f"距离特征 shape: {feat5.shape}")
print(f"距离特征 min={feat5.min():.4f}, max={feat5.max():.4f}")

result = gd.compute_drift_score(w_q5, c5)
print(f"\ncompute_drift_score 结果: {result}")
