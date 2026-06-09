#!/usr/bin/env python3
"""
Cold-tier latency benchmark: local FAISS (in-memory) vs Qdrant REST (container).

Methodology
-----------
  hot-tier  -> FAISS IndexFlatIP on hot KB (~750 docs, same as experiment)
  cold-tier -> Qdrant REST collection over cold pool (~5K docs)

For each tier run N_QUERIES queries; record p50/p90/p99 latency.
ratio = cold_p50 / hot_p50 -> empirical COLD_REMOTE_SCALE.

Prereq:  docker run -d -p 6377:6333 qdrant/qdrant:v1.9.2  (already running)
"""
from __future__ import annotations
import sys, json, time, numpy as np
from pathlib import Path

QDRANT_URL   = "http://localhost:6377"
COLLECTION   = "cold_pool_bench"
TOP_K        = 5
N_QUERIES    = 200
WARMUP       = 20
HOT_KB_SIZE  = 750
COLD_POOL    = 5_000
DIM          = 384
BATCH_UPSERT = 256

sys.path.insert(0, str(Path(__file__).parent.parent / 'motivation_1'))

print("=" * 64)
print("Loading StreamingQA docs …")
from loaders_temporal import load_streamingqa_temporal
doc_pool, queries, _ = load_streamingqa_temporal(n_distractors=5000)
doc_pool = doc_pool[:COLD_POOL]
queries  = queries[:N_QUERIES + WARMUP]

from sentence_transformers import SentenceTransformer
enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

doc_texts   = [d['text'][:512] for d in doc_pool]
query_texts = [q['question']   for q in queries]
print(f"Encoding {len(doc_texts):,} docs …")
doc_embs = enc.encode(doc_texts, batch_size=256, normalize_embeddings=True,
                      show_progress_bar=True).astype('float32')
print(f"Encoding {len(query_texts)} queries …")
q_embs   = enc.encode(query_texts, batch_size=256, normalize_embeddings=True,
                      show_progress_bar=False).astype('float32')

# -- HOT-TIER: FAISS in-memory ------------------------------------------------
import faiss
print(f"\nFAISS hot KB ({HOT_KB_SIZE} docs) …")
idx_flat = faiss.IndexFlatIP(DIM)
idx_flat.add(doc_embs[:HOT_KB_SIZE])
for i in range(WARMUP):
    idx_flat.search(q_embs[i:i+1], TOP_K)
hot_times = []
for i in range(WARMUP, WARMUP + N_QUERIES):
    t0 = time.perf_counter()
    idx_flat.search(q_embs[i:i+1], TOP_K)
    hot_times.append((time.perf_counter() - t0) * 1000)
hot_p50 = float(np.percentile(hot_times, 50))
hot_p90 = float(np.percentile(hot_times, 90))
hot_p99 = float(np.percentile(hot_times, 99))
print(f"  FAISS  p50={hot_p50:.3f}ms  p90={hot_p90:.3f}ms  p99={hot_p99:.3f}ms")

# -- COLD-TIER: Qdrant REST ---------------------------------------------------
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

print(f"\nQdrant cold pool ({len(doc_pool):,} docs) …")
client = QdrantClient(url=QDRANT_URL, timeout=30)
existing = [c.name for c in client.get_collections().collections]
if COLLECTION in existing:
    client.delete_collection(COLLECTION)
client.create_collection(COLLECTION,
    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
points = []
for i, (doc, emb) in enumerate(zip(doc_pool, doc_embs)):
    points.append(PointStruct(id=i, vector=emb.tolist(),
                               payload={"doc_id": doc['doc_id']}))
    if len(points) == BATCH_UPSERT:
        client.upsert(COLLECTION, points=points); points = []
if points:
    client.upsert(COLLECTION, points=points)
print(f"  Upserted {len(doc_pool):,} points.")
for i in range(WARMUP):
    client.search(COLLECTION, query_vector=q_embs[i].tolist(), limit=TOP_K)
cold_times = []
for i in range(WARMUP, WARMUP + N_QUERIES):
    t0 = time.perf_counter()
    client.search(COLLECTION, query_vector=q_embs[i].tolist(), limit=TOP_K)
    cold_times.append((time.perf_counter() - t0) * 1000)
cold_p50 = float(np.percentile(cold_times, 50))
cold_p90 = float(np.percentile(cold_times, 90))
cold_p99 = float(np.percentile(cold_times, 99))
print(f"  Qdrant p50={cold_p50:.3f}ms  p90={cold_p90:.3f}ms  p99={cold_p99:.3f}ms")

# -- Summary ------------------------------------------------------------------
ratio = cold_p50 / max(hot_p50, 0.001)
print("\n" + "=" * 64)
print(f"  HOT  FAISS in-mem KB={HOT_KB_SIZE}   p50 = {hot_p50:.3f} ms")
print(f"  COLD Qdrant REST  pool={len(doc_pool):,}  p50 = {cold_p50:.3f} ms")
print(f"  Empirical ratio  cold/hot = {ratio:.1f}x")
print(f"  (Model used COLD_REMOTE_SCALE = 5.0 per VectorDBBench)")
print("=" * 64)

result = {
    "setup": {
        "hot_kb_size": HOT_KB_SIZE, "cold_pool_size": len(doc_pool),
        "n_queries": N_QUERIES, "dim": DIM, "top_k": TOP_K,
        "qdrant_version": "v1.9.2", "encoder": "all-MiniLM-L6-v2",
    },
    "hot_faiss":   {"p50_ms": round(hot_p50,3),  "p90_ms": round(hot_p90,3),  "p99_ms": round(hot_p99,3)},
    "cold_qdrant": {"p50_ms": round(cold_p50,3), "p90_ms": round(cold_p90,3), "p99_ms": round(cold_p99,3)},
    "ratio_cold_hot": round(ratio, 2),
}
out = Path(__file__).parent / 'bench_qdrant_result.json'
out.write_text(json.dumps(result, indent=2))
print(f"\nResult saved -> {out}")
client.delete_collection(COLLECTION)
