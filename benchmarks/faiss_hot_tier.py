#!/usr/bin/env python3
"""真实 FAISS hot/cold tier 延迟与 replacement 微基准。

策略实验使用预计算 embedding 和 NumPy，适合比较缓存内容，但不代表线上索引
延迟。本脚本直接构建 ``IndexFlatIP`` / ``IndexIDMap2``，报告单 query 搜索延迟、
索引构建时间和 remove+add replacement 成本。它不测 LLM 生成或网络存储。
"""

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np


def _percentiles(samples_ms):
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(values.mean()), 6),
        "p50_ms": round(float(np.percentile(values, 50)), 6),
        "p95_ms": round(float(np.percentile(values, 95)), 6),
        "p99_ms": round(float(np.percentile(values, 99)), 6),
    }


def _search_latency(index, queries, top_k, warmup):
    for query in queries[:warmup]:
        index.search(query[None], top_k)
    samples = []
    for query in queries:
        started = time.perf_counter_ns()
        index.search(query[None], top_k)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return _percentiles(samples)


def _build_flat(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])
    started = time.perf_counter()
    index.add(vectors)
    return index, (time.perf_counter() - started) * 1000.0


def _replacement_latency(hot_vectors, cold_vectors, count):
    dimension = hot_vectors.shape[1]
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
    hot_ids = np.arange(len(hot_vectors), dtype=np.int64)
    index.add_with_ids(hot_vectors, hot_ids)
    count = min(count, len(hot_vectors), len(cold_vectors) - len(hot_vectors))
    samples = []
    for offset in range(count):
        victim = np.asarray([offset], dtype=np.int64)
        candidate_id = len(hot_vectors) + offset
        candidate = cold_vectors[candidate_id:candidate_id + 1]
        started = time.perf_counter_ns()
        index.remove_ids(victim)
        index.add_with_ids(candidate, np.asarray([candidate_id], dtype=np.int64))
        samples.append((time.perf_counter_ns() - started) / 1e6)
    result = _percentiles(samples)
    result["count"] = int(count)
    result["payload_bytes_per_write"] = int(dimension * 4)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-embeddings", required=True)
    parser.add_argument("--query-embeddings", required=True)
    parser.add_argument("--hot-ratio", type=float, default=0.1)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--replacements", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()

    faiss.omp_set_num_threads(max(1, args.threads))
    documents = np.ascontiguousarray(
        np.load(args.doc_embeddings).astype(np.float32))
    queries = np.ascontiguousarray(
        np.load(args.query_embeddings).astype(np.float32)[:args.queries])
    hot_size = max(1, int(round(len(documents) * args.hot_ratio)))
    hot_vectors = documents[:hot_size]

    cold_index, cold_build_ms = _build_flat(documents)
    hot_index, hot_build_ms = _build_flat(hot_vectors)
    result = {
        "pool_size": int(len(documents)),
        "hot_size": int(hot_size),
        "hot_ratio": float(hot_size / len(documents)),
        "dimension": int(documents.shape[1]),
        "queries": int(len(queries)),
        "top_k": int(args.top_k),
        "threads": int(args.threads),
        "cold_build_ms": round(float(cold_build_ms), 3),
        "hot_build_ms": round(float(hot_build_ms), 3),
        "cold_search": _search_latency(cold_index, queries, args.top_k, 20),
        "hot_search": _search_latency(hot_index, queries, args.top_k, 20),
        "replacement": _replacement_latency(
            hot_vectors, documents, args.replacements),
    }

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
