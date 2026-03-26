"""
benchmark/run_experiments.py — 真实 RAG Embedding 实验 CLI

使用 core/ 的真实 Embedding 模型 (nomic-embed-text-v1.5) 跑 KB 更新策略对比实验
数据源: benchmark/data/ (基于 topic 和随机游走构造的偏移数据)

实验场景:
  1. gradual_drift   — Gaussian 混合的渐变 topic 漂移 (WoW)
  2. sudden_shift    — Sigmoid 阶跃 topic 切换 (WoW)
  3. cyclic_return   — 周期性 topic 回归 (WoW)
  4. hotpotqa_walk   — HotpotQA 实体图随机游走

Usage:
    python -m benchmark.run_experiments --quick     # embedding 检索, 不调 LLM
    python -m benchmark.run_experiments --full      # 完整: embedding + reranker + LLM
    python -m benchmark.run_experiments --exp gradual_drift --quick
    python -m benchmark.run_experiments --quick --kb-budget 80 --queries 300
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.rag_pipeline import RAGPipeline
from benchmark.adapters import (
    QARCStrategyAdapter,
    ComRAGStrategyAdapter,
    ERASEStrategyAdapter,
)
from updator.base import (
    KBUpdateStrategy,
    StaticKBStrategy,
    RandomKBStrategy,
    ProcessResult,
)
from core.evaluator import (
    recall_at_k,
    gold_in_kb_rate,
    compute_adaptation_speed,
    kb_turnover_rate,
    sliding_window_recall,
    update_efficiency,
    cost_adjusted_recall,
    comprehensive_score,
)

logger = logging.getLogger(__name__)


# ============================================================
# 数据加载 — 全部来自 benchmark.data
# ============================================================

def _load_dataset(builder_fn, total_queries: int, **kwargs) -> Dict:
    """
    通用加载器: 调用 benchmark.data 的 builder, 返回统一格式

    Returns:
        {
            "queries": [{"query", "answer", "topic", "gold_doc_ids"}],
            "doc_pool": [{"doc_id", "text", "topic"}],
        }
    """
    dataset = builder_fn(total_queries=total_queries, **kwargs)

    doc_pool = [
        {"doc_id": d.doc_id, "text": d.text, "topic": d.topic}
        for d in dataset.document_pool
    ]

    # 构建 topic → doc_ids 映射，用于 topic-level 评估
    from collections import defaultdict
    topic_to_doc_ids = defaultdict(list)
    for d in doc_pool:
        topic_to_doc_ids[d["topic"]].append(d["doc_id"])

    queries = [
        {
            "query": q.question,
            "answer": q.answer,
            "topic": q.topic,
            "gold_doc_ids": q.gold_doc_ids,
            "topic_doc_ids": topic_to_doc_ids.get(q.topic, []),
        }
        for q in dataset.query_stream
    ]

    avg_topic_docs = sum(len(item["topic_doc_ids"]) for item in queries) / max(len(queries), 1)

    logger.info(
        f"Dataset loaded: {len(queries)} queries, {len(doc_pool)} pool docs, "
        f"topics={list(dataset.topics)[:6]}, "
        f"avg topic_docs/query={avg_topic_docs:.0f}"
    )
    return {"queries": queries, "doc_pool": doc_pool}


def load_gradual_drift(total_queries: int = 400, pool_size: int = 5000) -> Dict:
    """Gaussian 渐变 topic 漂移 (Wizard of Wikipedia)"""
    from benchmark.data import build_gradual_drift
    return _load_dataset(build_gradual_drift, total_queries, pool_size=pool_size)


def load_sudden_shift(total_queries: int = 400, pool_size: int = 5000) -> Dict:
    """Sigmoid 阶跃 topic 切换 (Wizard of Wikipedia)"""
    from benchmark.data import build_sudden_shift
    return _load_dataset(build_sudden_shift, total_queries, pool_size=pool_size)


def load_cyclic_return(total_queries: int = 400, pool_size: int = 5000) -> Dict:
    """周期性 topic 回归 (Wizard of Wikipedia)"""
    from benchmark.data import build_cyclic_return
    return _load_dataset(build_cyclic_return, total_queries, pool_size=pool_size)


def load_hotpotqa_walk(total_queries: int = 400, pool_size: int = 50000) -> Dict:
    """HotpotQA 实体图随机游走"""
    from benchmark.data import build_hotpotqa_entity_walk
    return _load_dataset(build_hotpotqa_entity_walk, total_queries, max_pool=pool_size)


# ============================================================
# 方法工厂
# ============================================================

def make_methods(kb_budget: int, window_size: int) -> List[KBUpdateStrategy]:
    """创建所有待对比的方法 (3 论文方法 + 2 baseline)"""
    return [
        QARCStrategyAdapter(kb_budget=kb_budget, window_size=window_size),
        ComRAGStrategyAdapter(kb_budget=kb_budget),
        ERASEStrategyAdapter(kb_budget=kb_budget),
        StaticKBStrategy(),
        RandomKBStrategy(seed=42, update_interval=50),
    ]


# ============================================================
# 单实验运行
# ============================================================

def run_single_experiment(
    pipeline: RAGPipeline,
    methods: List[KBUpdateStrategy],
    queries: List[Dict],
    doc_pool: List[Dict],
    doc_embeddings: np.ndarray,
    kb_budget: int = 50,
    use_generator: bool = False,
) -> Dict[str, Dict]:
    """
    对比所有方法在同一查询流上的表现

    Returns:
        {method_name: {"recalls", "avg_recall", "avg_gold_in_kb", ...}}
    """
    # 预计算查询嵌入
    query_texts = [q["query"] for q in queries]
    logger.info(f"Encoding {len(query_texts)} queries...")
    query_embeddings = pipeline.encode_queries(query_texts, batch_size=32)

    pool_id_to_idx = {d["doc_id"]: i for i, d in enumerate(doc_pool)}

    results = {}

    for method in methods:
        name = method.name
        logger.info(f"\n{'='*60}\n  Running: {name}\n{'='*60}")

        t0 = time.time()
        method.initialize(doc_pool, doc_embeddings, kb_budget)
        init_time = time.time() - t0

        recalls = []
        gold_kb_rates = []
        kb_sizes = []
        kb_snapshots = []  # 每步的 KB doc_id 快照 (用于 turnover)
        update_count = 0

        for step, q in enumerate(queries):
            proc = method.process_query(
                query_text=q["query"],
                query_embedding=query_embeddings[step],
                step=step,
                gold_doc_ids=q.get("gold_doc_ids"),
            )

            if proc.update_performed:
                update_count += 1

            gold_ids = q.get("gold_doc_ids", [])
            r = recall_at_k(proc.retrieved_doc_ids, gold_ids)
            recalls.append(r)

            kb_ids = method.get_kb_doc_ids()
            g = gold_in_kb_rate(kb_ids, gold_ids)
            gold_kb_rates.append(g)
            kb_sizes.append(proc.kb_size)
            kb_snapshots.append(set(kb_ids))

            # 进度
            if (step + 1) % 50 == 0 or step == len(queries) - 1:
                recent_r = np.mean(recalls[-50:]) if recalls else 0
                recent_g = np.mean(gold_kb_rates[-50:]) if gold_kb_rates else 0
                logger.info(
                    f"  [{name}] step={step+1}/{len(queries)} "
                    f"recall={recent_r:.4f} gold_kb={recent_g:.4f} "
                    f"kb={proc.kb_size} updates={update_count}"
                )

        total_time = time.time() - t0
        adapt_speed = compute_adaptation_speed(recalls, window_size=20)

        turnover = kb_turnover_rate(kb_snapshots)
        sw_recall = sliding_window_recall(recalls, window_size=20)

        avg_r = float(np.mean(recalls))
        comp = comprehensive_score(
            avg_recall=avg_r,
            total_updates=update_count,
            kb_turnover=turnover,
            recalls=recalls,
            total_queries=len(queries),
        )

        results[name] = {
            "recalls": [float(x) for x in recalls],
            "gold_in_kb_rates": [float(x) for x in gold_kb_rates],
            "kb_sizes": [int(x) for x in kb_sizes],
            "sliding_recall": sw_recall,
            "avg_recall": avg_r,
            "avg_gold_in_kb": float(np.mean(gold_kb_rates)),
            "kb_turnover": float(turnover),
            "adaptation_speed": float(adapt_speed),
            "total_updates": update_count,
            "total_time_sec": round(total_time, 2),
            "init_time_sec": round(init_time, 2),
            # 成本-精度综合评估
            "update_efficiency": comp["update_efficiency"],
            "cost_adjusted_recall": comp["cost_adjusted_recall"],
            "composite_score": comp["composite"],
            "efficiency_score": comp["efficiency_score"],
            "stability_score": comp["stability_score"],
            "precision_score": comp["precision_score"],
        }

        logger.info(
            f"  [{name}] DONE: recall={results[name]['avg_recall']:.4f} "
            f"gold_kb={results[name]['avg_gold_in_kb']:.4f} "
            f"adapt={adapt_speed:.1f}w updates={update_count} "
            f"time={total_time:.1f}s"
        )

    return results


# ============================================================
# 实验套件
# ============================================================

def _save_markdown_report(exp_name: str, results: Dict, path: str):
    """将实验结果保存为 Markdown 格式的可读报告。"""
    lines = []
    lines.append(f"# {exp_name} — Experiment Report\n")
    from datetime import datetime
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 基础指标表
    lines.append("## Performance Metrics\n")
    lines.append("| Method | Recall@10 | Gold_KB | Turnover | Adapt(w) | Updates | Time |")
    lines.append("|--------|-----------|---------|----------|----------|---------|------|")
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['avg_recall']:.4f} | {r['avg_gold_in_kb']:.4f} | "
            f"{r['kb_turnover']:.4f} | {r['adaptation_speed']:.1f} | "
            f"{r['total_updates']} | {r['total_time_sec']:.1f}s |"
        )

    # 成本-精度综合表
    lines.append("")
    lines.append("## Cost-Accuracy Tradeoff\n")
    lines.append("| Method | Recall | Updates | UE | CAR | Effic | Stab | Prec | Composite |")
    lines.append("|--------|--------|---------|------|------|-------|------|------|-----------|")
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['avg_recall']:.4f} | {r['total_updates']} | "
            f"{r['update_efficiency']:.4f} | {r['cost_adjusted_recall']:.4f} | "
            f"{r['efficiency_score']:.4f} | {r['stability_score']:.4f} | "
            f"{r['precision_score']:.4f} | {r['composite_score']:.4f} |"
        )

    # 排名
    ranked = sorted(results.items(), key=lambda x: x[1]['composite_score'], reverse=True)
    lines.append("")
    lines.append("## Ranking (by Composite Score)\n")
    for rank, (name, r) in enumerate(ranked, 1):
        tag = " ⭐" if name == "QARC" else ""
        lines.append(f"{rank}. **{name}** — composite={r['composite_score']:.4f}{tag}")

    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_experiment_suite(
    pipeline: RAGPipeline,
    exp_name: str,
    data: Dict,
    kb_budget: int,
    window_size: int,
    use_generator: bool,
    out_dir: str,
) -> Dict:
    """运行一个实验场景"""
    doc_pool = data["doc_pool"]
    queries = data["queries"]

    if not doc_pool or not queries:
        logger.error(f"No data for {exp_name}")
        return {}

    logger.info(f"\n{'#'*70}")
    logger.info(f"# Experiment: {exp_name}")
    logger.info(f"# Docs: {len(doc_pool)}, Queries: {len(queries)}, KB budget: {kb_budget}")
    logger.info(f"{'#'*70}")

    # 编码文档池
    doc_texts = [d["text"] for d in doc_pool]
    logger.info(f"Encoding {len(doc_texts)} documents...")
    doc_embeddings = pipeline.encode_documents(doc_texts, batch_size=64)

    # 创建方法并运行
    methods = make_methods(kb_budget, window_size)
    results = run_single_experiment(
        pipeline=pipeline,
        methods=methods,
        queries=queries,
        doc_pool=doc_pool,
        doc_embeddings=doc_embeddings,
        kb_budget=kb_budget,
        use_generator=use_generator,
    )

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{exp_name}_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved: {out_file}")

    # 保存可读的 Markdown 表格
    md_file = os.path.join(out_dir, f"{exp_name}_report.md")
    _save_markdown_report(exp_name, results, md_file)
    logger.info(f"Markdown report saved: {md_file}")

    # 汇总表
    print(f"\n{'='*75}")
    print(f"  {exp_name} — Results")
    print(f"{'='*75}")
    print(f"  {'Method':<12} {'Recall@10':>10} {'Gold_KB':>10} {'Turnover':>10} {'Adapt(w)':>10} {'Updates':>8} {'Time':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for name, r in results.items():
        print(
            f"  {name:<12} {r['avg_recall']:>10.4f} {r['avg_gold_in_kb']:>10.4f} "
            f"{r['kb_turnover']:>10.4f} "
            f"{r['adaptation_speed']:>10.1f} {r['total_updates']:>8d} "
            f"{r['total_time_sec']:>7.1f}s"
        )

    # ---- 成本-精度综合对比表 ----
    print(f"\n{'='*90}")
    print(f"  {exp_name} — Cost-Accuracy Tradeoff")
    print(f"{'='*90}")
    print(
        f"  {'Method':<12} {'Recall':>8} {'Updates':>8} {'UE':>8} {'CAR':>8} "
        f"{'Effic':>8} {'Stab':>8} {'Prec':>8} {'Composite':>10}"
    )
    print(
        f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} "
        f"{'-'*8} {'-'*8} {'-'*8} {'-'*10}"
    )
    for name, r in results.items():
        print(
            f"  {name:<12} {r['avg_recall']:>8.4f} {r['total_updates']:>8d} "
            f"{r['update_efficiency']:>8.4f} {r['cost_adjusted_recall']:>8.4f} "
            f"{r['efficiency_score']:>8.4f} {r['stability_score']:>8.4f} "
            f"{r['precision_score']:>8.4f} {r['composite_score']:>10.4f}"
        )
    # 按综合得分排序
    ranked = sorted(results.items(), key=lambda x: x[1]['composite_score'], reverse=True)
    print(f"\n  Ranking by Composite Score:")
    for rank, (name, r) in enumerate(ranked, 1):
        tag = " ★" if name == "QARC" else ""
        print(f"    #{rank} {name:<12} composite={r['composite_score']:.4f}{tag}")

    return results


# ============================================================
# CLI
# ============================================================

EXPERIMENTS = {
    "gradual_drift": load_gradual_drift,
    "sudden_shift":  load_sudden_shift,
    "cyclic_return": load_cyclic_return,
    "hotpotqa_walk": load_hotpotqa_walk,
}


def main():
    parser = argparse.ArgumentParser(
        description="KB update strategy comparison with real RAG embeddings"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="只用 embedding 检索 (不调 reranker/LLM)")
    mode.add_argument("--full", action="store_true",
                      help="embedding + reranker + LLM 端到端")

    parser.add_argument("--exp", choices=list(EXPERIMENTS.keys()),
                        help="运行指定实验 (默认跑全部)")
    parser.add_argument("--kb-budget", type=int, default=200)
    parser.add_argument("--pool-size", type=int, default=None,
                        help="文档池大小 (默认: quick=5000, full=50000)")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--queries", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    use_generator = args.full
    max_queries = args.queries or (200 if args.quick else 400)
    pool_size = args.pool_size or (5000 if args.quick else 50000)
    out_dir = args.output_dir or str(ROOT / "benchmark" / "results")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 初始化 pipeline
    pipeline = RAGPipeline(
        use_real_embeddings=True,
        use_reranker=args.full,
        use_generator=use_generator,
        device=args.device,
    )

    all_results = {}
    exps = [args.exp] if args.exp else list(EXPERIMENTS.keys())

    for exp_name in exps:
        loader = EXPERIMENTS[exp_name]
        data = loader(total_queries=max_queries, pool_size=pool_size)

        if not data["queries"]:
            logger.warning(f"No data for {exp_name}, skipping")
            continue

        results = run_experiment_suite(
            pipeline=pipeline,
            exp_name=exp_name,
            data=data,
            kb_budget=args.kb_budget,
            window_size=args.window_size,
            use_generator=use_generator,
            out_dir=out_dir,
        )
        all_results[exp_name] = results

    # 最终汇总
    print(f"\n{'='*85}")
    print(f"  FINAL SUMMARY — Real RAG Embeddings")
    print(f"{'='*85}")
    for exp_name, results in all_results.items():
        print(f"\n  [{exp_name}]")
        ranked = sorted(results.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        for name, r in ranked:
            print(
                f"    {name:<12} recall={r['avg_recall']:.4f}  "
                f"UE={r['update_efficiency']:.4f}  "
                f"CAR={r['cost_adjusted_recall']:.4f}  "
                f"composite={r['composite_score']:.4f}  "
                f"updates={r['total_updates']}"
            )


if __name__ == "__main__":
    main()
