"""
benchmarks/archive_legacy/run_experiments.py — 真实 RAG Embedding 实验 CLI

使用 core/ 的真实 Embedding 模型 (nomic-embed-text-v1.5) 跑 KB 更新策略对比实验
数据源: benchmarks/archive_legacy/data/ (基于 topic 和随机游走构造的偏移数据)

实验场景:
  1. gradual_drift   — Gaussian 混合的渐变 topic 漂移 (WoW)
  2. sudden_shift    — Sigmoid 阶跃 topic 切换 (WoW)
  3. cyclic_return   — 周期性 topic 回归 (WoW)
  4. hotpotqa_walk   — HotpotQA 实体图随机游走

Usage:
    python -m benchmarks.archive_legacy.run_experiments --quick     # embedding 检索, 不调 LLM
    python -m benchmarks.archive_legacy.run_experiments --full      # 完整: embedding + reranker + LLM
    python -m benchmarks.archive_legacy.run_experiments --exp gradual_drift --quick
    python -m benchmarks.archive_legacy.run_experiments --quick --kb-budget 80 --queries 300
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.archive_legacy.rag_pipeline import RAGPipeline
from benchmarks.archive_legacy.adapters import (
    DRIPStrategyAdapter,
)
from algorithms.base import (
    KBUpdateStrategy,
    StaticKBStrategy,
    RandomKBStrategy,
    ProcessResult,
)
from core.evaluator import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    gold_in_kb_rate,
    compute_adaptation_speed,
    kb_turnover_rate,
    sliding_window_recall,
    exact_match,
    token_f1,
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
    from benchmarks.archive_legacy.data import build_gradual_drift
    return _load_dataset(build_gradual_drift, total_queries, pool_size=pool_size)


def load_sudden_shift(total_queries: int = 400, pool_size: int = 5000) -> Dict:
    """Sigmoid 阶跃 topic 切换 (Wizard of Wikipedia)"""
    from benchmarks.archive_legacy.data import build_sudden_shift
    return _load_dataset(build_sudden_shift, total_queries, pool_size=pool_size)


def load_cyclic_return(total_queries: int = 400, pool_size: int = 5000) -> Dict:
    """周期性 topic 回归 (Wizard of Wikipedia)"""
    from benchmarks.archive_legacy.data import build_cyclic_return
    return _load_dataset(build_cyclic_return, total_queries, pool_size=pool_size)


def load_hotpotqa_walk(total_queries: int = 400, pool_size: int = 50000) -> Dict:
    """HotpotQA 实体图随机游走"""
    from benchmarks.archive_legacy.data import build_hotpotqa_entity_walk
    return _load_dataset(build_hotpotqa_entity_walk, total_queries, max_pool=pool_size)


# ============================================================
# 方法工厂
# ============================================================

def make_methods(kb_budget: int, window_size: int) -> List[KBUpdateStrategy]:
    """创建所有待对比的方法 (DRIP + 2 baseline)

    注: ComRAG / ERASE 已移除 (跨范式, 非固定预算缓存换入换出, 不可比)。
    cache replacement 主 baseline (LRU/TinyLFU/GPTCacheStyle/AgentRAGCache) 见
    algorithms/cache/registry.py, 由 motivation_2 测试台运行。
    """
    return [
        DRIPStrategyAdapter(kb_budget=kb_budget, window_size=window_size),
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
    pool_size = len(doc_pool)

    results = {}

    for method in methods:
        name = method.name
        logger.info(f"\n{'='*60}\n  Running: {name}\n{'='*60}")

        t0 = time.time()
        method.initialize(doc_pool, doc_embeddings, kb_budget)
        init_time = time.time() - t0

        recalls = []
        precisions = []
        mrrs = []
        gold_kb_rates = []
        kb_sizes = []
        kb_snapshots = []
        em_scores = []
        f1_scores = []
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
            p = precision_at_k(proc.retrieved_doc_ids, gold_ids)
            precisions.append(p)
            m = mean_reciprocal_rank(proc.retrieved_doc_ids, gold_ids)
            mrrs.append(m)

            # LLM 生成 + EM/F1 评估 (--full 模式)
            if use_generator and pipeline.use_generator:
                retrieved_texts = []
                for did in proc.retrieved_doc_ids[:5]:
                    idx = pool_id_to_idx.get(did)
                    if idx is not None:
                        retrieved_texts.append({"text": doc_pool[idx]["text"]})
                answer = pipeline.generate(q["query"], retrieved_texts)
                gold_answer = q.get("answer", "")
                if gold_answer:
                    em_scores.append(exact_match(answer, gold_answer))
                    f1_scores.append(token_f1(answer, gold_answer))

            kb_ids = method.get_kb_doc_ids()
            g = gold_in_kb_rate(kb_ids, gold_ids)
            gold_kb_rates.append(g)
            kb_sizes.append(proc.kb_size)
            kb_snapshots.append(set(kb_ids))

            # 进度
            if (step + 1) % 50 == 0 or step == len(queries) - 1:
                recent_r = np.mean(recalls[-50:]) if recalls else 0
                recent_g = np.mean(gold_kb_rates[-50:]) if gold_kb_rates else 0
                extra = ""
                if em_scores:
                    extra = f" em={np.mean(em_scores[-50:]):.4f} f1={np.mean(f1_scores[-50:]):.4f}"
                logger.info(
                    f"  [{name}] step={step+1}/{len(queries)} "
                    f"recall={recent_r:.4f} gold_kb={recent_g:.4f} "
                    f"kb={proc.kb_size} updates={update_count}{extra}"
                )

        total_time = time.time() - t0
        adapt_speed = compute_adaptation_speed(recalls, window_size=20)
        turnover = kb_turnover_rate(kb_snapshots)
        sw_recall = sliding_window_recall(recalls, window_size=20)

        avg_r = float(np.mean(recalls))
        avg_p = float(np.mean(precisions))
        avg_mrr = float(np.mean(mrrs))

        results[name] = {
            "recalls": [float(x) for x in recalls],
            "precisions": [float(x) for x in precisions],
            "mrrs": [float(x) for x in mrrs],
            "gold_in_kb_rates": [float(x) for x in gold_kb_rates],
            "kb_sizes": [int(x) for x in kb_sizes],
            "sliding_recall": sw_recall,
            "avg_recall": avg_r,
            "avg_precision": avg_p,
            "avg_mrr": avg_mrr,
            "avg_gold_in_kb": float(np.mean(gold_kb_rates)),
            "avg_em": float(np.mean(em_scores)) if em_scores else None,
            "avg_f1": float(np.mean(f1_scores)) if f1_scores else None,
            "kb_turnover": float(turnover),
            "adaptation_speed": float(adapt_speed),
            "total_updates": update_count,
            "total_time_sec": round(total_time, 2),
            "init_time_sec": round(init_time, 2),
            "pool_size": pool_size,
            "kb_budget": kb_budget,
        }

        logger.info(
            f"  [{name}] DONE: recall={avg_r:.4f} prec={avg_p:.4f} mrr={avg_mrr:.4f} "
            f"gold_kb={results[name]['avg_gold_in_kb']:.4f} "
            f"updates={update_count} time={total_time:.1f}s"
            + (f" em={results[name]['avg_em']:.4f} f1={results[name]['avg_f1']:.4f}"
               if results[name]['avg_em'] is not None else "")
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

    # 从任意方法获取 pool/kb 信息
    sample = next(iter(results.values()), {})
    pool_sz = sample.get("pool_size", "?")
    kb_sz = sample.get("kb_budget", "?")
    lines.append(f"Settings: Pool Size = {pool_sz}, KB Budget = {kb_sz}\n")

    has_gen = any(r.get("avg_em") is not None for r in results.values())

    # 检索指标表
    lines.append("## Retrieval Metrics\n")
    header = "| Method | Recall@10 | Prec@10 | MRR | Gold_KB | Turnover | Updates | Time |"
    sep = "|--------|-----------|---------|-----|---------|----------|---------|------|"
    lines.append(header)
    lines.append(sep)
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['avg_recall']:.4f} | {r['avg_precision']:.4f} | "
            f"{r['avg_mrr']:.4f} | {r['avg_gold_in_kb']:.4f} | "
            f"{r['kb_turnover']:.4f} | {r['total_updates']} | {r['total_time_sec']:.1f}s |"
        )

    # 生成指标表 (仅 --full 模式)
    if has_gen:
        lines.append("")
        lines.append("## Generation Metrics (LLM End-to-End)\n")
        lines.append("| Method | EM | Token F1 | Recall@10 | Updates |")
        lines.append("|--------|-----|----------|-----------|---------|")
        for name, r in results.items():
            em = r.get("avg_em")
            f1 = r.get("avg_f1")
            em_s = f"{em:.4f}" if em is not None else "—"
            f1_s = f"{f1:.4f}" if f1 is not None else "—"
            lines.append(
                f"| {name} | {em_s} | {f1_s} | {r['avg_recall']:.4f} | {r['total_updates']} |"
            )

    # 排名 (按 recall)
    ranked = sorted(results.items(), key=lambda x: x[1]['avg_recall'], reverse=True)
    lines.append("")
    lines.append("## Ranking (by Recall@10)\n")
    for rank, (name, r) in enumerate(ranked, 1):
        tag = " ⭐" if rank == 1 else ""
        lines.append(f"{rank}. **{name}** — recall={r['avg_recall']:.4f}{tag}")

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
    has_gen = any(r.get("avg_em") is not None for r in results.values())
    print(f"\n{'='*90}")
    print(f"  {exp_name} — Results  (pool={len(doc_pool)}, kb_budget={kb_budget})")
    print(f"{'='*90}")
    if has_gen:
        print(f"  {'Method':<12} {'Recall@10':>10} {'Prec@10':>10} {'MRR':>10} {'Gold_KB':>10} {'EM':>8} {'F1':>8} {'Updates':>8} {'Time':>8}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for name, r in results.items():
            em_s = f"{r['avg_em']:>8.4f}" if r.get('avg_em') is not None else f"{'—':>8}"
            f1_s = f"{r['avg_f1']:>8.4f}" if r.get('avg_f1') is not None else f"{'—':>8}"
            print(
                f"  {name:<12} {r['avg_recall']:>10.4f} {r['avg_precision']:>10.4f} "
                f"{r['avg_mrr']:>10.4f} {r['avg_gold_in_kb']:>10.4f} "
                f"{em_s} {f1_s} {r['total_updates']:>8d} "
                f"{r['total_time_sec']:>7.1f}s"
            )
    else:
        print(f"  {'Method':<12} {'Recall@10':>10} {'Prec@10':>10} {'MRR':>10} {'Gold_KB':>10} {'Turnover':>10} {'Updates':>8} {'Time':>8}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
        for name, r in results.items():
            print(
                f"  {name:<12} {r['avg_recall']:>10.4f} {r['avg_precision']:>10.4f} "
                f"{r['avg_mrr']:>10.4f} {r['avg_gold_in_kb']:>10.4f} "
                f"{r['kb_turnover']:>10.4f} {r['total_updates']:>8d} "
                f"{r['total_time_sec']:>7.1f}s"
            )

    # 按 recall 排序
    ranked = sorted(results.items(), key=lambda x: x[1]['avg_recall'], reverse=True)
    print(f"\n  Ranking by Recall@10:")
    for rank, (name, r) in enumerate(ranked, 1):
        tag = " ★" if rank == 1 else ""
        extra = ""
        if r.get("avg_em") is not None:
            extra = f"  em={r['avg_em']:.4f}  f1={r['avg_f1']:.4f}"
        print(f"    #{rank} {name:<12} recall={r['avg_recall']:.4f}  prec={r['avg_precision']:.4f}  mrr={r['avg_mrr']:.4f}{extra}{tag}")

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



def _print_cross_scenario_summary(all_results: Dict, out_dir: str):
    """打印并保存跨场景汇总表"""
    if not all_results:
        return

    # 收集所有方法名 (保持顺序)
    method_names = []
    for results in all_results.values():
        for name in results.keys():
            if name not in method_names:
                method_names.append(name)
    exp_names = list(all_results.keys())

    has_gen = any(
        r.get("avg_em") is not None
        for results in all_results.values()
        for r in results.values()
    )

    # == 控制台输出 ==
    print(f"\n{'='*120}")
    print(f"  CROSS-SCENARIO SUMMARY — Real RAG Embeddings")
    print(f"{'='*120}")

    # 每场景结果
    for exp_name, results in all_results.items():
        print(f"\n  [{exp_name}]")
        ranked = sorted(results.items(), key=lambda x: x[1]['avg_recall'], reverse=True)
        for name, r in ranked:
            extra = ""
            if r.get("avg_em") is not None:
                extra = f"  em={r['avg_em']:.4f}  f1={r['avg_f1']:.4f}"
            print(
                f"    {name:<12} recall={r['avg_recall']:.4f}  prec={r['avg_precision']:.4f}  "
                f"mrr={r['avg_mrr']:.4f}  gold_kb={r['avg_gold_in_kb']:.4f}  "
                f"updates={r['total_updates']}{extra}"
            )

    # 跨场景平均表
    print(f"\n{'='*120}")
    print(f"  OVERALL AVERAGE ACROSS {len(exp_names)} SCENARIOS")
    print(f"{'='*120}")
    if has_gen:
        print(f"  {'Method':<12} {'Recall@10':>10} {'Prec@10':>10} {'MRR':>10} {'Gold_KB':>10} "
              f"{'EM':>8} {'F1':>8} {'AvgUpdates':>12} {'Wins':>6}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} "
              f"{'-'*8} {'-'*8} {'-'*12} {'-'*6}")
    else:
        print(f"  {'Method':<12} {'Recall@10':>10} {'Prec@10':>10} {'MRR':>10} {'Gold_KB':>10} "
              f"{'AvgUpdates':>12} {'Wins':>6}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} "
              f"{'-'*12} {'-'*6}")

    # 统计每个方法赢了几个场景 (按 recall)
    wins = {name: 0 for name in method_names}
    for results in all_results.values():
        ranked = sorted(results.items(), key=lambda x: x[1]['avg_recall'], reverse=True)
        if ranked:
            wins[ranked[0][0]] += 1

    for name in method_names:
        vals = []
        for exp_name in exp_names:
            if name in all_results.get(exp_name, {}):
                vals.append(all_results[exp_name][name])
        if not vals:
            continue
        avg_recall = np.mean([v['avg_recall'] for v in vals])
        avg_prec = np.mean([v['avg_precision'] for v in vals])
        avg_mrr = np.mean([v['avg_mrr'] for v in vals])
        avg_gold = np.mean([v['avg_gold_in_kb'] for v in vals])
        avg_updates = np.mean([v['total_updates'] for v in vals])
        tag = " ★" if wins[name] == len(exp_names) else ""
        if has_gen:
            em_vals = [v['avg_em'] for v in vals if v.get('avg_em') is not None]
            f1_vals = [v['avg_f1'] for v in vals if v.get('avg_f1') is not None]
            em_s = f"{np.mean(em_vals):>8.4f}" if em_vals else f"{'—':>8}"
            f1_s = f"{np.mean(f1_vals):>8.4f}" if f1_vals else f"{'—':>8}"
            print(f"  {name:<12} {avg_recall:>10.4f} {avg_prec:>10.4f} {avg_mrr:>10.4f} "
                  f"{avg_gold:>10.4f} {em_s} {f1_s} "
                  f"{avg_updates:>12.1f} {wins[name]:>5d}{tag}")
        else:
            print(f"  {name:<12} {avg_recall:>10.4f} {avg_prec:>10.4f} {avg_mrr:>10.4f} "
                  f"{avg_gold:>10.4f} "
                  f"{avg_updates:>12.1f} {wins[name]:>5d}{tag}")

    # 按平均 recall 排名
    method_avg_recall = {}
    for name in method_names:
        recalls = [all_results[e][name]['avg_recall']
                   for e in exp_names if name in all_results.get(e, {})]
        if recalls:
            method_avg_recall[name] = np.mean(recalls)
    ranked_overall = sorted(method_avg_recall.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Overall Ranking (avg Recall@10):")
    for rank, (name, score) in enumerate(ranked_overall, 1):
        tag = " ★" if rank == 1 else ""
        print(f"    #{rank} {name:<12} avg_recall={score:.4f}  wins={wins.get(name,0)}/{len(exp_names)}{tag}")

    # == 保存 Markdown 汇总 ==
    _save_cross_scenario_markdown(all_results, method_names, exp_names, wins, out_dir)


def _save_cross_scenario_markdown(all_results, method_names, exp_names, wins, out_dir):
    """保存跨场景汇总 Markdown"""
    lines = []
    lines.append("# Cross-Scenario Summary Report\n")
    from datetime import datetime
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Scenarios: {', '.join(exp_names)}\n")

    has_gen = any(
        r.get("avg_em") is not None
        for results in all_results.values()
        for r in results.values()
    )

    # 每场景详细表
    for exp_name, results in all_results.items():
        lines.append(f"## {exp_name}\n")
        pool_sz = next(iter(results.values()), {}).get("pool_size", "?")
        kb_sz = next(iter(results.values()), {}).get("kb_budget", "?")
        lines.append(f"Pool Size: {pool_sz} | KB Budget: {kb_sz}\n")
        if has_gen:
            lines.append("| Method | Recall@10 | Prec@10 | MRR | Gold_KB | EM | F1 | Updates |")
            lines.append("|--------|-----------|---------|-----|---------|------|------|---------|")
        else:
            lines.append("| Method | Recall@10 | Prec@10 | MRR | Gold_KB | Turnover | Updates |")
            lines.append("|--------|-----------|---------|-----|---------|----------|---------|")
        ranked = sorted(results.items(), key=lambda x: x[1]['avg_recall'], reverse=True)
        for name, r in ranked:
            tag = " ⭐" if name == ranked[0][0] else ""
            if has_gen:
                em_s = f"{r['avg_em']:.4f}" if r.get('avg_em') is not None else "—"
                f1_s = f"{r['avg_f1']:.4f}" if r.get('avg_f1') is not None else "—"
                lines.append(
                    f"| {name}{tag} | {r['avg_recall']:.4f} | {r['avg_precision']:.4f} | "
                    f"{r['avg_mrr']:.4f} | {r['avg_gold_in_kb']:.4f} | "
                    f"{em_s} | {f1_s} | {r['total_updates']} |"
                )
            else:
                lines.append(
                    f"| {name}{tag} | {r['avg_recall']:.4f} | {r['avg_precision']:.4f} | "
                    f"{r['avg_mrr']:.4f} | {r['avg_gold_in_kb']:.4f} | "
                    f"{r['kb_turnover']:.4f} | {r['total_updates']} |"
                )
        lines.append("")

    # 总体汇总表
    lines.append("## Overall Average\n")
    if has_gen:
        lines.append("| Method | Avg Recall | Avg Prec | Avg MRR | Avg Gold_KB | Avg EM | Avg F1 | Avg Updates | Wins |")
        lines.append("|--------|------------|----------|---------|-------------|--------|--------|-------------|------|")
    else:
        lines.append("| Method | Avg Recall | Avg Prec | Avg MRR | Avg Gold_KB | Avg Updates | Wins |")
        lines.append("|--------|------------|----------|---------|-------------|-------------|------|")

    method_avg_recall = {}
    for name in method_names:
        vals = [all_results[e][name] for e in exp_names if name in all_results.get(e, {})]
        if not vals:
            continue
        avg_recall = np.mean([v['avg_recall'] for v in vals])
        avg_prec = np.mean([v['avg_precision'] for v in vals])
        avg_mrr = np.mean([v['avg_mrr'] for v in vals])
        avg_gold = np.mean([v['avg_gold_in_kb'] for v in vals])
        avg_updates = np.mean([v['total_updates'] for v in vals])
        method_avg_recall[name] = avg_recall
        tag = " ⭐" if wins.get(name, 0) == len(exp_names) else ""
        if has_gen:
            em_vals = [v['avg_em'] for v in vals if v.get('avg_em') is not None]
            f1_vals = [v['avg_f1'] for v in vals if v.get('avg_f1') is not None]
            em_s = f"{np.mean(em_vals):.4f}" if em_vals else "—"
            f1_s = f"{np.mean(f1_vals):.4f}" if f1_vals else "—"
            lines.append(
                f"| {name}{tag} | {avg_recall:.4f} | {avg_prec:.4f} | {avg_mrr:.4f} | "
                f"{avg_gold:.4f} | {em_s} | {f1_s} | {avg_updates:.0f} | "
                f"{wins.get(name,0)}/{len(exp_names)} |"
            )
        else:
            lines.append(
                f"| {name}{tag} | {avg_recall:.4f} | {avg_prec:.4f} | {avg_mrr:.4f} | "
                f"{avg_gold:.4f} | {avg_updates:.0f} | "
                f"{wins.get(name,0)}/{len(exp_names)} |"
            )

    # 排名
    ranked_overall = sorted(method_avg_recall.items(), key=lambda x: x[1], reverse=True)
    lines.append("")
    lines.append("## Overall Ranking\n")
    for rank, (name, score) in enumerate(ranked_overall, 1):
        tag = " ⭐" if rank == 1 else ""
        lines.append(f"{rank}. **{name}** — avg_recall={score:.4f}, wins={wins.get(name,0)}/{len(exp_names)}{tag}")

    lines.append("")
    import os
    md_path = os.path.join(out_dir, "cross_scenario_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Cross-scenario summary saved: {md_path}")


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
    _print_cross_scenario_summary(all_results, out_dir)


if __name__ == "__main__":
    main()
