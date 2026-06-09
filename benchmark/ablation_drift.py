"""
消融实验: QARC_Full vs QARC_NoDrift vs QARC_NoGap
验证 DriftLens 对齐特征是否有实际贡献
"""
import sys, numpy as np, time
sys.path.insert(0, "/data/jyliu/RAG-project")

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from benchmark.data import build_gradual_drift, build_sudden_shift, build_cyclic_return
from benchmark.rag_pipeline import RAGPipeline
from benchmark.adapters import QARCStrategyAdapter
from algorithms.qarc.detection.drift_detector import DriftResult
from algorithms.qarc.curation.interest_model import AlignmentGapResult
from algorithms.qarc.decision import kb_agent as _kb_agent_module
from core.evaluator import recall_at_k

# ─── 全局决策日志 ───
_decision_log = []
_orig_decide = _kb_agent_module.KBUpdateAgent.decide

def _patched_decide(self, drift_result, gap_result):
    decision = _orig_decide(self, drift_result, gap_result)
    _decision_log.append({
        "window": self._window_count,
        "fid": drift_result.fid_score,
        "threshold": drift_result.threshold,
        "is_drifted": drift_result.is_drifted,
        "gap": gap_result.gap,
        "gap_threshold": self.gap_threshold,
        "action": decision.action.value,
        "reason": decision.reason,
    })
    return decision

_kb_agent_module.KBUpdateAgent.decide = _patched_decide


def run_scenario(name, build_fn, pipeline, n_queries=300, pool_size=3000, kb_budget=150):
    print(f"\n{'='*70}")
    print(f"  场景: {name}  queries={n_queries}  pool={pool_size}  kb={kb_budget}")
    print(f"{'='*70}")

    dataset = build_fn(total_queries=n_queries, pool_size=pool_size)
    doc_pool = [
        {"doc_id": d.doc_id, "text": d.text, "topic": d.topic}
        for d in dataset.document_pool
    ]
    queries = [
        {"query": q.question, "gold_doc_ids": q.gold_doc_ids}
        for q in dataset.query_stream
    ]

    print(f"  编码 {len(queries)} 条 query + {len(doc_pool)} 条文档...")
    query_embeddings = pipeline.encode_queries([q["query"] for q in queries], batch_size=64)
    doc_embeddings   = pipeline.encode_queries([d["text"]  for d in doc_pool],  batch_size=64)

    results = {}

    for variant, drift_off, gap_off in [
        ("QARC_Full",    False, False),
        ("QARC_NoDrift", True,  False),
        ("QARC_NoGap",   False, True),
    ]:
        _decision_log.clear()

        adapter = QARCStrategyAdapter(kb_budget=kb_budget, window_size=8)
        adapter.initialize(doc_pool, doc_embeddings, kb_budget)

        # 注入 stub
        if drift_off:
            _orig_det = adapter._pipeline.detector.detect
            def _stub_det(X, _o=_orig_det):
                r = _o(X)
                return DriftResult(fid_score=r.fid_score, threshold=r.threshold,
                                   is_drifted=False)
            adapter._pipeline.detector.detect = _stub_det

        _im = None
        _orig_gap = None
        if gap_off:
            import algorithms.qarc.curation.interest_model as _im_mod
            _im = _im_mod
            _orig_gap = _im_mod.compute_alignment_gap
            def _stub_gap(*a, **kw):
                r = _orig_gap(*a, **kw)
                return AlignmentGapResult(gap=0.0, avg_max_sim=r.avg_max_sim,
                                          baseline_avg_sim=r.baseline_avg_sim,
                                          n_queries=r.n_queries)
            _im_mod.compute_alignment_gap = _stub_gap

        recalls = []
        for step, q in enumerate(queries):
            proc = adapter.process_query(
                query_text=q["query"],
                query_embedding=query_embeddings[step],
                step=step,
                gold_doc_ids=q["gold_doc_ids"],
            )
            r = recall_at_k(proc.retrieved_doc_ids, q["gold_doc_ids"])
            recalls.append(r)

        if gap_off and _im is not None:
            _im.compute_alignment_gap = _orig_gap

        avg_r  = np.mean(recalls)
        n_upd  = sum(1 for d in _decision_log if d["action"] != "no_op")
        n_drift= sum(1 for d in _decision_log if "FID" in d["reason"] or ("漂移" in d["reason"] and "Warmup" not in d["reason"] and "重校准" not in d["reason"]))
        n_gap  = sum(1 for d in _decision_log if "Gap 偏高" in d["reason"])
        n_warm = sum(1 for d in _decision_log if "Warmup" in d["reason"])
        n_recal= sum(1 for d in _decision_log if "重校准" in d["reason"] and "Warmup" not in d["reason"])
        n_cool = sum(1 for d in _decision_log if "冷却" in d["reason"])
        n_norm = sum(1 for d in _decision_log if "正常" in d["reason"])
        n_det_ready = sum(1 for d in _decision_log if d["threshold"] > 0)

        print(f"\n  [{variant:<16}] Recall={avg_r:.4f}  Updates={n_upd}")
        print(f"    触发: Warmup={n_warm}  Drift={n_drift}  Gap={n_gap}  Recalib={n_recal}")
        print(f"    抑制: Cooldown={n_cool}  Normal={n_norm}")
        print(f"    Detector已就绪窗口数: {n_det_ready}/{len(_decision_log)}")

        results[variant] = {"recall": avg_r, "updates": n_upd,
                            "drift": n_drift, "gap": n_gap, "warm": n_warm}

    print(f"\n  {'变体':<18} {'Recall':>8} {'Updates':>8} {'Drift触发':>10} {'Gap触发':>8}")
    print(f"  {'-'*56}")
    best_r = max(r["recall"] for r in results.values())
    for v, r in results.items():
        m = " ★" if r["recall"] == best_r else ""
        print(f"  {v:<18} {r['recall']:>8.4f} {r['updates']:>8} {r['drift']:>10} {r['gap']:>8}{m}")

    return results


if __name__ == "__main__":
    pipeline = RAGPipeline(device="cuda", use_generator=False)

    all_res = {}
    for name, fn in [
        ("gradual_drift", build_gradual_drift),
        ("sudden_shift",  build_sudden_shift),
        ("cyclic_return",  build_cyclic_return),
    ]:
        all_res[name] = run_scenario(name, fn, pipeline,
                                     n_queries=300, pool_size=3000, kb_budget=150)

    print("\n\n" + "="*70)
    print("  消融总结 — Recall 差值 (Full - NoDrift / Full - NoGap)")
    print("="*70)
    for name, res in all_res.items():
        full   = res["QARC_Full"]["recall"]
        nodrift= res["QARC_NoDrift"]["recall"]
        nogap  = res["QARC_NoGap"]["recall"]
        dt     = res["QARC_Full"]["drift"]
        print(f"  {name:<18} Drift贡献={full-nodrift:+.4f}  Gap贡献={full-nogap:+.4f}  Drift触发次数={dt}")
    print()
