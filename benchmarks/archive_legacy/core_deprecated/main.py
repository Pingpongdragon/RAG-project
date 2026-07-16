"""
主实验脚本:运行三个策略并对比
"""
from kb_base import ClusteredKnowledgeBase, load_kb_documents
from evaluator import load_test_data, compute_retrieval_score
from RAG_project.updator.static_updater import StaticUpdater
from RAG_project.updator.reactive_updater import ReactiveUpdater
from RAG_project.updator.cluster_updater import ClusteredAdaptiveUpdater
from RAG_project.core.detector.detector import AutoAdaptiveDetector


def run_experiment(updater, detector, queries, name: str):
    """运行实验（批量生成embedding版）"""
    from RAG_project.models.embeddings import embedding_service
    
    # 🔥 批量生成所有查询的 embedding
    print(f"   🔧 批量生成 {len(queries)} 个查询的 embedding...")
    query_texts = [q["query"] for q in queries]
    query_embeddings = embedding_service.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print(f"   ✅ Embedding 生成完成")
    
    metrics = {
        "total_updates": 0,
        "total_cost": 0,
        "retrieval_scores": [],
        "alert_count": 0
    }
    
    for i, query in enumerate(queries):
        query_vec = query_embeddings[i]
        
        # 直接调用 KB 的 search
        retrieved_docs = updater.kb.search(query_vec, query["domain"], step=i, top_k=10)
        
        # 计算召回率
        gold_doc_ids = query.get("gold_doc_ids", [])
        if gold_doc_ids and retrieved_docs:
            retrieved_ids = set(doc.doc_id for doc in retrieved_docs)
            gold_ids = set(gold_doc_ids)
            matched_count = len(retrieved_ids & gold_ids)
            score = matched_count / len(gold_ids)
        else:
            score = 0.0
        
        metrics["retrieval_scores"].append(score)
        
        # 检测
        detection_result = detector.detect(query["domain"], score, i)
        
        if detection_result.is_global_shift or detection_result.is_intra_degradation:
            metrics["alert_count"] += 1
        
        # 更新
        update_result = updater.update(detection_result, i)
        
        if update_result.get("action") not in ["no_update", None]:
            metrics["total_updates"] += 1
            metrics["total_cost"] += update_result.get("removed", 0) + update_result.get("added", 0)
            
            # ✅ 关键：更新后通知 detector 同步 KB 分布
            kb_stats = updater.kb.get_statistics()
            new_kb_dist = kb_stats["distribution"]
            detector.update_kb_distribution(new_kb_dist)
    
    avg_score = sum(metrics["retrieval_scores"]) / len(metrics["retrieval_scores"])
    
    print(f"\n{'='*50}")
    print(f"📊 {name} 结果:")
    print(f"{'='*50}")
    print(f"  平均检索得分: {avg_score:.3f}")
    print(f"  总更新次数: {metrics['total_updates']}")
    print(f"  总成本: {metrics['total_cost']}")
    print(f"  告警次数: {metrics['alert_count']}")
    
    return metrics


def init_kb(doc_pool, capacity=10000):
    """初始化 KB"""
    kb = ClusteredKnowledgeBase(capacity=capacity)
    
    for domain, docs in doc_pool.items():
        for doc in docs[:capacity//4]:
            kb.add_document(doc, step=0)
    
    return kb


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 KB Updater 实验开始 (真实评分版)")
    print("="*70)
    
    # 加载数据
    print("\n📚 加载数据...")
    try:
        doc_pool = load_kb_documents()
        print(f"   ✅ 文档池: {sum(len(docs) for docs in doc_pool.values())} 条")
    except FileNotFoundError as e:
        print(f"   ❌ 错误: {e}")
        exit(1)
    
    # 加载不同类型的domain shift数据集进行测试
    shift_types = ["sudden", "gradual", "recurring"]  # 可以添加 "recurring" 如果文件存在
    
    for shift_type in shift_types:
        print(f"\n{'='*70}")
        print(f"📊 测试 {shift_type.upper()} Domain Shift")
        print(f"{'='*70}")
        
        try:
            queries = load_test_data(shift_type=shift_type)
            print(f"   ✅ 查询数据: {len(queries)} 条")
        except FileNotFoundError as e:
            print(f"   ❌ 错误: {e}")
            continue
        
        # 初始化三个策略
        kb1 = init_kb(doc_pool)
        kb2 = init_kb(doc_pool)
        kb3 = init_kb(doc_pool)
        
        detector1 = AutoAdaptiveDetector()
        detector2 = AutoAdaptiveDetector()
        detector3 = AutoAdaptiveDetector()
        
        static_updater = StaticUpdater(kb1, doc_pool)
        reactive_updater = ReactiveUpdater(kb2, doc_pool)
        adaptive_updater = ClusteredAdaptiveUpdater(kb3, doc_pool)
        
        # 运行实验
        print(f"\n🔬 运行实验 ({shift_type})...")
        
        results = {}
        results["Static"] = run_experiment(static_updater, detector1, queries, f"Static ({shift_type})")
        results["Reactive"] = run_experiment(reactive_updater, detector2, queries, f"Reactive ({shift_type})")
        results["Adaptive"] = run_experiment(adaptive_updater, detector3, queries, f"Adaptive ({shift_type})")
        
        # 对比结果
        print(f"\n{'='*70}")
        print(f"📈 {shift_type.upper()} 对比总结")
        print(f"{'='*70}")
        for name, metrics in results.items():
            avg_score = sum(metrics["retrieval_scores"]) / len(metrics["retrieval_scores"])
            print(f"{name:12s} | 得分: {avg_score:.3f} | 更新: {metrics['total_updates']:3d} | 成本: {metrics['total_cost']:6d}")
    
    print("\n" + "="*70)
    print("✅ 实验完成!")
    print("="*70)