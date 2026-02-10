import random
from typing import Dict, List
from kb_base import ClusteredKnowledgeBase, KBDocument

class ComRAGUpdater:
    """
    ComRAG Updater: 增量式动态对齐 (Incremental Dynamic Alignment)
    Paper: ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores (ArXiv:2506.21098)
    """
    def __init__(self, kb: ClusteredKnowledgeBase, domain_pool: Dict[str, List[KBDocument]]):
        self.kb = kb
        self.domain_pool = domain_pool
        self.update_count = 0
        self.total_cost = 0
        self.retention_rate = 0.8  # 模拟记忆保留机制，避免剧烈抖动

    def should_update(self, detection_result) -> bool:
        # ComRAG 不仅响应全局漂移，也敏捷响应局部热点变化
        # 这里只要检测到分布不匹配就需要微调
        return detection_result.is_global_shift or detection_result.drift_score > 0.1

    def update(self, detection_result, step: int) -> Dict:
        """
        根据 Query 分布计算 Delta，仅增删必要的文档，而非全量替换。
        """
        if not self.should_update(detection_result):
            return {"action": "no_update"}

        target_dist = detection_result.query_dist
        current_stats = self.kb.get_statistics()
        current_buckets = self.kb.buckets  # 假设这是一个 {domain: [docs...]} 的字典
        
        removed_count = 0
        added_count = 0
        
        # 1. 计算每个 Domain 的目标容量
        # ComRAG 的核心是让存储结构(Vector Store)动态适应 Query 分布
        target_counts = {
            domain: int(self.kb.capacity * ratio) 
            for domain, ratio in target_dist.items()
        }
        
        # 2. 增量调整 (Incremental Adjustment)
        all_domains = set(self.kb.domains) | set(target_dist.keys())
        
        for domain in all_domains:
            current_docs = current_buckets.get(domain, [])
            current_len = len(current_docs)
            target_len = target_counts.get(domain, 0)
            
            delta = target_len - current_len
            
            if delta > 0:
                # [Injection] 需要补充文档：从 Pool 中采样新知识
                available = self.domain_pool.get(domain, [])
                # 找出不在库里的文档（去重）
                current_ids = {d.id for d in current_docs}
                candidates = [d for d in available if d.id not in current_ids]
                
                num_to_add = min(delta, len(candidates))
                if num_to_add > 0:
                    # 优先选择该 Domain 中最新的或最相关的文档(模拟Centroid-based selection)
                    # 这里简化为随机采样
                    sampled = random.sample(candidates, num_to_add)
                    for doc in sampled:
                        if self.kb.add_document(doc, step):
                            added_count += 1
                            
            elif delta < 0:
                # [Eviction] 需要缩减文档：移除该 Domain 中最旧或最不相关的
                num_to_remove = abs(delta)
                # 假设 bucket 是列表，移除头部(最早加入的)或随机移除
                # 模拟遗忘机制
                docs_to_remove = current_docs[:num_to_remove]
                for doc in docs_to_remove:
                    # 假设 kb 有 remove_document 方法，或者直接操作 bucket
                    # 这里模拟移除操作
                    if hasattr(self.kb, 'remove_document'):
                        self.kb.remove_document(doc)
                    else:
                        # Fallback: 如果没有 remove 方法，手动从 bucket 移除
                        # 注意：实际工程中需要处理索引删除
                        current_buckets[domain].remove(doc)
                        self.kb.total_docs -= 1
                    removed_count += 1

        self.update_count += 1
        # ComRAG 的 Cost 远低于 Reactive，因为只操作 Delta
        current_cost = removed_count + added_count
        self.total_cost += current_cost
        
        return {
            "action": "incremental_align", 
            "removed": removed_count, 
            "added": added_count,
            "cost_saving": "High"  # 相比 Reactive 的全量替换
        }

    def get_statistics(self) -> Dict:
        kb_stats = self.kb.get_statistics()
        return {
            "update_count": self.update_count,
            "total_cost": self.total_cost,
            "kb_size": kb_stats["total_docs"],
            "kb_distribution": kb_stats["distribution"],
            "strategy": "ComRAG (Dynamic Incremental)"
        }