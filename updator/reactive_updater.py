"""
Reactive Updater: 全量重建
"""
import random
from typing import Dict, List
from kb_base import ClusteredKnowledgeBase, KBDocument


class ReactiveUpdater:
    def __init__(self, kb: ClusteredKnowledgeBase, domain_pool: Dict[str, List[KBDocument]]):
        self.kb = kb
        self.domain_pool = domain_pool
        self.update_count = 0
        self.total_cost = 0
    
    def should_update(self, detection_result) -> bool:
        return detection_result.is_global_shift
    
    def update(self, detection_result, step: int) -> Dict:
        if not self.should_update(detection_result):
            return {"action": "no_update"}
        
        target_dist = detection_result.query_dist
        removed = sum(b.doc_count for b in self.kb.buckets.values())
        
        # 清空所有域
        for domain in self.kb.domains:
            self.kb.clear_domain(domain)
        
        # 按新分布重建
        added = 0
        for domain, ratio in target_dist.items():
            num_docs = int(self.kb.capacity * ratio)
            available = self.domain_pool.get(domain, [])
            
            if available:
                sampled = random.sample(available, min(num_docs, len(available)))
                for doc in sampled:
                    if self.kb.add_document(doc, step):
                        added += 1
        
        self.update_count += 1
        self.total_cost += removed + added
        
        return {"action": "full_replace", "removed": removed, "added": added}
    
    def get_statistics(self) -> Dict:
        kb_stats = self.kb.get_statistics()
        return {
            "update_count": self.update_count,
            "total_cost": self.total_cost,
            "kb_size": kb_stats["total_docs"],
            "kb_distribution": kb_stats["distribution"],
            "cluster_stats": kb_stats["buckets"]
        }