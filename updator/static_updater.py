"""
Static Updater: 从不更新
"""
from typing import Dict, List
from kb_base import ClusteredKnowledgeBase, KBDocument


class StaticUpdater:
    def __init__(self, kb: ClusteredKnowledgeBase, domain_pool: Dict[str, List[KBDocument]]):
        self.kb = kb
        self.domain_pool = domain_pool
        self.update_count = 0
        self.total_cost = 0
    
    def should_update(self, detection_result) -> bool:
        return False
    
    def update(self, detection_result, step: int) -> Dict:
        return {"action": "no_update"}
    
    def get_statistics(self) -> Dict:
        kb_stats = self.kb.get_statistics()
        return {
            "update_count": self.update_count,
            "total_cost": self.total_cost,
            "kb_size": kb_stats["total_docs"],
            "kb_distribution": kb_stats["distribution"],
            "cluster_stats": kb_stats["buckets"]
        }