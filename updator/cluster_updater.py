"""
Clustered-Adaptive Updater: 基于聚类 + 热度的自适应更新器
"""
import random
from typing import Dict, List
from kb_base import ClusteredKnowledgeBase, KBDocument


class ClusteredAdaptiveUpdater:
    """基于聚类 + 热度的自适应更新器"""
    def __init__(self, kb: ClusteredKnowledgeBase, domain_pool: Dict[str, List[KBDocument]],
                 global_ratio: float = 0.5, intra_ratio: float = 0.25, 
                 cold_threshold: float = 2.0):
        """
        Args:
            kb: 知识库实例
            domain_pool: 各domain的文档池
            global_ratio: 全局更新时每个domain添加的比例
            intra_ratio: 域内更新时添加的比例
            cold_threshold: 冷簇判定阈值（虚拟热度 < threshold 视为冷簇）
        """
        self.kb = kb
        self.domain_pool = domain_pool
        self.global_ratio = global_ratio
        self.intra_ratio = intra_ratio
        self.cold_threshold = cold_threshold
        self.update_count = 0
        self.total_cost = 0
    
    def should_update(self, detection_result) -> bool:
        """判断是否需要更新"""
        return detection_result.is_global_shift or detection_result.is_intra_degradation
    
    def update(self, detection_result, step: int) -> Dict:
        """统一更新入口"""
        if not self.should_update(detection_result):
            return {"action": "no_update"}
        
        if detection_result.is_global_shift:
            return self._global_update(detection_result, step)
        elif detection_result.is_intra_degradation:
            return self._intra_update(detection_result, step)
    
    def _global_update(self, detection_result, step: int) -> Dict:
        """
        全局更新策略（基于热度）：
        1. 先扩容高需求domain
        2. 再缩容低需求domain
        3. 最后淘汰所有domain的冷簇
        """
        target_dist = detection_result.query_dist
        total_capacity = self.kb.capacity
        added_total = 0
        removed_total = 0
        
        expand_domains = []
        shrink_domains = []
        
        # 计算每个domain的目标容量
        for domain in self.kb.domains:
            target_ratio = target_dist.get(domain, 0)
            new_capacity = int(total_capacity * target_ratio)
            bucket = self.kb.buckets[domain]
            
            if new_capacity > bucket.capacity:
                expand_domains.append((domain, bucket, new_capacity))
            elif new_capacity < bucket.capacity:
                shrink_domains.append((domain, bucket, new_capacity))
        
        # Phase 1: 扩容（为高需求domain添加新文档）
        for domain, bucket, new_capacity in expand_domains:
            bucket.capacity = new_capacity
            available_space = new_capacity - bucket.doc_count
            
            # ✅ 修复1：确保 num_to_add > 0
            if available_space <= 0:
                continue
            
            num_to_add = min(available_space, int(new_capacity * self.global_ratio))
            
            # ✅ 修复2：确保 num_to_add > 0
            if num_to_add <= 0:
                continue
            
            available_docs = self.domain_pool.get(domain, [])
            if not available_docs:
                continue
            
            # ✅ 修复3：防止采样数量超过可用文档数
            num_to_sample = min(num_to_add, len(available_docs))
            if num_to_sample <= 0:
                continue
            
            sampled = random.sample(available_docs, num_to_sample)
            for doc in sampled:
                if bucket.add_document(doc, step):
                    added_total += 1
        
        # Phase 2: 缩容（低需求domain自动触发淘汰）
        for domain, bucket, new_capacity in shrink_domains:
            old_capacity = bucket.capacity
            bucket.capacity = new_capacity
            
            # ✅ 修复4：只有在超出新容量时才淘汰
            if bucket.doc_count <= new_capacity:
                continue
            
            # 缩容时，超出容量部分会自动触发 _evict_coldest_cluster
            eviction_attempts = 0
            max_attempts = 10  # ✅ 修复5：防止无限循环
            
            while bucket.doc_count > new_capacity and len(bucket.clusters) > 1 and eviction_attempts < max_attempts:
                before_count = bucket.doc_count
                bucket._evict_coldest_cluster(current_step=step)
                after_count = bucket.doc_count
                
                removed_total += (before_count - after_count)
                eviction_attempts += 1
                
                # ✅ 修复6：如果没有移除任何文档，说明无法继续淘汰
                if before_count == after_count:
                    break
        
        # Phase 3: 主动淘汰冷簇（所有domain）
        for domain, bucket in self.kb.buckets.items():
            # ✅ 修复7：只有在有多余空间时才淘汰冷簇
            if bucket.doc_count >= bucket.capacity:
                continue
            
            cold_clusters = self._find_cold_clusters(bucket, step)
            
            # 每次最多淘汰2个冷簇（避免过度淘汰）
            if cold_clusters and len(bucket.clusters) > 2:
                evicted_count = 0
                for cold_cluster in cold_clusters[:2]:
                    if cold_cluster in bucket.clusters:
                        # ✅ 修复8：确保簇确实存在于列表中
                        try:
                            cluster_size = len(cold_cluster.docs)
                            bucket.clusters.remove(cold_cluster)
                            bucket.doc_count -= cluster_size
                            removed_total += cluster_size
                            evicted_count += 1
                        except ValueError:
                            # 簇可能已被其他操作移除
                            continue
        
        self.update_count += 1
        self.total_cost += added_total + removed_total
        
        return {
            "action": "global_update",
            "added": added_total,
            "removed": removed_total,
            "new_distribution": self.kb.get_distribution(),
            "step": step
        }
    
    def _intra_update(self, detection_result, step: int) -> Dict:
        """
        域内更新策略（基于热度）：
        1. 添加新文档
        2. 淘汰该domain的冷簇
        """
        degraded_domain = detection_result.degraded_domain
        bucket = self.kb.buckets.get(degraded_domain)
        
        if not bucket:
            return {"action": "no_update", "reason": "domain_not_found"}
        
        available_docs = self.domain_pool.get(degraded_domain, [])
        if not available_docs:
            return {"action": "no_update", "reason": "no_docs_available"}
        
        # 计算可添加的文档数
        num_to_add = min(
            int(bucket.capacity * self.intra_ratio),
            bucket.capacity - bucket.doc_count
        )
        
        if num_to_add <= 0:
            # 容量满时，先淘汰冷簇腾出空间
            cold_clusters = self._find_cold_clusters(bucket, step)
            if cold_clusters and len(bucket.clusters) > 1:
                cold_cluster = cold_clusters[0]
                cluster_size = len(cold_cluster.docs)
                bucket.clusters.remove(cold_cluster)
                bucket.doc_count -= cluster_size
                num_to_add = min(cluster_size, int(bucket.capacity * self.intra_ratio))
            else:
                return {"action": "no_update", "reason": "no_space"}
        
        # 添加新文档
        sampled = random.sample(available_docs, min(num_to_add, len(available_docs)))
        added = 0
        for doc in sampled:
            if bucket.add_document(doc, step):
                added += 1
        
        # 淘汰冷簇（如果有）
        removed = 0
        cold_clusters = self._find_cold_clusters(bucket, step)
        if cold_clusters and len(bucket.clusters) > 2:
            cold_cluster = cold_clusters[0]
            cluster_size = len(cold_cluster.docs)
            bucket.clusters.remove(cold_cluster)
            bucket.doc_count -= cluster_size
            removed = cluster_size
        
        self.update_count += 1
        self.total_cost += added + removed
        
        return {
            "action": "intra_update",
            "domain": degraded_domain,
            "added": added,
            "removed": removed,
            "new_clusters": len(bucket.clusters),
            "step": step
        }
    
    def _find_cold_clusters(self, bucket, current_step: int) -> List:
        """
        基于虚拟热度找出冷簇
        
        虚拟热度 = heat * (decay ** steps_since_last_access)
        """
        def get_virtual_heat(cluster):
            if cluster.last_access_step >= 0:
                steps_passed = current_step - cluster.last_access_step
                return cluster.heat * (0.95 ** steps_passed)
            return cluster.heat
        
        # 找出热度低于阈值的簇，并按热度排序
        cold_clusters = [
            c for c in bucket.clusters 
            if get_virtual_heat(c) < self.cold_threshold
        ]
        
        # 按虚拟热度升序排序（最冷的在前）
        cold_clusters.sort(key=get_virtual_heat)
        
        return cold_clusters
    
    def get_statistics(self) -> Dict:
        """获取更新器统计信息"""
        kb_total = self.kb.get_statistics()  # 返回总文档数
        
        # 手动收集详细统计
        bucket_stats = {}
        for domain, bucket in self.kb.buckets.items():
            bucket_stats[domain] = {
                "doc_count": bucket.doc_count,
                "cluster_count": len(bucket.clusters),
                "avg_cluster_size": bucket.doc_count / len(bucket.clusters) if bucket.clusters else 0,
                "avg_heat": sum(c.heat for c in bucket.clusters) / len(bucket.clusters) if bucket.clusters else 0
            }
        
        return {
            "update_count": self.update_count,
            "total_cost": self.total_cost,
            "kb_size": kb_total,
            "kb_distribution": self.kb.get_distribution(),
            "bucket_stats": bucket_stats
        }