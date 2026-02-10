"""
KB 基础数据结构 - 简化版
"""
import numpy as np
from typing import Dict, List
from pathlib import Path
import json
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from RAG_project.models.embeddings import embedding_service



class KBDocument:
    """知识库文档"""
    def __init__(self, doc_id: str, domain: str, content: str, title: str = "", embedding: np.ndarray = None):
        self.doc_id = doc_id
        self.domain = domain
        self.content = content
        self.title = title
        self.embedding = embedding
        self.access_count = 0
        self.last_access_step = -1


class TopicCluster:
    """域内的子话题簇"""
    def __init__(self, cluster_id: str, centroid: np.ndarray, domain: str):
        self.cluster_id = cluster_id
        self.centroid = centroid
        self.domain = domain
        self.docs: List[KBDocument] = []
        self.heat = 1.0
        self.last_access_step = -1
        self.creation_step = -1
        self.size = 0
    
    def add_doc(self, doc: KBDocument):
        """添加文档并更新质心"""
        self.docs.append(doc)
        self.size += 1
        
        # 重新计算质心（所有文档embedding的平均值）
        embeddings = np.array([d.embedding for d in self.docs])
        self.centroid = np.mean(embeddings, axis=0)
        
        # 归一化质心向量（用于余弦相似度计算）
        self.centroid = self.centroid / (np.linalg.norm(self.centroid) + 1e-8)
    
    def update_heat(self, step: int, decay: float = 0.95):
        """更新热度"""
        if self.last_access_step >= 0:
            steps_passed = step - self.last_access_step
            self.heat = self.heat * (decay ** steps_passed) + 1.0
        else:
            self.heat += 1.0
        self.last_access_step = step
    
    def compute_similarity(self, query_vec: np.ndarray) -> float:
        """计算相似度"""
        dot = np.dot(self.centroid, query_vec)
        norm1 = np.linalg.norm(self.centroid)
        norm2 = np.linalg.norm(query_vec)
        return dot / (norm1 * norm2 + 1e-8)


class DomainBucket:
    """单个 Domain 的存储桶"""
    def __init__(self, domain_name: str, capacity: int = 2000, 
                 similarity_threshold: float = 0.45, max_clusters: int = 12):
        self.domain_name = domain_name
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.max_clusters = max_clusters
        
        self.clusters: List[TopicCluster] = []
        self.doc_count = 0
    
    def add_document(self, doc: KBDocument, step: int) -> bool:
        """添加文档"""
        if doc.domain != self.domain_name:
            return False
        
        best_cluster = None
        max_sim = -1.0
        
        for cluster in self.clusters:
            sim = cluster.compute_similarity(doc.embedding)
            if sim > max_sim:
                max_sim = sim
                best_cluster = cluster
        
        if best_cluster and max_sim >= self.similarity_threshold:
            best_cluster.add_doc(doc)
            best_cluster.update_heat(step)
        else:
            if len(self.clusters) >= self.max_clusters:
                if best_cluster:
                    best_cluster.add_doc(doc)
                    best_cluster.update_heat(step)
                else:
                    return False
            else:
                new_cluster = TopicCluster(
                    cluster_id=f"{self.domain_name}_c{len(self.clusters)}",
                    centroid=doc.embedding.copy(),
                    domain=self.domain_name
                )
                new_cluster.add_doc(doc)
                new_cluster.update_heat(step)
                new_cluster.creation_step = step
                self.clusters.append(new_cluster)
        
        self.doc_count += 1
        
        while self.doc_count > self.capacity and len(self.clusters) > 1:
            self._evict_coldest_cluster(current_step=step)
        
        return True
    
    def _evict_coldest_cluster(self, current_step: int = 0):
        """简化版淘汰策略：只基于热度"""
        if not self.clusters:
            return
        
        # 计算每个簇的虚拟热度（考虑衰减）
        def get_virtual_heat(cluster):
            if cluster.last_access_step >= 0:
                steps_passed = current_step - cluster.last_access_step
                return cluster.heat * (0.95 ** steps_passed)
            return cluster.heat
        
        # 找到热度最低的簇
        target_cluster = min(self.clusters, key=get_virtual_heat)
        cluster_size = len(target_cluster.docs)
        
        # 简单策略：直接移除整个冷簇
        if cluster_size <= 50:  # 小簇直接删除
            removed_count = cluster_size
            self.clusters.remove(target_cluster)
        else:  # 大簇移除部分文档
            num_to_remove = max(1, int(cluster_size * 0.3))  # 移除30%
            
            # 按文档的访问时间排序，移除最旧的
            target_cluster.docs.sort(key=lambda d: (d.last_access_step, d.access_count))
            target_cluster.docs = target_cluster.docs[num_to_remove:]
            
            # 重新计算质心
            if target_cluster.docs:
                embeddings = np.array([d.embedding for d in target_cluster.docs])
                target_cluster.centroid = np.mean(embeddings, axis=0)
                target_cluster.centroid = target_cluster.centroid / (np.linalg.norm(target_cluster.centroid) + 1e-8)
            
            removed_count = num_to_remove
        
        self.doc_count -= removed_count
    
    def search(self, query_vec: np.ndarray, step: int, top_k: int = 5) -> List[KBDocument]:
        """检索文档 - 使用 retriever 进行精确检索"""
        if not self.clusters:
            return []
        
        # 1. 找到最相似的簇
        best_cluster = max(self.clusters, key=lambda c: c.compute_similarity(query_vec))
        best_cluster.update_heat(step)
        
        # 2. 簇内按相似度排序（保留原来的简单实现）
        doc_scores = []
        for doc in best_cluster.docs:
            sim = np.dot(query_vec, doc.embedding) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc.embedding) + 1e-8
            )
            doc_scores.append((doc, sim))
        
        # 3. 按相似度降序排序并取 top_k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 4. 更新访问统计
        results = []
        for doc, score in doc_scores[:top_k]:
            doc.access_count += 1
            doc.last_access_step = step
            results.append(doc)
        
        return results
    
    def get_all_docs(self) -> List[KBDocument]:
        """获取所有文档"""
        all_docs = []
        for cluster in self.clusters:
            all_docs.extend(cluster.docs)
        return all_docs
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "domain": self.domain_name,
            "doc_count": self.doc_count,
            "cluster_count": len(self.clusters),
            "avg_cluster_size": self.doc_count / len(self.clusters) if self.clusters else 0,
            "cluster_heats": [c.heat for c in self.clusters],
            
        }


class ClusteredKnowledgeBase:
    """基于聚类的知识库 - 简化版"""
    def __init__(self, capacity: int = 8000, encoder=None):
        self.capacity = capacity
        self.domains = ["0_entertainment", "1_stem", "2_humanities", "3_lifestyle"]
        self.encoder = encoder  # 外部传入的encoder
        
        per_domain_capacity = capacity // len(self.domains)
        
        self.buckets: Dict[str, DomainBucket] = {
            domain: DomainBucket(domain, per_domain_capacity)
            for domain in self.domains
        }
    
    def add_document(self, doc: KBDocument, step: int) -> bool:
        """添加文档到KB"""
        if doc.domain not in self.buckets:
            return False
        return self.buckets[doc.domain].add_document(doc, step)
    
    def search(self, query_vec: np.ndarray, query_domain: str, step: int, top_k: int = 5) -> List[KBDocument]:
        """向量检索"""
        if query_domain not in self.buckets:
            return []
        return self.buckets[query_domain].search(query_vec, step, top_k)
    
    def get_distribution(self) -> Dict[str, float]:
        """获取domain分布"""
        total_docs = sum(bucket.doc_count for bucket in self.buckets.values())
        if total_docs == 0:
            return {d: 0.0 for d in self.domains}
        return {
            domain: bucket.doc_count / total_docs
            for domain, bucket in self.buckets.items()
        }
    
    def get_statistics(self) -> Dict:
        """✅ 修复：返回完整的统计信息字典"""
        total_docs = sum(b.doc_count for b in self.buckets.values())
        
        return {
            "total_docs": total_docs,
            "capacity": self.capacity,
            "distribution": self.get_distribution(),  # ✅ 调用上面的方法
            "buckets": {
                domain: {
                    "doc_count": bucket.doc_count,
                    "capacity": bucket.capacity,
                    "cluster_count": len(bucket.clusters),
                    "utilization": bucket.doc_count / bucket.capacity if bucket.capacity > 0 else 0.0
                }
                for domain, bucket in self.buckets.items()
            }
        }
    
    def clear_domain(self, domain: str):
        """清空某个domain"""
        if domain in self.buckets:
            self.buckets[domain] = DomainBucket(domain, self.buckets[domain].capacity)


def load_kb_documents() -> Dict[str, List[KBDocument]]:
    """从 hotpot_kb 加载文档池"""
    HERE = Path(__file__).parent
    KB_DIR = HERE / "dataset_split_domain" / "hotpot_kb"
    
    if not KB_DIR.exists():
        raise FileNotFoundError(f"❌ KB 目录不存在: {KB_DIR}")
    
    pool = {}
    domains = ["0_entertainment", "1_stem", "2_humanities", "3_lifestyle"]
    
    for domain in domains:
        kb_file = KB_DIR / f"{domain}.jsonl"
        if not kb_file.exists():
            continue
        
        docs = []
        with open(kb_file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                doc = KBDocument(
                    doc_id=obj["doc_id"],
                    domain=domain,
                    content=obj["text"],
                    title=obj.get("title", ""),
                    embedding=None  # 先不设置embedding
                )
                docs.append(doc)
        
        # 批量生成 embedding（性能优化）
        if docs:
            print(f"  🔧 正在为 {domain} 生成 embedding ({len(docs)} 文档)...")
            texts = [doc.content for doc in docs]
            
            # 使用 SentenceTransformer 的 encode 方法批量编码
            embeddings = embedding_service.encode(
                texts,
                batch_size=32,  # 批量处理加速
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # 归一化（用于余弦相似度）
            )
            
            # 将 embedding 赋值给文档
            for doc, emb in zip(docs, embeddings):
                doc.embedding = emb
        
        pool[domain] = docs
    
    return pool