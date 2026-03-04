import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, deque
import logging
from scipy.spatial.distance import cosine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DynamicDetectionResult:
    step: int = 0
    is_distribution_shift: bool = False  # 是否发生了分布漂移（Global Shift）
    query_cluster_id: int = -1           # 当前 Query 归属的簇 ID
    is_new_cluster: bool = False         # 是否发现了新簇（新 Domain/Topic）
    cluster_distribution: Dict[int, float] = field(default_factory=dict) # 当前簇分布
    cluster_centroids: Dict[int, np.ndarray] = field(default_factory=dict) # 当前簇中心

class ClusteringDetector:
    """
    单层聚类检测器 (Single-Layer Clustering Detector)
    
    Inspired by ComRAG:
    - 维护动态的 Centroid 集合来表示 Domain/Cluster。
    - 在线更新 Centroid 位置。
    - 当 Query 距离所有已知 Centroid 过远时，创建新簇 (New Domain Discovery)。
    """
    def __init__(self, vector_dim: int = 768, similarity_threshold: float = 0.75, window_size: int = 50):
        self.vector_dim = vector_dim
        self.similarity_threshold = similarity_threshold  # Cosine Similarity 阈值 (0-1), 低于此值视为新簇
        
        # 簇状态
        self.centroids: Dict[int, np.ndarray] = {}  # id -> vector
        self.cluster_counts: Dict[int, int] = {}    # id -> count (total)
        self.next_cluster_id = 0
        
        # 短期窗口用于计算实时分布
        self.window = deque(maxlen=window_size)
        
        # 学习率 (用于移动 Centroid)
        self.alpha = 0.1 

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm_v1 * norm_v2)

    def detect(self, query_embedding: np.ndarray, step: int) -> DynamicDetectionResult:
        res = DynamicDetectionResult(step=step)
        
        # 1. 寻找最近的簇
        best_sim = -1.0
        best_cid = -1
        
        for cid, centroid in self.centroids.items():
            sim = self._cosine_similarity(query_embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cid = cid
        
        # 2. 判断是否创建新簇
        if best_cid == -1 or best_sim < self.similarity_threshold:
            # 创建新簇
            new_id = self.next_cluster_id
            self.centroids[new_id] = query_embedding
            self.cluster_counts[new_id] = 1
            self.next_cluster_id += 1
            
            res.query_cluster_id = new_id
            res.is_new_cluster = True
            res.is_distribution_shift = True # 发陈新 Domain 也是一种 Shift
            
            logger.info(f"🆕 [New Cluster] Created Cluster {new_id} (Sim={best_sim:.3f} < {self.similarity_threshold})")
            
            best_cid = new_id
        else:
            # 归属现有簇并更新 Centroid
            res.query_cluster_id = best_cid
            
            # 在线更新 Centroid: C_new = C_old + alpha * (x - C_old)
            # 或者简单的加权平均，这里用移动平均
            old_centroid = self.centroids[best_cid]
            new_centroid = old_centroid + self.alpha * (query_embedding - old_centroid)
            # Normalize to keep it unit length if needed, usually cosine sim handles magnitude but normalization helps stability
            self.centroids[best_cid] = new_centroid / np.linalg.norm(new_centroid)
            self.cluster_counts[best_cid] += 1

        # 3. 更新滑动窗口并计算分布
        self.window.append(best_cid)
        
        total_in_window = len(self.window)
        counts = Counter(self.window)
        current_distribution = {cid: count / total_in_window for cid, count in counts.items()}
        
        res.cluster_distribution = current_distribution
        res.cluster_centroids = self.centroids.copy()
        
        return res

class HierarchicalClusteringDetector(ClusteringDetector):
    """
    双层聚类检测器 (Hierarchical: Domain -> Topic)
    
    Structure:
    - Layer 1: Domains (Coarse clusters)
    - Layer 2: Topics (Fine clusters within each Domain)
    """
    def __init__(self, vector_dim: int = 768, domain_threshold: float = 0.6, topic_threshold: float = 0.85):
        super().__init__(vector_dim, domain_threshold)
        self.domain_threshold = domain_threshold
        self.topic_threshold = topic_threshold
        
        # Structure: domain_id -> {topic_id -> centroid}
        self.domain_topics: Dict[int, Dict[int, np.ndarray]] = {} 
        self.next_topic_ids: Dict[int, int] = {} # domain_id -> next_topic_id

    def detect(self, query_embedding: np.ndarray, step: int) -> DynamicDetectionResult:
        # 1. First Pass: Detect Domain (Coarse)
        # We reuse the logic from parent class for Domain detection, but with domain_threshold
        self.similarity_threshold = self.domain_threshold
        domain_res = super().detect(query_embedding, step)
        
        domain_id = domain_res.query_cluster_id
        
        # Ensure data structures exist for this domain
        if domain_id not in self.domain_topics:
            self.domain_topics[domain_id] = {}
            self.next_topic_ids[domain_id] = 0
            
        # 2. Second Pass: Detect Topic (Fine) within the identified Domain
        topics = self.domain_topics[domain_id]
        best_topic_sim = -1.0
        best_tid = -1
        
        for tid, centroid in topics.items():
            sim = self._cosine_similarity(query_embedding, centroid)
            if sim > best_topic_sim:
                best_topic_sim = sim
                best_tid = tid
                
        # Topic Logic
        if best_tid == -1 or best_topic_sim < self.topic_threshold:
            # Create New Topic
            new_tid = self.next_topic_ids[domain_id]
            topics[new_tid] = query_embedding
            self.next_topic_ids[domain_id] += 1
            logger.info(f"    ↳ 🆕 [New Topic] Created Topic {new_tid} in Domain {domain_id}")
        else:
            # Update Topic Centroid
            old_centroid = topics[best_tid]
            new_centroid = old_centroid + self.alpha * (query_embedding - old_centroid)
            topics[best_tid] = new_centroid / np.linalg.norm(new_centroid)
            
        # TODO: Refine the result to include topic info if needed
        # For now, we return the Domain result as the primary "Cluster" for the Updator
        # The Updator could be extended to handle hierarchical updates
        
        return domain_res
