import numpy as np
from typing import Dict, List, Optional
from collections import Counter, deque
from scipy.spatial.distance import jensenshannon
from dataclasses import dataclass
import logging

try:
    from river import drift
except ImportError:
    print("请先安装 river: pip install river")
    exit()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    step: int = 0
    is_global_shift: bool = False
    js_divergence: float = 0.0
    is_intra_degradation: bool = False
    degraded_domain: Optional[str] = None
    domain_score: float = 0.0
    query_dist: Dict[str, float] = None
    
    def __post_init__(self):
        if self.query_dist is None:
            self.query_dist = {}


class AutoAdaptiveDetector:
    """
    简化版自适应检测器：基于 JS 散度绝对阈值
    
    核心逻辑：
        - JS > 0.3 → 触发全局更新
        - Score 下降 > 5% → 触发域内更新
    """
    def __init__(self, domains: List[str] = None):
        self.domains = domains or ["0_entertainment", "1_stem", "2_humanities", "3_lifestyle"]
        
        # 查询窗口（用于计算当前分布）
        self.short_query_window = deque(maxlen=50)
        
        # KB 初始分布（均匀分布）
        self.kb_dist = {d: 1.0/len(self.domains) for d in self.domains}
        
        # ✅ 核心阈值
        self.js_threshold = 0.4  # JS 散度阈值（可调参数）
        self.score_drop_threshold = 0.02  # 分数下降阈值（5%）
        
        # 领域内检测器（用于检测分数下降）
        self.domain_adwins = {d: drift.ADWIN(delta=0.2) for d in self.domains}
        
        # 状态管理
        self.cooldown = 0
        
        logger.info(f"✅ 自适应检测器初始化完成 | JS阈值={self.js_threshold}")

    def update_kb_distribution(self, new_kb_dist: Dict[str, float]):
        """更新 KB 分布后，同步更新基线分布"""
        total = sum(new_kb_dist.values())
        self.kb_dist = {k: v / total for k, v in new_kb_dist.items()}
        
        # 重置域内检测器
        self.domain_adwins = {d: drift.ADWIN(delta=0.2) for d in self.domains}
        
        logger.info(f"🔄 KB 分布已更新: {self.kb_dist}")

    def detect(self, query_domain: str, retrieval_score: float, step: int) -> DetectionResult:
        """
        检测方法
        
        Args:
            query_domain: 当前查询所属的 domain
            retrieval_score: 当前查询的检索得分（Recall@k）
            step: 当前步数
        
        Returns:
            DetectionResult: 检测结果
        """
        res = DetectionResult(step=step)

        # 冷却期（避免短时间内重复检测）
        if self.cooldown > 0:
            self.cooldown -= 1
            return res

        # 1. 计算当前查询分布
        self.short_query_window.append(query_domain)
        q_counts = Counter(self.short_query_window)
        total_q = len(self.short_query_window)
        current_query_dist = {d: q_counts.get(d, 0)/total_q for d in self.domains}
        res.query_dist = current_query_dist
        
        # 2. 计算 JS 散度
        js_val = self._compute_js(current_query_dist, self.kb_dist)
        res.js_divergence = js_val
        
        # 3. ✅ 全局 Shift 检测：JS > 阈值
        if js_val > self.js_threshold:
            res.is_global_shift = True
            logger.warning(f"🚨 [Global Shift] Step {step} | JS={js_val:.3f} > 阈值={self.js_threshold}")
            logger.warning(f"   当前查询分布: {current_query_dist}")
            logger.warning(f"   当前 KB 分布: {self.kb_dist}")
            
            # 冷却期（避免同一变化重复检测）
            self.cooldown = 20

        # 4. 域内 Score 下降检测
        if query_domain in self.domain_adwins:
            adwin = self.domain_adwins[query_domain]
            prev_mean = adwin.estimation
            adwin.update(retrieval_score)
            current_mean = adwin.estimation
            
            if adwin.drift_detected:
                # 只有分数下降才报警
                if current_mean < prev_mean - self.score_drop_threshold:
                    res.is_intra_degradation = True
                    res.degraded_domain = query_domain
                    res.domain_score = retrieval_score
                    logger.warning(f"📉 [Intra Drop] Step {step} | Domain={query_domain} | Score={retrieval_score:.3f} (Was {prev_mean:.3f})")
                    
                    # 重置该 domain 的 ADWIN
                    self.domain_adwins[query_domain] = drift.ADWIN(delta=0.2)
                    
                    if not res.is_global_shift:  # 避免重复设置冷却
                        self.cooldown = 10

        return res

    def _compute_js(self, d1, d2):
        """计算 Jensen-Shannon 散度"""
        all_k = sorted(set(d1) | set(d2))
        p = np.array([d1.get(k, 0) for k in all_k]) + 1e-10
        q = np.array([d2.get(k, 0) for k in all_k]) + 1e-10
        return jensenshannon(p, q)