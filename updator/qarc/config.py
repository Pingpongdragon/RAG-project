"""
QARC 超参数配置 — 所有可调参数集中在此

使用方式:
    from updator.qarc.config import QARCConfig
    cfg = QARCConfig()                    # 全部默认值
    cfg = QARCConfig(window_size=16)      # 覆盖个别参数
    cfg = QARCConfig.from_dict({...})     # 从字典构造

修改建议:
    参数按模块分组, 每组有中文注释说明含义和调参方向
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class QARCConfig:
    """QARC 全部超参数。

    ┌─────────────────────────────────────────────────────────────┐
    │  一、窗口与检索                                               │
    │  window_size       每攒多少条 query 触发一次窗口处理           │
    │  retrieve_top_k    每条 query 从 KB 检索几篇文档               │
    │  query_history_max query 历史 ring buffer 最大容量             │
    ├─────────────────────────────────────────────────────────────┤
    │  二、KB 策展 (Curator)                                       │
    │  kb_budget         KB 最多装几篇文档                           │
    │  candidate_top_k   每个兴趣中心从文档池检索几篇候选             │
    ├─────────────────────────────────────────────────────────────┤
    │  三、对齐漂移检测 (Detector)                                  │
    │  drift_n_clusters  KMeans(KB) 聚几个主题                      │
    │  drift_top_n_sims  对齐特征取 top-N 相似度                     │
    │  drift_threshold_* 阈值校准参数                                │
    │  drift_cov_reg     协方差正则化系数 ε                          │
    ├─────────────────────────────────────────────────────────────┤
    │  四、Agent 决策                                               │
    │  agent_warmup_*    Warmup 期配置                              │
    │  agent_gap_*       AlignmentGap 自适应阈值                    │
    │  agent_lambda_*    更新替换比例                                │
    │  agent_eta_*       子模目标多样性系数                          │
    │  agent_cooldown_*  冷却期配置                                  │
    │  agent_recalibrate_after  连续漂移几次后重校准                 │
    ├─────────────────────────────────────────────────────────────┤
    │  五、AutoKMeans (兴趣聚类)                                    │
    │  kmeans_k_min/max  自动搜索 K 的范围                           │
    │  kmeans_seed       随机种子                                    │
    └─────────────────────────────────────────────────────────────┘
    """

    # ═══════ 一、窗口与检索 ═══════
    window_size: int = 8
    """每攒多少条 query 触发一次窗口处理。越小反应越快但估计越不稳定"""

    retrieve_top_k: int = 10
    """每条 query 从 KB 中检索的文档数"""

    query_history_max: int = 500
    """query 历史 ring buffer 容量。用于 DriftLens 基线建立和重校准"""

    # ═══════ 二、KB 策展 ═══════
    kb_budget: int = 50
    """KB 最大容量。越大覆盖越广但检索越慢"""

    candidate_top_k: int = 100
    """每个兴趣中心从文档池检索的候选文档数。越大候选越全但 curator 越慢"""

    # ═══════ 三、对齐漂移检测 ═══════
    drift_n_clusters: int = 5
    """KMeans(KB 文档) 的簇数 K。对齐特征的前 K 维 = [sim(q,c₁),...,sim(q,cₖ)]"""

    drift_top_n_sims: int = 10
    """对齐特征的后 N 维 = [top1_sim,...,topN_sim]。对齐特征总维度 = K + N"""

    drift_threshold_percentile: float = 95.0
    """FID 阈值取历史 FID 分布的第几百分位。95 = 5% 假阳率"""

    drift_threshold_n_samples: int = 500
    """校准阈值时随机采样多少个窗口"""

    drift_cov_reg: float = 1e-5
    """协方差正则化 ε: Σ = cov(X) + ε·I。防止奇异矩阵"""

    drift_random_state: int = 42
    """漂移检测器随机种子"""

    # ═══════ 四、Agent 决策 ═══════
    agent_warmup_windows: int = 3
    """前 N 个窗口始终激进更新 (冷启动快速收敛)"""

    agent_gap_ema_beta: float = 0.85
    """Gap EMA 平滑因子。越大越平滑, 对突变越迟钝"""

    agent_gap_k: float = 1.5
    """Gap 异常阈值 = EMA + k·MAD。越大越宽松 (越少触发 MILD_UPDATE)"""

    agent_lambda_mild: float = 0.2
    """轻度更新: 最多替换 KB 的 20%"""

    agent_lambda_aggressive: float = 0.5
    """激进更新: 最多替换 KB 的 50%"""

    agent_eta_mild: float = 0.1
    """轻度更新时子模目标的多样性系数 (越大越重视覆盖面)"""

    agent_eta_aggressive: float = 0.05
    """激进更新时多样性系数 (更偏向兴趣匹配)"""

    agent_cooldown_windows: int = 2
    """更新后冷却 N 个窗口不再更新 (防止连续无效替换)"""

    agent_recalibrate_after: int = 3
    """连续漂移 N 次后触发 RECALIBRATE (重建 DriftLens 基线 + 激进更新)"""

    # ═══════ 五、AutoKMeans 兴趣聚类 ═══════
    kmeans_k_min: int = 2
    """自动搜索 K 的下界"""

    kmeans_k_max: int = 10
    """自动搜索 K 的上界"""

    kmeans_seed: int = 42
    """KMeans 随机种子"""

    # ─── 工具方法 ───

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QARCConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def __repr__(self):
        lines = ["QARCConfig("]
        for k, v in self.to_dict().items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)
