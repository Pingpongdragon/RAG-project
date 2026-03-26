"""
benchmark 实验配置与数据结构定义。

包含两部分:
  1. 配置参数 — BenchmarkConfig 及子配置 (原 config.py)
  2. 核心数据类 — QueryItem, PoolDocument, ExperimentDataset (原 data_structures.py)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union


# ============================================================
#  Topic 分组常量（WoW 数据集中的真实 topic 名称）
# ============================================================

FOOD_CHAIN = ["Pasta", "Pizza", "Baking", "Wine tasting"]
HEALTH_CHAIN = ["Obesity", "Physical fitness", "Chronic fatigue syndrome"]
DIVERSE_TOPICS = ["Red", "Manta ray", "Superman", "Niagara Falls", "Ferrari"]
CYCLE_TOPICS = ["Pasta", "Ferrari", "Hair coloring"]
BIG_TOPICS = ["Pasta", "Brown hair", "Jazz", "Ferrari", "Obesity"]


# ============================================================
#  数据集构建参数
# ============================================================

@dataclass
class WoWConfig:
    """WoW 数据源配置"""
    split: str = "validation"
    min_conversations: int = 15       # select_topics 时要求的最少对话数


@dataclass
class PoolConfig:
    """文档池配置"""
    max_total: int = 5000             # 文档池上限
    keep_all_gold: bool = True        # 始终保留所有 gold 文档


@dataclass
class GradualDriftConfig:
    """Exp 1: Gradual Drift 参数"""
    total_queries: int = 300
    sigma: float = 0.18               # 高斯标准差，越大重叠越多
    n_topics: int = 10                # 选取的 topic 数量
    preferred_topics: List[str] = field(default_factory=lambda: list(BIG_TOPICS))
    seed: int = 42
    pool: PoolConfig = field(default_factory=PoolConfig)
    wow: WoWConfig = field(default_factory=WoWConfig)


@dataclass
class SuddenShiftConfig:
    """Exp 2: Sudden Shift 参数"""
    total_queries: int = 300
    steepness: float = 30.0           # sigmoid 陡峭程度，越大切换越突然
    n_topics: int = 10
    preferred_topics: List[str] = field(default_factory=lambda: list(DIVERSE_TOPICS))
    seed: int = 42
    pool: PoolConfig = field(default_factory=PoolConfig)
    wow: WoWConfig = field(default_factory=WoWConfig)


@dataclass
class CyclicReturnConfig:
    """Exp 3: Cyclic Return 参数"""
    total_queries: int = 300
    n_cycles: int = 2                 # 周期数
    sigma: float = 0.08               # 每个高斯峰宽度
    n_topics: int = 5
    preferred_topics: List[str] = field(default_factory=lambda: list(CYCLE_TOPICS))
    seed: int = 42
    pool: PoolConfig = field(default_factory=PoolConfig)
    wow: WoWConfig = field(default_factory=WoWConfig)


@dataclass
class HotpotQAConfig:
    """Exp 4: HotpotQA Entity Walk 参数"""
    total_queries: int = 400
    split: str = "validation_distractor"
    seed: int = 42
    pool: PoolConfig = field(default_factory=lambda: PoolConfig(max_total=50000))


@dataclass
class BenchmarkConfig:
    """全局配置 — 聚合所有实验的参数"""
    gradual_drift: GradualDriftConfig = field(default_factory=GradualDriftConfig)
    sudden_shift: SuddenShiftConfig = field(default_factory=SuddenShiftConfig)
    cyclic_return: CyclicReturnConfig = field(default_factory=CyclicReturnConfig)
    hotpotqa: HotpotQAConfig = field(default_factory=HotpotQAConfig)

    @classmethod
    def default(cls) -> "BenchmarkConfig":
        return cls()

    @classmethod
    def quick_test(cls) -> "BenchmarkConfig":
        """缩小规模的快速测试配置"""
        small_pool = PoolConfig(max_total=500)
        return cls(
            gradual_drift=GradualDriftConfig(total_queries=30, n_topics=3, pool=small_pool),
            sudden_shift=SuddenShiftConfig(total_queries=30, n_topics=3, pool=small_pool),
            cyclic_return=CyclicReturnConfig(total_queries=30, n_topics=3, pool=small_pool),
            hotpotqa=HotpotQAConfig(total_queries=40, pool=PoolConfig(max_total=1000)),
        )


# ============================================================
#  核心数据类型
# ============================================================

@dataclass
class QueryItem:
    """查询流中的单条查询。

    Attributes:
        query_id:      唯一标识，如 "wow_a3f1c2_t0"
        question:      用户问题文本
        answer:        参考答案（WoW 中为 Wizard 的回复）
        topic:         自然 topic 标签（WoW 的 topics[0]）
        gold_doc_ids:  该查询的 gold 文档 ID 列表
        metadata:      扩展字段（conv_id, turn 等）
    """
    query_id: str
    question: str
    answer: str
    topic: str
    gold_doc_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolDocument:
    """候选文档池中的一篇文档。

    Attributes:
        doc_id:   唯一标识，如 "wow_a3f1c2d4e5f6"
        text:     文档正文
        topic:    所属 topic
        title:    文档标题（WoW 中为知识段落的 title）
        metadata: 扩展字段
    """
    doc_id: str
    text: str
    topic: str
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentDataset:
    """一个完整的实验数据集。

    Attributes:
        name:               实验名称 (gradual_drift / sudden_shift / ...)
        description:        简短描述
        query_stream:       按时间排列的查询序列
        document_pool:      候选文档池（KB 从中选取子集）
        topics:             涉及的所有 topic 列表
        schedule_type:      调度类型 (gaussian / sigmoid / cyclic / entity_walk)
        topic_schedule_log: 每个时刻的 topic 概率分布记录
    """
    name: str
    description: str
    query_stream: List[QueryItem]
    document_pool: List[PoolDocument]
    topics: List[str]
    schedule_type: str
    topic_schedule_log: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """序列化为可 JSON 化的 dict。"""
        return {
            "name": self.name,
            "description": self.description,
            "topics": self.topics,
            "schedule_type": self.schedule_type,
            "topic_schedule_log": self.topic_schedule_log,
            "query_stream": [
                {
                    "query_id": q.query_id,
                    "question": q.question,
                    "answer": q.answer,
                    "topic": q.topic,
                    "gold_doc_ids": q.gold_doc_ids,
                    "metadata": q.metadata,
                }
                for q in self.query_stream
            ],
            "document_pool": [
                {
                    "doc_id": d.doc_id,
                    "text": d.text,
                    "topic": d.topic,
                    "title": d.title,
                    "metadata": d.metadata,
                }
                for d in self.document_pool
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentDataset":
        """从 dict 反序列化。"""
        return cls(
            name=data["name"],
            description=data["description"],
            topics=data["topics"],
            schedule_type=data["schedule_type"],
            topic_schedule_log=data.get("topic_schedule_log", []),
            query_stream=[
                QueryItem(
                    query_id=q["query_id"],
                    question=q["question"],
                    answer=q["answer"],
                    topic=q["topic"],
                    gold_doc_ids=q.get("gold_doc_ids", []),
                    metadata=q.get("metadata", {}),
                )
                for q in data["query_stream"]
            ],
            document_pool=[
                PoolDocument(
                    doc_id=d["doc_id"],
                    text=d["text"],
                    topic=d["topic"],
                    title=d.get("title", ""),
                    metadata=d.get("metadata", {}),
                )
                for d in data["document_pool"]
            ],
        )

    def save_json(self, path: Union[str, Path]) -> Path:
        """保存为 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "ExperimentDataset":
        """从 JSON 文件加载。"""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
