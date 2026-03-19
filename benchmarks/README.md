# benchmarks/ — 实验数据集构造模块

## 概览

本模块负责为 RAG 知识库更新策略的对比实验构造数据集。

**核心思路:** 将用户查询建模为一个随时间变化的流（stream），通过不同的 topic 概率调度器模拟各种兴趣漂移模式，然后评估各 KB 更新策略（QARC、ComRAG、ERASE 等）在漂移环境下的检索效果。

### 为什么文档池需要非 gold 文档？

文档池（`document_pool`）模拟真实检索环境：pool 中包含大量候选文档（默认 5000 篇），只有极少数是当前查询的 gold 文档。KB 更新策略需要自己判断保留哪些文档——如果 pool 中只有 gold docs，任何策略检索正确率都接近 100%，实验就失去了意义。

```
document_pool (5000 docs)
├── Gold docs (每个查询 1-2 篇)    ← 策略要找到这些
├── 同 topic 非 gold docs          ← 合理的干扰项
└── 其他 topic docs                ← 无关噪声
```

`merge_pool()` 确保所有 gold 文档必定存在于池中，其余位置随机填充。

### 数据流

```
                        ┌─────────────────────┐
 原始数据               │  WoW / HotpotQA     │
 (datasets/)            │  JSON 文件           │
                        └──────────┬──────────┘
                                   │ load_wow() / load_hotpotqa()
                                   ▼
                        ┌─────────────────────┐
 数据提取               │  extract_topic_data  │  按 topic 分组提取
                        │  / build_entity_graph│  查询和文档
                        └──────────┬──────────┘
                                   │
                                   ▼
┌──────────────────┐    ┌─────────────────────┐
│  TopicSchedule   │───▶│  build_stream()     │  按时变概率采样
│  P(topic | t)    │    │  / greedy_walk()     │  生成查询序列
└──────────────────┘    └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
 输出                   │  ExperimentDataset   │
                        │  .query_stream       │  300~400 条查询
                        │  .document_pool      │  5000 篇候选文档
                        └─────────────────────┘
```

## 模块结构

```
benchmarks/
├── __init__.py                    # 统一导出层
├── config.py                      # 所有可调参数（集中管理）
├── data_structures.py             # QueryItem, PoolDocument, ExperimentDataset
├── datasets/                      # 数据加载器
│   ├── __init__.py
│   ├── wow_loader.py              # WoW 数据加载、topic 提取、流构建
│   └── hotpotqa_loader.py         # HotpotQA 加载、实体图游走
├── schedules/                     # Topic 概率调度器
│   ├── __init__.py
│   └── topic_schedules.py         # TopicSchedule + 3 种实现
├── builders/                      # 实验构建器
│   ├── __init__.py
│   └── experiment_builders.py     # 4 种 build_* 函数
└── README.md                      # 本文档
```

## 配置系统（config.py）

所有可调参数集中在 `config.py`，修改参数无需改动任何 loader/builder 代码。

### 配置类层次

```python
BenchmarkConfig                    # 全局配置（聚合所有实验）
├── GradualDriftConfig             # Exp 1 参数
│   ├── total_queries = 300
│   ├── sigma = 0.18               # 高斯标准差
│   ├── n_topics = 10
│   ├── preferred_topics = BIG_TOPICS
│   ├── pool: PoolConfig
│   │   ├── max_total = 5000       # 文档池上限
│   │   └── keep_all_gold = True
│   └── wow: WoWConfig
│       ├── split = "validation"
│       └── min_conversations = 15
├── SuddenShiftConfig              # Exp 2 参数
│   ├── steepness = 30.0           # sigmoid 陡峭程度
│   └── ...
├── CyclicReturnConfig             # Exp 3 参数
│   ├── n_cycles = 2
│   ├── sigma = 0.08
│   └── ...
└── HotpotQAConfig                 # Exp 4 参数
    ├── total_queries = 400
    ├── split = "validation_distractor"
    └── pool: PoolConfig(max_total=50000)
```

### Topic 分组常量

| 变量 | Topics | 用途 |
|------|--------|------|
| `BIG_TOPICS` | Pasta, Brown hair, Jazz, Ferrari, Obesity | 渐进漂移（数据充足） |
| `DIVERSE_TOPICS` | Red, Manta ray, Superman, Niagara Falls, Ferrari | 突变转移（语义距离大） |
| `CYCLE_TOPICS` | Pasta, Ferrari, Hair coloring | 周期回归 |
| `FOOD_CHAIN` | Pasta, Pizza, Baking, Wine tasting | 语义近邻链 |
| `HEALTH_CHAIN` | Obesity, Physical fitness, Chronic fatigue syndrome | 健康领域链 |

### 使用示例

```python
from benchmarks.config import BenchmarkConfig, GradualDriftConfig

# 使用默认配置
cfg = BenchmarkConfig.default()

# 快速测试（缩小规模）
cfg = BenchmarkConfig.quick_test()

# 自定义单个实验
from benchmarks.config import PoolConfig
my_cfg = GradualDriftConfig(
    total_queries=500,
    sigma=0.25,
    pool=PoolConfig(max_total=8000),
)
```

## 数据结构（data_structures.py）

### QueryItem — 单条查询
| 字段 | 类型 | 含义 |
|------|------|------|
| `query_id` | str | 唯一 ID，如 `"wow_a3f1c2_t0"` |
| `question` | str | 用户问题 |
| `answer` | str | 参考答案 |
| `topic` | str | 自然 topic 标签 |
| `gold_doc_ids` | List[str] | 该查询的 gold 文档 ID |
| `metadata` | Dict | 扩展字段 (conv_id, turn 等) |

### PoolDocument — 候选文档
| 字段 | 类型 | 含义 |
|------|------|------|
| `doc_id` | str | 唯一 ID |
| `text` | str | 文档正文 |
| `topic` | str | 所属 topic |
| `title` | str | 标题 |

### ExperimentDataset — 完整实验数据集
| 字段 | 类型 | 含义 |
|------|------|------|
| `name` | str | 实验名称 |
| `query_stream` | List[QueryItem] | 按时间排列的查询序列 |
| `document_pool` | List[PoolDocument] | 候选文档池（含 gold + 干扰项） |
| `topics` | List[str] | 涉及的所有 topic |
| `topic_schedule_log` | List[Dict] | 每步的 topic 概率分布 |

## Topic 调度器（schedules/）

调度器定义 `P(topic | t)`，t ∈ [0, 1] 是归一化时间位置。

### GaussianDriftSchedule — 渐进漂移
```
    Topic A    Topic B    Topic C
      ╭╮         ╭╮         ╭╮
     ╱  ╲       ╱  ╲       ╱  ╲
───╱──────╲───╱──────╲───╱──────╲──▶ t
```
- 每个 topic 一个高斯激活曲线，`sigma` 控制重叠（默认 0.18）

### SigmoidShiftSchedule — 突变转移
```
    Topic A  │  Topic B  │  Topic C
    ━━━━━━━━━┿━━━━━━━━━━━┿━━━━━━━━━━
             ↑           ↑
        突然切换     突然切换
```
- sigmoid 阶跃切换，`steepness` 控制速度（默认 30.0）

### CyclicSchedule — 周期回归
```
    A → B → C → A → B → C
```
- topics 周期性激活，`n_cycles` 控制总周期数（默认 2）

## 数据加载器（datasets/）

### wow_loader.py

```
load_wow() → extract_topic_data() → build_stream() → merge_pool()
```

WoW 对话天然标注 topic，利用这些标签构造漂移查询流。`build_stream()` 按调度器概率每步采样一个 topic，再从该 topic 的查询池中取下一条。

### hotpotqa_loader.py

```
load_hotpotqa() → build_entity_graph() → greedy_walk() → hotpotqa_item_to_query_and_docs()
```

HotpotQA 共享 supporting fact title 的问题形成邻接图。贪心游走优先访问未去过的邻居，产生自然的 topic 漂移。

## 四种实验（builders/）

### Exp 1: Gradual Drift（渐进漂移）
```python
ds = build_gradual_drift(total_queries=300, sigma=0.18)
```

### Exp 2: Sudden Shift（突变转移）
```python
ds = build_sudden_shift(total_queries=300, steepness=30.0)
```

### Exp 3: Cyclic Return（周期回归）
```python
ds = build_cyclic_return(total_queries=300, n_cycles=2)
```

### Exp 4: HotpotQA Entity Walk
```python
ds = build_hotpotqa_entity_walk(total_queries=400)
```

### 一键构建
```python
all_ds = build_all_datasets()
```

## 与其他模块的关系

```
benchmarks/              ← 数据集构造（你在这里）
    │
    │  ExperimentDataset
    ▼
test/experiment_framework.py  ← 实验执行框架
    │                           （Adapter 包装各 KB 更新策略）
    ├── QARCAdapter      ← updator/qarc/
    ├── ComRAGAdapter    ← updator/comrag/
    ├── ERASEAdapter     ← updator/erase/
    ├── StaticKBBaseline
    └── RandomKBBaseline
    │
    ▼
test/run_experiments.py  ← 实验入口

注意: core/ (retriever, reranker, generator 等) 是独立的 RAG 管线，
      实验代码不依赖 core/。
```

## 向后兼容

`test/experiment_datasets.py` 是薄兼容层，旧 import 继续工作：

```python
# 旧写法（继续有效）
from test.experiment_datasets import build_gradual_drift, QueryItem

# 新写法（推荐）
from benchmarks import build_gradual_drift, QueryItem
from benchmarks.config import BenchmarkConfig
```
