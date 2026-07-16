# DRIP 实验入口

## 当前方法

当前 registry 只保留一个论文类 `DRIP`。默认配置是 Reactive；显式提供冷库
topic partition 时启用 `drip-reactive-v2-topicstate-v1` 的 TopicState 分支。算法定义见：

```text
algorithms/drip/README.md
algorithms/drip/PARAMETER_MAPPING.md
```

数据构造协议见：

```text
docs/experiments/DATASET_CONSTRUCTION.md
```

2026-07-14 及之前的日期型报告是历史研究档案，其中的 `DRIPNOdetector`、
`DRIP-WorkingSet`、Forecast、detector 和 hidden 结果不能作为当前实现的主表。

## 当前可运行 workload

| Runner name | 协议 | 作用 |
|---|---|---|
| `squad_direct` | factorized recurring evidence regimes | 高复用 direct 主机制实验 |
| `fever` | factorized evidence regimes + offline topic metadata | 高复用 cross-topic 扩展 |
| `streamingqa_official` | 官方 `question_ts` 自然流 | temporal 边界实验 |
| `mind_news_context` | 真实点击时间流 | agent access-cache 外部有效性 |
| `mtrag_human` | 官方 conversation turn order | 多轮 agent RAG evidence residency |
| `wizard_of_wikipedia` | dialogue turn order + explicit topic | multi-session cross-topic 扩展 |
| `2wikimultihopqa` | factorized hidden evidence regimes | hidden-evidence 边界实验 |

## Cache-ratio sweep

从项目根目录运行。输出前缀默认是 `drip_reactive_sweep`，不会把旧
`working_set_sweep` JSON 当作当前方法结果：

```bash
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python

$PY motivation/run_cache_ratio_sweep.py squad_direct \
  --ratios 0.01 0.02 0.05 0.10 --gpu 0
$PY motivation/run_cache_ratio_sweep.py streamingqa_official \
  --ratios 0.01 0.02 0.05 0.10 --gpu 1
```

默认策略为：

```text
LRU FIFO TinyLFU GPTCacheStyle Proximity AgentRAGCache DRIP
```

只看命令与结果位置，不执行：

```bash
$PY motivation/run_cache_ratio_sweep.py squad_direct --dry-run
```

输出位置由 runner 决定：

```text
experiments/direct/data/drip_reactive_sweep_<dataset>_rXXXX.json
experiments/agent/data/<agent_protocol>.json
```

cache-ratio sweep 仍测试默认 Reactive QA 路径；真实 access trace 的正式入口是
`experiments/agent/run_access_trace.py` 和
`experiments/agent/run_mtrag_trace.py`。

活跃 loader 已按论文证据角色收缩。HotpotQA/2Wiki comparison、TriviaQA、
TREC-COVID、Mind2Web、hidden HotpotQA 和 MuSiQue 的历史结果仍可审计，但不再从
当前 registry 启动。

## 当前结果状态

当前权威结果、协议与 artifact 清单见：

```text
docs/experiments/DRIP_TOPIC_STATE_RESULTS_2026-07-16.md
```

旧 B=24、pre-disjoint SQuAD JSON 仅是历史调试结果，不能进入论文主表。
