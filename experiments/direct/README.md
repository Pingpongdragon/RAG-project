# Direct-Evidence Experiments

本目录是 DRIP 的 direct-evidence 正式实验入口，覆盖两类流：

- `natural_temporal`：使用数据集官方时间顺序，不合成 topic；
- `factorized_*`：在独立的 sparse evidence 空间构造 one-shot、gradual、recurring
  等受控 evidence-regime，不使用 DRIP 的 dense embedding 定义漂移。

当前正式 loader 只注册三个有明确论文角色的数据集：SQuAD 是受控
evidence-domain 主结果，FEVER 是带 gold Wikipedia evidence 的高复用
cross-topic 扩展，StreamingQA 是可选的自然时间外部有效性。HotpotQA/2Wiki
comparison、TriviaQA 与 TREC-COVID 已从活跃 direct 代码移除；历史 JSON 不删除。

## 代码结构

| 文件 | 职责 |
|---|---|
| `run.py` | 薄 CLI：解析参数、保存 JSON、触发绘图 |
| `experiment.py` | 单数据集实验编排 |
| `workload.py` | 统一数据读入、流构造、warm-up 与初始 KB |
| `evaluation.py` | 逐窗口先服务计分、后更新策略 |
| `metrics.py` | Has-Answer 与 evidence residency 原始统计 |
| `reporting.py` | AMAT、Recall、Replacements、延迟和 JSON 汇总 |
| `plotting.py` | 正式实验曲线 |
| `loaders.py` | 普通 QA 数据集 loader |
| `loaders_temporal.py` | StreamingQA 官方时间流 loader |
| `config.py` | 数据集、策略、容量和系统成本配置 |

主调用链：

```text
run.py
  -> experiment.py
       -> workload.py
       -> evaluation.py
       -> reporting.py
  -> plotting.py
```

## 运行示例

```bash
cd /data/jyliu/RAG-project
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python

$PY experiments/direct/run.py \
  --datasets streamingqa_official \
  --workload natural_temporal \
  --n-windows 50 --window-size 500 \
  --kb-pool-ratio 0.1 \
  --strategies LRU FIFO TinyLFU AgentRAGCache DRIP Oracle \
  --output streamingqa_direct.json

$PY experiments/direct/run.py \
  --datasets squad fever \
  --workload factorized_recurring \
  --n-windows 50 --window-size 50 \
  --kb-pool-ratio 0.1 \
  --strategies LRU FIFO TinyLFU AgentRAGCache DRIP Oracle \
  --output controlled_direct.json
```

结果写入 `experiments/direct/data/`，图写入 `experiments/direct/figures/`。
数据集构造与 detector 解耦的协议见
[`experiments/common/factorized_workload.py`](../../experiments/common/factorized_workload.py)，
自然时间流和泄漏审计见
[`experiments/common/stream_protocol.py`](../../experiments/common/stream_protocol.py)。
