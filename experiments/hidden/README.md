# Hidden-Evidence Experiments

本目录保存不能由 direct dense signal 完整覆盖的边界实验：

- 2Wiki 的 inference、compositional 与 bridge-comparison evidence；
- graph 或 entity-expansion retrieval。

它与 `experiments/direct/` 共享策略注册表、因果 stream protocol 和 factorized
workload，但保留独立 runner，因为这里还需要 question-type filtering、graph
retrieval 与 query expansion。不要把不同控制流重新塞回一个巨型
`run.py`。

## 当前入口

```bash
cd /data/jyliu/RAG-project
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python

$PY experiments/hidden/run.py \
  --datasets 2wikimultihopqa \
  --expanded --q-type bridge_comparison \
  --workload factorized_recurring \
  --retrieval graph \
  --n-source 2000 --n-windows 20 --window-size 25 \
  --kb-pool-ratio 0.1 \
  --strategies LRU FIFO TinyLFU AgentRAGCache DRIP Oracle \
  --output 2wiki_hidden.json
```

可用数据集由 `config.py::DATASET_CONFIGS` 和 `loaders.py::LOADERS` 共同定义。
当前只保留 `2wikimultihopqa`：它是 hidden-evidence 机制边界，不承担
cross-domain cache necessity 的主证据。HotpotQA 与 MuSiQue loader 已从活跃代码
移除；原始数据与历史 JSON/figure 不删除。MIND、MT-RAG 与 Wizard of Wikipedia
位于 `experiments/agent/`。

2Wiki bridge-comparison 使用统一 loader，不再在 direct 分支复制一份：

```bash
$PY experiments/hidden/run.py \
  --datasets 2wikimultihopqa --expanded \
  --q-type bridge_comparison ...
```

结果写入 `data/`，图写入 `figures/`。

正式 DRIP 实现仍只有一份，位于 [`algorithms/drip/`](../../algorithms/drip/)；
本目录只负责实验协议，不复制算法。
