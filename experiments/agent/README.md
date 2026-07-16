# Agent RAG Access Experiments

本目录保存真实 agent、session 与用户访问日志实验。它和 `direct/`、`hidden/`
最重要的区别不是 evidence 一定可见或不可见，而是请求具有真实执行上下文：时间戳、
会话、任务步骤，以及服务完成后才出现的 evidence feedback。

## 数据与入口

| 数据集 | 顺序协议 | 正式入口 |
|---|---|---|
| MIND-small | `behaviors.tsv` 真实时间戳 | `run_access_trace.py` |
| MT-RAG | 保持每个 conversation 的 turn order | `run_mtrag_trace.py` |
| Wizard of Wikipedia | 保持每个 dialogue 的 turn order；topic 为显式 domain | `loaders.py`，待接 exact-residency runner |

Mind2Web 已从活跃 loader 删除：旧 controlled stream 有较高 exact-query repetition，
对当前 cross-domain 论证不如 Wizard of Wikipedia 干净。Wizard 的 selected knowledge
不会写入 query，只在当前 turn 评分后作为 access feedback；数据集没有全局 timestamp，
因此只能使用 session round-robin 或明确标注的 controlled domain schedule。

## 运行命令

MIND 的 exact evidence-residency replay：

```bash
cd /data/jyliu/RAG-project
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python

$PY experiments/agent/run_access_trace.py \
  --windows 100 --window-size 500 --warmup-windows 3 \
  --cache-size 144 --write-budget 9 --candidate-budget 72 \
  --output experiments/agent/data/mind_access.json
```

MT-RAG 的 conversation replay：

```bash
$PY experiments/agent/run_mtrag_trace.py \
  --protocol session_round_robin --query-view rewrite \
  --window-size 25 --max-queries 5000 \
  --output experiments/agent/data/mtrag_round_robin.json
```

两个 runner 都先测量当前 cache，再把本次 evidence 当作后验反馈交给策略，避免未来
support 泄漏。受控 recurring-domain 只用于单独的可预测性诊断，并在输出中标为
synthetic，不能冒充真实时间漂移。

Wizard loader 快速审计：

```bash
$PY -c "from experiments.agent.loaders import load_wizard_of_wikipedia; \
d,q,m=load_wizard_of_wikipedia(); print(len(d), len(q), len(m))"
```
