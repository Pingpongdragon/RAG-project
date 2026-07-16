# DRIP 数据集构造协议

当前实验明确区分两类 workload：自然时间流和受控 evidence-regime 流。二者不能混用
同一种重排方式。

## 1. 自然时间流

适用数据集：

```text
streamingqa_official  按官方 question_ts 排序
mind_news_context     按真实点击 event_ts 排序
mtrag_human           保持 conversation 内 turn_idx 顺序
wizard_of_wikipedia   保持 dialogue 内 turn_idx；无全局 timestamp
```

带 `preserve_order=True` 的 loader 输出只能选择 `natural_temporal`。构造器不会做
KMeans、topic 分箱、循环 query 或随机重排。

流程：

1. loader 按官方 timestamp 排序全部事件；
2. 取互不重叠的 warm-up prefix 和 evaluation events；
3. `prefix` 使用日志前缀，`window_span` 在完整时间跨度上等距选择连续事件块；
4. 初始 hot cache 只由 warm-up query 的 cold retrieval 构造；
5. evaluation query 不参与初始化，策略只在服务窗口后更新下一窗口缓存。

代码：

```text
experiments/direct/loaders_temporal.py::load_streamingqa_official
experiments/agent/loaders.py::load_mind_news_context
experiments/agent/mtrag_loader.py::load_mtrag_human
experiments/agent/loaders.py::load_wizard_of_wikipedia
experiments/common/stream_protocol.py::chronological_sample
experiments/common/session_workload.py
experiments/common/stream_protocol.py::causal_prefix_init_kb
```

## 2. 受控 QA 漂移

适用数据集：SQuAD 与 FEVER 等无官方时间顺序、但具备 gold evidence 的 QA
数据。raw loader 只加载文档、query 和离线 gold support，不再自己制造 head/tail 流。

构造流程：

1. 只在离线 benchmark constructor 中，将 gold evidence 文本编码为 sparse TF-IDF；
2. 在 evidence space 中用 MiniBatchKMeans 得到 latent evidence topics；
3. 按 query mass 将 topics 平衡合并为 working-set regimes；
4. evidence family sampler 优先选择能在多次 regime visit 中复用的 support family，
   但每条 exact query 只允许出现一次；
5. 通过 regime schedule 独立控制漂移结构：

```text
factorized_one_shot    前半旧 regime，后半新 regime
factorized_gradual     两端稳定，中间线性迁移
factorized_recurring   regime 周期回访，默认受控主实验
factorized_shuffled    保持 regime 边际频率，打乱可预测转移
factorized_stationary  各 regime 比例保持不变
```

online policy 不会收到 `workload_topic` 或 `workload_regime`，这些 latent constructor
字段只用于离线协议审计。受控 evidence-residency trace 会先冻结 `K_t` 并完成当前请求
计分，再把该请求的 gold evidence ID 作为 post-service access feedback 交给所有策略；
所以它只能更新 `K_{t+1}`。该协议测量的是 oracle evidence-demand residency，不是
端到端 RAG retrieval。构造空间是 evidence TF-IDF，而 DRIP 的冷库 topic directory
来自独立 dense semantic pages，因此避免用同一标签同时构造漂移和直接通知策略。

代码：

```text
experiments/direct/loaders.py          raw direct QA loaders
experiments/hidden/loaders.py          raw hidden multi-hop QA loaders
experiments/agent/loaders.py           real access/action trace loaders
experiments/agent/mtrag_loader.py      official MT-RAG conversation loader
experiments/common/factorized_workload.py      完整受控 QA 构造器
```

## 3. 必须随结果报告的构造统计

每个 JSON 的 `config` 都保存：

```text
stream_sampling       exact-query 重复率，主实验必须为 0
warmup_audit          warm-up/evaluation overlap，必须为 0
support_reuse         历史 support 复用与相邻窗口 overlap
query_drift           query centroid shift 与 regime JS divergence
workload_factors      drift magnitude、within-regime reuse、转移可预测性
factorized_construction  topic/regime/family 构造明细
temporal_sampling     自然日志的时间跨度覆盖
```

这套构造不是为了保证 DRIP 胜过 LRU。它把三个因素分开报告：跨阶段 evidence overlap
决定 drift magnitude，阶段内 evidence reuse 决定缓存是否有价值，重复 transition
决定预取是否可能。自然流用于外部有效性，受控流用于机制验证。

为避免只挑一个有利的 recurring 流，论文主实验必须同时满足：

1. recurring 与 matched shuffled 使用相同数据量、regime 边际和 cache ratio；
2. 至少报告一个自然 chronological workload；
3. 每个 workload 报告上述 factor diagnostics 和多个随机 seed；
4. `k` 与 replacement target 只做全局选择，不按数据集分别调参；
5. 若 DRIP 只在高 reuse、可重复 transition 上领先，应把它写成适用条件，不写成
   对任意漂移普遍领先。

## 4. 最小复现命令

从项目根目录运行：

```bash
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python

# 受控 recurring evidence-regime workload
$PY experiments/direct/run.py \
  --datasets squad --n-source 4000 \
  --n-windows 50 --window-size 300 \
  --workload factorized_recurring \
  --warmup-windows 3 --kb-pool-ratio 0.10 \
  --strategies LRU DRIP \
  --output drip_squad_recurring_r1000.json

# StreamingQA 官方时间流
$PY experiments/direct/run.py \
  --datasets streamingqa_official \
  --n-windows 50 --window-size 500 \
  --workload natural_temporal --temporal-sampling window_span \
  --warmup-windows 3 --kb-pool-ratio 0.10 \
  --strategies LRU DRIP \
  --output drip_streamingqa_natural_r1000.json
```

结果分别写入对应 runner 目录下的 `data/`。
