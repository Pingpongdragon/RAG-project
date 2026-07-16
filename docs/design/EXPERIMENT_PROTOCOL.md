# Current Stream Construction Protocol

本文件只描述当前可运行的数据流协议。旧的 query-embedding KMeans、head/tail
混合比例、`--drift sudden/gradual`、手工 bridge-reuse 流和 future-gold 初始化均已
从 runner 删除。

## 1. 唯一入口

两个 runner 使用相同协议：

```text
experiments/direct/run.py
experiments/hidden/run.py
```

推荐参数是 `--workload auto`：

| 数据类型 | `auto` 选择 | 是否重排 |
|---|---|---|
| 带 `preserve_order` 的真实日志 | `natural_temporal` | 否 |
| 普通 QA / agent 数据 | `factorized_recurring` | 是，仅离线构造 |

显式指定与数据协议冲突的 workload 会直接报错，不再静默覆盖。

## 2. 自然时间流

实现位于 `experiments/common/stream_protocol.py::chronological_sample`。

- `prefix`：使用日志开头的连续事件；
- `window_span`：在完整时间范围选择连续窗口块，保留窗口内 locality；
- 前 `warmup_windows * window_size` 条只用于初始缓存；
- 后续恰好 `n_windows * window_size` 条用于评估；
- warm-up 与 evaluation 的 exact query 必须无交集。

StreamingQA 只使用官方 `question_ts` loader。旧的“从问题文本抽年份再分箱”代理
已经删除。

## 3. 受控 Evidence Drift

实现位于 `experiments/common/factorized_workload.py`，构造标签不提供给在线策略。

1. 用 gold evidence 文本的 sparse TF-IDF 表示聚类 latent evidence topics；
2. 将 topics 平衡映射为 working-set regimes；
3. 从初始 regime 先取互不重复的 warm-up query；
4. 每个 evaluation query 最多出现一次；
5. 通过不同 regime 日程独立控制漂移与可预测性。

| workload | 日程含义 |
|---|---|
| `factorized_recurring` | regime 周期复现，用于检验可预测转移与 evidence reuse |
| `factorized_shuffled` | 保持 regime 边际频率，打乱转移顺序 |
| `factorized_one_shot` | 旧 working set 到新 working set 的一次转移 |
| `factorized_stationary` | 所有 regime 的稳定混合对照 |

证据家族默认规则：单 support 按完整 support；多 support direct query 按共享 anchor。
复用的是 evidence family，不是重复 query 文本。

## 4. 初始缓存与容量

所有策略共用 `causal_prefix_init_kb`：只根据 warm-up query 的冷库检索结果初始化，
不读取评估流的 gold support 或未来 `ctx_titles`。

```text
B = round(kb_pool_ratio * |D|)
```

默认 `kb_pool_ratio=0.1`，`--kb-budget` 可覆盖绝对容量。

## 5. 必须保存的构造审计

每个结果 JSON 的 `config` 保存：

- `workload` 与 `initialization`；
- `stream_sampling`：exact-query 重复率；
- `warmup_audit`：warm-up/evaluation overlap；
- `support_reuse`：阶段内 evidence reuse；
- `query_drift`：仅作事后 query-space 诊断；
- `workload_factors`：跨 regime evidence overlap、漂移幅度与转移可预测性；
- `factorized_construction` 或 `temporal_sampling`。

## 6. 示例命令

受控 direct workload：

```bash
python experiments/direct/run.py \
  --datasets hotpotqa_comparison \
  --n-windows 50 --window-size 50 \
  --workload factorized_recurring \
  --kb-pool-ratio 0.1 --warmup-windows 3
```

自然时间 workload：

```bash
python experiments/direct/run.py \
  --datasets streamingqa_official \
  --n-windows 50 --window-size 500 \
  --workload natural_temporal \
  --temporal-sampling window_span \
  --kb-pool-ratio 0.1 --warmup-windows 3
```

容量 sweep 的统一入口是 `motivation/run_cache_ratio_sweep.py`。
