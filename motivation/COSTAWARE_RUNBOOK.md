# CostAwareDRIP 实验 Runbook

主指标明确为 4 个：

| Metric | 用途 | 是否 ARC 原文主指标 |
|---|---|---|
| **Has-Answer Rate↑** | 热缓存里是否已经有回答所需 support | 是 |
| **AMAT↓** | 平均访问时间；经典形式是 `T_hot + miss_rate * T_miss` | 是 |
| **Recall@5↑** | 从 effective cache 检索 top-5 的质量 | 否，我们补充 |
| **Replacements↓** | cache 换入换出次数；固定容量下每次 admission 通常对应一次 eviction | 否，我们补充 |

诊断指标不放主表，只用于解释原因：

| Diagnostic | 解释 | 什么时候看 |
|---|---|---|
| **Replacement Count↓** | cache replacement 总次数；固定容量下通常等于写入新文档并换出旧文档的次数 | 主方法是否频繁抖动 |
| **Eviction Count↓** | 被换出的 resident 文档数量 | 分析替换压力；CostAwareDRIP 会单独记录 |
| **Cache Writes / Write Traffic↓** | 写入 cache 的流量；在本实验里等同 `cache_writes` / `update_cost` | 系统维护成本、存储写放大 |
| **Churn Rate↓** | replacement 的归一化变化率，即每窗口替换掉多少比例的 KB | 是否频繁换入换出、cache 是否不稳定 |
| **Miss Rate↓ / Hit Rate↑** | `miss_rate = 1 - has_answer_rate`；`hit_rate = has_answer_rate` | 解释 AMAT 的来源 |


这份文档记录早期 CostAwareDRIP 实验。正式实验入口现分为 direct QA、hidden QA
和真实 agent trace 三类协议：

| 实验 | 问题 | Runner | 主数据 |
|---|---|---|---|
| E1 temporal boundary | 真实时间漂移下 LRU/FIFO 是否已经很强？同等质量下谁写得少？ | `experiments/direct/run.py` | StreamingQA temporal；TREC-COVID temporal 可选 |
| E2 direct topic drift | query-visible direct evidence 的 topic drift 下，成本感知 admission 是否更稳？ | `experiments/direct/run.py` | 2Wiki comparison；HotpotQA comparison；MuSiQue direct subset |
| E3 cost/churn stress | KB 更紧或 probe/write 更贵时，谁在质量-成本 frontier 上更好？ | `experiments/direct/run.py` | E2 同数据，扫 `--kb-budget` |
| Hidden evidence | query-hidden evidence 下图扩展和证据补全是否有效？ | `experiments/hidden/run.py` | bridge / bridge_comparison QA |
| Agent access | 真实时序或会话访问中 evidence 是否在请求前驻留？ | `experiments/agent/` | MIND、MT-RAG、Mind2Web |

当前主方法：

```text
CostAwareDRIP
CostAwareDRIP-NoDrift
CostAwareDRIP-NoChurn
```

当前主线暂时不使用 `MultiAgentDriftDetector` 做论证；它保留给旧
`DRIPCore` / hidden diagnostic。

## 统一指标口径

质量指标：

```text
recall@5_h2
has_answer_h2 / has_answer_rate
amat / amat_normalized
miss_rate
l2_accesses_per_query
support_coverage_h2 / support_coverage_rate
kb_coverage_h2 或 cov_h2
```

AMAT 采用经典 cache 口径：

```text
AMAT = T_hot + L2AccessRate * T_miss

默认：
T_hot  = 1
T_miss = 10

L2AccessRate = max(1 - HasAnswerRate, serve_fetches_per_query)
```

其中 `serve_fetches_per_query` 用来处理 OnDemandFetch 这类服务时真的访问
full index 的策略。参数已经配置在：

```text
experiments/direct/config.py
experiments/hidden/config.py
algorithms/cache/params.py
```

默认值：

```text
AMAT_HIT_COST = 1
AMAT_MISS_PENALTY = 10
```

如果要模拟不同系统论文里的延迟设定，可以运行前用环境变量覆盖。

成本指标：

```text
replacement_count    # 推荐论文主表使用：cache replacement 次数
replacement_rate_per_query
replacement_rate_per_window
cache_churn_rate     # 诊断使用：平均每窗口替换掉的 KB 比例
cache_churn_rate_pct
cache_writes         # 同 update_cost，表示写入/换入次数
update_cost          # 旧字段名，保留兼容历史脚本
evictions            # CostAwareDRIP 内部记录的换出次数
churn_rate_mean      # 平均每窗口写入占 KB 的比例
maint_retrieval_cost # 维护阶段从 L2/full pool probe 的成本
serve_retrieval_cost # serve-time fallback 成本，OnDemandFetch 主要看这个
```

Cache replacement / churn 采用统一公式：

```text
R = total number of cache replacements
Q = total number of served queries
T = number of windows
C = cache capacity / KB budget

ReplacementCount = R
ReplacementRate_per_query = R / Q
ReplacementRate_per_window = R / T
CacheChurnRate = R / (T * C)
```

解释：

```text
ReplacementCount 看总换入换出量；
ReplacementRate_per_query 看每个 query 平均触发多少写入；
CacheChurnRate 看每个窗口平均替换了多少比例的 hot cache。
```

论文主表建议只写 `Replacements`，不要写旧字段 `update_cost`。`CacheChurnRate`
放 appendix 或诊断图。

论文解释时优先看：

```text
同等 recall / has-answer / AMAT 下：replacements 更少。
```

不要把 E1 写成 CostAware 必须赢 LRU/FIFO。E1 是 temporal visible drift 的边界：
LRU/FIFO 强是正常的。

## 通用环境

```bash
cd /home/jyliu/RAG-project

export PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CAD_PROBE_TOPK=8
```

`CAD_PROBE_TOPK=8` 是当前默认推荐值：比 12 更省维护检索成本，direct smoke
中质量基本不掉。

## E1: Real Temporal Visible Drift

### E1-a StreamingQA temporal

推荐主跑：

```bash
$PY experiments/direct/run.py \
  --datasets streamingqa_temporal \
  --n-windows 100 \
  --window-size 50 \
  --drift temporal \
  --kb-budget 400 \
  --strategies LRU FIFO TemporalAware RecencyTTL CostAwareDRIP CostAwareDRIP-NoDrift CostAwareDRIP-NoChurn OnDemandFetch Oracle \
  --output costaware_streamingqa_temporal_100w50_kb400.json
```

已跑过的一轮结果文件：

```text
experiments/direct/data/costaware_streamingqa_temporal_100w50_kb400.json
```

这轮的读法：

```text
LRU/FIFO recall 更强，但写入更多；
CostAwareDRIP 写入更少，但质量不一定追上 LRU/FIFO；
NoChurn 往往质量更高，但写入和 churn 更高，正好证明 churn control 是质量-成本取舍。
```

### E1-b TREC-COVID temporal，可选

```bash
$PY experiments/direct/run.py \
  --datasets trec_covid_temporal \
  --n-windows 20 \
  --window-size 5 \
  --drift temporal \
  --strategies LRU FIFO TemporalAware RecencyTTL CostAwareDRIP CostAwareDRIP-NoDrift CostAwareDRIP-NoChurn OnDemandFetch Oracle \
  --output costaware_trec_covid_temporal_20w5.json
```

注意：TREC-COVID 的 qrels 每个 query 有大量 relevant docs，`has_answer` 很容易全低，
区分度不如 StreamingQA。建议作为 appendix / robustness，不做主图。

## E2: Controlled Direct-Evidence Topic Drift

### E2-a 2Wiki comparison，快速 smoke

这是目前最快、最稳的 direct evidence 检查：

```bash
$PY experiments/hidden/run.py \
  --datasets 2wikimultihopqa \
  --expanded \
  --q-type comparison \
  --n-source 800 \
  --n-stream-queries 200 \
  --n-windows 8 \
  --window-size 25 \
  --drift full_gradual \
  --workload cluster_shift \
  --retrieval graph \
  --kb-budget 1250 \
  --strategies ARC LRU FIFO CostAwareDRIP CostAwareDRIP-NoDrift CostAwareDRIP-NoChurn OnDemandFetch Oracle \
  --output costaware_2wiki_comp_direct_8w25_kb1250.json
```

### E2-b 2Wiki comparison，主跑

```bash
$PY experiments/hidden/run.py \
  --datasets 2wikimultihopqa \
  --expanded \
  --q-type comparison \
  --n-source 2000 \
  --n-stream-queries 500 \
  --n-windows 20 \
  --window-size 25 \
  --drift full_gradual \
  --workload cluster_shift \
  --retrieval graph \
  --kb-budget 6250 \
  --strategies ARC LRU FIFO CostAwareDRIP CostAwareDRIP-NoDrift CostAwareDRIP-NoChurn OnDemandFetch Oracle \
  --output costaware_2wiki_comp_direct_20w25_kb6250.json
```

### E2-c HotpotQA comparison，中等规模

```bash
$PY experiments/hidden/run.py \
  --datasets hotpotqa \
  --expanded \
  --q-type comparison \
  --n-source 3500 \
  --n-stream-queries 500 \
  --n-windows 20 \
  --window-size 25 \
  --drift full_gradual \
  --workload cluster_shift \
  --retrieval graph \
  --kb-budget 9600 \
  --strategies ARC LRU FIFO CostAwareDRIP CostAwareDRIP-NoDrift CostAwareDRIP-NoChurn OnDemandFetch Oracle \
  --output costaware_hotpot_comp_direct_20w25_kb9600.json
```

这个规模的 pool 大约二三万；如果没有 embedding cache，BGE-large 编码会比较慢。

### E2-d HotpotQA comparison，对齐图中 pool≈87k

更接近旧图 `(pool≈87,175, KB=9,600)`，但会很慢：

```bash
$PY experiments/hidden/run.py \
  --datasets hotpotqa \
  --expanded \
  --q-type comparison \
  --n-source 18943 \
  --n-stream-queries 1000 \
  --n-windows 40 \
  --window-size 25 \
  --drift full_gradual \
  --workload cluster_shift \
  --retrieval graph \
  --kb-budget 9600 \
  --strategies ARC LRU FIFO CostAwareDRIP CostAwareDRIP-NoDrift CostAwareDRIP-NoChurn OnDemandFetch Oracle \
  --output costaware_hotpot_comp_direct_40w25_pool87k_kb9600.json
```

### E2-e MuSiQue direct-ish

MuSiQue 没有同样干净的 `q-type comparison`，建议当作 direct-ish / robustness：

```bash
$PY experiments/hidden/run.py \
  --datasets musique \
  --expanded \
  --n-source 1200 \
  --n-stream-queries 400 \
  --n-windows 16 \
  --window-size 25 \
  --drift full_gradual \
  --workload cluster_shift \
  --retrieval graph \
  --kb-budget 2500 \
  --strategies ARC LRU FIFO CostAwareDRIP CostAwareDRIP-NoDrift CostAwareDRIP-NoChurn OnDemandFetch Oracle \
  --output costaware_musique_directish_16w25_kb2500.json
```

## E3: Cost/Churn Stress

固定 E2-b 其他参数，只扫：

```text
--kb-budget 1250
--kb-budget 2500
--kb-budget 6250
```

要画的不是单条 recall 曲线，而是 Pareto 图：

```text
x-axis: update_cost / evictions / churn_rate_mean / maint_retrieval_cost
y-axis: recall@5_h2 / has_answer_rate / support_coverage_rate
```

预期解释：

```text
CostAwareDRIP 不一定 raw recall 最高；
NoChurn 可能更高但写得更多；
CostAwareDRIP 的价值是同等质量下更少写、更少 churn，或者同等写入预算下更稳。
```

## Diagnostic: Hidden Evidence

hidden 不删，但先不要放主线：

```bash
$PY experiments/hidden/run.py \
  --datasets 2wikimultihopqa \
  --expanded \
  --q-type bridge_comparison \
  --n-source 2000 \
  --n-stream-queries 500 \
  --n-windows 20 \
  --window-size 25 \
  --drift full_gradual \
  --workload cluster_shift \
  --retrieval graph \
  --kb-budget 6250 \
  --strategies ARC LRU FIFO DRIP-QueryVisible DRIP-QueryHidden CostAwareDRIP Oracle \
  --output costaware_hidden_diagnostic_20w25_kb6250.json
```

## 结果读取

### Direct 结果

```bash
jq '.streamingqa_temporal.summary | {
  LRU:{has:.LRU.has_answer_rate, amat:.LRU.amat, r5_h2:.LRU["recall@5_h2"], repl:.LRU.replacement_count, churn:.LRU.cache_churn_rate_pct},
  FIFO:{has:.FIFO.has_answer_rate, amat:.FIFO.amat, r5_h2:.FIFO["recall@5_h2"], repl:.FIFO.replacement_count, churn:.FIFO.cache_churn_rate_pct},
  CostAwareDRIP:{
    has:.CostAwareDRIP.has_answer_rate,
    amat:.CostAwareDRIP.amat,
    r5_h2:.CostAwareDRIP["recall@5_h2"],
    repl:.CostAwareDRIP.replacement_count,
    churn:.CostAwareDRIP.cache_churn_rate_pct,
    evictions:.CostAwareDRIP.evictions,
    internal_churn:.CostAwareDRIP.churn_rate_mean
  },
  NoDrift:{
    has:."CostAwareDRIP-NoDrift".has_answer_rate,
    amat:."CostAwareDRIP-NoDrift".amat,
    r5_h2:."CostAwareDRIP-NoDrift"["recall@5_h2"],
    repl:."CostAwareDRIP-NoDrift".replacement_count,
    churn:."CostAwareDRIP-NoDrift".cache_churn_rate_pct,
    evictions:."CostAwareDRIP-NoDrift".evictions,
    internal_churn:."CostAwareDRIP-NoDrift".churn_rate_mean
  },
  NoChurn:{
    has:."CostAwareDRIP-NoChurn".has_answer_rate,
    amat:."CostAwareDRIP-NoChurn".amat,
    r5_h2:."CostAwareDRIP-NoChurn"["recall@5_h2"],
    repl:."CostAwareDRIP-NoChurn".replacement_count,
    churn:."CostAwareDRIP-NoChurn".cache_churn_rate_pct,
    evictions:."CostAwareDRIP-NoChurn".evictions,
    internal_churn:."CostAwareDRIP-NoChurn".churn_rate_mean
  }
}' experiments/direct/data/costaware_streamingqa_temporal_100w50_kb400.json
```

### Hidden 结果

把文件名换成你实际输出的 JSON：

```bash
jq '.["2wikimultihopqa"].summary // .hotpotqa.summary // .musique.summary | {
  ARC:{has:.ARC.has_answer_rate, amat:.ARC.amat, r5_h2:.ARC["recall@5_h2"], repl:.ARC.replacement_count, churn:.ARC.cache_churn_rate_pct, maint:.ARC.maint_retrieval_cost},
  LRU:{has:.LRU.has_answer_rate, amat:.LRU.amat, r5_h2:.LRU["recall@5_h2"], repl:.LRU.replacement_count, churn:.LRU.cache_churn_rate_pct, maint:.LRU.maint_retrieval_cost},
  FIFO:{has:.FIFO.has_answer_rate, amat:.FIFO.amat, r5_h2:.FIFO["recall@5_h2"], repl:.FIFO.replacement_count, churn:.FIFO.cache_churn_rate_pct, maint:.FIFO.maint_retrieval_cost},
  CostAwareDRIP:{
    has:.CostAwareDRIP.has_answer_rate,
    amat:.CostAwareDRIP.amat,
    r5_h2:.CostAwareDRIP["recall@5_h2"],
    repl:.CostAwareDRIP.replacement_count,
    churn:.CostAwareDRIP.cache_churn_rate_pct,
    evictions:.CostAwareDRIP.evictions,
    internal_churn:.CostAwareDRIP.churn_rate_mean,
    maint:.CostAwareDRIP.maint_retrieval_cost
  },
  NoDrift:{
    has:."CostAwareDRIP-NoDrift".has_answer_rate,
    amat:."CostAwareDRIP-NoDrift".amat,
    r5_h2:."CostAwareDRIP-NoDrift"["recall@5_h2"],
    repl:."CostAwareDRIP-NoDrift".replacement_count,
    churn:."CostAwareDRIP-NoDrift".cache_churn_rate_pct,
    evictions:."CostAwareDRIP-NoDrift".evictions,
    internal_churn:."CostAwareDRIP-NoDrift".churn_rate_mean,
    maint:."CostAwareDRIP-NoDrift".maint_retrieval_cost
  },
  NoChurn:{
    has:."CostAwareDRIP-NoChurn".has_answer_rate,
    amat:."CostAwareDRIP-NoChurn".amat,
    r5_h2:."CostAwareDRIP-NoChurn"["recall@5_h2"],
    repl:."CostAwareDRIP-NoChurn".replacement_count,
    churn:."CostAwareDRIP-NoChurn".cache_churn_rate_pct,
    evictions:."CostAwareDRIP-NoChurn".evictions,
    internal_churn:."CostAwareDRIP-NoChurn".churn_rate_mean,
    maint:."CostAwareDRIP-NoChurn".maint_retrieval_cost
  }
}' experiments/hidden/data/costaware_2wiki_comp_direct_8w25_kb1250.json
```

## 文件夹读法

```text
experiments/direct/
  run.py              temporal 和 direct-topic 统一入口
  loaders_temporal.py StreamingQA / TREC-COVID temporal loader
  loaders.py          direct QA loaders
  data/               direct JSON outputs

experiments/hidden/
  run.py              hidden-evidence 和 agent 实验入口
  loaders.py          HotpotQA / 2Wiki / MuSiQue loaders
  data/               hidden JSON outputs

algorithms/drip/cache_manager/cost_aware.py
  CostAwareDRIP 主方法逻辑

algorithms/drip/cache_manager/cost_aware_config.py
  CostAwareDRIP 参数与主公式

algorithms/drip/cache_manager/query_router.py
  visible/direct vs hidden router

algorithms/drip/detection/multi_agent_drift.py
  旧 DRIPCore/hidden diagnostic detector，当前主线先放着
```
