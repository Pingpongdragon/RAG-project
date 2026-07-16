# DRIP-WorkingSet Smoke Test (2026-07-13)

## 目的

本轮只验证新策略端到端可运行、无未来泄漏、recency fallback 能减少无效写入、
hidden bundle 能全写或全拒绝。4 个窗口的结果不能作为论文主结果。

## 自动测试

项目环境没有安装 `pytest`，因此直接导入并执行全部 `algorithms/drip/tests/test_*.py`
中的测试函数：

```text
79 DRIP tests passed
```

新增边界测试覆盖：首次 A→B 不预取、重复 A→B 才预取、recency 权重更新、Top-1
fallback、二元 bundle 整体准入和预算不足时零写入。

## Direct smoke

命令：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python experiments/direct/run.py \
  --datasets hotpotqa_comparison --drift full_gradual \
  --n-source 400 --n-stream-queries 100 --n-windows 4 --window-size 25 \
  --kb-pool-ratio 0.1 --drip-ablation \
  --strategies LRU DRIP DRIP-WorkingSet \
  --output smoke_working_set_hotpot_fallback_4w25.json
```

结果文件：
`experiments/direct/data/smoke_working_set_hotpot_fallback_4w25.json`

| Method | Has-Answer | AMAT | Full Recall@5 | Replacements |
|---|---:|---:|---:|---:|
| LRU | 4.0 | 10.60 | 16.0 | 71 |
| DRIP | 4.0 | 10.60 | 15.5 | 87 |
| DRIP-WorkingSet | 4.0 | 10.60 | 15.5 | 77 |

窗口 0--1 两个 expert 尚未分出优劣，reactive probe 为 Top-8；窗口 2--3 recency
权重更高，probe 自动收缩为 Top-1。Has-Answer/Recall 不变时，相对当前 DRIP 少 10
次 replacements，并接近 LRU 的 71 次。该趋势需要在完整流和多 seed 上确认。

四个窗口均没有可靠重复 topic transition，因此 topic candidates 和 prefetch writes
均为 0，符合保守 fallback 预期。

## Hidden smoke

命令：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python experiments/hidden/run.py \
  --datasets 2wikimultihopqa --expanded --q-type bridge_comparison \
  --workload topic_shift_bridge_reuse --retrieval graph \
  --n-source 500 --n-stream-queries 100 --n-windows 4 --window-size 25 \
  --kb-pool-ratio 0.1 --drip-ablation \
  --strategies DRIP DRIP-WorkingSet \
  --output smoke_working_set_2wiki_hidden_fallback_4w25.json
```

结果文件：
`experiments/hidden/data/smoke_working_set_2wiki_hidden_fallback_4w25.json`

| Method | Has-Answer | AMAT | Full Recall@5 | Replacements |
|---|---:|---:|---:|---:|
| DRIP | 1.0 | 10.90 | 16.8 | 93 |
| DRIP-WorkingSet | 1.0 | 10.90 | 16.8 | 94 |

窗口 1 产生 5 个 relation bundles：1 个整体准入并写入 2 个成员，4 个整体拒绝；
没有单成员部分写入。这个抽样未找到足够的 topic-shift groups，runner 回退到了
multi-agent bridge reuse，因此这里只证明代码链完整，不能据此判断 hidden 效果。

## 当前结论

1. Recency fallback 已经能在不损失本 smoke 质量的情况下压低 replacements。
2. Topic predictor 在没有重复可学习转移时会 abstain，不会强行预取。
3. Hidden writer 已满足二元 evidence bundle 的原子准入语义。
4. 下一步必须运行设计文档中的完整 temporal/direct/hidden/agent 四组实验，并报告
   恢复 lag、prefetch precision、all-support Has-Answer 和 replacements Pareto。

