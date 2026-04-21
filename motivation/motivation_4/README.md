# Motivation 4: Query Distribution Drift 实验

## 目录结构

```
motivation_4/
├── config.py          # 全局常量、路径、数据集配置、策略超参数
├── loaders.py         # 数据集加载器（5个loader）
├── strategies.py      # 5种KB更新策略实现
├── utils.py           # 共享工具函数（嵌入、聚类、评估）
├── run.py             # 统一实验入口（CLI）
├── cache/             # 嵌入向量缓存（.npy）
├── data/              # 实验结果JSON
└── figures/           # 生成的可视化图表
```

## 运行方式

```bash
# 默认：20窗口 × 50/25 queries，sudden drift，3数据集5策略
python run.py

# 50窗口扩展实验（使用完整train+dev数据）
python run.py --n-windows 50 --window-size 50 --expanded

# 只跑特定数据集/策略
python run.py --datasets hotpotqa --strategies Static QueryDriven Oracle

# Gradual drift 模式
python run.py --drift gradual
```

---

## 模块详解

### config.py — 全局配置

| 变量 | 说明 |
|---|---|
| `SEED=42` | 全局随机种子，确保可复现 |
| `EMBED_MODEL='all-MiniLM-L6-v2'` | Sentence-BERT 嵌入模型（384维） |
| `SF_HIT_THRESH=0.55` | 余弦相似度阈值，判定文档是否与query相关（Supporting Fact 匹配） |
| `K_LIST=[1,5,10,20]` | Recall@K 的 K 值列表 |
| `DATASET_CONFIGS` | 每个数据集的窗口数、窗口大小、聚类数、KB预算比例等 |
| `DOC_ARRIVE=80, DOC_ADD_CAP=40` | DocArrival策略的新文档到达数和每窗口最大添加数 |
| `EDIT_BATCH=30` | KnowledgeEdit策略每窗口随机替换的文档数 |
| `QD_TOP_K=50, QD_REPLACE_CAP=60` | QueryDriven策略的候选池大小和每窗口最大替换数 |
| `STRATEGY_ORDER` | 策略执行和显示顺序 |
| `STRATEGY_STYLES` | 各策略在图表中的颜色、标记和线型 |

### loaders.py — 数据集加载

提供5个加载函数，统一返回 `(doc_pool, queries, title_to_idx)` 三元组：

| 函数 | 数据集 | 说明 |
|---|---|---|
| `load_hotpotqa()` | HotpotQA validation | 从 distractor 格式解析context paragraphs和supporting facts |
| `load_2wikimultihopqa()` | 2WikiMultihopQA dev | 解析 dev.json 的 context + supporting_facts |
| `load_musique()` | MuSiQue dev | 解析 `paragraphs` 字段，标记 `is_supporting` |
| `load_musique_expanded(n_source)` | MuSiQue train+dev | 合并训练集和验证集，用于50窗口大规模实验 |
| `load_2wiki_expanded(n_source)` | 2Wiki train+dev | 合并训练集和验证集，pool可达38万+ |

每个 `doc_pool` 元素: `{'doc_id': str, 'title': str, 'text': str}`
每个 `queries` 元素: `{'qidx': int, 'question': str, 'sf_titles': [str, ...]}`

### utils.py — 工具函数

#### `compute_embeddings(doc_pool, queries, tag)`
用 SentenceTransformer 编码所有文档和query为384维向量。结果缓存到 `cache/{tag}_*.npy`，避免重复计算。

#### `cluster_and_build_stream(queries, query_embs, cfg, drift_mode)`
**参考论文思路：** 用 K-Means 对 query 向量聚类，模拟真实场景中 query 分布随时间漂移的现象。

- **Sudden drift（默认）：** 前半段从"头部"聚类（最大的几个）采样，后半段切换到"尾部"聚类（最小的几个），模拟话题突变。
- **Gradual drift：** 头部比例线性递减，尾部比例线性递增，模拟渐进式话题漂移。

返回 `(stream, centroids, head_set)`:
- `stream`: 按窗口顺序排列的query列表
- `centroids`: 聚类中心向量
- `head_set`: 头部聚类的query索引集合

#### `focus_pool(doc_pool, title_to_idx, doc_embs, stream)`
将 doc_pool 缩减为只包含 stream 中出现的 supporting fact 文档 + 部分随机噪声文档。用于小数据集避免KB过大而recall过低。

#### `head_biased_init_kb(doc_pool, doc_embs, centroids, head_set, kb_budget, stream)`
**初始化KB的核心逻辑：** 优先放入头部（前半段）query 的 supporting facts，再用与头部聚类中心最相似的文档填满预算。这确保：
- 前半段（drift前）recall较高
- 后半段（drift后）recall自然下降，体现distribution drift的影响

#### `recall_at_k(kb, window_queries, d2p, doc_embs, title_to_idx, query_embs)`
对当前窗口内的每个 query，在 KB 中用余弦相似度检索 top-K 文档，计算是否命中 supporting fact title（阈值 `SF_HIT_THRESH=0.55`）。返回各 K 值的平均 Recall。

### strategies.py — KB更新策略

所有策略继承 `BaseStrategy`，实现 `step(window_queries, window_query_embs, window_idx)` 方法：

| 策略 | 类名 | 说明 |
|---|---|---|
| **Static** | `Static` | 不做任何更新，KB始终保持初始状态。作为 baseline |
| **DocArrival** | `DocArrival` | 每窗口从 pool 中随机添加 `DOC_ARRIVE` 个新文档候选，选最相似的 `DOC_ADD_CAP` 个加入KB。模拟新文档自然流入 |
| **KnowledgeEdit** | `KnowledgeEdit` | 每窗口随机替换KB中 `EDIT_BATCH` 个文档为pool中随机文档。模拟知识定期批量更新 |
| **QueryDriven** | `QueryDriven` | 用当前窗口query的平均嵌入评估KB相关性，替换KB中与当前query最不相关的文档为pool中最相关的。模拟基于用户反馈的主动KB维护 |
| **Oracle** | `Oracle` | 后半段（drift后）直接将所有tail query的supporting facts加入KB。代表理论上限（完美知道未来query需要什么） |

### run.py — 实验入口

- `run_dataset()`: 对一个数据集执行完整实验流程（加载→嵌入→构造stream→初始化KB→逐窗口评估5种策略）
- `print_summary()`: 格式化打印 Recall@K 的 H1/H2 半段对比表
- `generate_figures()`: 生成 per-window Recall@5 曲线图（PDF + PNG）
- `main()`: CLI 解析，支持 `--n-windows`, `--expanded`, `--drift`, `--datasets`, `--strategies` 等参数

---

## 最新实验结果

### 20窗口 Sudden Drift（默认配置）

#### HotpotQA (pool=9,783, KB=1,000, stream=20×50)

| Strategy | R@5 H1 | R@5 H2 | Δ | Cost |
|---|---|---|---|---|
| Static | 23.6% | 0.5% | -23.1 | 0 |
| DocArrival | 21.8% | 2.2% | -19.6 | 265 |
| KnowledgeEdit | 22.6% | 2.2% | -20.4 | 513 |
| QueryDriven | 22.2% | 7.5% | -14.7 | 731 |
| Oracle | 23.6% | 74.5% | +50.9 | 995 |

#### 2WikiMultihopQA (pool=6,119, KB=600, stream=20×25)

| Strategy | R@5 H1 | R@5 H2 | Δ | Cost |
|---|---|---|---|---|
| Static | 26.1% | 0.5% | -25.6 | 0 |
| DocArrival | 24.7% | 2.1% | -22.6 | 160 |
| KnowledgeEdit | 21.7% | 3.0% | -18.7 | 593 |
| QueryDriven | 24.6% | 2.7% | -21.9 | 626 |
| Oracle | 26.1% | 60.9% | +34.8 | 590 |

#### MuSiQue (pool=9,838, KB=1,000, stream=20×25)

| Strategy | R@5 H1 | R@5 H2 | Δ | Cost |
|---|---|---|---|---|
| Static | 21.3% | 2.3% | -19.0 | 0 |
| DocArrival | 20.4% | 2.7% | -17.7 | 271 |
| KnowledgeEdit | 18.6% | 1.4% | -17.2 | 578 |
| QueryDriven | 21.2% | 8.8% | -12.4 | 464 |
| Oracle | 21.3% | 55.1% | +33.8 | 900 |

### 关键发现

1. **Distribution drift 造成严重性能退化**：Static baseline 在 drift 后 Recall@5 从 20-26% 骤降至 0.5-2.3%
2. **被动策略效果有限**：DocArrival 和 KnowledgeEdit 仅带来微弱改善（+1-2%），因为它们不感知 query 分布变化
3. **QueryDriven 是最优可行策略**：利用当前 query 信号主动更新 KB，在 HotpotQA 和 MuSiQue 上将 H2 recall 提升至 7-9%
4. **Oracle 上限巨大**：完美预知下 recall 可达 55-75%，说明 KB 维护的潜力空间极大
5. **motivation 明确**：需要一种能感知 query 分布变化并自适应更新 KB 的机制
