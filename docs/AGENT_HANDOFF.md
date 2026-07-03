# Agent Handoff — RAG Project Motivation Experiments

> 创建时间：2026-05-12，用于换服务器后继续实验。  
> GitHub: https://github.com/Pingpongdragon/RAG-project.git  branch: master

---

## 0. 快速上手

```bash
git clone https://github.com/Pingpongdragon/RAG-project.git
cd RAG-project

# Python 环境（需安装依赖）
conda activate <your_env>
pip install sentence-transformers ir_datasets torch numpy matplotlib tqdm

# TREC-COVID 数据会自动下载到 ~/.ir_datasets/
```

---

## 1. 项目背景

回应 reviewer 对 motivation 实验的三点拒稿意见：
1. **weak baselines**：原来用 RandomFIFO/Static 等太弱 → 换成 LRU / GPTCacheStyle / MemGPTStyle
2. **no real temporal data**：只有合成漂移 → 新增 TREC-COVID 真实时序数据集
3. **no system-level metrics**：没有延迟/开销数据 → 新增 step_latency / step_overhead

---

## 2. Python 环境

- 路径（原服务器）：`/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python3`
- 关键包：`sentence-transformers`, `ir_datasets`, `torch`, `faiss-cpu`, `numpy`, `matplotlib`
- **GPU 注意**：`utils.py` 已强制 `device='cpu'`，有 GPU 可改回 `device='cuda'`

---

## 3. 已完成的代码改动（均已 commit a98799a）

### motivation_1/（mo1：KB update strategy 对比）

| 文件 | 状态 | 说明 |
|------|------|------|
| `strategies.py` | ✅ 已改 | 删除 RandomFIFO/OnDemandFetch/LogDrivenArrival；新增 LRU、GPTCacheStyle、MemGPTStyle |
| `config.py` | ✅ 已改 | 更新 STRATEGY_ORDER/LABELS/STYLES；新增 trec_covid 数据集配置 |
| `run.py` | ✅ 已改 | 新增 step_latency/step_overhead 计时；路由 TREC-COVID loader |
| `loaders_temporal.py` | ✅ 新建 | TREC-COVID loader，返回 (doc_pool, queries, title_to_idx) |
| `utils.py` | ✅ 已改 | `SentenceTransformer(EMBED_MODEL, device='cpu')` |

### STRATEGY_FACTORIES keys（motivation_1）
```
['Static', 'DocArrival', 'KnowledgeEdit', 'LRU', 'GPTCacheStyle', 'MemGPTStyle', 'DRIP-Dense', 'Oracle']
```

### motivation.tex
- ✅ 删除 Para 5（7数据集枚举段）
- ✅ Para 6 开头改为 "Across both regimes"

---

## 4. 尚未完成（待新服务器继续）

### 4.1 冒烟测试（优先）
```bash
cd RAG-project/motivation/motivation_1
python3 run.py \
  --datasets fever \
  --n-windows 4 --window-size 20 \
  --drift gradual \
  --strategies Static DocArrival LRU GPTCacheStyle MemGPTStyle DRIP-Dense Oracle
```
预期：4 个窗口正常跑完，打印 summary 表格，JSON 写入 `data/` 目录。

### 4.2 motivation_2/ 同步改动（❌ 未做）

motivation_2 是 mo2 实验（KB retrieval 覆盖率 vs 漂移），需做相同 baseline 替换：

```bash
# 参考 motivation_1/strategies.py 的改法，对 motivation_2/strategies.py 做同样修改：
# 1. 删除 RandomFIFO / OnDemandFetch / LogDrivenArrival
# 2. 新增 LRU / GPTCacheStyle / MemGPTStyle
# 3. 更新 STRATEGY_FACTORIES
# 4. motivation_2/utils.py 加 device='cpu'（或 'cuda'）
```

### 4.3 全量实验运行（❌ 未做）
```bash
# mo1 全量（gradual drift，主要数据集）
cd motivation/motivation_1
python3 run.py \
  --datasets fever 2wiki hotpot musique \
  --n-windows 50 --window-size 100 \
  --drift gradual \
  --strategies Static DocArrival LRU GPTCacheStyle MemGPTStyle DRIP-Dense Oracle

# mo1 TREC-COVID（真实时序）
python3 run.py \
  --datasets trec_covid \
  --n-windows 20 \
  --drift gradual \
  --strategies Static DocArrival LRU GPTCacheStyle MemGPTStyle DRIP-Dense Oracle
```

### 4.4 重新生成图表（❌ 未做）
```bash
cd motivation/motivation_1
python3 plot.py   # 生成 figures/mo1_gradual.pdf/png 等
```

### 4.5 motivation.tex 补充（❌ 未做）
- 在 §1 约 Para4 后新增 ~250 词段落，解释 QD 框架的 **Trigger / Selection / Action** 三步机制（reviewer #3 要求）
- 更新图表引用（新 baselines 出现在图注中）

---

## 5. 关键常量

```python
SF_HIT_THRESH = 0.55   # semantic hit 阈值
WRITE_CAP     = 200    # KB 最大写入条数
PROBE_TOPK    = 50     # 检索 top-k
SEED          = 42
EMBED_MODEL   = 'all-MiniLM-L6-v2'
```

---

## 6. 三个新 Baseline 的实现思路（供验证）

| Baseline | 核心逻辑 |
|----------|----------|
| **LRU** | 维护访问时间戳；新文档入库时淘汰最久未访问的旧条目 |
| **GPTCacheStyle** | 用语义相似度做去重缓存；相似度 > 阈值则跳过写入 |
| **MemGPTStyle** | 重要性分 = access_freq × recency_decay；低分条目优先淘汰 |

---

## 7. 文件结构（motivation_1）

```
motivation/motivation_1/
├── run.py              # 主运行脚本
├── strategies.py       # 所有 KB 更新策略
├── config.py           # 常量 + 可视化配置
├── loaders.py          # 原有数据集 loader
├── loaders_temporal.py # TREC-COVID loader（新建）
├── utils.py            # embedding 工具（已改 cpu）
├── plot.py             # 画图脚本
├── data/               # 实验结果 JSON 输出
└── figures/            # 生成图表
```

