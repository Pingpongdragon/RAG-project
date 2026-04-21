# Motivation 1 — Single-Hop Sanity Check

> **目的**：在进入多跳 KB 漂移实验（Motivation 2，目录 `motivation_4/`）之前，先在**单跳检索任务**上验证 *QueryDriven (QD) 的稠密对齐信号是真实有效的*。这样后续 Motivation 2 中 QD 在多跳上的失败才能被解读为"任务复杂度问题"而非"QD 信号本身有问题"。

---

## 1. 实验设置

| 项目 | 值 |
|------|----|
| 数据集 | **HotpotQA `comparison` 子集**（train+dev，共 18,943 条；按 SEED=42 抽样 8,000） |
| Pool | 51,366 unique titles |
| Queries | 8,000（平均 2 个 SF/query；均为 comparison 类型，单跳） |
| 流长度 | 50 windows × 50 queries = 2,500（head=1,249 / tail=1,251） |
| KB 预算 | `max(300, ⌈head_ctx_docs × 1.2 / 50⌉ × 50)` ≈ 13k–18k |
| Embedding | `all-MiniLM-L6-v2` |
| 主指标 | **KB Coverage** = `|effective_KB ∩ window_gold_SFs| / |window_gold_SFs|` |
| 辅助指标 | Recall@5（仅作 sanity，不作为对比依据，因为受 retriever 影响） |

**为什么用 comparison 子集做"单跳"？**
- Comparison 题型要求比较两个实体（"哪部电影更早？"），**两个 SF 段落都可以由原始 query 直接对齐**（实体词在 query 中明示），不需要"先找实体A，再桥接到实体B"的二阶推理。
- 与 Motivation 2 共享同一个 pool/embedding/evaluator → 严格控制变量。

**为什么主指标用 Coverage 而不是 Recall？**
- Coverage 只衡量"KB 选择质量"，与具体的 retriever（dense / BM25 / hybrid / rerank）正交。
- Recall 同时受 KB 选择 ⊕ retriever 排序两层影响。我们的论点是 **KB 写入策略**，不是 retriever 升级，所以选 Coverage 作为主轴。
- Recall@5 仍记录在 JSON 中作为下游 sanity check。

---

## 2. 关键结果

### Sudden drift（W26 后骤变到 tail 簇）

```
            Strategy | Cov H1 Cov H2     Δ |  CumH1  CumH2     Δ | R@5 H1 R@5 H2 | Writes
              Static |  97.1%   9.2% -87.9 |  96.3%  68.3% -28.0 |  80.0%   7.6% |      0
          RandomFIFO |  83.0%  16.5% -66.5 |  83.4%  46.6% -36.8 |  68.8%  14.9% |   7497
          DocArrival |  96.9%   9.2% -87.7 |  96.0%  67.9% -28.1 |  79.9%   7.6% |    105
       KnowledgeEdit |  84.8%  11.5% -73.3 |  84.6%  51.0% -33.6 |  70.7%  10.2% |   9757
       OnDemandFetch |  97.1%  22.3% -74.8 |  96.7%  74.8% -21.9 |  79.5%  19.8% |      0
    LogDrivenArrival |  95.9%  13.7% -82.2 |  95.3%  65.2% -30.1 |  78.5%  11.7% |   1800
        QueryDriven  |  95.2%  22.4% -72.8 |  94.7%  62.0% -32.7 |  77.8%  19.8% |   4761
              Oracle | 100.0% 100.0%  +0.0 |  60.5%  29.7% -30.8 |  81.8%  86.0% | 151901
```

**读法**：
- **QD H2 Cov = 22.4%**，**比 Static / DocArrival (9.2%) 高出 +13.2pp**，与 OnDemandFetch (22.3%) 并列第一（两者机制不同：OnDemand 是 query-time 全池检索，QD 是 background 写入）。
- DocArrival 几乎不写（105 writes）→ 自我节流（详见 motivation_4 ETHOS 笔记），表现等同 Static。
- **结论：在单跳任务上，QD 的稠密对齐信号确实能把 KB 移向 tail 簇。**

### Gradual drift（W11 起 tail 比例线性上升）

```
            Strategy | Cov H1 Cov H2     Δ |  CumH1  CumH2     Δ | R@5 H1 R@5 H2 | Writes
              Static |  97.1%  54.2% -42.9 |  96.4%  86.8%  -9.6 |  78.3%  43.9% |      0
        RandomFIFO  |  89.4%  47.4% -42.0 |  88.3%  67.8% -20.5 |  72.6%  40.1% |   6492
        DocArrival  |  96.9%  54.1% -42.8 |  96.2%  86.6%  -9.6 |  78.2%  43.8% |     80
       KnowledgeEdit|  88.4%  47.3% -41.1 |  87.4%  70.4% -17.0 |  71.5%  39.6% |   9636
       OnDemandFetch|  97.1%  58.9% -38.2 |  96.7%  89.1%  -7.6 |  78.0%  48.2% |      0
    LogDrivenArrival|  96.2%  54.3% -41.9 |  95.7%  83.9% -11.8 |  77.3%  43.4% |   1800
        QueryDriven |  95.9%  56.2% -39.7 |  95.5%  82.4% -13.1 |  77.1%  45.7% |   4018
              Oracle | 100.0% 100.0%  +0.0 |  70.8%  51.9% -18.9 |  80.9%  83.7% | 169472
```

**读法**：
- 渐变下 head/tail 长期共存，Static 也能维持 54%（因为 init KB 已含约 26% tail）；QD 仍**优于 Static +2.0pp**，但优势远小于骤变。
- 这说明 QD 的价值与"漂移速度 × KB 剩余预算"成正比 —— 与多跳实验的发现一致。

### 图

`figures/coverage_50w_sudden.pdf` 与 `figures/coverage_50w_gradual.pdf`：上排为 per-window cov，下排为 cumulative cov。

---

## 3. 与 Motivation 2 的对接

| 维度 | Motivation 1（本文，单跳 comparison） | Motivation 2（`motivation_4/`，多跳 bridge） |
|------|-----|-----|
| QD vs Static H2 Δ | **+13pp（sudden）** | **0 ~ −2pp（HotpotQA bridge / 2Wiki / MuSiQue）** |
| 结论 | **QD 信号 *有效*** | **QD 信号 *不足以*应对桥接缺口** |
| 因果 | 稠密对齐能找到 SF | 桥接 SF 与 query 不在同一稠密邻域 |

⇒ 两个实验放在一起：QD 不是"坏方法"，而是"在单跳上够用、在多跳上需要图扩散补齐"的方法。这正是后续引入 HippoRAG-style PPR 作为 V2 升级的动机。

---

## 4. 复现

```bash
cd /data/jyliu/RAG-project/motivation/motivation_1
python run.py --n-windows 50 --window-size 50 --drift sudden  --output results_50w_sudden.json
python run.py --n-windows 50 --window-size 50 --drift gradual --output results_50w_gradual.json
```

输出：
- `data/results_50w_{sudden,gradual}.json` — 完整 per-window 数值
- `figures/coverage_50w_{sudden,gradual}.{pdf,png}` — 主图

---

## 5. 已知限制 / 注意事项

- **Oracle CumH2 < CumH1**：Oracle 每窗重建 KB（只装当窗 gold），所以累计交集会随累计 gold 增长被稀释。这是 Oracle 设计本身的产物，不影响 *per-window* coverage = 100% 的上界含义。
- **KB 预算策略与 Motivation 2 一致**（`kb_head_mult=1.2`），保证 init KB 容得下绝大多数 head SF，又留少量 tail 余量逼出策略差异。
- 共享 `strategies.py / utils.py / config.py 中的策略超参` 与 Motivation 2 完全一致（`WRITE_CAP=200`, `PROBE_TOPK=50` 等），便于横向比较。
