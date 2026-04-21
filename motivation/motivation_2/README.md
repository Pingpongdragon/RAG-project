# Motivation 2 — 现行持续 RAG 策略在多跳分布漂移下为何全部失败

> ** **Motivation 2** 引用，前置实验为 [`../motivation_1/`](../motivation_1/README.md) — 单跳 sanity 验证 QD 信号本身有效。

## TL;DR

在 HotpotQA / 2WikiMultihopQA / MuSiQue（多跳 QA，扩展池：33k / 385k / 84k；3,500 子采查询；50 窗口 × 50 查询）上，**突变**（W26 跳簇）和**渐变**（线性插值）两种漂移模式下，**给每种策略充分预算**（每窗口最多写 200 篇 KB、每查询取 50 候选）后：

- **HotpotQA** 上排序如预期：`Static < DocArrival < OnDemandFetch < KnowledgeEdit ≈ RandomFIFO < QueryDriven << Oracle`。突变 H2：QD 21.6% vs Static 9.2%（**+135%**）；渐变 H2：QD 43.5% vs Static 37.2%（**+17%**）。但 QD 离 Oracle (69.6%) 仍**差 48 pp**。
- **2Wiki / MuSiQue** 上出现**反向**结果：QD/KnowledgeEdit/Random 在 H2 R@5 反而**低于** Static（如 2Wiki 渐变：QD 10.6% < Static 16.2%，KE 11.1%）。这不是"预算不够"，而是 **dense cosine 信号在大池/复杂关系上失真** — 加预算只放大噪声。

**结论**：差距不是吞吐，是**选择信号**。所有上线方法都用同一种"对查询/文档算 cosine 取 top-K"的密集检索去挑候选。在小池（HotpotQA）上信号还能勉强工作 → 现行方法能跑出 ~40% 的 H2，但仍达不到 Oracle 的 70%；在大池/多跳关系（2Wiki/MuSiQue）上信号已经噪声化 → 写得越多反而越脏。**唯一出路是结构性变更**（图增强 PPR + 覆盖率驱动），不是再调超参。

---

## 1. 实验设置

| 项 | 值 |
|---|---|
| 数据集 | HotpotQA (train+dev distractor)、2WikiMultihopQA、MuSiQue (train+dev) |
| 文档池 | 32,941 / 384,857 / 84,459 |
| 子采样查询 | 3,500（先采查询再以查询所引用文档构池） |
| 簇划分 | KMeans k=8（密集向量空间）；3 head + 5 tail |
| KB 容量 | `kb_head_mult=1.2 × #head-context-docs`（自适应） |
| 流 | 50 窗口 × 50 查询 = 2,500 次查询 |
| 突变漂移 | W1–W25 仅 head 簇；W26–W50 仅 tail 簇 |
| 渐变漂移 | head→tail 比例线性插值 |
| 嵌入 | BAAI/bge-small-en-v1.5（缓存） |

**H1 / H2** = 第 1–25 / 26–50 窗口的平均；**Δ = H2 − H1**。

### 1.1 KB 容量为什么按 head-context 数算

旧实验用 `pool_size × 0.10`：HotpotQA 池 33k → KB 3.3k 只能装 28% 的 head-context（共 12k）。结果 H1 已被容量天花板压住，看起来"所有策略都很差"，掩盖了策略差异。

新版 `KB = 1.2 × |head-context|`：

| 数据集 | head-ctx 数 | KB 容量 | 池占比 | 初始 head SF 覆盖 | 初始 tail SF 覆盖 |
|---|---|---|---|---|---|
| HotpotQA | 12,041 | 14,450 | 44% | **100%** | 35% |
| 2Wiki    | 7,275 | 8,750  | 2.3% | 100% | 47% |
| MuSiQue  | 12,000 | 14,750 | 17% | 100% | 65% |

H1 不再被容量卡死；H2 的崩盘**纯粹**来自策略选错了文档。

---

## 2. 写预算如何确定 — 让每种方法发挥最佳性能

> 旧版用 `WRITE_CAP=50, PROBE_TOPK=20`。结果是所有策略在 H2 都被"喂不饱"，与 Static 几乎平 → motivation 反而被实验设计掩盖。

### 2.1 设计原则（修正）

> "给每种方法**够大的预算**让它发挥自己的最佳；最佳之间的相对排序才是真信号。"

约束：

1. **写额度上限**：`WRITE_CAP × n_H2_windows ≥ 2 × |tail SF docs|` — 让任何策略**理论上**能在 H2 装齐 tail SF，并留 2× 余量。tail SF ≈ 2,500–5,000 → cap ≈ **200 / 窗** 给 5,000 写额度。
2. **探测宽度**：`PROBE_TOPK=50` — 每窗口约 25 失败查询 × 50 候选 = 1,250 raw 候选，去重后能稳定喂满写 cap。
3. **公平共享**：所有 writer 共用同一 `WRITE_CAP`、所有 prober 共用同一 `PROBE_TOPK`，差异只来自"选择信号"。

### 2.2 共享常量（`config.py`）

| 常量 | 值 | 谁用 |
|---|---|---|
| `WRITE_CAP` | **200** | `DOC_ADD_CAP` / `EDIT_BATCH` / `QD_REPLACE_CAP` / `FIFO_BATCH` / `LOG_FIX_CAP` |
| `PROBE_TOPK` | **50** | `QD_TOP_K` / `FETCH_TOP_K` / `LOG_FIX_TOP_K` |

### 2.3 验证：预算是否真的"够用"

每种 writer 在 H2 实际写入数（HotpotQA 突变）：

| 方法 | H2 实际写入 | 占 cap×25=5000 |
|---|---:|---:|
| RandomFIFO | 5,561 | 111% (full) |
| KnowledgeEdit | 8,857 | 全程触发 |
| QueryDriven | 6,133 | 123% |
| LogDrivenArrival | 1,800 | 36%（lag 5 窗触发） |
| DocArrival | 52 | 1%（**池随机采样找不到 tail**，受限于其方法本性而非预算） |

写入次数都已经接近或超过物理需求 → 排序差异**不再是带宽问题**。

---

## 3. 三档成本核算

| 档 | 字段 | 含义 | 延迟特征 |
|---|---|---|---|
| KB 写入 | `Writes` | 持久化插入/替换 KB 中的文档 | 一次写多次用 |
| 后台检索 | `MaintR` | 后台批量扫池（DocArrival 采样 / KE / QD / Log 的密集 top-K） | 离线、可批处理 |
| 在线检索 | `ServeR` | 用户请求路径上的池查询（仅 OnDemandFetch） | **在用户延迟里** |

`MaintR` 和 `ServeR` 不能合并：60k 次 `MaintR` 可以放到夜间 CPU 上跑；60k 次 `ServeR` 直接进了请求 P99。

---

## 4. 主结果（50 × 50，扩展池，WRITE_CAP=200, PROBE_TOPK=50）

> **L1 Cov** = 当前窗口的黄金 SF 文档真实落入 KB 的比例。R@5 = L1 (KB 选对) × L2 (检索器排对)。

### 4.1 突变漂移（W26 跳簇）

#### HotpotQA  (pool = 32,941, KB = 14,450) — **排序符合预期**

| 策略 | R@5 H1 | R@5 H2 | Δ | Cov H1 | Cov H2 | Δ | Writes | MaintR | ServeR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static | 65.5 | 9.2 | -56.3 | 97.2 | 13.4 | -83.8 | 0 | 0 | 0 |
| DocArrival | 65.5 | 9.1 | -56.4 | 97.1 | 13.3 | -83.8 | 52 | 4,000 | 0 |
| OnDemandFetch | 65.5 | 15.3 | -50.2 | 97.2 | 21.3 | -75.9 | 0 | 0 | 67,800 |
| LogDrivenArrival | 64.3 | 15.1 | -49.2 | 95.1 | 21.5 | -73.6 | 1,800 | 61,700 | 0 |
| KnowledgeEdit | 59.8 | 16.3 | -43.5 | 88.6 | 23.5 | -65.1 | 8,857 | 10,000 | 0 |
| RandomFIFO | 60.2 | 21.1 | -39.1 | 88.9 | 28.8 | -60.1 | 5,561 | 10,000 | 0 |
| **QueryDriven** | 63.6 | **21.6** | -42.0 | 93.7 | **30.9** | -62.8 | 6,133 | 54,950 | 0 |
| **Oracle (上界)** | 69.8 | 69.6 | -0.2 | 100.0 | 100.0 | 0.0 | 128,198 | 0 | 0 |

**ordering 满足**：`Static ≈ DocArrival < OnDemand ≈ LogDriven < KE < Random < QD << Oracle`。QD 比 Static **+12.4 pp（+135%）**，比 Random 高 0.5 pp（且 H1 损失更小）。距离 Oracle 仍差 **48 pp** → §6 V2 的合法动机。

#### 2WikiMultihopQA  (pool = 384,857, KB = 8,750) — **反向：信号失真**

| 策略 | R@5 H1 | R@5 H2 | Δ | Cov H1 | Cov H2 | Δ | Writes | MaintR | ServeR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static | 52.5 | **2.6** | -49.9 | 98.1 | 14.8 | -83.3 | 0 | 0 | 0 |
| DocArrival | 52.3 | 2.5 | -49.8 | 97.7 | 14.7 | -83.0 | 113 | 4,000 | 0 |
| OnDemandFetch | 52.2 | **3.0** | -49.2 | 98.1 | 15.5 | -82.6 | 0 | 0 | 62,700 |
| LogDrivenArrival | 49.8 | 2.3 | -47.5 | 95.1 | 15.4 | -79.7 | 1,800 | 61,750 | 0 |
| KnowledgeEdit | 41.5 | 2.2 | -39.3 | 77.0 | 10.2 | -66.8 | 9,999 | 10,000 | 0 |
| RandomFIFO | 39.6 | 1.7 | -37.9 | 71.8 | 4.5 | -67.3 | 9,762 | 10,000 | 0 |
| QueryDriven | 44.6 | 1.3 | -43.3 | 88.9 | 13.7 | -75.2 | 8,119 | 62,300 | 0 |
| **Oracle (上界)** | 56.0 | **64.0** | +8.0 | 100.0 | 100.0 | 0.0 | 123,081 | 0 | 0 |

**关键反例**：QD 写了 8k 篇，R@5 反而只有 1.3% **低于** Static 的 2.6%。OnDemandFetch 是唯一比 Static 略高的"现行"策略，但也只有 3.0% — 距离 Oracle 64% 差 **61 pp**。Random 和 KE 把 H1 也搞砸了（39.6% / 41.5% vs Static 52.5%）。**dense top-K 在 385k 池上选的根本不是 SF 文档**。

#### MuSiQue  (pool = 84,459, KB = 14,750) — **反向：与 2Wiki 同因**

| 策略 | R@5 H1 | R@5 H2 | Δ | Cov H1 | Cov H2 | Δ | Writes | MaintR | ServeR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static | 44.4 | **10.7** | -33.7 | 98.1 | 33.5 | -64.6 | 0 | 0 | 0 |
| DocArrival | 44.3 | 10.7 | -33.6 | 97.9 | 33.6 | -64.3 | 80 | 4,000 | 0 |
| OnDemandFetch | 44.1 | **11.7** | -32.4 | 98.1 | 37.1 | -61.0 | 0 | 0 | 86,300 |
| LogDrivenArrival | 43.9 | 8.8 | -35.1 | 96.8 | 29.7 | -67.1 | 1,800 | 86,300 | 0 |
| QueryDriven | 43.1 | 9.3 | -33.8 | 94.7 | 26.6 | -68.1 | 6,566 | 85,000 | 0 |
| KnowledgeEdit | 39.0 | 9.3 | -29.7 | 84.4 | 25.1 | -59.3 | 9,902 | 10,000 | 0 |
| RandomFIFO | 39.4 | 10.0 | -29.4 | 86.1 | 25.8 | -60.3 | 8,246 | 10,000 | 0 |
| **Oracle (上界)** | 46.5 | **49.1** | +2.6 | 100.0 | 100.0 | 0.0 | 204,091 | 0 | 0 |

OnDemandFetch 是唯一略胜 Static 的（+1.0 pp）；QD/KE/Random 全部低于 Static。Oracle 仍能维持 49% → 上界存在，但所有 dense-cosine 策略都到不了。

### 4.2 渐变漂移（线性插值）

#### HotpotQA — **排序符合预期**

| 策略 | R@5 H1 | R@5 H2 | Δ | Cov H1 | Cov H2 | Δ | Writes | MaintR | ServeR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static | 65.1 | 37.2 | -27.9 | 97.6 | 54.8 | -42.8 | 0 | 0 | 0 |
| DocArrival | 65.1 | 37.4 | -27.7 | 97.5 | 55.0 | -42.5 | 43 | 4,000 | 0 |
| OnDemandFetch | 65.0 | 39.5 | -25.5 | 97.6 | 58.2 | -39.4 | 0 | 0 | 46,850 |
| RandomFIFO | 62.0 | 40.3 | -21.7 | 92.4 | 58.6 | -33.8 | 4,033 | 10,000 | 0 |
| LogDrivenArrival | 64.1 | 40.9 | -23.2 | 95.8 | 59.1 | -36.7 | 1,800 | 42,400 | 0 |
| KnowledgeEdit | 61.1 | 41.0 | -20.1 | 92.0 | 59.7 | -32.3 | 7,969 | 10,000 | 0 |
| **QueryDriven** | 64.3 | **43.5** | -20.8 | 96.0 | **62.5** | -33.5 | 4,653 | 39,000 | 0 |
| **Oracle (上界)** | 68.2 | 67.6 | -0.6 | 100.0 | 100.0 | 0.0 | 128,085 | 0 | 0 |

QD 41.5 → 43.5 比 Static 高 6.3 pp，离 Oracle 还差 **24 pp**。

#### 2WikiMultihopQA — **反向**

| 策略 | R@5 H1 | R@5 H2 | Δ | Cov H1 | Cov H2 | Δ | Writes | MaintR | ServeR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static | 52.0 | **16.2** | -35.8 | 98.1 | 40.4 | -57.7 | 0 | 0 | 0 |
| DocArrival | 51.8 | 16.2 | -35.6 | 97.9 | 40.3 | -57.6 | 109 | 4,000 | 0 |
| OnDemandFetch | 51.7 | **16.5** | -35.2 | 98.1 | 40.9 | -57.2 | 0 | 0 | 51,400 |
| LogDrivenArrival | 50.0 | 14.8 | -35.2 | 95.8 | 39.0 | -56.8 | 1,800 | 51,250 | 0 |
| KnowledgeEdit | 43.2 | 11.1 | -32.1 | 80.6 | 26.0 | -54.6 | 10,000 | 10,000 | 0 |
| QueryDriven | 46.4 | 10.6 | -35.8 | 91.4 | 34.1 | -57.3 | 7,858 | 53,250 | 0 |
| RandomFIFO | 42.0 | 9.6 | -32.4 | 76.3 | 18.7 | -57.6 | 9,723 | 10,000 | 0 |
| **Oracle (上界)** | 55.3 | **61.2** | +5.9 | 100.0 | 100.0 | 0.0 | 144,491 | 0 | 0 |

#### MuSiQue — **反向**

| 策略 | R@5 H1 | R@5 H2 | Δ | Cov H1 | Cov H2 | Δ | Writes | MaintR | ServeR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static | 43.9 | **22.7** | -21.2 | 98.1 | 57.6 | -40.5 | 0 | 0 | 0 |
| DocArrival | 43.8 | 22.7 | -21.1 | 98.0 | 57.5 | -40.5 | 88 | 4,000 | 0 |
| OnDemandFetch | 43.5 | **23.6** | -19.9 | 98.1 | 59.8 | -38.3 | 0 | 0 | 78,050 |
| LogDrivenArrival | 43.6 | 21.2 | -22.4 | 97.1 | 53.1 | -44.0 | 1,800 | 78,300 | 0 |
| QueryDriven | 42.6 | 20.1 | -22.5 | 95.4 | 48.2 | -47.2 | 5,849 | 78,400 | 0 |
| KnowledgeEdit | 39.6 | 18.9 | -20.7 | 88.7 | 46.1 | -42.6 | 9,913 | 10,000 | 0 |
| RandomFIFO | 40.0 | 18.8 | -21.2 | 88.5 | 44.7 | -43.8 | 7,854 | 10,000 | 0 |
| **Oracle (上界)** | 45.6 | **47.3** | +1.7 | 100.0 | 100.0 | 0.0 | 221,170 | 0 | 0 |

### 4.3 关键观察

#### (a) HotpotQA 上排序对了 — 信号"勉强能用"

小池（33k）+ 主题相对独立的多跳 → dense cosine 还能粗略圈出相关文档。所以 QD > Random > KE > OnDemand > Static 的顺序成立，QD 漂亮地拉出 12 pp 优势。**但 QD 距 Oracle 仍差 48 pp** — 这个 gap 必须用 §6 PPR 关掉。

#### (b) 2Wiki / MuSiQue 上**写得越多越差** — 信号失真

385k pool / 84k pool + 实体型多跳 → 桥接文档跟原查询在 dense 空间几乎无关联。每个失败查询取 top-50 全是噪声，写进 KB 后**挤掉**了原本（init KB 选过的）相对相关的 head 文档：

- 2Wiki 突变 H2：QD Cov 13.7 < Static 14.8（QD 写了 8k 篇还**降低**了覆盖率！）
- 2Wiki 渐变 H1：Random 42.0% / KE 43.2% / QD 46.4% **远低于** Static 52.0% — 写入污染连 H1 head 都伤了。

OnDemandFetch 是唯一活着的（不写持久 KB，仅按需取）但 ceiling 也只 16.5%。**dense cosine 已经从"弱信号"退化到"误导信号"**。

#### (c) Oracle 上界稳定 47–70%

无论 dense 信号多差，Oracle 都能维持 ≥ 47%（直接读 ground-truth SF id）。**这证明上界存在、问题确实是 selection signal**，不是任务本身不可解。

#### (d) 三类失败模式总结

| 失败类型 | 出现于 | 现象 | 根因 |
|---|---|---|---|
| **带宽瓶颈** | （旧版 cap=50 时） | 所有方法 ≈ Static | 写预算太小 — 已修复 |
| **信号弱** | HotpotQA | QD > Static 但 ≪ Oracle | dense top-K 漏桥接 |
| **信号失真** | 2Wiki / MuSiQue | 写得越多越差 | dense 在大池/多跳上完全错配 |

后两类 **同根**：dense cosine 对桥接节点不可见。这就是 §6 PPR 必须做的事。

---

## 5. 论文图

`figures/recall_drift_50w_sudden.pdf`、`figures/recall_drift_50w_gradual.pdf`：

- 每个数据集 2 行：上行 = Recall@5，下行 = L1 KB 覆盖率
- Oracle 红色实线作为**全程上界**
- QueryDriven 蓝色加粗 + 星号标记
- 灰色虚线 = drift onset（W26 / 渐变中点）
- Type-42 字体 + 色盲友好调色板，可直接放进 LaTeX

---

## 6. 算法建议（V2 设计方向）

数据指向**结构性变更**，不是参数调优。HotpotQA 上 QD 能跑出来但只到 Oracle 的 30%；2Wiki/MuSiQue 上 QD 直接被 dense 信号拖到 Static 之下。两条相互正交的修法：

### 6.1 用**图锚点 + 个性化 PageRank** 替换密集 top-K 选候选

复用 HippoRAG / G-Retriever 思路，在文档池上构建 OpenIE 实体-关系图。失败查询触发时：

1. 抽取查询实体；
2. 从这些实体起点跑 Personalized PageRank；
3. 把 PPR 高分**邻居**的文档作为候选 — 这正是密集 cosine 漏掉的桥接节点。

直接攻 §4.3(b)(c) 的 L1 失败根因：让二跳桥接文档变得"图上可达"。预期收益：

- HotpotQA：QD 21.6% → 40–50%（关掉一半 QD-Oracle gap）
- 2Wiki / MuSiQue：从"反向"（不如 Static）翻转到 +10–20 pp 优势

### 6.2 用**需求驱动淘汰** 替换"top-K 写入"

为每篇 KB 文档维护**滚动覆盖率**（被多少近期失败查询所需要）。每窗口：

- 淘汰覆盖率衰减到阈值以下的 KB 文档；
- 写入 §6.1 中能覆盖**当前失败查询集**的候选。

同样的 `WRITE_CAP=200` 才真正花在刀刃上 — 解决 §4.3(b) 在 2Wiki 上观察到的"写入污染 head"问题。

### 6.3 路线图

| 阶段 | 内容 |
|---|---|
| V2.0 | 接 HippoRAG 的 OpenIE 抽取，离线建图（一次性） |
| V2.1 | 在 QueryDriven 中替换"密集 top-K 候选" → "PPR top-K 邻居" |
| V2.2 | 加覆盖率统计 + 软淘汰（保留 `WRITE_CAP=200` 不变，只换打分） |
| V2.3 | 跑同一套 motivation_2 流水线，期望 H2 R@5 拉到 Oracle 的 50–80% |

---

## 7. 复现

```bash
cd motivation/motivation_2
python run.py --n-windows 50 --window-size 50 --expanded --n-source 3500 \
              --drift sudden  --output results_50w_expanded_sudden.json
python run.py --n-windows 50 --window-size 50 --expanded --n-source 3500 \
              --drift gradual --output results_50w_expanded_gradual.json
```

输出：

- `data/results_50w_expanded_{sudden,gradual}.json` — 每窗口 R@K + L1 覆盖率 + 三档成本
- `figures/recall_drift_50w_{sudden,gradual}.{pdf,png}` — 论文级双行图

主要超参（`config.py`）：

| 常量 | 值 | 含义 |
|---|---|---|
| `WRITE_CAP` | 200 | 每窗口最多写入 KB 篇数（所有 writer 共用） |
| `PROBE_TOPK` | 50 | 每个失败查询取 top-K 候选（所有 prober 共用） |
| `kb_head_mult` | 1.2 | KB 容量 = head-context 文档数 × 1.2 |

`--datasets` `--n-windows` `--window-size` `--n-source` 都可 CLI 覆盖。
