# 局部 PPR 支撑流：面向多跳缓存的 demand/serve 重设计

状态：设计提案（尚未实现）
取代：`algorithms/drip/cache_manager/graph_index.py` 中的单边 `E_graph` path-evidence 分支
（`_path_evidence`，第 238 行），以及 `algorithms/drip/cache_manager/__init__.py`
中扁平的 `serve+demand` 优先级（第 316 行）。

## 摘要

当前 bridge 分支用单条 `A -> B` 边给第二跳文档 `B` 打分（对 `a_sim`、`link`、
`relation` 取几何平均）。这带来三个问题：(a) 只能到一跳；(b) bridge 候选**只赚得到
demand、永远赚不到 serve**，在 admission 竞争里结构性吃亏；(c) demand/serve 的平衡
被隐式塞进 decay 常数里，没有显式可解释的权重。

本设计用**局部 Personalized PageRank (PPR)** 替换 bridge 的 demand 信号：每个 query
在一个有界邻域子图上计算（思路源自 HippoRAG，但做成局部、适配流式场景）。同时围绕两个
可解释权重 `beta`（多跳偏好）和 `gamma`（已实现价值 vs 预期价值）重建优先级，并按
route 分桶写预算。

## 问题（承接前几轮 review）

- `FINAL_METHOD.md:197` —— "只处理一跳 bridge q->A->B"。
- `DRIP_ENGINEERING_REVIEW.md:20` —— 实体链"太弱……无法证明候选就是缺失的第二跳
  证据……H2 coverage 没有提升"。
- `__init__.py:316` —— `base = serve + demand` 把两个不同量纲直接相加，没有显式权重；
  平衡被藏在 `demand_decay`/`serve_decay` 里。
- `__init__.py:247` `_credit_serve` 只给**已常驻且被检索命中**的文档发 serve，所以
  查询语义够不着的 bridge 文档永远攒不到 serve。

## 各组件

### 0. 局部子图（每个 query 一次）

```
seeds(q) = TopK0_d  max(0, sim(q, d))
V_q      = BFS(seeds, depth <= R, per-hop degree <= d_cap)
```

- `R` = BFS 最大半径（默认 3）。
- `d_cap` = 实体度数上限，复用 `entity_degree_power` 的意图，做成硬上限，让通用 hub
  实体在子图构造阶段就被截断。
- direct 路由的 query 完全跳过这一步（由 QueryRouter 分流）。

### 1. 转移矩阵 W（复用现有 link strength）

对 V_q 中共享实体的文档节点 A、B：

```
w(A, B) = sum_{e in ents(A) ∩ ents(B)}  (idf(e)/idf_max) / degree(e)^p
W       = row_normalize(w)
```

`w` 就是 `_link_strength`（graph_index.py:232）在共享实体上的求和，不引入任何新变量。

### 2. 种子（personalization）向量 s

```
s(d) = max(0, sim(q, d)) / sum_{d' in seeds} max(0, sim(q, d'))
```

仅在 `seeds(q)` 上非零；归一化到和为 1。

### 3. 局部 PPR：截断幂迭代  ->  E 值

```
pi^(0) = s
pi^(l) = c * s + (1 - c) * W^T pi^(l-1)      for l = 1..L
E_graph(d) = pi^(L)(d)
```

- `c` = 重启概率（默认 0.5，HippoRAG 取值）。
- `L` = 截断步数（默认 3）；L 即有效最大跳数。重启 + 截断双重收敛控制，不会无界爆炸。
- 只在 V_q 上跑：每步是一次小规模稀疏矩阵乘。

### 4. demand 双通道

```
D_dir(d) <- lambda_dir * D_dir(d) + max(0, sim(q, d))
D_brg(d) <- lambda_brg * D_brg(d) + E_graph(d)
```

### 5. serve 双通道（bridge 那条腿是关键修复）

```
S_dir(d) <- lambda_s * S_dir(d) + 1/len(pos)                  # 现有直接命中
S_brg(d) <- lambda_s * S_brg(d) + pi^(L)(d) / sum_{d' in KB} pi^(L)(d')
```

`S_brg` 在 BRIDGE 路由的 query 答完后发放，只对常驻（KB 内）文档，按其 PPR 质量分配。
annotation-free；与直接 serve 同量纲（都归一到 1）。有 PPR 质量的中间跳文档现在也能
赚到 serve，从而在淘汰中存活。

### 6. 归一化 + 双权重合成

```
Xhat = X / (q95(X) + eps)      对 D_dir, D_brg, S_dir, S_brg 各自归一

D(d) = (1 - beta) * Dhat_dir + beta * Dhat_brg
S(d) = (1 - beta) * Shat_dir + beta * Shat_brg
P(d) = gamma * S(d) + (1 - gamma) * D(d) - rho * max(0, red(d) - tau_red)
```

### 7. 分桶预算 + admission

```
f_brg      = 窗口内 BRIDGE query 占比
budget_brg = ceil(f_brg * budget)
budget_dir = budget - budget_brg

admit c iff  D(c) > gain_margin * P(victim)
```

bridge 候选只在 `budget_brg` 内竞争；direct 候选在 `budget_dir` 内竞争。

## 参数迁移表

| 旧 | 新 | 默认 | 说明 |
|---|---|---|---|
| `bridge_demand_gain` | `beta` | 0.5 | 只调偏好，不偷预算 |
| 隐式的 `demand/serve_decay` 平衡 | `gamma` | 0.3 -> 0.5 | 显式 serve/demand 权重；随窗口爬升（serve 是滞后信号，冷启动近 0） |
| 单边 `_path_evidence` | `c`、`L` | 0.5、3 | 局部 PPR 重启 / 截断 |
| `entity_degree_power` | 折进 `W` + `d_cap` | 沿用 | 子图构造期截断 hub |
| `admission_gain_margin` | `gain_margin` | 1.0 | 归一化后成为真正的比值 |
| （新增） | `R`、`K0`、`d_cap` | 3、top-K、cap | 局部子图边界 |

## 复杂度

每个 BRIDGE query：一次有界 BFS 构造 V_q + L 次 V_q 上的稀疏矩阵乘。
靠 `d_cap` 和 `R` 把 `|V_q|` 压在几百到几千节点，单 query 亚毫秒级。
direct query 不付这个成本（已被路由分流）。只有 `f_brg` 比例的 query 承担 PPR 开销。

## 待定决策

- `beta` 固定 vs 自适应（绑到 `f_brg` 或 FID drift 信号）。先固定 0.5 验证根因；
  之后再做自适应，接上 drift-aware 的叙事主线。
- `c`（起 0.5）：c 越大越接近直接相似、bridge 越弱；c 越小走得越远、爆炸风险越高。
- HippoRAG 2（arXiv 2502.14802）在 PPR 之上加了更深的 passage 整合；其确切公式本文
  未对照原文核实（网络受限）。若要严格对齐 HippoRAG 2，在此处补齐。

## 验证协议

1. **先做根因验证（最便宜）：** 保留单边证据，只加分桶预算 + `S_brg`。如果 H2 coverage
   在 20-window bridge run 上就动了，说明缺陷是预算混池 + 缺 serve 腿，而非匹配太弱。
2. **再上 PPR：** 把 `E_graph` 换成局部 PPR（`c=0.5, L=3`）。对比 H2 coverage、support
   recall、AMAT、update cost，与第 1 步及 DRIP-Dense baseline 比较。
3. **消融：** L ∈ {2,3,4}；c ∈ {0.3,0.5,0.7}；beta ∈ {0, 0.5, 自适应}；gamma 固定 vs
   爬升。报告相对 ARC / DRIP-Dense 的 bridge 增益分离度。
4. **成本护栏：** 跟踪单 query 的 PPR 耗时与 `|V_q|` 分布；确认 direct query 延迟不变。
