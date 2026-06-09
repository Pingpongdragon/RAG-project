# Motivation Storyline v4 (UPDATED — 2026-05-26)

> v4 改动（针对评审意见 [评审意见.md](评审意见.md)）：
> - **核心 framing 升级**：从单一的 "query distribution drift" 升级为 **"Agentic Shared RAG Access Patterns"**，把 Fig 0 的宏观时间漂移与 Fig 2 的桥接多跳串成同一个 system audit。
> - **三级访问模式（L1/L2/L3）作为叙事主轴**：L1 = temporal locality（集体话题漂移），L2 = direct semantic multi-hop（实体在 query 中显式可见），L3 = bridge/topological multi-hop（second-hop 实体被隐藏）。Fig 0 / Fig 1 / Fig 2 不再是"先试 A 不行再试 B"的流水账，而是按三级访问模式分开的诊断证据。
> - **Fig 0 计划补 (c) 子图**：从 AgentBench / WebShop 等真实 agent trace 中提取链式关联查询比例，把 Fig 0 从"宏观集体话题漂移"扩展到"Agent 微观链式访问也是高频模式"，弥合评审指出的 macroscopic↔microscopic 断层（实施在下一轮，见 [NEXT_STEPS_AUDIT_TODO.md](NEXT_STEPS_AUDIT_TODO.md)）。
> - **并发动机实验**：评审意见 §三.3 提议在 100-agent 批次上测桥接实体共享率。本轮先在 motivation.tex 里把这一段落 framing 写出来，但用诚实的占位说明数字 pending audit；具体跑数留下一轮（见 [NEXT_STEPS_AUDIT_TODO.md](NEXT_STEPS_AUDIT_TODO.md)）。
> - **SemFlow 重定位**：不再写成 "纯语义 baseline"，而是定位为"针对 L2 direct semantic multi-hop 的 admission engine"；它在 L1 上不输 LRU 是合理的（L1 已被 access history 解决），在 L2 上 +9.2pp 是 positive evidence，在 L3 上的 21pp Oracle gap 是引出 [METHOD] 的 negative evidence。

---

## 0. 一句话主线 (logline)

> 在多用户并发智能体共享 RAG 的部署场景里，hot-tier 缓存同时面临**宏观时间漂移（L1）**与**微观链式拓扑访问（L2/L3）**两类访问压力。本文先把 agent 工作负载抽象为 L1/L2/L3 三级访问模式，做一个统一的 system audit：L1 用 access history（LRU）就够了；L2 由我们提出的 SemFlow 通过 query 语义扩散解决；L3 桥接多跳暴露出语义信号触达不到的"second-hop 不可达"结构性缺陷。我们因此提出 [METHOD] —— 一个 drift-aware routed cache，把 SemFlow（L1/L2 的 demand engine）与 entity-chained graph prefetch（L3 的拓扑预取 engine）耦合起来。

---

## 1. 五幕结构（Intro 的五个 paragraph 对应五个节拍）

### § 1.1  场景设定 — Multi-user agent 的 shared external memory
**[论证目标]** 把读者带入"被研究的物体"是 hot-tier KB cache，不是 agent 私有记忆，不是 corpus 训练更新。

**节拍内容**
- Long-running agents = agent-local memory + shared external knowledge
- 前者已被大量研究（MemGPT, MemoryBank, A-Mem, surveys）→ 一句带过 + 集中引用
- 我们聚焦后者：long-lived corpus-backed KB，对延迟敏感 → **two-tier 部署**：
  - cold corpus（完整，~10⁵–10⁷ docs）
  - hot tier（内存内 ANN，~10³–10⁴ docs，SLO < 10ms）
- Hot tier 必须持续对齐当下需求，否则 SLO 内拿不到证据

**[支撑材料]** zhang2024survey / hu2025memory / packer2023memgpt / jayaram2019diskann / chen2021spann / malkov2018efficient 等；无图

---

### § 1.2  Agentic shared-RAG 的两层 locality —— 宏观 + 微观
**[论证目标]** 这是评审意见点出的**关键节拍**。要让读者一上来就明白：agentic 工作负载同时产生**两层 locality**，所以 system audit 必须分级。

**节拍内容**
- 宏观（macroscopic）：多个用户/agent 在小时-月-年的时间尺度上，集体地把 attention 在 topic 之间转移。这是 L1。
- 微观（microscopic）：每一个 agent 自己在几秒-几轮的尺度上，沿实体/关系链做检索：读完 entity A 的文档 → 下一条 query 通过 A 的某条边引出 B。这是 L2/L3。
- 关键 framing：这两层不是相互替代，也不是先后顺序的"做着做着跑题了"。它们是 agentic 共享 RAG 同时面临的两类访问压力，**system-level cache 必须同时回答这两类问题**。
- 因此后面的 Fig 0（宏观集体话题漂移）与 Fig 2（微观链式桥接访问）不是两个不同的故事，而是同一个 audit 的两端。

**[支撑材料]** Fig 0 的 (a)WildChat + (b)Trends 撑宏观；(c) 子图（计划补：从 AgentBench / WebShop / 合成多跳日志中提取"链式关联查询占比"）撑微观；引用 ReAct, AgentBench 等。

**[关键 framing 调整]（堵评审 #1 — macroscopic↔microscopic 断层）**
- 必须在 §1.2 就明确两层 locality 同时存在，不要等到 §1.5 才把"链式访问"从天而降。

---

### § 1.3  L1 诊断 — 时间漂移：access history 已经够用
**[论证目标]** 证明面对纯单跳的时间漂移，传统 access history（LRU）已经接近上限；同时拆掉"是不是 timestamp 信号没用上"的潜在质疑。

**节拍内容**
- StreamingQA 14 年新闻流 → 把它定位成 **L1 cleanest natural temporal drift testbed**
- Miss-driven LRU H2 R@5 = 28.8%，SemFlow = 27.1% —— **诚实承认 SemFlow 在 L1 上不胜出**
- 把 Recency-TTL（oracle year）当作 timestamp 信号的天花板：仍然只到 6.4%，因为 timestamp 在 candidate 内没有区分度（同 era 几千篇）
- OnDemandFetch H2 R@5 = 46.4% 但代价是 1.45×10⁵ 次 cold-tier 检索，是 fallback 不是常驻策略

**[关键 framing 调整]**
- 不写"我们的方法没赢" → 写**"L1 已经被 access history 解决，所以本文的 contribution 在 L2 与 L3"**
- 这个节拍的功能是**精确划界**：给读者一张明确的"问题难度地图"

**[支撑材料]** Fig 1（StreamingQA signal audit）

---

### § 1.4  L2 诊断 — 直接语义多跳：query embedding 邻域可达，SemFlow 起作用
**[论证目标]** SemFlow 不是"通用方法"，而是专门针对 L2 设计的 admission engine。HotpotQA-comparison 中 +9.2pp 是 positive evidence。

**节拍内容**
- 情境：HotpotQA-comparison query 把两个被对比实体都暴露在文本里，两跳证据都落在 query embedding 的邻域
- SemFlow 算法回顾：tier-1 LRU floor + tier-2 失败 query 在 top-K 冷库邻居上累积 demand + 冗余惩罚淘汰
- 数据：SemFlow R@5 H2 = 54.3%, LRU = 45.1%（+9.2pp）
- **claim**：query-side semantic diffusion 在 "实体显式可见" 的多跳上是有效的 admission 信号

**[支撑材料]** Fig 2(a) — HotpotQA-comparison

---

### § 1.5  L3 诊断 — 桥接多跳：query embedding 触达不到 second-hop，必须用图
**[论证目标]** 通过 2Wiki bridge-comparison 暴露 SemFlow 的结构性缺陷，引出 [METHOD] 的图预取模块。

**节拍内容**
- 情境：2Wiki bridge-comparison query 只暴露 first-hop 实体，second-hop 实体只能从第一跳 doc 推出
- second-hop doc 在 query embedding 空间里**与 query 不近** → SemFlow 的 tier-2 邻域 demand 抓不到
- 数据：SemFlow R@5 H2 = 36.0%, LRU = 29.0%（+7.0pp），Oracle = 57.0% → **21pp 缺口**；OnDemandFetch = 48.8% → **13pp 缺口**
- 这不是 SemFlow 调参问题，是**信号空间**问题：要触达 second-hop，必须沿实体/关系边在图结构上预取
- → 直接 motivate [METHOD] 的 R3 模块（entity-chained / graph-structured prefetch）

**[支撑材料]** Fig 2(b) — 2WikiMultihopQA bridge-comparison

---

### § 1.6  Why this is a shared-cache problem —— 防"是不是单 session memory"质疑
**[论证目标]** 评审意见 §三.3 的"护城河"。要让审稿人知道：bridge 不是一个 agent 自己的事，是多 agent 同时撞到的共享压力。

**节拍内容**
- 模拟 100 个 agent 并发跑 2Wiki/HotpotQA bridge 任务
- 报告两个 cache-independent 指标：
  - 同一 batch 内，**至少与另一 agent 共享 ≥1 个 hidden bridge entity 的 agent 比例**（pairwise overlap rate）
  - 同一 batch 内，**hidden bridge evidence 重复读取的比例**（duplicate cold reads / total cold reads）
- 话术：哪怕只看 gold sf 的桥接实体重叠（保守估计，不计 distractor 与一般 cold miss），共享率仍然不可忽略 → per-session memory 装不下、不能跨 agent 复用 → 必须做 global hot tier
- **重要**：本轮 motivation.tex 中的具体数字（13.2% / 3.6%）是占位，**真实数据下一轮跑**（见 [NEXT_STEPS_AUDIT_TODO.md](NEXT_STEPS_AUDIT_TODO.md)）。在数字落定前，tex 里要么改成 "preliminary X% pending audit"，要么暂时只保留定性 claim。

**[支撑材料]** 计划补一个柱状图（可放 appendix 或 Fig 2c），数据从 motivation_2/data/results_*_2wiki_bridge_*.json + 100-batch overlap script

---

### § 1.7  因此提出 [METHOD] — 三级访问模式 → 三个 repair routine
**[论证目标]** 用三级访问模式把 contribution 串起来，让读者看到 [METHOD] 不是堆模块，而是**对 L1/L2/L3 各自的对症下药**。

**一段总览**
- [METHOD] = drift detector + failure classifier + 3 repair routines
- R1 = demand-aware replacement（处理 L1 + 部分 L2，复用 SemFlow 引擎）
- R2 = online cold-tier retrieval（兜底，处理"hot tier 完全无证据"）
- R3 = entity-chained / graph-structured prefetch（处理 L3 桥接多跳）
- 价值主张："Where pure query embedding blindly diffuses, [METHOD] detects when to act and routes by access regime."

**三个 contribution**
1. **Problem formalization**：把 hot-tier admission for shared agentic RAG 形式化为 L1/L2/L3 三级访问模式问题
2. **Empirical characterization**：跨 L1/L2/L3 的 system audit（Fig 0/1/2）+ 100-agent 共享压力诊断
3. **[METHOD] framework**：drift-aware routed cache，把 SemFlow（L1/L2）与 entity-chained graph prefetch（L3）耦合起来；quality / latency / update cost 三轴 dominate SemFlow 与 OnDemand 的连线

---

## 2. 图–文映射（LOCKED v4）

| Fig | 数据 | 论证哪一幕 | 状态 |
|---|---|---|---|
| **Fig 0(a/b)** | WildChat + Google Trends | §1.2 宏观 L1：集体话题漂移 | ✅ 已有 |
| **Fig 0(c)** | AgentBench / WebShop / 合成多跳日志的链式关联查询占比 | §1.2 微观 L2/L3：Agent 链式访问也高频 | 🔜 下一轮跑 |
| **Fig 1** | StreamingQA, 8 strategies | §1.3 L1 诊断：access history 够用 | ✅ 已有 |
| **Fig 2(a)** | HotpotQA-comparison | §1.4 L2 诊断：SemFlow 起作用 | ✅ 已有 |
| **Fig 2(b)** | 2WikiMultihopQA bridge-comparison | §1.5 L3 诊断：SemFlow 不够，引出图预取 | ✅ 已有 |
| **Fig 2(c) / appendix bar** | 100-agent 批次桥接实体共享率 | §1.6 防单 session memory 质疑 | 🔜 下一轮跑 |

---

## 3. 执行顺序

```
[Tier A — 已完成] storyline + tex framing
[1] STORYLINE 升级到 v4 ✅ (本文档)
[2] motivation.tex 处理占位数字 + 加 §1.2 微观铺垫 + 打磨 caption
[3] 落 NEXT_STEPS_AUDIT_TODO.md

[Tier B — 下一轮]
[4] 写 batch_overlap.py：100-agent 并发桥接共享实验
[5] 抽 Fig 0(c) 数据：AgentBench / WebShop / 合成多跳的链式占比
[6] 重画 Fig 0 (3-panel)、补 Fig 2(c) 或 appendix bar
[7] 把 §1.6 的真实数字写进 tex

[Tier C — 待后]
[8] 实现 [METHOD] + main result
[9] 完整论文
```

---

## 4. 已决定

- **算法名字**：占位 [METHOD]，所有 tex/图/表都用 placeholder，最后统一替换
- **Fig 1 保留 OnDemand**（作为 quality 天花板参照）
- **SemFlow 定位**：明确为 L2 的 admission engine，不是 universal baseline
- **timestamp 论证**：核心反驳是"timestamp 在 candidate 内没有区分度"（Recency-TTL with oracle year 仍只到 6.4%）
- **Fig 0(c) 数据来源**：评审意见建议 AgentBench / WebShop / 合成多跳日志，**优先使用真实 agent trace**（AgentBench 或 WebShop），合成日志作为 fallback
- **并发实验**：保守报告，只统计 gold sf 的桥接实体重叠（不算 distractor 与一般 cold miss），承认数字是 lower bound
