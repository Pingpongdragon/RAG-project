# 设计note：从「热库子模策展」转向「漂移感知的跨层预取」

> office-hours 产出（2026-06）。记录框架转向的决策、理由、被否决的方案。
> 旧方法 = QARC/DRYAD（漂移检测 → 热库子模换入换出）。
> 新方法 = drift-aware prefetch（漂移检测 → 从向量库预取进 cache；换出交给 LRU/ARC）。

## Summary

把"筛选/换入换出"从 backing store 移走，只在 cache 层做。漂移检测的动作从
"对热库做子模策展"改为"触发从 L2 向量库预取一批文档进 L1 cache"。换出由成熟
cache 策略（LRU / Agent-RAG ARC 2511.02919）负责，不自己重造。

## Problem

审稿人反馈："冷库做筛选没必要，通常都是一直加。" —— 工业界 backing store
(Milvus 等) 确实是 append-only，eviction 只在 cache 层。旧 QARC 在"冷库"做
子模筛选，与真实系统分层不符，且把贡献押在一个被质疑的环节上。

## 系统框架（两层，工业标准）

```
L2  向量数据库 (Milvus/Faiss, 全量, append-only, 慢)
  │   ↑ drift 触发：按新 query 分布(+实体链 bridge) 从 L2 召回 → 预取进 L1
  ▼   │
L1  cache (固定预算, 内存, 快) ── 在此检测 query 漂移；LRU/ARC 管换入换出
```

- 检测信号来自 L1（miss 模式 / query–KB 对齐漂移）。
- 动作 = L2→L1 主动预取（区别于 OnDemandFetch/CRAG 的被动逐条 fetch）。
- 消化 = L1 内部 eviction，交给 LRU / ARC-2511（非本文贡献，是下游组件）。
- 类比：CPU prefetcher（你的贡献）+ cache replacement policy（现成的）。Hennessy-Patterson framing。

## User

固定缓存预算下、query 分布随时间漂移的 RAG/agent 部署（edge/mobile、多 agent 共享后端）。

## Constraints（硬）

- 预算约束只在 L1。L2 视为全量、不删。
- 单机静态语料下，"往冷库加"= no-op（文档已在 L2）；本文的边是 **L2→L1 预取**，
  除非引入外部文档流式到达（另一设定，v1 不做）。
- 评测口径对齐 ARC-2511：has-answer rate、AMAT、storage%。

## Proposed wedge（v1）

**drift-aware proactive prefetch**：检测到 query–KB 对齐漂移时，按当前 query 分布
（可选沿实体链拉 bridge 文档）从 L2 预取一批进 L1；L1 用 LRU/ARC 消化。

两条贡献线（用户确认两者都要，但分主次）：
1. **主**：预取调度——何时触发（漂移检测）、预取什么分布、是否拉 bridge。
2. **支撑**：换入换出直接用 LRU/ARC-2511，证明不重造轮子、即插即用。

## Architecture direction

- 检测器（可插拔，消融）：NoDetector / ADWIN（标量）/ raw-embedding MMD / 对齐特征 FID(ours)。
- 预取边 baseline：无预取 / 固定周期预取 / OnDemandFetch(CRAG, 被动) / ARC-2511 几何打分预取。
- 换出边（L1 管理器，可替换）：LRU / TinyLFU / ARC-2511。

## Rejected alternatives

- **经典 ARC (Adaptive Replacement Cache, FAST'03)**：纯 access-history 页面替换，
  与语义检索缓存差太远，比了是另一种稻草人。不做。
- **ComRAG / ERASE 作主 baseline**：知识更新/对话记忆范式，非固定预算换入换出，
  指标不可比 → 已删除（见 algorithms/__init__.py 注释）。
- **L1 memory / 总结成文档（agent-mem 风格）**：v1 非目标。理由：
  (1) 评测变脏——总结质量只能 LLM-judge / 下游 QA，贵且主观；
  (2) 与"抗漂移"论点自相矛盾——总结比原始 passage 更易过期；
  (3) 把最近邻威胁从 ARC-2511 扩大到整片 agent-memory 文献，护城河变窄；
  (4) ARC_COMPARISON.md 已定 memory 层"不作主卖点"。
  若必做，最窄版 = "compressed synthesis 作为与原始文档抢同一预算的 cache 条目"，
  评测仍用 has-answer-per-byte，不单独证明总结质量。列 future work。

## Risks

- 预取增益若拉不开 vs 固定周期预取 / OnDemandFetch → 被指"LRU + 周期召回"。
  必做消融：drift-triggered vs 固定周期 vs 无预取，做出差距。
- bridge 增益在 2Wiki 上偏小（+0.8~2.7pp）→ 需在更纯/更大 pool 放大。

## Next step

1. 删 ComRAG/ERASE（进行中）。
2. 新增 AgentRAGCache baseline (2511.02919) 进 algorithms/cache/。
3. 新增检测器 baseline (NoDetector/ADWIN/MMD) 与 DriftLensDetector 同接口。
4. 命名：QARC/DRYAD → 待定（DAP / TierShift / DriftFetch），代码统一替换。
5. 回填 ARC_COMPARISON.md。
