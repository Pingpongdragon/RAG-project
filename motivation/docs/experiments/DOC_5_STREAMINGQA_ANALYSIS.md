# DOC_5: StreamingQA 实验分析 + 延迟基准完整记录

生成时间：2026-05-13
关联代码：motivation_1/loaders_temporal.py, plotting/plot_main.py, plotting/bench_qdrant_result.json
实验结果：motivation_1/data/results_streamingqa_temporal.json

---

## 1. Pool 和 KB 的大小及来源

| 名称 | 大小 | 来源 |
|---|---|---|
| cold pool | 29,819 docs | 4,128条 dated query 的 gold context（去重）+ 27,691条 distractor context（从36K undated 采样去重） |
| hot KB | 750 docs | head_biased_init_kb 根据 H1 query 的 gold doc 命中率决定；kb_head_mult=1.6 → kb_budget=750 |
| cold/hot 比 | ~40× | 29819 / 750 |
| stream | 50 windows × 50 queries = 2500 | 前25窗口从H1(R1-R2, 2008-2013)采样；后25从H2(R3-R5, 2014-2020)采样 |

KB初始化逻辑：把所有H1 query的gold SF doc收集为head_ctx，× kb_head_mult=1.6，取50的倍数。
因此KB在H1时期 gold doc coverage=100%（实测 Static cov_h1=100%），H2时期完全错配。

---

## 2. 现有方法在 StreamingQA 上全部失效的原因

### 2.1 实测数据

Strategy        R@5-H1  R@5-H2  CovH1  CovH2  Writes/win
Static           84.5%   3.0%   100%   3.3%     0
LRU              26.1%   3.2%   29.8%  4.1%    78
GPTCacheStyle    44.9%   2.6%   52.6%  3.2%    75
MemGPTStyle      27.8%   2.7%   31.6%  2.8%    78
KnowledgeEdit    25.2%   2.0%   29.9%  2.8%   198
OnDemandFetch    64.4%  58.6%   100%  75.5%     0
DRIP-Dense      79.2%  17.2%   94.1% 19.2%    91
Oracle           81.0%  76.2%   100%  100%    380

H1→H2 Static R@5 从84.5%→3.0%，drift极其剧烈（金融危机时代 vs Brexit/COVID，gold docs无重叠）。

### 2.2 各类方法失效根因

① LRU / MemGPTStyle：Arrival-side信号盲区（H2 R@5 仅 3-4%）

更新信号来自随机doc arrival（_ArrivalCacheBase）：每window采样约80篇文档。
Pool有29,819篇，H2 gold docs散落在其中（~2000篇）。
每次随机采样命中概率 80/29819 = 0.27%，且H2 gold docs与H1 KB的cos sim < 0.3
→ LRU的DEDUP/相似度门槛会进一步过滤掉这些"离当前KB太远"的文档。
虽然每window写入约78篇，但写进去的都是pool里的随机文档，H2覆盖率仍只有4.1%。

② GPTCacheStyle：semantic dedup门槛反向杀掉最需要的文档（H2 R@5仅2.6%，最差）

DEDUP_LOW=0.30：arrival doc与KB最大cos sim < 0.3 则拒绝，认为是噪声。
H1→H2剧烈漂移时，H2 gold docs与H1 KB的cos sim普遍 < 0.3。
这个"过滤噪声"的门槛在drift场景下反而成了"过滤最需要文档"的门槛。
比LRU还差，是所有方法中H2 R@5最低的（2.6%）。

③ KnowledgeEdit：KB内部相似度搜索不含drift信号（H2 R@5仅2.0%）

选KB内"陈旧"文档，在pool里找语义最接近的替代文档。
H2期间KB全是H1文档，pool里语义最接近KB的依然是H1时代新闻，而非H2 gold docs。
错误信号来源：用KB内部结构而非query失败信号。虽然总写入9896次（最多），H2覆盖率仍2.8%。
本质：知识编辑解决的是"KB内部冗余/过时"问题，而非"workload需要什么"问题。

---

## 3. Static 在 H2 为何还有 3% 而不是 0%

recall@5_per_window (W26-W50):
[10, 4, 4, 2, 4, 2, 4, 6, 6, 2, 0, 2, 4, 0, 4, 0, 2, 2, 0, 0, 4, 2, 2, 4, 0]

三个来源：
1. 少量跨时代通用文档：部分"常绿"文档（政策背景/知名人物等）H1初始化进KB，碰巧和H2 query相关。
2. 跨round语义连接：R5(2019-2020)部分query涉及跨越多年的事件/人物，与R1-R2 KB有微弱连接。
3. top-5余弦"退而求其次"：KB里同主题的非gold文档可能cos sim勉强超过threshold(0.5)。

结论：3%是噪声级别，无统计意义。真正的H2 baseline应以OnDemandFetch(58.6%)为参照。

---

## 4. 为什么 DRIP-Dense 有效（H2 R@5=17.2%）

核心：使用失败查询的embedding探测pool，把"被workload证明需要"的文档写入KB。

writes_per_window完整序列：
[0,0,0,0,0,0,1,1,0,3,3,10,17,3,5,8,14,14,38,51,65,57,114,72,110,
 128,151,153,179,173,138,138,125,144,140,149,161,147,123,125,129,134,150,163,184,176,210,205,210,210]

Step 1：H1期间几乎不写（W1-W12写入0-3次）
  KB覆盖率94.1%，大多数query成功，fail集合极小，demand无法积累到超过serve的门槛。

Step 2：H1末期（W19-W25）漂移信号提前积累
  W19-W25写入量从38骤增到110，因为stream开始采样R3过渡期query：
  失败query的embedding投影到pool，H2文档的cos sim高→demand快速积累→门槛被突破。

Step 3：H2期间持续写入但recall受限（平均17.2%）
  受两重限制：
  a. KB budget上限：750槽，H2约有2000个不同gold docs，理论命中率上限750/2000=37.5%
  b. 时序错配：H2的stream先采R3(W26-W33)，而pool里R5(2019-20)文档最多(48%)，
     DRIP-Dense优先写R5相关文档，但这批文档对R3的query没用→写错了方向。

对比OnDemandFetch(H2 R@5=58.6%)：
  OnDemand每次query即时从pool拉取，天然回避了时序错配和KB容量限制。
  代价：serve latency 7.1ms/query vs DRIP-Dense 1.6ms/query（4.5×差距）。

cov_h2(19.2%) > recall@5_h2(17.2%)的差距说明：
  部分gold doc虽然进了KB，但被更多噪声文档稀释，检索时没排进top-5。

---

## 5. 对论文叙事的启示

| 维度 | 结论 |
|---|---|
| Drift幅度 | StreamingQA的H1→H2 drift是灾难性的：所有cache类方法R@5-H2 < 4%，与Static持平 |
| Cache类方法根本缺陷 | 更新信号来自supply-side（池子里随机来了什么），而非demand-side（query失败需要什么） |
| DRIP-Dense有效理由 | 用失败query的embedding精准探测pool；写入量自动随drift程度扩展（drift越严重写得越多） |
| OnDemand的代价 | 服务延迟4.5×，R@5-H2是QD的3.4倍（58.6% vs 17.2%）：知道答案但付出延迟 vs 延迟低但覆盖率低 |
| Oracle差距揭示 | Oracle R@5-H2=76.2%，远高于QD 17.2%：QD的时序错配+KB容量上限压制了准确率上界 |
| 结论句 | 在14年真实新闻流中，supply-side cache与静态KB无显著差异；demand-side failure signal方法可追踪drift，但最优解(OnDemand)以延迟为代价 |

---

## 6. 延迟基准：Qdrant v1.9.2 Docker 实测

配置：5K docs, MiniLM-L6-v2(dim=384), top-5, n=200, warmup=20, localhost。
结果文件：plotting/bench_qdrant_result.json

| 配置 | p50(ms) | p90(ms) | vs FAISS |
|---|---|---|---|
| FAISS in-mem 5K docs | 0.288 | 0.293 | 1.0× |
| Qdrant REST no-keepalive | 6.43 | 7.16 | 22.4× |
| Qdrant REST keep-alive | 4.04 | 4.29 | 14.0× |
| VectorDBBench gRPC 100K+ docs | — | — | 3-8× |

我们用5×而非14×的理由：
- 14×是localhost REST测得，REST比生产gRPC慢2-3×
- VectorDBBench在真实部署(gRPC)报告3-8×
- 论文实验cold pool为29K-200K docs，同等规模FAISS约1-3ms，比5K规模慢→比值缩小
- 5.0是VectorDBBench生产场景3-8×的中位数，保守合理

论文引用建议：
  We measured FAISS in-memory (KB=750) at 0.05ms/q and Qdrant REST (pool=5K, Docker)
  at 4.0-6.4ms/q. For production gRPC at 100K+ docs, VectorDBBench [2024] reports 3-8×;
  we adopt 5× as conservative (actual Docker REST: 14-22×).
