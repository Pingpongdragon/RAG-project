# 文档二：逻辑质疑与回答

---

## Q1: 为什么现有范式的更新效率低？

### 回答：信号来源的结构性缺陷

现有RAG KB更新范式有两种信号来源，**都没有demand signal**：

| 范式 | 代表系统 | 更新信号 | 根本问题 |
|---|---|---|---|
| Document Arrival | HippoRAG, LightRAG | "新文档到了" | 新文档是随机到达的，与当前用户需求无关 |
| Knowledge Edit | RECIPE | "某个事实变了" | 编辑请求针对已有知识，不知道用户需要什么新知识 |

**关键insight**: 这两种范式在**架构层面**就无法获取"用户当前需要什么"这个信号。不是算法差，是**范式blind spot**。

**数据证据**: 
- DocArrival-Heavy用了2.5×操作预算（498 vs QD的731 swaps on HotpotQA），但H2 R@5只有3.9%
- 原因：它的498次swap中，每次选的文档都是从pool随机到达的，跟H2的query完全无关
- 换个角度：即使给DocArrival无限预算，它也只会把pool里的随机文档换进去，不会比QD好多少

**操作效率对比**:
- DocArrival: 0.009% recall per swap（每次替换平均只贡献0.009%的recall提升）
- Oracle: 0.075% recall per swap（每次替换贡献0.075%）
- **Oracle效率是DocArrival的8倍**——说明问题不在于换多少，在于换什么

### 一句话总结
> 现有范式低效是因为它们的替换是"盲的"——不知道用户当前需要什么，所以换进去的文档大概率也是无关的。

---

## Q2: 现有数据集的场景真是这么构造的吗？

### 回答：数据集是标准的，漂移构造是我们设计的

**数据集本身**: HotpotQA / 2WikiMultihopQA / MuSiQue 都是标准benchmark，不是我们造的。HippoRAG论文用的同一批数据。

**漂移构造是我们的贡献**，但有充分依据：

1. **聚类漂移方法**: 用KMeans将query按语义分成8类，H1集中在head 3类(97%)，H2翻转到tail 5类(97%)。这模拟的是：用户兴趣从"热门话题"转向"冷门话题"。

2. **文献依据**: Gama et al. (2014) 定义了concept drift的标准分类：
   - Sudden drift: 分布突变（我们的97/3→3/97）
   - Gradual drift: 分布逐步变化（我们的线性插值实验）
   - 我们两种都做了。

3. **为什么没有现成的query drift数据集**:
   - **HoH** (ACL 2025): 研究outdated info → 事实变了，不是query变了
   - **DynaQuest** (ACL Findings 2025): Wikipedia Infobox temporal QA → 也是事实变了
   - **DailyQA** (2025): Weekly Wikipedia revision → 还是事实变了
   - **MAB-RAG** (AAAI 2025): 切换检索方法 → 跟KB内容无关
   - **没有任何现有数据集研究 query distribution drift + KB content management**

4. **97/3比例的合理性**:
   - 模拟极端但现实的场景：一个新闻事件导致用户搜索方向突然转变
   - 97/3不是50/50，因为现实中query分布本来就是skewed的（少数话题占绝大多数流量）
   - gradual drift实验(88/12→3/97)证明即使漂移更温和，结论不变

### 潜在审稿人质疑及防御

| 质疑 | 防御 |
|---|---|
| 97/3太极端 | Gradual drift实验证明结论在温和漂移下同样成立 |
| 为什么不用真实drift数据 | 不存在这样的公开数据集，这正是我们的研究gap |
| 聚类真的能代表"话题"吗 | MiniLM的embedding在语义空间确实按话题聚类，这是well-established的 |

---

## Q3: 普通的QD也这么低吗？

### 回答：是的，有三个结构性原因

QD的H2 Recall@5: HotpotQA **7.5%**, 2Wiki **2.7%**, MuSiQue **8.8%**

这确实很低。但看per-window轨迹就能理解为什么：

**HotpotQA QD per-window R@5:**
```
H1: [25, 27, 27, 22, 22, 30, 19, 15, 23, 12]  mean=22.2%
H2: [ 2,  2,  7,  4, 11, 10,  5, 11, 11, 12]  mean=7.5%
       ^  ^                                 ^
       |  |                                 |
       前3窗几乎0           后3窗恢复到11-12%
```

**三个结构性瓶颈**:

1. **适应延迟 (Adaptation Latency)**
   - QD必须先**观察到**失败query才能反应
   - H2前2-3个窗口：QD还在用H1的旧KB，recall接近0
   - 到H2后半段才开始恢复（HotpotQA后3窗: 11.3%，vs H1均值22.2%，恢复率51%）

2. **替换预算限制 (Budget Constraint)**
   - 每窗口最多60次swap，KB有1000个doc
   - 完全替换KB需要 ~17个窗口，但H2只有10个窗口
   - QD总计731次swap，只替换了73%的KB
   
3. **Content Selection Precision**
   - QD用embedding相似度选候选文档
   - MiniLM-L6的query-SF相似度: HotpotQA均值=0.523，>0.55的只有47%
   - 意味着53%的gold SF在embedding空间根本不会被选为候选
   - **这是最关键的瓶颈**：即使QD知道哪些query失败了，它用embedding heuristic也找不到正确的文档

### 2Wiki特殊情况

2Wiki上QD只有2.7%，甚至比DocArrival-Heavy(2.8%)还低！原因：
- 2Wiki的embedding质量更差，QD的候选选择更不精确
- 但Oracle在2Wiki上达到60.9%——证明正确文档确实在pool里
- 这进一步说明：**demand signal是必要的但不充分的**，还需要更精确的content selection

### 恢复率分析

| 数据集 | QD H2后3窗均值 | H1均值 | 恢复率 |
|---|---|---|---|
| HotpotQA | 11.3% | 22.2% | 51% |
| 2Wiki | 5.3% | 24.6% | 22% |
| MuSiQue | 12.3% | 21.2% | 58% |

即使给QD更多时间（更多H2窗口），它也被embedding precision上界卡死。

---

## Q4: 你这里提出用什么来进行更新？

### 回答：Motivation实验不提出具体方法

**Motivation的角色**是证明"需要一个更强的方法"，而**不是**展示那个方法。

具体来说，motivation证明了：

1. **现有supply-side范式（HippoRAG/RECIPE）缺少demand signal** → 需要demand-driven更新
2. **简单的demand-driven方法(QD)也不够** → 需要比embedding heuristic更强的content selection
3. **Oracle证明KB budget本身够用** → 瓶颈明确是"选什么文档"，不是"换多少文档"

**这motivate了一个系统需要具备**:
- ✅ Demand-driven: 从query失败信号触发更新（不是等文档到达）
- ✅ Semantic understanding: 比embedding相似度更强的文档选择能力
- ✅ Budget-efficient: 在固定KB预算下精准替换

**最终算法(QARC)在正文提出**，不在motivation中透露。

---

## Q5: 用了之后效果提升明显吗？

### 回答：这个问题在motivation层面用Oracle回答

Motivation不展示最终算法，但Oracle证明了**提升空间巨大**:

| 数据集 | QD H2 R@5 | Oracle H2 R@5 | Gap | 理论提升上限 |
|---|---|---|---|---|
| HotpotQA | 7.5% | 74.5% | 67.0pp | 10× |
| 2Wiki | 2.7% | 60.9% | 58.2pp | 23× |
| MuSiQue | 8.8% | 55.1% | 46.3pp | 6× |

这意味着：**如果能缩小这个gap的30-50%，就已经是非常显著的贡献**。

例如在HotpotQA上：
- 缩小30%: 7.5% + 20pp = **27.5%** (3.7×提升)
- 缩小50%: 7.5% + 33pp = **40.5%** (5.4×提升)

---

## Q6: 这个上界(Oracle)这么高真能达到吗？

### 回答：Oracle不是目标，而是ceiling证明

**Oracle不能被达到**。它使用了两个不可能的能力：
1. **看到所有未来H2 query**（实际系统只能看到当前窗口的query）
2. **知道每个query的gold supporting facts**（实际系统不知道答案）

**Oracle的数字也不是100%**:

| 数据集 | Oracle H2 R@5 | post-rebuild R@5 | 为什么不是100% |
|---|---|---|---|
| HotpotQA | 74.5% | 82.7% | 第一个H2窗口在rebuild前测量(1.0%拉低均值) |
| 2Wiki | 60.9% | 67.7% | 同上 + embedding排序噪声 |
| MuSiQue | 55.1% | 61.1% | 同上 + multi-hop复杂度 |

**Oracle不是100%的三个原因**:
1. **测量artifact**: 第一个H2窗口在rebuild执行前被测量，R@5≈0-1%
2. **Embedding ranking noise**: 即使正确文档在KB里，MiniLM可能排不进top-5
3. **Multi-hop**: 每个query需要2+个SF都在top-K中，难度倍增

**正确的理解方式**:
- Oracle 不是说"你应该达到75%"
- Oracle 是说"**在同样的KB预算下，如果你选对了文档，recall可以从7.5%到74.5%**"
- 这证明了KB预算不是瓶颈，content selection是瓶颈
- 任何对content selection的改进都能在这个67pp的空间内获得收益

**最终算法的合理预期**:
- 目标不是match Oracle（那不可能）
- 目标是**显著缩小QD到Oracle的gap**
- 哪怕缩小30%（+20pp），也是对现有paradigm的巨大改进

---

## 总结：五个核心论点的证据链

| # | 论点 | 证据 |
|---|---|---|
| 1 | Query drift causes catastrophic collapse | Static H1→H2: 23.6→0.5% (HotpotQA) |
| 2 | Supply-side paradigms are blind | DocArrival-Heavy 498 swaps → 3.9% H2 |
| 3 | Demand signal helps but isn't enough | QD 731 swaps → 7.5% H2 |
| 4 | KB budget is sufficient | Oracle same budget → 74.5% H2 |
| 5 | Content selection precision is the bottleneck | Oracle 0.075%/op vs QD 0.010%/op |
