# RAG Runtime Core

`core/` 保存与具体实验无关的 RAG 主流程组件：

```text
data_processor.py    输入文档处理
kb_base.py           KB 抽象
retriever.py         检索
reranker.py          重排
generator.py         生成
evaluator.py         通用 RAG 指标
query_strategies.py  查询策略
utils/               运行时通用工具
```

漂移构造、warm-up、session interleaving 等实验协议位于
`experiments/common/`，不属于线上 RAG runtime。

