# algorithms/

这里放所有 cache policy 与 DRIP 方法。

## 当前 DRIP 方法

当前主实验入口是：

```text
DRIPNOdetector
```

含义：不依赖 drift detector，只使用 query-visible direct evidence、
serve/demand 账本和 replacement-aware admission。

实验从统一 registry 取策略：

```python
from algorithms.cache.registry import STRATEGY_FACTORIES

strategy = STRATEGY_FACTORIES["DRIPNOdetector"](
    doc_pool, doc_embs, title_to_idx
)
```

## DRIP 文件

DRIP 代码在 `algorithms/drip/`。当前文件分工见：

```text
algorithms/drip/README.md
algorithms/drip/PARAMETER_MAPPING.md
```

核心文件：

```text
algorithms/drip/cache_manager/core.py
  cache manager 主窗口循环。

algorithms/drip/cache_manager/policies.py
  DRIP / DRIPNOdetector 策略入口。

algorithms/drip/cache_manager/evidence_core.py
  direct evidence、hidden diagnostic、replacement writer。

algorithms/drip/cache_manager/drip_config.py
  当前唯一 active config。
```

## Baselines

cache baseline 在 `algorithms/cache/`：

```text
recency/        LRU, FIFO, temporal variants
frequency/      TinyLFU
semantic/       GPTCache-style / Proximity
paradigm_ref/   AgentRAGCache 等参考方法
oracle/         Belady-style upper bound
```
