# Algorithms

## 论文主方法

唯一 DRIP 入口是：

```python
from algorithms.drip import DRIP, DRIPConfig
```

代码阅读顺序：

```text
algorithms/drip/config.py
algorithms/drip/controller.py
algorithms/drip/index.py
algorithms/drip/topic_partition.py
algorithms/drip/topic_dynamics.py
algorithms/drip/policy.py
```

完整公式与代码映射见：

```text
algorithms/drip/PARAMETER_MAPPING.md
```

## Baselines

```text
algorithms/cache/recency/       LRU, FIFO
algorithms/cache/frequency/     TinyLFU
algorithms/cache/semantic/      GPTCacheStyle, Proximity
algorithms/cache/paradigm_ref/  AgentRAGCache
algorithms/cache/oracle/        Oracle upper bound
```

统一策略注册表位于 `algorithms/cache/registry.py`。其中 `DRIP` 默认使用
evidence-only placement；TopicDynamics 由 runner 显式传入冷库分区配置。LRU 是独立
baseline，不参与 DRIP utility。旧 hard-state、Fixed-Share、hidden bridge 与 ablation
prototype 不再作为 active 策略注册。
