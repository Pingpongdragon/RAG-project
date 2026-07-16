# Motivation figure data

Panel (a) reproduces Table 1 (bge-small-en) from arXiv:2511.02919.
Panels (b,c) aggregate five held-out SQuAD construction seeds (42--46).
All local methods use cache size B=32 and per-window write cap W=4.

| Panel | Method / dataset | Condition | Mean | Std. | Reads | Writes |
|---|---|---|---:|---:|---:|---:|
| a | AgentRAGCache / MMLU | fixed benchmark | 62.63 | -- | -- | -- |
| a | AgentRAGCache / AdversarialQA | fixed benchmark | 71.18 | -- | -- | -- |
| a | AgentRAGCache / SQuAD | fixed benchmark | 79.80 | -- | -- | -- |
| b | LRU | stationary | 79.76 | 1.05 | -- | -- |
| b | LRU | recurring | 28.12 | 0.52 | -- | -- |
| b | AgentRAGCache | stationary | 53.88 | 0.27 | -- | -- |
| b | AgentRAGCache | recurring | 25.08 | 0.52 | -- | -- |
| b | ClassicalARC | stationary | 68.32 | 2.17 | -- | -- |
| b | ClassicalARC | recurring | 23.80 | 0.63 | -- | -- |
| c | LRU | recurring | 28.12 | 0.52 | 359.4 | 76.0 |
| c | ClassicalARC | recurring | 23.80 | 0.63 | 381.0 | 67.4 |
| c | AgentRAGCache | recurring | 25.08 | 0.52 | 374.6 | 60.0 |
| c | DRIP-TopicState | recurring | 32.76 | 0.54 | 376.2 | 50.6 |
