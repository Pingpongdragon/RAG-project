| Workload | Source type | Turnover | Past-support availability | LRU hot answer (10%) | Source JSON |
|:---|:---|---:|---:|---:|:---|
| MIND (shared users; natural) | multi-user access proxy | 68.55% | 95.97% | 96.00% | `experiments/agent/data/working_set_sweep_mind_news_access_r1000.json` |
| Mind2Web (multi-session; controlled) | multi-session agent trace | 91.67% | 82.88% | 37.80% | `experiments/agent/data/working_set_sweep_mind2web_agent_r1000.json` |
| StreamingQA (single source; temporal) | single-source temporal RAG | 88.20% | 63.80% | 48.20% | `experiments/direct/data/working_set_sweep_streamingqa_temporal_r1000.json` |
| HotpotQA (low-reuse control) | controlled QA negative control | 99.54% | 4.48% | 6.80% | `experiments/direct/data/working_set_sweep_hotpotqa_comparison_r1000.json` |
| 2Wiki (low-reuse control) | controlled QA negative control | 99.96% | 0.24% | 1.80% | `experiments/direct/data/working_set_sweep_2wiki_comparison_r1000.json` |
