# Figure 2 audit: cacheable domain-mixture drift

The plot source is `motivation/plotting/plot_intro_domain_adapt.py`.  Its three
reuse values use the cache object and domain definition of the corresponding
trace; they are not query-duplicate rates.

| Trace | Domain/order evidence | In-domain reusable request rate | Audit |
|---|---|---:|---|
| MIND | Official click timestamps; natural user/category mixture | 93.58% | Among 25,000 evaluation clicks after a 1,500-click warm-up, the clicked news ID had previously been clicked by a different user. There are 11,450 evaluation users and 1,603 unique clicked articles. |
| SQuAD | Four controlled recurring evidence regimes | 79.8% | `workload_factors.within_regime_repeated_support_rate` in `experiments/direct/data/baseline_matrix_smoke_squad_recurring_seed42_k4.json`. |
| WoW | Eight controlled recurring coarse domains; dialogue turn order preserved | 44.8% | Mean exact selected-knowledge reuse within the same coarse domain over seeds 42--46: 44.14%, 45.40%, 44.18%, 45.42%, and 44.86%. |

For MIND, the corresponding repeated-event rate without the different-user
restriction is 93.588%; the cross-user rate is 93.58%.  In addition, 59.83%
of the 1,603 clicked articles are accessed by more than one evaluation user.

Panel (b) uses the seed-42 matched SQuAD audit reported in the paper: stationary
versus recurring hit rates are LRU 76.2/22.0, classical ARC 74.8/19.2, and
AgentRAGCache 43.4/18.4.  The source pool, cache capacity, write cap, and support
marginals are held fixed; only the request schedule changes.
