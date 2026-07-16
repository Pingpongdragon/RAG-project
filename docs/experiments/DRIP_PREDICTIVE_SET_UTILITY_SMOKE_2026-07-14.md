# DRIP Predictive Set Utility: Unified Smoke Study (2026-07-14)

## 1. What changed

The current method does not maintain separate rules for reactive admission,
prefetching, and hidden-evidence bundles. Every candidate generator proposes an
evidence group `G`; the cache optimizer evaluates the same prospective set
exchange:

```text
Delta_t(G,V)
  = U_hat_t((K_t \ V) union (G \ K_t))
    - U_hat_t(K_t)
    - lambda_t |G \ K_t|.

execute (G,V) iff Delta_t(G,V) > 0.
```

`U_hat_t` contains calibrated singleton utility and relation complementarity:

```text
U_hat_t(K)
  = sum_{d in K} u_hat_t(d)
    + sum_{g in E_t} m_hat_t(g) I[g subseteq K].
```

Direct evidence and forecast evidence are singleton groups. Hidden evidence is
an atomic hyperedge, so a partial reasoning chain receives no relation utility.
Victims are chosen by the true marginal loss of the same set function. The
primal-dual shadow price `lambda_t` is the only write-cost term.

Active prefetching is described as **transition-conditioned evidence
forecasting**, not as a heuristic writer. From causal history through window
`t`, the forecaster estimates the next evidence distribution:

```text
x_hat^M_{t+1}(d) = sum_a x_t(a) P_hat_t(d | a)

x_hat^K_{t+1}(d | z_t)
  = sum_{i<t} w_i(z_t) x_{i+1}(d)

F_t(d) = |Q_t| c_t x_hat_{t+1}(d).
```

The training pair `(z_i,x_{i+1})` is recorded only after window `i+1` arrives.
The one-step Forecasted Evidence Utility `F_t` expires before the next update and
enters `U_hat_t`; it never bypasses the unified action rule.

## 2. Protocol changes

The controlled workload now separates four measurable factors: evidence-set
drift, within-regime evidence reuse, causal transition predictability, and
evidence visibility.

- SQuAD and HotpotQA regimes are built offline from sparse TF-IDF features of
  evidence text. The online policy and detector use dense support proxies, so
  the stream is not created and validated with the same embedding.
- Exact query text is never duplicated. Reuse comes from distinct queries that
  share evidence families.
- `factorized_recurring` and `factorized_shuffled` preserve regime marginals and
  support reuse while changing only transition order.
- HotpotQA can use an anchor evidence family and a declared minimum support
  frequency instead of fabricating repeated queries.
- StreamingQA preserves official chronology. MIND context queries use only
  pre-click context rather than the target news title.

## 3. Smoke results

These are one-seed mechanism checks, not final paper numbers. `NoF` is the
identical unified optimizer with the forecast generator disabled. `FWrite` is
the number of admitted documents whose action source is forecast evidence;
`PHit` counts predicted candidates observed in the next support state.

| Workload | Reuse | Causal transition acc. | Policy | HasAns | AMAT | Repl. | FWrite | PHit |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| SQuAD recurring | 0.800 | 0.789 | LRU | 1.8 | 10.82 | 467 | 0 | 0 |
|  |  |  | DRIP-NoF | 32.0 | 7.80 | 168 | 0 | 0 |
|  |  |  | DRIP-Predictive | **32.8** | **7.72** | 173 | 25 | 75 |
| SQuAD matched shuffled | 0.800 | 0.158 | LRU | 14.4 | 9.56 | 418 | 0 | 0 |
|  |  |  | DRIP-NoF | **28.8** | **8.12** | 165 | 0 | 0 |
|  |  |  | DRIP-Predictive | **28.8** | **8.12** | 163 | 23 | 25 |
| HotpotQA reusable direct | 0.303 | 0.429 | LRU | 13.5 | 9.65 | **97** | 0 | 0 |
|  |  |  | DRIP-NoF | 14.5 | 9.55 | 152 | 0 | 0 |
|  |  |  | DRIP-Predictive | **15.0** | **9.50** | 156 | 2 | 0 |
| MIND causal context prefix | 0.505 | 0.000 | LRU | **50.5** | **5.95** | 95 | 0 | 0 |
|  |  |  | DRIP-NoF | 43.0 | 6.70 | 48 | 0 | 0 |
|  |  |  | DRIP-Predictive | 43.5 | 6.65 | **43** | 4 | 4 |
| StreamingQA chronological prefix | 0.170 | 0.000 | LRU | **17.5** | **9.25** | 146 | 0 | 0 |
|  |  |  | DRIP-NoF | 7.5 | 10.25 | **115** | 0 | 0 |
|  |  |  | DRIP-Predictive | 7.5 | 10.25 | 125 | 6 | 0 |

## 4. Interpretation

The matched SQuAD pair is the cleanest mechanism test. With the same reuse and
regime marginals, forecast improves HasAns by 0.8 points only when the transition
order is causally predictable; it is neutral after matched shuffling. This is
the desired qualitative interaction, but one seed and a 0.8-point gain are not
enough for a paper claim.

HotpotQA and MIND show only 0.5-point changes. HotpotQA records no next-state
prediction hit, so its small difference cannot be credited confidently to
forecasting. The MIND and StreamingQA prefix runs cover only about 0.55% of
their source timeline and must not be presented as full natural-trace evidence.

StreamingQA is the important negative control. It has only 17% repeated support
and no learned transition recurrence in this prefix. Predictive DRIP adds ten
replacements over NoF without improving answerability. The unified formula is
correctly shared, but forecast confidence and the dual price are not yet
conservative enough in a non-predictable natural stream.

The wider forecast-support experiment was also negative: expanding beyond the
sparse top-k support reduced SQuAD recurring HasAns from 32.8 to 31.2. The code
therefore keeps a sparse forecast support tied to the existing cold-probe top-k.

## 5. Result files

```text
experiments/direct/data/
  smoke_factorized_squad_recurring_unified_v2_20w25_kb05.json
  smoke_factorized_squad_shuffled_unified_v2_20w25_kb05.json
  smoke_factorized_hotpot_reusable_conditional_8w25.json
  smoke_streamingqa_natural_forecast_ablation_8w25.json

experiments/hidden/data/
  smoke_mind_context_prefix_forecast_ablation_8w25.json
```

## 6. Reproduction commands

Run from `/data/jyliu/RAG-project`:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DRIP_REPLACEMENT_TARGET=0.25
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python
```

SQuAD predictable recurrence:

```bash
CUDA_VISIBLE_DEVICES=0 $PY experiments/direct/run.py \
  --datasets squad --n-source 1000 --n-windows 20 --window-size 25 \
  --workload factorized_recurring --drip-ablation \
  --strategies LRU DRIP-WorkingSet-NoForecast DRIP-WorkingSet \
  --kb-pool-ratio 0.05 --warmup-windows 1 \
  --output smoke_factorized_squad_recurring_unified_v2_20w25_kb05.json
```

Matched low-predictability control:

```bash
CUDA_VISIBLE_DEVICES=0 $PY experiments/direct/run.py \
  --datasets squad --n-source 1000 --n-windows 20 --window-size 25 \
  --workload factorized_shuffled --drip-ablation \
  --strategies LRU DRIP-WorkingSet-NoForecast DRIP-WorkingSet \
  --kb-pool-ratio 0.05 --warmup-windows 1 \
  --output smoke_factorized_squad_shuffled_unified_v2_20w25_kb05.json
```

HotpotQA reusable direct-evidence core:

```bash
CUDA_VISIBLE_DEVICES=1 $PY experiments/direct/run.py \
  --datasets hotpotqa_comparison --n-source 4000 \
  --n-windows 8 --window-size 25 --workload factorized_recurring \
  --factorized-min-support-frequency 2 --factorized-family-mode anchor \
  --drip-ablation \
  --strategies LRU DRIP-WorkingSet-NoForecast DRIP-WorkingSet \
  --kb-pool-ratio 0.1 --warmup-windows 1 \
  --output smoke_factorized_hotpot_reusable_conditional_8w25.json
```

MIND causal context view:

```bash
CUDA_VISIBLE_DEVICES=2 $PY experiments/hidden/run.py \
  --datasets mind_news_context --workload natural_temporal \
  --n-windows 8 --window-size 25 --temporal-sampling prefix \
  --drip-ablation \
  --strategies LRU DRIP-WorkingSet-NoForecast DRIP-WorkingSet \
  --kb-pool-ratio 0.01 --warmup-windows 1 \
  --output smoke_mind_context_prefix_forecast_ablation_8w25.json
```

StreamingQA official chronology:

```bash
CUDA_VISIBLE_DEVICES=3 $PY experiments/direct/run.py \
  --datasets streamingqa_official --n-windows 8 --window-size 25 \
  --workload natural_temporal --temporal-sampling prefix \
  --drip-ablation \
  --strategies LRU DRIP-WorkingSet-NoForecast DRIP-WorkingSet \
  --kb-pool-ratio 0.1 --warmup-windows 1 \
  --output smoke_streamingqa_natural_forecast_ablation_8w25.json
```

## 7. Required next experiments

1. Repeat recurring/shuffled matched pairs with at least five data seeds.
2. Sweep cache ratio and `DRIP_REPLACEMENT_TARGET` jointly and report the
   HasAns--replacement Pareto frontier rather than selecting one point.
3. Use `window_span` for StreamingQA and MIND so the sampled workload covers the
   natural timeline while preserving local contiguous windows.
4. Add forecast precision, coverage, pollution, and avoided cold accesses to
   the main result schema.
5. Build the hidden protocol separately from relation-complete evidence units;
   do not infer hidden success from the direct workloads above.
