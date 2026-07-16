# DRIP Budgeted-Replacement SOTA (2026-07-15)

## Claim boundary

The defensible SOTA setting is **natural, write-constrained evidence
replacement**, not unconstrained synthetic QA recall:

- MIND-small official positive-click stream in timestamp order;
- 51,282 cold items and a 513-item hot tier (1%);
- 1,500 causal warm-up events, disjoint from evaluation;
- 200,000 evaluation accesses (400 windows x 500);
- at most 10 physical writes per window (2% of requests);
- the clicked item is revealed only after serving the request and can affect
  the next window, exactly as in a standard cache-access trace.

The exploratory transition forecast and entity-neighbour expansion are disabled
in this result. They did not improve the strict 2Wiki bridge stream and are not
part of the main formula below.

## Verified result

The authoritative 200k-event run is
`experiments/agent/data/sota_mind_authoritative_cli_cap10_400w500_s42.json`.

| Policy | Has-Answer / R@5 | AMAT | Writes | Maintenance reads |
|---|---:|---:|---:|---:|
| LRU | 86.5 | 2.350 | 3,997 | 22,966 |
| TinyLFU | 44.9 | 6.510 | 4,000 | 107,002 |
| AgentRAGCache | 70.7 | 3.930 | 1,150 | 58,578 |
| **DRIP** | **90.6** | **1.939** | **3,996** | **18,776** |

Against LRU, DRIP gains 4.069 points averaged over 400 paired windows; a paired
window bootstrap gives a 95% CI of `[3.697, 4.464]` points. It wins 374 windows,
ties 11, reduces misses by 30.37%, AMAT by 17.35%, and maintenance reads by
18.24%, with one fewer physical write.

The 25k-event full-baseline check was repeated with seeds 42--46. DRIP is
`87.4 +/- 0.0`, AgentRAGCache `85.9 +/- 0.0`, LRU `80.78 +/- 0.18`, TinyLFU
`79.46 +/- 0.34`, Proximity `42.8 +/- 0.0`, and GPTCache-style
`41.48 +/- 0.82`. The fixed natural event sequence makes most variation come
only from seeded initialization/tie breaking.

## Final online objective

Let `K_t` be the hot tier before window `t`, `C` its capacity, `A_t` admitted
items, `V_t` evicted residents, `R_t = |A_t| = |V_t|`, and `B_t` the hard
write cap. DRIP solves the causal proxy problem

```text
maximize    sum_t [ sum_(c in A_t) U_t(c) - sum_(v in V_t) U_t(v) ]
subject to  K_(t+1) = (K_t \ V_t) union A_t,
            |K_t| <= C,
            |A_t| = |V_t|,
            R_t <= B_t,
            limsup_T (1/T) sum_(t=1)^T R_t / max(1,B'_t) <= b_rep,
```

where `B'_t = min(B_t, M_t)` and `M_t` is the number of under-covered
requests observed in window `t`. No future request or gold support enters
`U_t`.

## Observable evidence

For an item-access trace, the request key `x` is observed after service:

```text
E_t(x,d) = 1[d = x].
```

For ordinary QA where the evidence key is unknown, a bounded miss probe gives

```text
g_t(q,d) = 1[d in TopK(q)] *
           { s(q,d) / [r_q(d) (epsilon + 1 - s(q,d))^alpha]
             + b_1 1[r_q(d)=1] },

E_t(q,d) = g_t(q,d) / sum_j g_t(q,j).
```

Thus each miss contributes one unit of measured evidence flow (MEF), independent
of probe width.

## Demand, serve, and the two experts

```text
D_t(d) = beta_D D_(t-1)(d) + sum_(q in misses_t) E_t(q,d),
S_t(d) = beta_S S_(t-1)(d) + A_t^serve(d).
```

Let `Pct(.)` be the tie-aware percentile normalization over residents and
candidates. The recency and reactive experts are

```text
z_t^LRU(d) = Pct(last_seen_t(d)),
z_t^RE (d) = Pct(S_t(d) + D_t(d)).
```

Using the current observable support proxy set `O_t`, delayed expert loss is

```text
ell_(t,k) = (1/|O_t|) sum_(d in O_t) [1 - z_(t-1)^k(d)].
```

Fixed-Share then updates

```text
eta_t       = min(1, sqrt(2 log 2 / t)),
h_t(k)      = exp(-eta_t L_t(k)) / sum_j exp(-eta_t L_t(j)),
alpha_t     = min(0.25, 1/sqrt(t+1)),
w_t(k)      = (1-alpha_t) h_t(k) + alpha_t/2,
U_t(d)      = w_t(LRU) z_t^LRU(d) + w_t(RE) z_t^RE(d).
```

## Budgeted replacement rule

For the highest-utility candidate `c` and lowest-utility resident `v`,

```text
Delta_t(c,v) = U_t(c) - U_t(v) - lambda_t.
```

DRIP swaps `v` for `c` iff `Delta_t(c,v) > 0`, the semantic duplicate gate is
not violated, and fewer than `B'_t` items have been written. The shadow price is

```text
rho_t        = R_t / max(1,B'_t),
gamma_t      = 1/sqrt(t+1),
lambda_(t+1) = max(0, lambda_t + gamma_t (rho_t - b_rep)).
```

The SOTA operating point uses `C=513`, `B_t=10`, and `b_rep=1.0`. Lower
`b_rep` values trace the quality-write Pareto frontier without changing the
admission formula.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
EMBED_MODEL=all-MiniLM-L6-v2 \
EXPERIMENT_SEED=42 DATA_SEED=42 \
DRIP_REPLACEMENT_TARGET=1.0 \
DRIP_FORECAST_TOPK=0 DRIP_BRIDGE_TOPK=0 \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python \
experiments/hidden/run.py \
  --datasets mind_news_access \
  --n-windows 400 --window-size 500 \
  --workload natural_temporal --temporal-sampling prefix \
  --warmup-windows 3 --kb-pool-ratio 0.01 --write-cap 10 \
  --strategies LRU TinyLFU AgentRAGCache DRIP \
  --output sota_mind_authoritative_cli_cap10_400w500_s42.json
```
