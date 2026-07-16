# Causal Region-Shadow Prefetch Audit

Audit version: `causal-region-shadow-v1`

This audit tests a semantic-only minimum of Approach B: region-level forecasting, document-level selection, and a TTL=1 shadow buffer. `CausalShadow` uses only support feedback observed through window t. `OracleDocShadow` sees the next window and is an unattainable mechanism upper bound, not a method result. Total cache capacity is fixed.

## Command

```bash
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python benchmarks/audit_region_shadow_prefetch.py --datasets squad_direct hotpotqa_comparison 2wiki_comparison streamingqa_official --region-sizes 128 256 512 --region-widths 1 2 4 --shadow-fractions 0.05 0.10 0.20 --cache-ratio 0.10 --output docs/experiments/REGION_SHADOW_AUDIT_2026-07-15.json --report docs/experiments/REGION_SHADOW_AUDIT_2026-07-15.md
```

## Best Causal Setting Per Dataset

| Dataset | Region | Setting | Locality | Markov cov. | Freq. cov. | Selected-doc cov. | Full LRU HA | Hot-only HA | Causal HA | Oracle HA | Gain | Prefetch precision | Write ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| squad_direct | 512 | `L4_S0.05` | 100.0% | 94.7% | 100.0% | 16.2% | 84.8% | 68.8% | 84.2% | 87.8% | -0.6 pp | 91.7% | 3.86x |
| hotpotqa_comparison | 256 | `L1_S0.05` | 0.0% | 5.7% | 2.0% | 0.0% | 26.0% | 24.0% | 24.5% | 96.5% | -1.5 pp | 0.0% | 1.04x |
| 2wiki_comparison | 512 | `L2_S0.10` | 0.0% | 6.9% | 0.0% | 0.0% | 14.0% | 13.5% | 14.0% | 92.5% | +0.0 pp | 0.0% | 1.02x |
| streamingqa_official | 128 | `L4_S0.05` | 2.6% | 1.9% | 1.8% | 0.0% | 2.1% | 2.1% | 2.1% | 46.0% | -0.0 pp | 0.0% | 1.00x |

## Verdict

- **squad_direct:** best `512/L4_S0.05` gives -0.6 pp Has-Answer versus full-capacity LRU. Markov region coverage is -5.3 pp versus historical frequency; selected-document coverage is 16.2%, and prefetch precision is 91.7%.
- **hotpotqa_comparison:** best `256/L1_S0.05` gives -1.5 pp Has-Answer versus full-capacity LRU. Markov region coverage is +3.7 pp versus historical frequency; selected-document coverage is 0.0%, and prefetch precision is 0.0%.
- **2wiki_comparison:** best `512/L2_S0.10` gives +0.0 pp Has-Answer versus full-capacity LRU. Markov region coverage is +6.9 pp versus historical frequency; selected-document coverage is 0.0%, and prefetch precision is 0.0%.
- **streamingqa_official:** best `128/L4_S0.05` gives -0.0 pp Has-Answer versus full-capacity LRU. Markov region coverage is +0.1 pp versus historical frequency; selected-document coverage is 0.0%, and prefetch precision is 0.0%.

The causal method beats full-capacity document LRU on 0/4 datasets. A positive oracle gap alone does not validate forecasting: it only says that perfect future information would make reserved shadow capacity useful.

## Interpretation Rules

- Markov coverage must exceed the frequency baseline; otherwise the model learns popularity, not a transition.
- Selected-document coverage measures the full forecasting bottleneck after region routing and within-region ranking.
- `Hot-only HA` exposes the capacity tax caused by reserving shadow space.
- `Oracle HA` separates a bad causal predictor from a fundamentally useless shadow mechanism.
- The semantic-only version is a prerequisite test. An entity view is justified only where relation feedback exists and adds coverage beyond semantic regions.
