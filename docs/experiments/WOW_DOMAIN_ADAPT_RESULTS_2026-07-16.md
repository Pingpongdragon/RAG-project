# WoW query-adaptive cache audit (2026-07-16)

## Protocol correction

Wizard of Wikipedia exposes turn order inside each dialogue but no timestamp
across dialogues. The formal runner therefore reports:

1. `session_round_robin`: seeded interleaving with strict dialogue-order
   preservation and no claim of natural chronology;
2. `controlled_recurring_domain`: offline clustering of 591 dialogue topic
   names into eight coarse domains, followed by recurring 25-request domain
   blocks. Clustering reads neither selected knowledge nor stream order.

Using all 591 raw topics as domains is invalid for a 5,000-request recurring
experiment: the stream cannot complete one 591-domain cycle and therefore has
no domain recurrence. The formal results below use the eight-domain protocol.

The cache object is the selected knowledge sentence. Candidate knowledge IDs
are observable current-query retrieval candidates; the selected label is
revealed only after the request is scored. Each seed uses 500 warm-up and 5,000
evaluation turns, 25 turns per window, calibration-only cache sizing (21--23
sentences), and candidate budget 24.

## Five-seed result

Mean strict evidence hit percentage +/- sample standard deviation; writes are
mean physical cache writes over 5,000 evaluation requests.

### Session round-robin

| Write cap | LRU | TinyLFU | DRIP-Reactive | DRIP-DomainAdapt |
|---:|---:|---:|---:|---:|
| 1 | 6.16 +/- 0.39 / 200 | 6.77 +/- 0.58 / 200 | 10.19 +/- 0.56 / 50.0 | **10.89 +/- 0.52 / 49.8** |
| 2 | 5.57 +/- 0.22 / 400 | 7.82 +/- 0.63 / 400 | 10.39 +/- 0.27 / 103.8 | **10.86 +/- 0.22 / 103.4** |
| 3 | 5.20 +/- 0.38 / 600 | 8.12 +/- 0.70 / 600 | 9.99 +/- 0.45 / 156.6 | **10.46 +/- 0.35 / 156.2** |
| 5 | 4.45 +/- 0.34 / 1000 | 8.19 +/- 0.94 / 1000 | 8.98 +/- 0.37 / 262.8 | **9.51 +/- 0.44 / 262.2** |

### Controlled recurring eight-domain schedule

| Write cap | LRU | TinyLFU | DRIP-Reactive | DRIP-DomainAdapt |
|---:|---:|---:|---:|---:|
| 1 | 5.40 +/- 0.26 / 200 | 6.92 +/- 0.44 / 200 | 9.17 +/- 0.45 / 50.4 | **9.48 +/- 0.45 / 50.4** |
| 2 | 5.64 +/- 0.93 / 400 | 7.56 +/- 1.03 / 400 | 9.04 +/- 0.77 / 102.0 | **9.21 +/- 1.05 / 102.2** |
| 3 | 3.75 +/- 0.51 / 600 | 8.06 +/- 0.60 / 600 | 8.55 +/- 0.86 / 154.6 | **8.66 +/- 0.90 / 154.6** |
| 5 | 4.09 +/- 1.20 / 1000 | **7.89 +/- 0.57 / 1000** | 7.26 +/- 0.91 / 261.6 | 7.30 +/- 0.92 / 261.4 |

## Interpretation

- At write cap 1, DomainAdapt improves over TinyLFU by 4.12 points on
  round-robin and 2.56 points on recurring domains while using about 75% fewer
  writes.
- Candidate routing adds 0.70 points over Reactive on round-robin and 0.31
  points on recurring domains. The increment is modest but repeats across the
  five-seed mean.
- Increasing the hard cap can reduce hit rate because LRU/FIFO and eventually
  DRIP replace useful evidence too aggressively. Therefore the paper must show
  the hit--write Pareto curve, not select the cap-1 point alone.
- WoW validates multi-turn/multi-session evidence placement. It does not
  validate natural global time, and its coarse-domain schedule is controlled.

Reproduction:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python experiments/agent/run_wow_trace.py \
  --seed 42 --output experiments/agent/data/domain_adapt_wow_formal_s42.json
```
