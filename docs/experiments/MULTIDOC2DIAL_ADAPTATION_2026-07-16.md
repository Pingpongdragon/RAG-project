# MultiDoc2Dial query-adaptation audit (2026-07-16)

## Decision

MultiDoc2Dial is the closest current benchmark to the paper's intended setting:
document-grounded agent requests switch documents inside a session, while the
same cold documents are reused across many requests.  It supports a defensible
two-stage claim:

1. **query-adaptive retrieval** selects a small domain/region for the current
   request and ranks concrete documents inside it;
2. **persistent placement** retains only documents supported by post-service
   evidence feedback under a switching budget.

The experiment does **not** support future-topic prediction or whole-topic
cache loading.  Current routed candidates are useful for serving the current
request, but they add almost no reliable signal about future residency.

## Dataset audit

The official archive contains 488 documents in four domains.  On the training
split, agent responses contain 24,603 document-evidence occurrences over 415
unique documents; 98.31% of occurrences are repeats after the first occurrence.
At span granularity, 68.91% of 39,304 occurrences repeat.  Moreover, 79.85% of
dialogues use more than one document, and consecutive evidence-bearing agent
turns change document 21.08% of the time.

These statistics establish both sides of the workload: document shift exists,
and exact evidence reuse is high enough for caching to be meaningful.  The
released JSON order is not a global timestamp.  We therefore use either a
seeded session round-robin or a clearly labelled controlled recurring-domain
schedule; both preserve turn order inside every dialogue.

## Causal query-adaptive retrieval

Protocol: official test split, 4,335 user-to-agent requests, current query plus
causal dialogue history, TF-IDF domain centroids, and in-domain Top-4 document
ranking.  Gold references are used only after routing for offline evaluation.

| Retrieval path | Domains routed | Docs scanned/query | Support recall@4 |
|---|---:|---:|---:|
| Full-pool ranking | 4 | 488.0 | 80.30% |
| Query-adaptive routing | 1 | 125.4 | 77.72% |
| Query-adaptive routing | 2 | 249.7 | 79.93% |

Top-2 routing preserves all but 0.37 percentage points of full-pool recall
while scanning 48.8% fewer documents.  This is a retrieval-efficiency result,
not a persistent-cache hit.

## Persistent document residency

The formal replay uses 500 warm-up and 3,800 evaluation requests, window size
25, capacity calibrated only from the warm-up, and exact document IDs as
post-service feedback.  Values below are strict document hit rate / cache
writes.  The routed candidate list is current-query-only and contains no gold
label at construction time.

### Session round-robin

| Write cap/window | LRU | TinyLFU | DRIP-Reactive | DRIP-DomainAdapt |
|---:|---:|---:|---:|---:|
| 1 | 24.26 / 152 | 23.16 / 152 | **30.45 / 38** | 30.34 / **37** |
| 3 | 17.97 / 456 | 28.79 / 456 | 29.76 / 117 | **30.00 / 117** |
| 5 | 17.82 / 760 | 28.42 / 760 | 28.00 / 191 | **28.50 / 189** |

### Controlled recurring domains

| Write cap/window | LRU | TinyLFU | DRIP-Reactive | DRIP-DomainAdapt |
|---:|---:|---:|---:|---:|
| 1 | 22.18 / 152 | **27.45 / 152** | 26.74 / 39 | 27.05 / **38** |
| 3 | 18.39 / 456 | **25.50 / 456** | 24.74 / 108 | 24.87 / **105** |
| 5 | 14.13 / 760 | **24.21 / 760** | 23.63 / **176** | 23.71 / **176** |

The clean Pareto point is session round-robin with write cap 1: DRIP-Reactive
improves LRU by 6.19 hit points with 75% fewer writes.  Candidate-based
DomainAdapt changes persistent hit by at most 0.50 points and sometimes hurts.
Thus adaptation should be claimed as a current-retrieval layer; the cache core
should remain downstream-evidence-informed, switching-cost-aware placement.

## Reproduction

```bash
python experiments/agent/audit_multidoc2dial.py \
  --output experiments/agent/data/multidoc2dial_cacheability_audit.json

python experiments/agent/run_multidoc2dial_adaptation.py \
  --split test --protocol both --warmup-size 500 --evaluation-size 3800 \
  --window-size 25 --block-size 25 --route-width 1 --retrieve-topk 4 \
  --candidate-budget 4 --write-budgets 1 3 5 --seed 42 \
  --output experiments/agent/data/multidoc2dial_domain_adaptation_s42.json
```

Top-2 retrieval trade-off:

```bash
python experiments/agent/run_multidoc2dial_adaptation.py \
  --split test --protocol session_round_robin --warmup-size 500 \
  --evaluation-size 3800 --window-size 25 --route-width 2 \
  --retrieve-topk 4 --candidate-budget 4 --write-budgets 1 \
  --policies DRIP-Reactive DRIP-DomainAdapt --seed 42 \
  --output experiments/agent/data/multidoc2dial_domain_adaptation_route2_s42.json
```
