# Motivation Storyline — DRIP-Dense, DRIP-ESC, and DRIP-ESC-Lease

> Purpose: one-day presentation note for the current preliminary motivation result.

## Core Position

`DRIP-Dense` is the query-visible dense/direct channel. ARC shows that
query-side demand can be made more algorithmic with DRF and hubness. Therefore,
the paper should not frame the final contribution as "we score document
demand." The stronger story is:

> Query-side admission, including `DRIP-Dense` and ARC-style DRF+hubness, is
> still blind to hidden second-hop documents in bridge multi-hop RAG. DRIP adds
> evidence-conditioned hidden-support completion (`DRIP-ESC`) and pair lease
> retention (`DRIP-ESC-Lease`) for reusable A+B support pairs in the shared hot tier.

## Latest MO Preview

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python experiments/hidden/run.py \
  --expanded \
  --q-type bridge_comparison \
  --n-source 2000 \
  --strategies Static LRU TinyLFU AgentRAGCache AgentRAGCache_NoHub DRIP-Dense DRIP-ESC DRIP-ESC-Lease Oracle \
  --datasets 2wikimultihopqa \
  --n-windows 8 \
  --window-size 25 \
  --output mo_story_2wiki_bridge_8w25_n2000_drip_ablation.json
```

Result file:

`experiments/hidden/data/mo_story_2wiki_bridge_8w25_n2000_drip_ablation.json`

Setting: `2Wiki bridge_comparison`, pool `10809`, KB `850`, stream `8x25`, sudden drift.

| Method | Role | R@5 H1 | R@5 H2 | Writes | MaintR |
|---|---|---:|---:|---:|---:|
| LRU | access-history cache | 52.8 | 2.2 | 99 | 113 |
| TinyLFU | frequency cache | 52.8 | 2.2 | 99 | 113 |
| AgentRAGCache_NoHub | ARC ablation, DRF only | 32.5 | 1.8 | 1074 | 143 |
| AgentRAGCache | ARC full, DRF + hubness | 38.5 | 3.0 | 1062 | 130 |
| DRIP-Dense | dense/direct channel only | 52.8 | 2.2 | 98 | 565 |
| DRIP-ESC | + evidence-conditioned hidden support | 46.2 | 5.2 | 1483 | 920 |
| DRIP-ESC-Lease | + pair lease | 46.2 | 5.3 | 858 | 928 |
| Oracle | upper bound | 53.5 | 50.8 | 2228 | 0 |

## What This Shows

1. **DRIP-Dense is not enough on bridge H2.**
   `DRIP-Dense` has H2 `2.2`, the same as LRU/TinyLFU in this preview. It preserves H1 but does not admit hidden second-hop documents.

2. **ARC is a strong nearest-neighbor threat, but still query/geometry-side.**
   ARC full improves H2 to `3.0`, while DRF-only is `1.8`. Hubness helps ARC, but full ARC still does not close the bridge gap and pays many writes.

3. **ESC is the first real bridge signal.**
   `DRIP-ESC` raises H2 from `DRIP-Dense`'s `2.2` to `5.2` by using `q -> A -> B` evidence conditioning. This is the main evidence that the missing signal is hidden-support completion, not just a better query-demand formula.

4. **DRIP-ESC-Lease's role is retention.**
   `DRIP-ESC-Lease` keeps the completed A+B pair resident after the hidden support is found. This tests whether pair lease turns a one-time bridge completion into reusable cache value.

## Slide Story

### Slide 1: Problem

In Agent RAG, the full corpus is L2 and the L1 hot tier is small. Under query drift and bridge multi-hop, the second-hop document is often not directly similar to the query.

```text
q -> A -> B
```

`A` is query-visible. `B` is hidden unless we inspect content/entities from `A`.

### Slide 2: Nearest Threat

ARC already has a sophisticated query/geometry score:

```text
DRF(p) = sum_q 1 / (rank(q,p) * dist(q,p)^alpha)
h_k(p) = number of docs for which p is a kNN neighbor
Priority(p) ~= beta * log(h_k(p)+1) + (1-beta) * DRF(p)
```

This means the novelty is not "query demand scoring." ARC and DRIP-Dense are both in the query-side admission family.

### Slide 3: Failure of Query-Side Admission

On 2Wiki bridge:

```text
LRU H2             2.2
DRIP-Dense H2         2.2
ARC full H2        3.0
Oracle H2         50.8
```

Even stronger query/geometry scoring leaves a large gap, because hidden `B` is not visible from `sim(q, doc)`.

### Slide 4: Our Added Signal

DRIP-ESC adds evidence-conditioned hidden-support completion:

```text
failed q
  -> retrieve or observe first-hop A
  -> condition on A's evidence
  -> score hidden support candidates B
  -> credit bridge demand to B
```

Result:

```text
DRIP-Dense H2       2.2
DRIP-ESC H2   5.2
```

### Slide 5: DRIP-ESC-Lease Role

DRIP-ESC-Lease wraps DRIP-ESC with:

```text
LEASE: protect completed A+B support pairs
ADMIT: DRIP-Dense + ESC candidates compete in one support-priority ledger
```

Current result:

```text
DRIP-ESC H2   5.2, writes 1483
DRIP-ESC-Lease H2  5.3, writes 858
```

So the current lease-ablation role is:

> DRIP-ESC supplies the hidden-support signal; DRIP-ESC-Lease makes completed support pairs survive long enough to be reused.

## Main Claim

Do not claim:

> We introduce demand-based caching.

Claim:

> We show that query-side demand admission, even ARC-style DRF+hubness, is insufficient for bridge multi-hop Agent RAG. DRIP extends `DRIP-Dense` with evidence-conditioned hidden-support completion and pair lease retention in the shared L1 document cache.
