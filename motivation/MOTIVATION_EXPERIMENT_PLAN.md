# Motivation Experiment Plan

## Purpose

This document turns the motivation story into an executable experimental plan. The core claim is not that all dynamic RAG methods fail, but that they optimize different actions than the one we care about here: selective persistent KB evolution under a fixed budget.

## Current Framing

We study a broad operational regime where a RAG system must maintain a bounded persistent KB for future queries and cannot rely on unlimited live search or unbounded storage growth.

## What This Plan Must Prove

1. This setting is practically important rather than a narrow corner case.
2. Adjacent dynamic-RAG methods solve related but different problems.
3. Query-exposed support failures are a useful signal for deciding whether to update the persistent KB and what to admit.
4. Under the same KB budget, methods should be judged by future persistent benefit rather than current-query repair alone.

## Claims and Non-Claims

### Claims

1. Persistent, budget-limited KB maintenance is a common deployment regime.
2. Existing update families focus on ingestion, retention, correction, or current-query adaptation, but rarely optimize expected future support per admitted item.
3. A query-driven update policy should reduce repeated unsupported queries more efficiently than supply-side or reuse-only baselines.

### Non-Claims

1. We do not claim persistent KB evolution is the only valid answer to dynamic knowledge.
2. We do not claim external search or retrieval-time correction are weak baselines for current-query quality.
3. We do not claim every unsupported query should cause a KB update.

## Baseline Families to Compare

1. Frozen KB: static retrieval only.
2. Supply-side incremental update: append or integrate newly arriving documents without using query-side failure signals.
3. Data-centric edit or prune: improve compactness or quality of already available content.
4. Retention and caching: reuse admitted content or past interactions more efficiently.
5. Current-query repair: retrieval correction or live search for the present request.
6. Query-driven persistent update: our target family.

## Experimental Logic

The paper should separate four quantities that are often conflated:

1. Current-query repair: did the present unsupported query get fixed?
2. Future support gain: did the update reduce unsupported failures on later queries?
3. Update cost: how much budget, churn, or admission effort was consumed?
4. Admission waste: how much persisted content produced little future reuse?

If a baseline improves the first quantity but not the second, it is solving a different problem than selective persistent KB evolution.

## Core Metrics

1. Future Support Gain (FSG): the reduction in unsupported future queries attributable to an update, measured over the next $h$ windows under the same KB budget.
2. Update Utility (UU): FSG divided by update cost. This is the main efficiency metric.
3. Wasted Update Rate (WUR): fraction of admitted items that are never used again, or never improve future support above a minimum threshold.
4. Persistent Benefit Ratio (PBR): fraction of total performance gain that comes from persisted knowledge rather than one-off current-query repair.
5. Unsupported Recovery Precision (URP): among updates triggered by support failures, how often the admitted item later helps a truly unsupported query rather than a query that was unanswerable in principle.
6. Reuse Distance (RD): number of future queries or windows before an admitted item is used again; lower is better if utility is similar.
7. Tail Support Rate (TSR): support coverage on tail or low-frequency query clusters.
8. Budgeted Coverage@B: future support coverage under a fixed KB size $B$.

## Experiment A: Diagnostic of Existing Paradigms

### Goal

Show that common update paradigms can look reasonable on aggregate quality while still underperforming on future persistent benefit.

### Reuse Existing Artifact

Use [motivation/motivation_4/existing_paradigm_diagnostic.py](motivation/motivation_4/existing_paradigm_diagnostic.py) as the starting point. It already provides:

1. Head-biased initialization.
2. Head-to-tail query drift.
3. Fixed KB budget.
4. Static, ERASE-like, ComRAG-like, and query-conditioned update baselines.

### Required Outputs

1. Per-window support coverage.
2. Tail support coverage.
3. OOS false answer rate.
4. Update cost.
5. FSG, UU, WUR, and RD.

### Reviewer-Facing Result

Show that methods focused on retention or ingestion may preserve high head performance, but spend budget on low-future-value admissions when the stream shifts toward tail or unsupported demand.

## Experiment B: Persistent Update vs Current-Query Repair

### Goal

Demonstrate that a method can repair the current query without creating durable future benefit.

### Setup

1. For each unsupported query, allow two action types: one-off repair and persistent admission.
2. Evaluate both under the same future stream.
3. Track how much of the gain survives after the current query ends.

### Key Comparison

1. Current-query correction methods should improve immediate success.
2. Query-driven persistent update should improve later support more efficiently per unit budget.

### Reviewer-Facing Result

This experiment supports the claim that neighboring work is not wrong; it is optimized for a different objective.

## Experiment C: Generality Stress Test

### Goal

Show that the regime is broad, not narrow.

### Variants

1. Head-to-tail drift.
2. Rare unsupported clusters recurring intermittently.
3. Temporal novelty where missing knowledge appears after model pretraining.
4. Mixed regime combining stable head demand with sparse tail bursts.

### Reviewer-Facing Result

If the same update logic remains useful across these variants, we can argue that persistent budgeted KB evolution is a general operating regime even if it is not the whole dynamic-RAG space.

## Data and Labeling Requirements

1. Distinguish unsupported-but-fixable queries from fundamentally unanswerable queries.
2. Label whether a later success came from persistent KB support or one-off repair.
3. Log which admitted item caused which later support gain.
4. Track admission, eviction, and reuse history for every KB item.

## Failure Modes to Guard Against

1. Confusing current-query repair with persistent update benefit.
2. Counting repeated head answers as evidence of good update strategy.
3. Updating on noisy failures that are actually unanswerable.
4. Allowing methods to exceed the same effective KB budget through hidden caches.

## Implementation Checklist

1. Extend the existing diagnostic script to log per-item admission time, reuse count, and future support gain.
2. Add a flag that separates one-off repair from persistent admission.
3. Compute the new metrics per window and cumulatively.
4. Plot tail support, UU, WUR, and PBR against update cost.
5. Report both overall averages and post-drift averages.

## Recommended Paper Story

1. Dynamic RAG is broader than persistent KB evolution.
2. Persistent, budget-limited KB maintenance is still a central deployment regime.
3. Existing methods usually optimize a neighboring action.
4. The missing question is which query-exposed failures deserve persistence.
5. Our evaluation shows that this question requires different metrics than standard answer accuracy.

---

## Engineering Specification (Focused)

### CEO-Review Verdict: Scope Reduction

The three shortfalls stated in the paper map exactly to three metrics.
That is the whole diagnostic.
Experiments B, C, D from the earlier plan are out.
The motivation needs one figure, three panels, one run.

| Paper claim | Metric | Status in existing code |
|---|---|---|
| Supply-side admission wastes budget | Wasted Update Rate (WUR) | Not yet computed |
| Budget allocation misses future demand | Future Support Gain / Update Utility (FSG, UU) | Not yet computed |
| Methods break under demand drift | Tail Support Coverage (TSC) | **Already computed** as `tail_cov_bin` per window |

TSC requires zero new code.
Only WUR and FSG/UU need to be added.

---

### What to Add: Exactly Three Things

**1. One line in `run()` — snapshot KB before `step()`**

```python
# BEFORE st.step(wq, we, w), inside the per-window loop:
kb_snapshots[w] = set(st.kb)      # add this line
st.step(wq, we, w)
```

No other change to any existing class.

**2. One new function — `compute_fsg_wur()`**

```python
def compute_fsg_wur(
    kb_snapshots: dict[int, set],   # {window: kb_set BEFORE step()}
    kb_after:     dict[int, set],   # {window: kb_set AFTER step()} = kb_snapshots[w+1]
    stream:       list[dict],
    dembs:        np.ndarray,
    t2i:          dict,
    id2pi:        dict,
    horizon:      int = 5,
) -> dict:
    """
    Returns:
      fsg_per_window  : list[float]  mean SF-coverage lift over [w+1, w+horizon]
                        caused by the docs admitted at window w
      uu_per_window   : list[float]  fsg / n_admitted  (0 if nothing admitted)
      wur             : float        fraction of admitted docs with fsg_item == 0
    """
    n_windows = len(kb_snapshots)
    fsg_wins, uu_wins = [], []
    waste_count = total_admitted = 0

    for w in range(n_windows):
        kb_pre  = kb_snapshots[w]
        kb_post = kb_snapshots.get(w + 1, kb_pre)  # kb after step w
        admitted = kb_post - kb_pre
        n_adm = len(admitted)
        total_admitted += n_adm

        if n_adm == 0 or w + horizon >= n_windows:
            fsg_wins.append(0.0)
            uu_wins.append(0.0)
            continue

        future_qs = stream[(w + 1) * WINDOW_SIZE : (w + horizon) * WINDOW_SIZE]
        base_cov  = sf_cov_binary(kb_pre,  future_qs, id2pi, dembs, t2i)
        full_cov  = sf_cov_binary(kb_post, future_qs, id2pi, dembs, t2i)
        fsg_w     = max(0.0, full_cov - base_cov)

        for did in admitted:
            kb_minus = kb_post - {did}
            cov_minus = sf_cov_binary(kb_minus, future_qs, id2pi, dembs, t2i)
            if full_cov - cov_minus < 1e-4:   # item contributed nothing
                waste_count += 1

        fsg_wins.append(fsg_w)
        uu_wins.append(fsg_w / n_adm)

    wur = waste_count / total_admitted if total_admitted > 0 else 0.0
    return {
        'fsg_per_window': fsg_wins,
        'uu_per_window':  uu_wins,
        'fsg_mean':       float(np.mean(fsg_wins)),
        'uu_mean':        float(np.mean(uu_wins)),
        'wur':            wur,
    }
```

**3. One new figure — `make_figure_three_metrics()`**

Three-panel figure, one row, same x-axis (window index):

| Panel | y-axis | What it shows |
|---|---|---|
| Left | Tail Support Coverage (`tail_cov_bin`) | Drift degradation per strategy |
| Middle | Cumulative FSG (sum of `fsg_per_window`) | Future benefit accumulated over time |
| Right | WUR (single bar per strategy) | Admission waste rate |

```python
def make_figure_three_metrics(results, fsg_wur):
    """
    results   : existing dict from run(), contains 'tail_cov_bin' per window
    fsg_wur   : dict keyed by strategy name, each value from compute_fsg_wur()
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    w = np.arange(N_WINDOWS)

    # Panel 1: Tail Support Coverage over windows
    ax = axes[0]
    for name in ORDER:
        m = results[name]
        tsc = [x['tail_cov_bin'] for x in m]
        ax.plot(w, tsc, label=LABELS[name],
                color=STYLES[name]['c'], ls=STYLES[name]['ls'],
                marker=STYLES[name]['m'], markevery=4, markersize=4)
    ax.axvline(N_WINDOWS // 2, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Window'); ax.set_ylabel('Tail Support Coverage')
    ax.set_title('(a) TSC under demand drift')
    ax.legend(fontsize=7)

    # Panel 2: Cumulative FSG
    ax = axes[1]
    for name in ORDER:
        fsg = fsg_wur[name]['fsg_per_window']
        ax.plot(w, np.cumsum(fsg), label=LABELS[name],
                color=STYLES[name]['c'], ls=STYLES[name]['ls'])
    ax.set_xlabel('Window'); ax.set_ylabel('Cumulative FSG')
    ax.set_title('(b) Future Support Gain (cumulative)')

    # Panel 3: WUR bar
    ax = axes[2]
    wurs = [fsg_wur[n]['wur'] for n in ORDER]
    bars = ax.bar(ORDER, wurs, color=[STYLES[n]['c'] for n in ORDER], width=0.5)
    ax.set_ylabel('Wasted Update Rate'); ax.set_ylim(0, 1)
    ax.set_title('(c) Admission Waste Rate')
    ax.set_xticklabels([LABELS[n].split()[0] for n in ORDER], fontsize=8)

    plt.tight_layout()
    save_fig(fig, FIG_DIR / 'fig_three_metrics.pdf')
```

---

### Patch to `run()` — Complete Minimal Diff

```python
def run(doc_pool, dembs, stream, se, t2i, qembs, centroids, head_set):
    ...
    results = {}
    for name, st in strats.items():
        st.set_kb(init)
        metrics = []
        kb_snapshots = {}                           # NEW
        for w in range(N_WINDOWS):
            kb_snapshots[w] = set(st.kb)            # NEW  ← before step
            a, b = w*WINDOW_SIZE, (w+1)*WINDOW_SIZE
            wq = stream[a:b]; we = se[a:b]
            # ... existing metric computation unchanged ...
            st.step(wq, we, w)
        kb_snapshots[N_WINDOWS] = set(st.kb)        # NEW  ← final snapshot

        post = compute_fsg_wur(
            kb_snapshots,
            {w: kb_snapshots[w+1] for w in range(N_WINDOWS)},
            stream, dembs, t2i, id2pi,
        )                                           # NEW
        results[name] = {'windows': metrics, 'post': post}   # shape change
    return results
```

---

### What Not to Build

- Experiment B (repair vs persist): not needed for motivation
- Experiment C (generality stress test): not needed for motivation
- Experiment D (signal quality ablation): not needed for motivation
- `AdmissionLedger` class: the snapshot dict is sufficient; a full ledger is over-engineering for this diagnostic
- `compute_pbr`, `compute_urp`: not needed for the three-panel figure
- Per-item marginal counterfactual for WUR: the full-vs-minus-one loop above is the right granularity

### Output Expected

Running the patched script produces:
- `figures/fig_three_metrics.pdf` — the motivation figure
- `results[name]['post']` — scalar WUR, FSG mean, UU mean per strategy
- Per-window `tail_cov_bin` — already in `results[name]['windows']`

The expected result, if the framing is correct:

- Static and ComRAG: low FSG, high WUR, TSC collapses after drift midpoint
- ERASE: moderate TSC (supply helps initially), WUR improves slightly, FSG still low
- QARC (ours): higher FSG, lower WUR, TSC degrades less sharply

If all four methods show equally low FSG and high WUR, the metric design needs revisiting before the paper is submitted.
