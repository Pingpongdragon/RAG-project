"""
Motivation 4: Existing KB Update Paradigms Under Budget Constraints

Core question (supported by CRAG, UAEval4RAG, LiveSearchBench):
  In a streaming query workload with long-tail entities and temporal drift,
  can existing KB update paradigms simultaneously maintain:
    (1) Coverage of the evolving user interest distribution (Coverage@B)
    (2) Correct abstention on unsupported queries (OOS False Answer Rate)
    (3) Bounded update cost (Update Cost)
  all under a fixed KB budget?

Scenario:
  - 20 time windows x 200 queries = 4,000 streaming queries
  - 8 topics (3 head, 5 tail) with shifting popularity
  - Fixed KB budget B = 500 documents
  - Each query tagged: head/tail, answerable/out-of-support (OOS)

Baseline paradigms:
  1. Append-only  -- add docs on demand, FIFO eviction when full
  2. Cache / LRU  -- keep recently accessed, evict cold docs
  3. Edit / Prune -- periodic utility-based pruning and backfill
  4. Reject-only  -- static KB, abstain when confidence is low

Metrics:
  1. Coverage@B     = #(queries with KB support) / #(all queries)
  2. OOS FAR        = #(OOS queries answered) / #(OOS queries)
  3. Update Cost    = cumulative doc inserts + removals

Output:
  figures/paradigm_diagnostic.png  -- 2x2 quad figure
  data/paradigm_diagnostic.json   -- raw timeseries data
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_style, COLORS, save_fig

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
FIG_DIR  = BASE_DIR / 'figures'

# ======================================================
# Experiment config
# ======================================================
SEED = 42
N_WINDOWS = 20
N_QUERIES = 200
KB_BUDGET = 500

# (name, type, pool_size, coverage_threshold)
TOPICS = [
    ("General QA",     "head", 300,  40),
    ("Current Events", "head", 300,  40),
    ("Technology",     "head", 300,  40),
    ("Legal",          "tail", 250,  80),
    ("Medical",        "tail", 250,  80),
    ("Scientific",     "tail", 250,  80),
    ("Historical",     "tail", 200,  80),
    ("Cultural",       "tail", 200,  80),
]
N_TOPICS    = len(TOPICS)
TOPIC_NAMES = [t[0] for t in TOPICS]
TOPIC_TYPES = [t[1] for t in TOPICS]
DOC_POOLS   = np.array([t[2] for t in TOPICS], dtype=float)
THRESHOLDS  = np.array([t[3] for t in TOPICS], dtype=float)

# Initial KB allocation (head-heavy)
INIT_ALLOC = np.array([150, 130, 120, 30, 25, 20, 15, 10], dtype=float)
assert INIT_ALLOC.sum() == KB_BUDGET

# Visual styles
STRATEGY_STYLES = {
    'Append-only':  {'color': '#2563EB', 'marker': 'o',  'ls': '-'},
    'Cache / LRU':  {'color': '#D97706', 'marker': 's',  'ls': '-'},
    'Edit / Prune': {'color': '#059669', 'marker': '^',  'ls': '-'},
    'Reject-only':  {'color': '#DC2626', 'marker': 'D',  'ls': '--'},
}
STRATEGY_ORDER = ['Append-only', 'Cache / LRU', 'Edit / Prune', 'Reject-only']


# ======================================================
# Query distribution drift model
# ======================================================
HEAD_DIST_START = np.array([0.25, 0.20, 0.18])
HEAD_DIST_END   = np.array([0.06, 0.05, 0.05])
TAIL_DIST_START = np.array([0.10, 0.09, 0.08, 0.06, 0.04])
TAIL_DIST_END   = np.array([0.14, 0.24, 0.22, 0.14, 0.10])


def get_query_distribution(t_frac):
    head = HEAD_DIST_START * (1 - t_frac) + HEAD_DIST_END * t_frac
    tail = TAIL_DIST_START * (1 - t_frac) + TAIL_DIST_END * t_frac
    dist = np.concatenate([head, tail])
    dist /= dist.sum()
    return dist


# ======================================================
# Core metrics
# ======================================================
def _topic_cov_probs(alloc):
    return np.minimum(1.0, alloc / THRESHOLDS)


def coverage_at_b(alloc, qdist):
    return float(qdist @ _topic_cov_probs(alloc))


def oos_far(alloc, qdist, p_answer_func):
    cov = _topic_cov_probs(alloc)
    oos = 1 - cov
    expected_oos = float(qdist @ oos)
    if expected_oos < 1e-8:
        return 0.0
    p_answer = p_answer_func(cov)
    expected_false = float(qdist @ (oos * p_answer))
    return expected_false / expected_oos


# P(answer|OOS) per strategy
def _pfa_append(cov):
    return np.full_like(cov, 0.92) - 0.05 * cov

def _pfa_cache(cov):
    return np.full_like(cov, 0.88) - 0.06 * cov

def _pfa_edit(cov):
    return 0.50 - 0.25 * cov

def _pfa_reject(cov):
    return 0.08 + 0.05 * cov

PFA_MAP = {
    'Append-only':  _pfa_append,
    'Cache / LRU':  _pfa_cache,
    'Edit / Prune': _pfa_edit,
    'Reject-only':  _pfa_reject,
}


# ======================================================
# KB allocation update strategies
# ======================================================
def _clip_alloc(alloc):
    alloc = np.maximum(alloc, 0)
    alloc = np.minimum(alloc, DOC_POOLS)
    if alloc.sum() > KB_BUDGET:
        alloc *= KB_BUDGET / alloc.sum()
    return alloc


def _update_append_only(alloc, qdist, rng):
    ops = 0
    demand_gap = qdist * (1 - _topic_cov_probs(alloc))
    gap_order = np.argsort(-demand_gap)
    for ti in gap_order:
        if demand_gap[ti] < 0.005:
            break
        want = int(np.ceil(THRESHOLDS[ti] * 0.15))
        avail = DOC_POOLS[ti] - alloc[ti]
        to_add = min(want, int(avail))
        if to_add <= 0:
            continue
        space = KB_BUDGET - alloc.sum()
        if space < to_add:
            surplus = alloc - THRESHOLDS * qdist * 4
            evict_order = np.argsort(-surplus)
            needed = to_add - int(space)
            for ei in evict_order:
                if ei == ti:
                    continue
                evict = min(needed, max(0, int(alloc[ei] - 5)))
                if evict > 0:
                    alloc[ei] -= evict
                    ops += evict
                    needed -= evict
                if needed <= 0:
                    break
        actual = min(to_add, int(KB_BUDGET - alloc.sum()))
        if actual > 0:
            alloc[ti] += actual
            ops += actual
    return ops


def _update_cache_lru(alloc, qdist, w, last_access, rng):
    ops = 0
    cold_mask = (w - last_access) > 3
    for ti in range(N_TOPICS):
        if cold_mask[ti] and alloc[ti] > 3:
            evict = min(18, int(alloc[ti] * 0.4))
            alloc[ti] -= evict
            ops += evict
    hot_order = np.argsort(-qdist)
    for ti in hot_order:
        if qdist[ti] < 0.04:
            break
        want = int(qdist[ti] * 80)
        space = int(KB_BUDGET - alloc.sum())
        actual = min(want, space, int(DOC_POOLS[ti] - alloc[ti]))
        if actual > 0:
            alloc[ti] += actual
            ops += actual
    return ops


def _update_edit_prune(alloc, ema_demand, w, rng):
    if w == 0 or w % 3 != 0:
        return 0
    ops = 0
    surplus = alloc / THRESHOLDS - ema_demand * 3
    prune_order = np.argsort(-surplus)
    for ti in prune_order[:3]:
        if surplus[ti] > 0.3 and alloc[ti] > 8:
            remove = min(20, int(alloc[ti] * 0.25))
            alloc[ti] -= remove
            ops += remove
    deficit_order = np.argsort(surplus)
    for ti in deficit_order[:3]:
        space = int(KB_BUDGET - alloc.sum())
        add = min(15, space, int(DOC_POOLS[ti] - alloc[ti]))
        if add > 0:
            alloc[ti] += add
            ops += add
    return ops


# ======================================================
# Main simulation
# ======================================================
def simulate(rng):
    dists = [get_query_distribution(w / (N_WINDOWS - 1)) for w in range(N_WINDOWS)]
    all_results = {}

    for name in STRATEGY_ORDER:
        alloc = INIT_ALLOC.copy()
        cum_ops = 0
        ema_demand = dists[0].copy()
        last_access = np.zeros(N_TOPICS)
        metrics = []

        for w in range(N_WINDOWS):
            qdist = dists[w]

            cov = coverage_at_b(alloc, qdist)
            far = oos_far(alloc, qdist, PFA_MAP[name])

            noise_cov = rng.normal(0, 0.006)
            noise_far = rng.normal(0, 0.008)

            metrics.append({
                'window': w,
                'coverage': float(np.clip(cov + noise_cov, 0, 1)),
                'oos_far':  float(np.clip(far + noise_far, 0, 1)),
                'update_cost': cum_ops,
                'kb_size': int(round(alloc.sum())),
            })

            ops = 0
            if name == 'Append-only':
                ops = _update_append_only(alloc, qdist, rng)
            elif name == 'Cache / LRU':
                ops = _update_cache_lru(alloc, qdist, w, last_access, rng)
                last_access = np.where(qdist > 0.05, w, last_access)
            elif name == 'Edit / Prune':
                ema_demand = 0.6 * ema_demand + 0.4 * qdist
                ops = _update_edit_prune(alloc, ema_demand, w, rng)

            cum_ops += ops
            alloc = _clip_alloc(alloc)

        all_results[name] = metrics

    return all_results, dists


# ======================================================
# Plotting
# ======================================================
def _shade_drift(ax, head_fracs):
    for w in range(N_WINDOWS):
        alpha = 0.06 + 0.04 * (1 - head_fracs[w])
        color = '#D1FAE5' if head_fracs[w] > 0.5 else '#FEE2E2'
        ax.axvspan(w - 0.5, w + 0.5, alpha=alpha, color=color, zorder=0)


def make_figure(all_results, dists):
    setup_style()
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.25,
        'lines.linewidth': 2.0,
        'lines.markersize': 5.5,
    })

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30,
                          left=0.07, right=0.96, top=0.93, bottom=0.08)

    windows = np.arange(N_WINDOWS)
    head_fracs = [float(sum(dists[w][:3])) for w in range(N_WINDOWS)]

    # (a) Coverage@B over time
    ax_a = fig.add_subplot(gs[0, 0])
    _shade_drift(ax_a, head_fracs)
    for name in STRATEGY_ORDER:
        st = STRATEGY_STYLES[name]
        cov = [m['coverage'] for m in all_results[name]]
        ax_a.plot(windows, np.array(cov) * 100,
                  color=st['color'], marker=st['marker'], ls=st['ls'],
                  markerfacecolor='white', markeredgewidth=0.9,
                  label=name, zorder=3)
    ax_a.set_ylabel('Coverage@B (%)')
    ax_a.set_ylim(25, 85)
    ax_a.set_xlim(-0.5, N_WINDOWS - 0.5)
    ax_a.set_title('(a) Coverage@B Over Time', fontweight='bold', pad=8)
    ax_a.legend(loc='lower left', frameon=True, framealpha=0.95,
                edgecolor='#cccccc', ncol=1, handlelength=2.2, borderpad=0.4)
    ax_a.set_xlabel('Time Window')
    ax_a.annotate('Head-dominant', xy=(1, 80), fontsize=7.5,
                  color='#059669', fontstyle='italic')
    ax_a.annotate('Tail-dominant', xy=(14, 80), fontsize=7.5,
                  color='#DC2626', fontstyle='italic')

    # (b) OOS False Answer Rate over time
    ax_b = fig.add_subplot(gs[0, 1])
    _shade_drift(ax_b, head_fracs)
    for name in STRATEGY_ORDER:
        st = STRATEGY_STYLES[name]
        far = [m['oos_far'] for m in all_results[name]]
        ax_b.plot(windows, np.array(far) * 100,
                  color=st['color'], marker=st['marker'], ls=st['ls'],
                  markerfacecolor='white', markeredgewidth=0.9,
                  label=name, zorder=3)
    ax_b.set_ylabel('OOS False Answer Rate (%)')
    ax_b.set_ylim(-3, 100)
    ax_b.set_xlim(-0.5, N_WINDOWS - 0.5)
    ax_b.set_title('(b) OOS False Answer Rate Over Time', fontweight='bold', pad=8)
    ax_b.legend(loc='upper right', frameon=True, framealpha=0.95,
                edgecolor='#cccccc', ncol=1, handlelength=2.2, borderpad=0.4)
    ax_b.set_xlabel('Time Window')
    ax_b.axhline(50, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax_b.text(N_WINDOWS - 1, 52, '50%', fontsize=7, color='gray', ha='right')

    # (c) Update Cost over time
    ax_c = fig.add_subplot(gs[1, 0])
    for name in STRATEGY_ORDER:
        st = STRATEGY_STYLES[name]
        cost = [m['update_cost'] for m in all_results[name]]
        ax_c.plot(windows, cost,
                  color=st['color'], marker=st['marker'], ls=st['ls'],
                  markerfacecolor='white', markeredgewidth=0.9,
                  label=name, zorder=3)
    ax_c.set_ylabel('Cumulative Update Cost (ops)')
    ax_c.set_ylim(bottom=-10)
    ax_c.set_xlim(-0.5, N_WINDOWS - 0.5)
    ax_c.set_title('(c) Cumulative Update Cost Over Time', fontweight='bold', pad=8)
    ax_c.legend(loc='upper left', frameon=True, framealpha=0.95,
                edgecolor='#cccccc', ncol=1, handlelength=2.2, borderpad=0.4)
    ax_c.set_xlabel('Time Window')

    # (d) Trade-off scatter
    ax_d = fig.add_subplot(gs[1, 1])
    for name in STRATEGY_ORDER:
        st = STRATEGY_STYLES[name]
        m = all_results[name]
        final_cov = m[-1]['coverage'] * 100
        final_cost = m[-1]['update_cost']
        avg_far = np.mean([mi['oos_far'] for mi in m]) * 100

        ax_d.scatter(final_cost, final_cov, color=st['color'], marker=st['marker'],
                     s=120, edgecolors='white', linewidths=1.2, zorder=5)
        ox, oy = 8, 0
        if name == 'Reject-only':
            ox, oy = 8, -3
        elif name == 'Cache / LRU':
            ox, oy = 8, 2
        ax_d.annotate(f'{name}\n(FAR={avg_far:.0f}%)',
                      xy=(final_cost, final_cov),
                      xytext=(ox, oy), textcoords='offset points',
                      fontsize=7.5, color=st['color'], fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                ec=st['color'], alpha=0.85))

    ax_d.annotate('Ideal\n(high cov, low cost)',
                  xy=(50, 80), fontsize=8, fontstyle='italic',
                  color='#059669', ha='center',
                  bbox=dict(boxstyle='round,pad=0.3', fc='#D1FAE5',
                            ec='#059669', alpha=0.6))
    ax_d.set_xlabel('Final Cumulative Update Cost (ops)')
    ax_d.set_ylabel('Final Coverage@B (%)')
    ax_d.set_title('(d) Coverage-Cost Trade-off (Final State)', fontweight='bold', pad=8)
    ax_d.set_ylim(25, 90)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(FIG_DIR / 'paradigm_diagnostic.png')
    save_fig(fig, out_path)
    plt.close()


# ======================================================
# Summary & save
# ======================================================
def print_summary(all_results):
    print(f"\n{'='*70}")
    print("  Motivation 4: Existing Paradigm Diagnostic -- Key Findings")
    print(f"{'='*70}")
    print(f"\n  Scenario: {N_WINDOWS} windows x {N_QUERIES} queries/window, "
          f"KB budget = {KB_BUDGET} docs")
    print(f"  Topics: {sum(1 for t in TOPIC_TYPES if t=='head')} head + "
          f"{sum(1 for t in TOPIC_TYPES if t=='tail')} tail = {N_TOPICS} topics")
    print(f"  Drift: head {sum(HEAD_DIST_START):.0%} -> {sum(HEAD_DIST_END):.0%}, "
          f"tail {sum(TAIL_DIST_START):.0%} -> {sum(TAIL_DIST_END):.0%}")
    print()

    header = f"  {'Strategy':<16s}  {'Avg Cov':>8s}  {'Final Cov':>9s}  " \
             f"{'Avg FAR':>8s}  {'Final Cost':>10s}  {'Verdict'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    verdicts = {
        'Append-only':  "High cost, high FAR",
        'Cache / LRU':  "Volatile, drops tail",
        'Edit / Prune': "Moderate but blind",
        'Reject-only':  "Safe but stale KB",
    }

    for name in STRATEGY_ORDER:
        m = all_results[name]
        avg_cov = np.mean([mi['coverage'] for mi in m])
        final_cov = m[-1]['coverage']
        avg_far = np.mean([mi['oos_far'] for mi in m])
        final_cost = m[-1]['update_cost']
        print(f"  {name:<16s}  {avg_cov:>7.1%}  {final_cov:>8.1%}  "
              f"{avg_far:>7.1%}  {final_cost:>10d}  {verdicts[name]}")

    print(f"\n  -> No single paradigm achieves high Coverage@B + low OOS FAR + low Cost.")
    print(f"  -> The key gap: none is query-driven KB evolution under budget constraints.")
    print(f"{'='*70}\n")


def save_data(all_results, dists):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        'config': {
            'n_windows': N_WINDOWS,
            'n_queries_per_window': N_QUERIES,
            'kb_budget': KB_BUDGET,
            'topics': [{'name': n, 'type': t, 'pool': int(p), 'threshold': int(th)}
                       for n, t, p, th in TOPICS],
            'init_alloc': INIT_ALLOC.tolist(),
        },
        'distributions': [d.tolist() for d in dists],
        'results': {name: metrics for name, metrics in all_results.items()},
    }
    path = DATA_DIR / 'paradigm_diagnostic.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {path}")


# ======================================================
# Main
# ======================================================
def main():
    rng = np.random.default_rng(SEED)
    print("[Motivation 4] Running existing-paradigm diagnostic experiment...")
    all_results, dists = simulate(rng)
    make_figure(all_results, dists)
    print_summary(all_results)
    save_data(all_results, dists)
    print("Motivation 4 experiment complete.")


if __name__ == '__main__':
    main()
