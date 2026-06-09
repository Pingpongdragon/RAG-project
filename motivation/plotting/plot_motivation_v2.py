"""Plot Fig.1 (StreamingQA L1 audit) + Fig.2 (L2/L3 diagnostic).

Current framing:
  Fig.1 -- L1 temporal-locality diagnosis: access history handles simple
            temporal drift; timestamps and freshness are too coarse.
  Fig.2 -- L2/L3 diagnosis: SemFlow helps direct semantic multi-hop, but
            bridge/topological access requires entity-chained prefetch.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'motivation_1' / 'data'
OUT  = ROOT / 'paper_figs' / 'intro'
OUT.mkdir(parents=True, exist_ok=True)

# ── style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

COLOR = {
    'Static':        '#9CA3AF',
    'DocArrival':    '#059669',
    'KnowledgeEdit': '#7C3AED',
    'TinyLFU':       '#D97706',
    'LRU':           '#D97706',
    'MissLRU':       '#F472B6',
    'MissTinyLFU':   '#A78BFA',
    'GPTCacheStyle': '#0891B2',
    'MemGPTStyle':   '#BE185D',
    'TemporalAware': '#2563EB',
    'RecencyTTL':    '#1E40AF',
    'OnDemandFetch': '#0F766E',
    'QueryDriven':   '#10B981',
    'Oracle':        '#DC2626',
}
LABEL = {
    'Static':        'Static',
        'TinyLFU':       'TinyLFU',
        'LRU':           'LRU',
    'DocArrival':    'IndexGrowth\n(bounded)',
    'TinyLFU':       'TinyLFU',
    'LRU':           'LRU',
    'MissLRU':       'LRU',
    'MissTinyLFU':   'TinyLFU',
    'GPTCacheStyle': 'GPTCache',
    'MemGPTStyle':   'MemGPT',
    'TemporalAware': 'Temporal-Aware\n(decay)',
    'RecencyTTL':    'Recency-TTL\n(oracle ts)',
    'OnDemandFetch': 'On-Demand Fetch',
    'QueryDriven':   'SemFlow\n(ours)',
    'Oracle':        'Oracle',
}
SIGNAL_FAMILY = {
    'Static':        'none',
    'DocArrival':    'idx',
    'KnowledgeEdit': 'idx',
    'TinyLFU':       'cache',
    'LRU':           'cache',
    'MissLRU':       'query',
    'MissTinyLFU':   'query',
    'GPTCacheStyle': 'cache',
    'MemGPTStyle':   'cache',
    'TemporalAware': 'time',
    'RecencyTTL':    'time',
    'QueryDriven':   'query',
    'OnDemandFetch': 'oracle',
    'Oracle':        'oracle',
}

def load(name):
    p = DATA / f'results_{name}.json'
    d = json.load(open(p))
    k = list(d.keys())[0]
    return d[k]

# ── Fig.1 — Integrated: per-window R@5 (top) + 2-panel cost audit (bottom) ──
def fig1():
    """Three panels: (a) per-window R@5, (b) online query lat / cache-hit SLO,
    (c) offline per-window maintenance latency (linear scale).
    8 strategies covering 5 signal families + 2 bounds.
    """
    from matplotlib.transforms import blended_transform_factory

    d = load('streamingqa_temporal')
    s = d['summary']
    cfg = d['config']
    nw = cfg['n_windows']

    ORDER = ['Static', 'DocArrival', 'RecencyTTL',
             'MissTinyLFU', 'MissLRU',
             'QueryDriven', 'OnDemandFetch', 'Oracle']
    ORDER = [n for n in ORDER if n in s]
    cols  = [COLOR[n] for n in ORDER]

    XLBL = {
        'Static':        'Static',
        'TinyLFU':       'TinyLFU',
        'LRU':           'LRU',
        'MissLRU':       'LRU',
        'MissTinyLFU':   'TinyLFU',
        'DocArrival':    'IndexGrowth\n(bounded)',
        'RecencyTTL':    'RecencyTTL',
        'QueryDriven':   'SemFlow',
        'OnDemandFetch': 'OnDemand',
        'Oracle':        'Oracle (dense)',
        'OracleEntityExpand': 'Oracle+EntityExpand',
    }
    xlbls = [XLBL[n] for n in ORDER]

    LW = {
        'Oracle':        (2.8, '-'),
        'OnDemandFetch': (2.4, '-'),
        'QueryDriven':   (2.4, '-'),
        'RecencyTTL':    (1.6, ':'),
        'DocArrival':    (1.4, '--'),
        'Static':        (1.6, '-'),
        'TinyLFU':       (1.4, ':'),
        'LRU':           (1.4, ':'),
        'MissLRU':       (2.0, '-.'),
        'MissTinyLFU':   (1.6, '-.'),
    }

    fig = plt.figure(figsize=(14.0, 7.2))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1.75, 1.0],
                           hspace=0.68, wspace=0.38)
    axL = fig.add_subplot(gs[0, :])   # (a) line chart full width
    axB = fig.add_subplot(gs[1, 0])   # (b) online query latency
    axC = fig.add_subplot(gs[1, 1])   # (c) cumulative cache-miss fetches

    # ── (a) line chart ──────────────────────────────────────────────
    for name in ORDER:
        pw = s[name].get('recall@5_per_window', [])
        if not pw:
            continue
        y = np.array(pw)
        k = min(5, len(y))
        kern = np.ones(k) / k
        # Causal MA: pad only at front so window-0 shows the true starting value
        ys = np.convolve(np.concatenate([np.full(k - 1, y[0]), y]), kern, mode='valid')[:len(y)]
        lw, ls = LW.get(name, (1.3, '-'))
        axL.plot(np.arange(len(ys)), ys, color=COLOR[name], lw=lw, ls=ls,
                 label=LABEL[name].replace('\n', ' '))

    # era bands + year labels via blended transform (no y-overflow)
    n_rounds = 5
    per = nw // n_rounds
    ROUND_LABELS = ['R1  2008-10', 'R2  2011-13', 'R3  2014-16',
                    'R4  2017-18', 'R5  2019-20\n(COVID)']
    trans = blended_transform_factory(axL.transData, axL.transAxes)
    for ri in range(n_rounds):
        x0, x1 = ri * per, (ri + 1) * per
        if ri % 2 == 1:
            axL.axvspan(x0, x1, alpha=0.055, color='steelblue', zorder=0)
        axL.text((x0 + x1) / 2, 0.975, ROUND_LABELS[ri],
                 ha='center', va='top', fontsize=8.0, color='#374151',
                 linespacing=1.15, transform=trans)

    axL.set_xlabel('Window index  (50 queries / window; 5 eras x 20 windows)',
                   fontsize=10.5)
    axL.set_ylabel('Recall@5 (%)', fontsize=11)
    axL.set_title(
        '(a) L1 temporal-locality audit on the natural StreamingQA stream  '
        f'(pool = {cfg["pool_size"]:,},  hot KB = {cfg["kb_budget"]:,})',
        fontsize=11)
    axL.legend(ncol=4, fontsize=8.4, framealpha=0.92,
               loc='upper center', bbox_to_anchor=(0.5, -0.22))
    axL.grid(alpha=0.28)
    axL.set_xlim(0, nw - 1)
    axL.set_ylim(0, 100)

    x = np.arange(len(ORDER))
    BAR_KW = dict(edgecolor='#111827', linewidth=0.7, width=0.65)

    # ── (b) Cold-tier fetches per query (hardware-agnostic) ─────────────────
    _total_q = cfg.get('n_windows', 100) * cfg.get('window_size', 50)
    qfetch = [round(s[n].get('serve_retrieval_cost', 0) / _total_q, 2)
              for n in ORDER]
    _ymax_b = max(max(qfetch) * 1.45, 1.0)
    bB = axB.bar(x, qfetch, color=cols, **BAR_KW)
    axB.set_ylim(0, _ymax_b)
    for i, (r, v) in enumerate(zip(bB, qfetch)):
        if 'OnDemandFetch' in ORDER and i == ORDER.index('OnDemandFetch'):
            continue  # drawn as white inside-bar label below
        lbl = f'{v:.1f}' if v >= 1 else '0'
        axB.text(r.get_x() + r.get_width() / 2, v + _ymax_b * 0.025, lbl,
                 ha='center', va='bottom', fontsize=7.8)
    axB.set_xticks(x)
    axB.set_xticklabels(xlbls, fontsize=7.0, rotation=25, ha='right')
    axB.set_ylabel('Cold-tier fetches per query', fontsize=9.5)
    axB.set_title('(b) Serve-time cold-tier retrieval cost\n(per query; hardware-agnostic)',
                  fontsize=10.0)
    axB.grid(axis='y', alpha=0.3)
    # Annotate OnDemand bar
    od_idx = ORDER.index('OnDemandFetch') if 'OnDemandFetch' in ORDER else None
    if od_idx is not None:
        serve_cnt = s['OnDemandFetch'].get('serve_retrieval_cost', 0)
        axB.text(od_idx, qfetch[od_idx] * 0.45, f'{qfetch[od_idx]:.0f}',
                 ha='center', va='center', fontsize=9, color='white',
                 fontweight='bold', zorder=5)
        axB.text(od_idx, qfetch[od_idx] + _ymax_b * 0.12,
                 f'({serve_cnt:,} total)',
                 ha='center', va='bottom', fontsize=6.5, color='#0F766E',
                 style='italic')
    # ── (c) offline update / maintenance latency per window (linear) ────────
    ulat_raw = [s[n].get('update_latency_mean_ms', 0.0) for n in ORDER]
    bC = axC.bar(x, ulat_raw, color=cols, hatch='//', alpha=0.85, **BAR_KW)
    axC.set_ylim(0, max(ulat_raw) * 1.45)
    for r, v in zip(bC, ulat_raw):
        lbl = f'{v:.0f}'
        axC.text(r.get_x() + r.get_width() / 2, v + 0.3, lbl,
                 ha='center', va='bottom', fontsize=7.8)
    axC.set_xticks(x)
    axC.set_xticklabels(xlbls, fontsize=7.0, rotation=25, ha='right')
    axC.set_ylabel('Per-window maint. latency (ms)')
    axC.set_title('(c) Algorithmic maintenance overhead\n'
                  '(per window; simulated CPU, not disk I/O)', fontsize=10.0)
    axC.grid(axis='y', alpha=0.3)

    fig.savefig(OUT / 'fig1_intro_streamingqa_signal_audit.pdf',
                dpi=180, bbox_inches='tight')
    fig.savefig(OUT / 'fig1_intro_streamingqa_signal_audit.png',
                dpi=180, bbox_inches='tight')
    print('  wrote fig1_intro_streamingqa_signal_audit.{pdf,png}')
    plt.close(fig)


# ── Fig.2 — Absolute Recall@5 curves on multi-hop datasets ─────────────────
def fig2():
    DATA2 = ROOT / 'motivation_2' / 'data'
    HOTPOT_FILE = DATA2 / 'results_hotpotqa_comp_full_gradual_q2k.json'
    WIKI_FILE   = DATA2 / 'results_2wiki_bc_entity_expand_gradual_q2k.json'
    d_hot  = json.load(open(HOTPOT_FILE))['hotpotqa']
    d_wiki = json.load(open(WIKI_FILE))['2wikimultihopqa']

    COLOR2 = {**COLOR}
    LABEL2 = {
        'Static':        'Static (frozen)',
        'TinyLFU':       'TinyLFU',
        'LRU':           'LRU',
        'MissLRU':       'LRU',
        'MissTinyLFU':   'TinyLFU',
        'DocArrival':    'IndexGrowth (bounded)',
        'QueryDriven':   'SemFlow (ours)',
        'OnDemandFetch': 'On-Demand Fetch',
        'Oracle':        'Oracle',
    }
    LW2 = {
        'Static': 1.3, 'TinyLFU': 1.4, 'LRU': 1.4, 'DocArrival': 1.5,
        'MissLRU': 2.2, 'MissTinyLFU': 1.6,
        'QueryDriven': 2.8, 'OnDemandFetch': 2.0, 'Oracle': 2.2,
    }
    LS2 = {
        'Static': '--', 'TinyLFU': ':', 'LRU': ':', 'DocArrival': '--',
        'MissLRU': '-.', 'MissTinyLFU': ':',
        'QueryDriven': '-', 'OnDemandFetch': '--', 'Oracle': '-',
    }
    ORDER_LINE = ['Static', 'DocArrival', 'MissTinyLFU', 'MissLRU', 'QueryDriven', 'OnDemandFetch', 'Oracle']

    def ma(arr, w=5):
        a = np.array(arr, dtype=float)
        if len(a) < w:
            return a
        c = np.convolve(a, np.ones(w)/w, mode='valid')
        pad_l = (len(a) - len(c)) // 2
        pad_r = len(a) - len(c) - pad_l
        return np.pad(c, (pad_l, pad_r), mode='edge')

    def add_gradual_bg(ax, n_w, y_max):
        left  = np.array([219, 234, 254], dtype=float) / 255.0
        right = np.array([254, 243, 199], dtype=float) / 255.0
        xs = np.linspace(0.0, 1.0, 512)
        grad = np.zeros((2, xs.size, 3), dtype=float)
        for ci in range(3):
            grad[:, :, ci] = left[ci] * (1 - xs) + right[ci] * xs
        ax.imshow(grad, extent=[0, n_w - 1, 0, y_max], aspect='auto',
                  origin='lower', alpha=0.55, zorder=0)
        ax.text(8,       y_max * 0.96, 'more head', ha='left',   va='top',
                fontsize=7.5, color='#1E40AF', style='italic')
        ax.text(n_w / 2, y_max * 0.96, 'full gradual drift', ha='center', va='top',
                fontsize=7.5, color='#475569', style='italic')
        ax.text(n_w - 8, y_max * 0.96, 'more tail', ha='right',  va='top',
                fontsize=7.5, color='#92400E', style='italic')

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=False)
    fig.subplots_adjust(wspace=0.32, top=0.84)

    def draw_recall_ax(ax, data, panel_lbl, ds_title, annot_text, annot_color, annot_bg):
        """Panel a: plain Recall@5 curves (HotpotQA comparison)."""
        cfg  = data['config']
        summ = data['summary']
        names = [n for n in ORDER_LINE if n in summ]
        wx = np.arange(cfg['n_windows'])

        all_vals = []
        for n in names:
            all_vals.extend(summ[n].get('recall@5_per_window', []))
        y_max = min(100, max(all_vals) * 1.15) if all_vals else 100
        y_max = max(y_max, 60)

        add_gradual_bg(ax, cfg['n_windows'], y_max)

        for n in names:
            pw = np.array(summ[n].get('recall@5_per_window', []), dtype=float)
            alpha  = 0.95 if n in ('QueryDriven', 'Oracle', 'OnDemandFetch') else 0.65
            zorder = 3 if n == 'QueryDriven' else 2
            ax.plot(wx, ma(pw),
                    color=COLOR2.get(n, '#888'),
                    lw=LW2.get(n, 1.3),
                    ls=LS2.get(n, '-'),
                    label=LABEL2.get(n, n),
                    alpha=alpha, zorder=zorder)

        ax.text(0.04, 0.06, annot_text,
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=9, fontweight='bold', color=annot_color,
                bbox=dict(fc=annot_bg, ec=annot_color,
                          boxstyle='round,pad=0.4', alpha=0.92))
        ax.set_xlim(0, cfg['n_windows'] - 1)
        ax.set_ylim(0, y_max)
        ax.set_xlabel('Window index', fontsize=9.5)
        ax.set_ylabel('Recall@5 (%)', fontsize=10)
        ax.set_title(
            f'$\\mathbf{{({panel_lbl})}}$  {ds_title}\n'
            f'pool={cfg["pool_size"]:,}   KB={cfg["kb_budget"]:,}',
            fontsize=10.5)
        ax.grid(alpha=0.22, zorder=1)

    draw_recall_ax(ax_a, d_hot,
               panel_lbl='a', ds_title='L2 direct multi-hop: HotpotQA-comp  (both entities in query)',
               annot_text='SemFlow +9.2 pp over LRU  (H2: 54.3%  vs  45.1%)\nQuery-embedding neighborhood reaches both hops',
               annot_color='#065F46', annot_bg='#D1FAE5')

    draw_recall_ax(ax_b, d_wiki,
               panel_lbl='b', ds_title='L3 bridge access: 2WikiMultihopQA  (2nd-hop entity hidden)',
               annot_text=('SemFlow +7.0 pp over LRU  (H2: 36.0%  vs  29.0%)\n'
                           'BUT 21 pp gap to Oracle: 2nd-hop docs are not\n'
                           'reachable from query embedding alone\n'
                           '→ motivates entity-chained prefetch'),
               annot_color='#92400E', annot_bg='#FEF3C7')

    handles_a, labels_a = ax_a.get_legend_handles_labels()
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    seen = set(labels_a)
    for h, l in zip(handles_b, labels_b):
        if l not in seen:
            handles_a.append(h); labels_a.append(l); seen.add(l)
    fig.legend(handles_a, labels_a,
               loc='lower center', bbox_to_anchor=(0.5, -0.14),
               ncol=5, fontsize=8.5, framealpha=0.92,
               columnspacing=1.0, handlelength=2.2)

    fig.suptitle('L2 vs. L3 Agentic RAG access: SemFlow helps direct multi-hop but not hidden bridge paths',
                 fontsize=11.5, y=0.98)

    for ext in ('pdf', 'png'):
        out = OUT / f'fig2_intro_qd_multihop_gap.{ext}'
        fig.savefig(out, dpi=180, bbox_inches='tight')
        print(f'  wrote {out}')
    plt.close(fig)


if __name__ == '__main__':
    print('=== Fig.1 (integrated) ===')
    fig1()
    print('=== Fig.2 ===')
    fig2()
    print('done.')
