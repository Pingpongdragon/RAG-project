"""Draw the first motivation figure: aggregate domain mixtures can change.

The figure preserves the original WildChat and Google Trends evidence.  It is
descriptive evidence for a realistic operating condition, not a claim that
every RAG deployment must drift.
"""
import json
import re
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import setup_style, save_fig
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

LINE_STYLES = [
    {'color': '#1f77b4', 'marker': 'o',  'ls': '-'},  # Blue
    {'color': '#ff7f0e', 'marker': 's',  'ls': '-'},  # Orange
    {'color': '#2ca02c', 'marker': '^',  'ls': '-'},  # Green
    {'color': '#d62728', 'marker': 'D',  'ls': '-'},  # Red
    {'color': '#9467bd', 'marker': 'v',  'ls': '-'},  # Purple
    {'color': '#8c564b', 'marker': 'P',  'ls': '--'}, # Brown
    {'color': '#e377c2', 'marker': 'X',  'ls': '--'}, # Pink
    {'color': '#7f7f7f', 'marker': 'o',  'ls': ':'},  # Gray
]

GT_STYLES = {
    'AI chatbot':    {'color': '#1f77b4', 'marker': 'o',  'ls': '-'},
    'AI coding':     {'color': '#ff7f0e', 'marker': 's',  'ls': '-'},
    'AI healthcare': {'color': '#2ca02c', 'marker': '^',  'ls': '-'},
    'AI education':  {'color': '#d62728', 'marker': 'D',  'ls': '-'},
    'AI agent':      {'color': '#9467bd', 'marker': 'v',  'ls': '-'},
}

AREA_COLORS = {
    'Coding': '#4C78A8',
    'Image Gen.': '#F58518',
    'Creative Writing': '#54A24B',
    'Education': '#EECA3B',
    'Knowledge QA': '#B279A2',
    'Entertainment': '#FF9DA6',
    'Business': '#9D755D',
    'Other': '#BAB0AC',
    'AI chatbot': '#4C78A8',
    'AI coding': '#F58518',
    'AI healthcare': '#54A24B',
    'AI education': '#E45756',
    'AI agent': '#B279A2',
}


WC_ORDER = [
    'Coding', 'Creative Writing', 'Image Gen.', 'Education',
    'Knowledge QA', 'Business', 'Entertainment', 'Other'
]
GT_ORDER = ['AI chatbot', 'AI coding', 'AI healthcare', 'AI education', 'AI agent']

WILDCHAT_RULES = {
    'Coding': [
        r'\b(code|coding|program|python|java\b|javascript|html|css|sql|api|function|'
        r'class\b|variable|compile|debug|algorithm|regex|git\b|docker|linux|'
        r'bash|library|framework|react|node\.?js|django|flask|'
        r'implement|bug|error|exception|database|backend|frontend|script|'
        r'array|loop|recursion|pytorch|tensorflow|pandas|numpy|json|yaml|'
        r'software|developer|programming|c\+\+|rust\b|golang|php|ruby|'
        r'vba|terminal|server|deploy|aws|azure|syntax|compiler)\b',
    ],
    'Creative Writing': [
        r'\b(write\s+(a\s+)?(story|novel|poem|chapter|scene|dialogue|script|episode|'
        r'fanfic|fiction|narrative|short\s+story|opening|narration|essay|article|'
        r'letter\b|text|paragraph|speech|monologue|lyrics|haiku|limerick))',
        r'\b(roleplay|character\s+(sheet|description|backstory)|'
        r'worldbuilding|fantasy|plot|protagonist|villain|lore|'
        r'SCP|crossover|fanfiction|creative\s+writ)\b',
        r'\b(story\s+prompt|write.*story|continue.*story|story.*about)\b',
        r'\b(screenwriter|screenplay|act\s+as\s+a|imagine\s+you\s+are)\b',
        r'(In the clubroom|In a.*biome|\[player\])',
    ],
    'Image Gen.': [
        r'\b(midjourney|stable\s*diffusion|dall-?e|image\s+prompt|'
        r'art\s+style|graphic\s+design|logo\s+design|illustration|'
        r'prompt\s+generator.*generative|generate.*image|image.*generat|'
        r'etsy|t-?shirt|mug\s+design|print\s+on\s+demand|'
        r'illustrator|photoshop|canva|art\s*work)\b',
    ],
    'Education': [
        r'\b(homework|assignment|exam|quiz|test\s+question|'
        r'math\b|calculus|algebra|geometry|physics\b|chemistry\b|biology\b|'
        r'history\s+(of|lesson)|economics|'
        r'student|teacher|school|course|textbook|'
        r'solve\b|equation|formula|theorem|proof|'
        r'science|scientific|research)\b',
        r'\b(explain\s+(how|what|why|the)|teach\s+me|learn\s+about|tutor)\b',
    ],
    'Business': [
        r'\b(business|marketing|strategy|startup|management|'
        r'resume|cv\b|cover\s+letter|interview|salary|career|'
        r'email|meeting|presentation|project\s+plan|budget|'
        r'linkedin|corporate|client|proposal|fiverr|freelanc|'
        r'investment|stock|financ|economy|revenue|profit)\b',
    ],
    'Entertainment': [
        r'\b(movie|film|tv\s+show|series|anime|manga|game|gaming|'
        r'music|song|album|lyrics|netflix|youtube|twitch|'
        r'fortnite|minecraft|roblox|nba|nfl|soccer|football|'
        r'pokemon|marvel|dc\s+comic|video\s+game|sport)\b',
    ],
    'Knowledge QA': [
        r'\b(what\s+is|what\s+are|what\s+was|who\s+is|who\s+was|'
        r'how\s+does|how\s+do|how\s+did|how\s+is|'
        r'when\s+did|when\s+was|where\s+is|where\s+was|'
        r'why\s+is|why\s+are|why\s+did|why\s+do|'
        r'tell\s+me\s+about|can\s+you\s+tell|give\s+me\s+(a\s+)?list|'
        r'difference\s+between|compare|definition\s+of)\b',
    ],
}

_WC_COMPILED = {t: [re.compile(p, re.IGNORECASE) for p in ps]
                for t, ps in WILDCHAT_RULES.items()}

def classify_query(text):
    for topic, rxs in _WC_COMPILED.items():
        for rx in rxs:
            if rx.search(text):
                return topic
    return 'Other'

def analyze_wildchat():
    path = os.path.join(DATA_DIR, 'wildchat_sampled.json')
    print(f"[WildChat] Loading {path}")
    with open(path) as f:
        data = json.load(f)
    en = [d for d in data if d.get('lang') == 'English']
    print(f"[WildChat] {len(en)} English queries")
    mt = defaultdict(lambda: defaultdict(int))
    tt = Counter()
    for d in en:
        m, t = d['ts'][:7], classify_query(d['query'])
        mt[m][t] += 1
        tt[t] += 1
    topics = sorted([t for t in tt if t != 'Other'], key=lambda t: tt[t], reverse=True)
    topics.append('Other')
    months = sorted(mt.keys())
    prop = np.zeros((len(months), len(topics)))
    for i, m in enumerate(months):
        s = sum(mt[m].values())
        for j, t in enumerate(topics):
            prop[i, j] = mt[m].get(t, 0) / s
    jsd = [_jsd(prop[i-1], prop[i]) for i in range(1, len(months))]
    return months, topics, prop, jsd

GT_TOPICS = ["AI chatbot", "AI coding", "AI healthcare", "AI education", "AI agent"]
GT_TIMEFRAME = '2022-01-01 2026-03-01'
GT_CACHE = os.path.join(DATA_DIR, 'google_trends_cache.json')

def fetch_google_trends():
    if os.path.exists(GT_CACHE):
        with open(GT_CACHE) as f:
            c = json.load(f)
        df = pd.DataFrame(c['data'], index=pd.to_datetime(c['index']))
        df.columns = c['columns']
        return df
    from pytrends.request import TrendReq
    pt = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
    pt.build_payload(GT_TOPICS, timeframe=GT_TIMEFRAME, geo='')
    df = pt.interest_over_time()
    if 'isPartial' in df.columns:
        df = df.drop(columns='isPartial')
    cache = {'index': [str(d) for d in df.index],
             'columns': list(df.columns), 'data': df.values.tolist()}
    with open(GT_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)
    return df

def analyze_google_trends():
    weekly = fetch_google_trends()
    monthly = weekly.resample('ME').mean()
    monthly = monthly.loc[monthly.sum(axis=1) > 0]
    proportions = monthly.div(monthly.sum(axis=1), axis=0).fillna(0)
    months = [d.strftime('%Y-%m') for d in proportions.index]
    topics = list(proportions.columns)
    prop = proportions.values
    jsd = [_jsd(prop[i-1], prop[i]) for i in range(1, len(months))]
    return months, topics, prop, jsd

def _jsd(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    sp, sq = p.sum(), q.sum()
    if sp == 0 or sq == 0:
        return 0.0
    p, q = p / sp, q / sq
    m = 0.5 * (p + q)
    eps = 1e-12
    return float(0.5 * (np.sum(p * np.log((p+eps)/(m+eps))) +
                        np.sum(q * np.log((q+eps)/(m+eps)))))


def _reorder_topics(topics, prop, preferred_order):
    order = [topic for topic in preferred_order if topic in topics]
    remainder = [topic for topic in topics if topic not in order]
    ordered_topics = order + remainder
    idx = [topics.index(topic) for topic in ordered_topics]
    return ordered_topics, prop[:, idx]


def _stacked_share_panel(ax, x, topics, prop, title, xlabel, x_pad=1.6,
                         label_threshold=6.0, min_gap=6.2, force_labels=None,
                         right_pad=3.0):
    colors = [AREA_COLORS.get(topic, '#999999') for topic in topics]
    values = prop.T * 100.0
    ax.stackplot(x, values, colors=colors, alpha=0.94, linewidth=0)

    cumulative = np.cumsum(values, axis=0)
    for idx in range(len(topics)):
        ax.plot(x, cumulative[idx], color='white', lw=0.95, alpha=0.88, zorder=3)

    forced = set(force_labels or [])
    end_centers = []
    for idx, topic in enumerate(topics):
        lower = cumulative[idx - 1] if idx > 0 else np.zeros_like(x, dtype=float)
        center = 0.5 * (lower + cumulative[idx])
        end_centers.append((topic, center[-1], colors[idx], values[idx, -1]))

    visible = [item for item in end_centers if item[3] >= label_threshold or item[0] in forced]
    visible.sort(key=lambda item: item[1])
    placed = []
    for topic, y, color, share in visible:
        if placed:
            y = max(y, placed[-1][1] + min_gap)
        placed.append((topic, min(y, 98.0), color, share))
    for idx in range(len(placed) - 2, -1, -1):
        topic, y, color, share = placed[idx]
        placed[idx] = (topic, min(y, placed[idx + 1][1] - min_gap), color, share)

    label_x = x[-1] + x_pad
    for topic, y, color, share in placed:
        topic_label = topic.replace('AI ', '').replace('Creative Writing', 'Creative writing')
        ax.plot([x[-1], label_x - 0.12], [y, y], color=color, lw=1.2, alpha=0.95, zorder=4)
        ax.text(label_x, y, topic_label, fontsize=8.0, color=color,
                fontweight='bold', ha='left', va='center', zorder=5)

    ax.set_ylabel('Share (%)')
    ax.set_ylim(0, 100)
    ax.set_xlim(x[0], x[-1] + x_pad + right_pad)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=5)
    ax.grid(axis='y', alpha=0.22)
    ax.grid(axis='x', alpha=0.0)
    ax.set_axisbelow(True)


def _set_month_ticks(ax, months, stride):
    """Label the data region with calendar months, excluding annotation space."""
    tick_idx = list(range(0, len(months), stride))
    if tick_idx[-1] != len(months) - 1:
        if len(months) - 1 - tick_idx[-1] < stride / 2:
            tick_idx[-1] = len(months) - 1
        else:
            tick_idx.append(len(months) - 1)
    labels = [pd.to_datetime(months[idx]).strftime("%b '%y") for idx in tick_idx]
    ax.set_xticks(tick_idx, labels)


def make_figure(wc, gt):
    wc_months, wc_topics, wc_prop, _wc_jsd = wc
    gt_months, gt_topics, gt_prop, _gt_jsd = gt

    setup_style()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5,
        'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0,
        'legend.fontsize': 8.0, 'axes.linewidth': 0.8,
        'grid.linewidth': 0.6, 'grid.alpha': 0.4, 'grid.color': '#D3D3D3',
        'grid.linestyle': '--',
        'lines.linewidth': 2.0, 'lines.markersize': 5.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    fig = plt.figure(figsize=(3.35, 3.45))
    gs = fig.add_gridspec(2, 1, hspace=0.45,
                          left=0.16, right=0.98, top=0.95, bottom=0.12)

    wc_topics, wc_prop = _reorder_topics(wc_topics, wc_prop, WC_ORDER)
    gt_topics, gt_prop = _reorder_topics(gt_topics, gt_prop, GT_ORDER)

    ax_a = fig.add_subplot(gs[0, 0])
    x_wc = np.arange(len(wc_months))
    _stacked_share_panel(
        ax_a, x_wc, wc_topics, wc_prop,
        '(a) WildChat query-topic mixture',
        'Month',
        x_pad=2.4,
        right_pad=7.5,
        label_threshold=3.0,
        min_gap=8.2,
        force_labels=['Coding', 'Creative Writing', 'Image Gen.', 'Education',
                      'Knowledge QA', 'Business', 'Entertainment', 'Other'],
    )
    _set_month_ticks(ax_a, wc_months, stride=4)

    ax_b = fig.add_subplot(gs[1, 0])
    x_gt = np.arange(len(gt_months))
    _stacked_share_panel(
        ax_b, x_gt, gt_topics, gt_prop,
        '(b) Google Trends application mixture',
        'Month',
        x_pad=5.0,
        right_pad=15.0,
        min_gap=9.5,
    )
    _set_month_ticks(ax_b, gt_months, stride=12)

    out_dir = os.path.join(BASE_DIR, 'figures')
    save_fig(fig, os.path.join(out_dir, 'user_query_topic_drift.png'))
    plt.close()

if __name__ == '__main__':
    wc = analyze_wildchat()
    gt = analyze_google_trends()
    make_figure(wc, gt)
