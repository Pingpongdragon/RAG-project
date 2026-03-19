"""
Motivation 1: LLM 用户查询主题随时间显著漂移

(a) WildChat-1M 真实用户 query 的 topic 分布变化 (2023.4-2024.4)
(b) Google Trends LLM 应用领域搜索热度变化 (2022-2026)
底部共享: 月间 Jensen-Shannon Divergence
"""
import json
import re
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from plot_config import setup_style, save_fig
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ═══════════════════════════════════════════════════════════
# 学术配色 + marker (每条线可区分, 打印友好)
# ═══════════════════════════════════════════════════════════
LINE_STYLES = [
    {'color': '#2166AC', 'marker': 'o',  'ls': '-'},   # dark blue
    {'color': '#D6604D', 'marker': 's',  'ls': '-'},   # brick red
    {'color': '#4DAF4A', 'marker': '^',  'ls': '-'},   # green
    {'color': '#FF7F00', 'marker': 'D',  'ls': '-'},   # orange
    {'color': '#984EA3', 'marker': 'v',  'ls': '-'},   # purple
    {'color': '#A65628', 'marker': 'P',  'ls': '--'},  # brown
    {'color': '#377EB8', 'marker': 'X',  'ls': '--'},  # blue
    {'color': '#999999', 'marker': '.',  'ls': ':'},   # gray
]

GT_STYLES = {
    'AI chatbot':    {'color': '#2166AC', 'marker': 'o',  'ls': '-'},
    'AI coding':     {'color': '#D6604D', 'marker': 's',  'ls': '-'},
    'AI healthcare': {'color': '#4DAF4A', 'marker': '^',  'ls': '-'},
    'AI education':  {'color': '#FF7F00', 'marker': 'D',  'ls': '-'},
    'AI agent':      {'color': '#984EA3', 'marker': 'v',  'ls': '-'},
}

# ═══════════════════════════════════════════════════════════
# Part A: WildChat topic 分类
# ═══════════════════════════════════════════════════════════
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

_WC_COMPILED = {}
for _t, _ps in WILDCHAT_RULES.items():
    _WC_COMPILED[_t] = [re.compile(p, re.IGNORECASE) for p in _ps]

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

# ═══════════════════════════════════════════════════════════
# Part B: Google Trends
# ═══════════════════════════════════════════════════════════
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
    cache = {'index': [str(d) for d in df.index], 'columns': list(df.columns), 'data': df.values.tolist()}
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

# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════
def _jsd(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    sp, sq = p.sum(), q.sum()
    if sp == 0 or sq == 0:
        return 0.0
    p, q = p / sp, q / sq
    m = 0.5 * (p + q)
    eps = 1e-12
    return float(0.5 * (np.sum(p * np.log((p+eps)/(m+eps))) + np.sum(q * np.log((q+eps)/(m+eps)))))

# ═══════════════════════════════════════════════════════════
# 学术折线图
# ═══════════════════════════════════════════════════════════
def make_figure(wc, gt):
    wc_months, wc_topics, wc_prop, wc_jsd = wc
    gt_months, gt_topics, gt_prop, gt_jsd = gt

    setup_style()
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7.5,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.6,
        'lines.markersize': 4.5,
    })

    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.8, 1], hspace=0.08, wspace=0.28,
                          left=0.07, right=0.95, top=0.91, bottom=0.12)

    # ── (a) WildChat 折线图 ─────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    x_wc = np.arange(len(wc_months))

    for j, topic in enumerate(wc_topics):
        st = LINE_STYLES[j % len(LINE_STYLES)]
        ax_a.plot(x_wc, wc_prop[:, j] * 100,
                  color=st['color'], marker=st['marker'], linestyle=st['ls'],
                  markerfacecolor='white', markeredgewidth=1.0,
                  label=topic, zorder=3)

    ax_a.set_ylabel('Proportion (%)')
    ax_a.set_ylim(bottom=0)
    ax_a.set_xlim(0, len(wc_months) - 1)
    ax_a.set_title('(a) Real User Query Topics (WildChat-1M, English)',
                   fontsize=11, fontweight='bold', pad=8)
    ax_a.legend(loc='upper right', fontsize=7, frameon=True, framealpha=0.95,
                edgecolor='#cccccc', ncol=2, columnspacing=0.8,
                handlelength=2.0, borderpad=0.4)
    ax_a.tick_params(axis='x', labelbottom=False)

    # ── (a-bottom) WildChat JSD ─────────────────────────────
    ax_ajsd = fig.add_subplot(gs[1, 0], sharex=ax_a)
    jx_wc = np.arange(1, len(wc_months))
    lbl_wc = [m[2:].replace('-', '/') for m in wc_months]

    ax_ajsd.fill_between(jx_wc, wc_jsd, alpha=0.15, color='#2166AC')
    ax_ajsd.plot(jx_wc, wc_jsd, color='#2166AC', linewidth=1.4, marker='o',
                 markersize=3.5, markerfacecolor='white', markeredgewidth=0.8)
    avg_wc = np.mean(wc_jsd)
    ax_ajsd.axhline(avg_wc, color='#D6604D', linestyle='--', linewidth=1.0,
                     alpha=0.8, label=f'Mean = {avg_wc:.4f}')
    ax_ajsd.set_ylabel('JSD')
    ax_ajsd.set_xlabel('Month')
    ax_ajsd.set_xticks(x_wc)
    ax_ajsd.set_xticklabels(lbl_wc, rotation=45, ha='right')
    ax_ajsd.legend(fontsize=7, loc='upper right')
    ax_ajsd.set_ylim(bottom=0)

    # ── (b) Google Trends 折线图 ────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    x_gt = np.arange(len(gt_months))

    for topic in gt_topics:
        j = gt_topics.index(topic)
        st = GT_STYLES[topic]
        label = topic.replace('AI ', '').capitalize()
        ax_b.plot(x_gt, gt_prop[:, j] * 100,
                  color=st['color'], marker=st['marker'], linestyle=st['ls'],
                  markerfacecolor='white', markeredgewidth=0.8,
                  markersize=3.5, label=label, zorder=3)

    ax_b.set_ylabel('Proportion (%)')
    ax_b.set_ylim(bottom=0)
    ax_b.set_xlim(0, len(gt_months) - 1)
    ax_b.set_title('(b) LLM Application Interest (Google Trends, 2022–2026)',
                   fontsize=11, fontweight='bold', pad=8)
    ax_b.legend(loc='upper left', fontsize=7.5, frameon=True, framealpha=0.95,
                edgecolor='#cccccc', ncol=1, handlelength=2.0, borderpad=0.4)
    ax_b.tick_params(axis='x', labelbottom=False)

    # ── (b-bottom) Trends JSD ───────────────────────────────
    ax_bjsd = fig.add_subplot(gs[1, 1], sharex=ax_b)
    jx_gt = np.arange(1, len(gt_months))
    tick_gt = list(range(0, len(gt_months), 6))
    lbl_gt = [gt_months[i][2:].replace('-', '/') for i in tick_gt]

    ax_bjsd.fill_between(jx_gt, gt_jsd, alpha=0.15, color='#2166AC')
    ax_bjsd.plot(jx_gt, gt_jsd, color='#2166AC', linewidth=1.2, marker='o',
                 markersize=2.5, markerfacecolor='white', markeredgewidth=0.6)
    avg_gt = np.mean(gt_jsd)
    ax_bjsd.axhline(avg_gt, color='#D6604D', linestyle='--', linewidth=1.0,
                     alpha=0.8, label=f'Mean = {avg_gt:.4f}')
    ax_bjsd.set_ylabel('JSD')
    ax_bjsd.set_xlabel('Month')
    ax_bjsd.set_xticks(tick_gt)
    ax_bjsd.set_xticklabels(lbl_gt, rotation=45, ha='right')
    ax_bjsd.legend(fontsize=7, loc='upper right')
    ax_bjsd.set_ylim(bottom=0)

    out_dir = os.path.join(BASE_DIR, 'figures')
    save_fig(fig, os.path.join(out_dir, 'user_query_topic_drift.png'))
    plt.close()


def print_summary(wc, gt):
    wc_months, wc_topics, wc_prop, wc_jsd = wc
    gt_months, gt_topics, gt_prop, gt_jsd = gt
    print(f"\n{'='*60}")
    print("Key Findings")
    print(f"{'='*60}")
    print(f"\n[WildChat Real User Queries]")
    print(f"  Period:     {wc_months[0]} → {wc_months[-1]} ({len(wc_months)} months)")
    print(f"  Avg JSD:    {np.mean(wc_jsd):.4f}")
    print(f"  Max JSD:    {max(wc_jsd):.4f}")
    ft = wc_topics[np.argmax(wc_prop[0])]
    lt = wc_topics[np.argmax(wc_prop[-1])]
    print(f"  Dominant:   {ft} ({wc_prop[0].max()*100:.0f}%) → {lt} ({wc_prop[-1].max()*100:.0f}%)")
    print(f"\n[Google Trends]")
    print(f"  Period:     {gt_months[0]} → {gt_months[-1]} ({len(gt_months)} months)")
    print(f"  Avg JSD:    {np.mean(gt_jsd):.4f}")
    for y in [2022, 2024, 2026]:
        idxs = [i for i, m in enumerate(gt_months) if m.startswith(str(y))]
        if idxs:
            avg = gt_prop[idxs].mean(axis=0)
            t = gt_topics[np.argmax(avg)]
            print(f"  {y} dominant: {t} ({avg.max()*100:.0f}%)")
    print(f"\n→ Topic drift is pervasive in LLM usage.")
    print(f"{'='*60}")


def save_timeseries(wc, gt):
    ts = {
        'wildchat': {'months': wc[0], 'topics': wc[1], 'proportions': wc[2].tolist(), 'jsd': wc[3]},
        'google_trends': {'months': gt[0], 'topics': gt[1], 'proportions': gt[2].tolist(), 'jsd': gt[3]},
    }
    path = os.path.join(DATA_DIR, 'query_topic_timeseries.json')
    with open(path, 'w') as f:
        json.dump(ts, f, indent=2)
    print(f"✅ Saved: {path}")


def main():
    wc = analyze_wildchat()
    gt = analyze_google_trends()
    make_figure(wc, gt)
    print_summary(wc, gt)
    save_timeseries(wc, gt)


if __name__ == '__main__':
    main()
