"""
Motivation 1B: 学术研究热点随时间演变 (OpenAlex API)
输出: motivation_1/research_topic_shift.png
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_style, COLORS, save_fig

# ==========================================
# 关键词 & 配色
# ==========================================
KEYWORDS = [
    "Cloud Computing", "3D Printing", "Blockchain", "IoT",
    "ChatGPT", "Mamba", "Decision Tree", "Big Data"
]

TOPIC_STYLES = {
    'Cloud Computing': {'color': '#1f77b4', 'marker': 'o'},
    '3D Printing':     {'color': '#ff7f0e', 'marker': 's'},
    'Blockchain':      {'color': '#2ca02c', 'marker': '^'},
    'IoT':             {'color': '#d62728', 'marker': 'v'},
    'ChatGPT':         {'color': '#9467bd', 'marker': 'D'},
    'Mamba':           {'color': '#8c564b', 'marker': 'p'},
    'Decision Tree':   {'color': '#e377c2', 'marker': 'P'},
    'Big Data':        {'color': '#7f7f7f', 'marker': '^'},
}

START_YEAR = 2010
END_YEAR = 2025


def fetch_from_openalex():
    """从 OpenAlex API 拉取数据"""
    import requests

    years = list(range(START_YEAR, END_YEAR + 1))
    all_counts = {y: {} for y in years}

    for kw in KEYWORDS:
        print(f"  🔍 Fetching: {kw}")
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"title_and_abstract.search:{kw},publication_year:{START_YEAR}-{END_YEAR}",
            "group_by": "publication_year",
            "mailto": "example@test.com"
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                for g in r.json().get('group_by', []):
                    y = int(g['key'])
                    if START_YEAR <= y <= END_YEAR:
                        all_counts[y][kw] = g['count']
        except Exception as e:
            print(f"  ❌ {kw}: {e}")
        time.sleep(0.5)

    return years, all_counts



def run():
    setup_style()

    print("📊 Motivation 1B: Research Topic Evolution")
    print("=" * 60)

    years, all_counts = fetch_from_openalex()
    # 检查是否真的拿到了数据
    has_data = any(all_counts[y].get(KEYWORDS[0], 0) > 0 for y in years)
    if not has_data:
        raise ValueError("API returned empty data")

    # ---- 计算 JS 散度 ----
    distributions = []
    for y in years:
        dist = np.array([all_counts[y].get(kw, 0) for kw in KEYWORDS], dtype=float)
        total = dist.sum()
        if total > 0:
            dist /= total
        else:
            dist = np.ones(len(KEYWORDS)) / len(KEYWORDS)
        distributions.append(dist)

    js_divs = [0.0]
    for i in range(1, len(distributions)):
        p = np.clip(distributions[i - 1], 1e-10, None)
        q = np.clip(distributions[i], 1e-10, None)
        p /= p.sum()
        q /= q.sum()
        js_divs.append(float(jensenshannon(p, q)))

    # ---- 画图 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                    gridspec_kw={'height_ratios': [2.5, 1]})

    # 上图: 论文数趋势
    for kw in KEYWORDS:
        counts_series = [all_counts[y].get(kw, 0) for y in years]
        style = TOPIC_STYLES.get(kw, {'color': 'gray', 'marker': '.'})
        ax1.plot(years, counts_series, color=style['color'], marker=style['marker'],
                 linewidth=2, markersize=5, label=kw)

    ax1.set_ylabel("Number of Papers", fontsize=13)
    ax1.set_title(f"Evolution of Research Topics ({START_YEAR}-{END_YEAR})",
                  fontsize=15, fontweight='bold')
    ax1.legend(loc='upper left', title='Topics', fontsize=9, title_fontsize=10,
               bbox_to_anchor=(1, 1))
    ax1.set_xlim(START_YEAR - 0.5, END_YEAR + 0.5)

    # 关键事件
    for yr, label in [(2017, "Transformer"), (2022, "ChatGPT")]:
        ax1.axvline(x=yr, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax1.text(yr + 0.2, ax1.get_ylim()[1] * 0.9, label,
                 fontsize=9, color='gray', fontstyle='italic')

    # 下图: JS 散度
    ax2.plot(years, js_divs, color=COLORS['secondary'], linewidth=2.5, marker='o', markersize=5)
    ax2.fill_between(years, js_divs, color=COLORS['secondary'], alpha=0.1)
    ax2.set_ylabel("Drift (JS Divergence)", fontsize=12, color=COLORS['secondary'])
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_title("Year-over-Year Distribution Shift Magnitude", fontsize=12)
    ax2.tick_params(axis='y', labelcolor=COLORS['secondary'])

    max_idx = int(np.argmax(js_divs))
    ax2.annotate(f'Peak drift: {years[max_idx]}',
                 xy=(years[max_idx], js_divs[max_idx]),
                 xytext=(years[max_idx] + 1, js_divs[max_idx] + 0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                 fontsize=10, fontweight='bold')

    plt.tight_layout()
    out_path = str(Path(__file__).resolve().parent / "research_topic_shift.png")
    save_fig(fig, out_path)
    plt.close()


if __name__ == "__main__":
    run()