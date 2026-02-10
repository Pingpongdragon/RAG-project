import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import jensenshannon
from time import sleep

# ================= 配置 =================
# 想要对比的学术关键词 (代表不同时代的 AI 热点)
KEYWORDS = [
    "Support Vector Machine", 
    "Random Forest",
    "Deep Learning", 
    "Convolutional Neural Network",
    "Transformer",
    "Large Language Model"
]

START_YEAR = 2010
END_YEAR = 2024

def fetch_openalex_counts(keyword, start_year, end_year):
    """
    从 OpenAlex API 获取指定关键词每年的论文数量
    """
    url = "https://api.openalex.org/works"
    counts = {}
    
    print(f"🔍 Fetching data for: '{keyword}' ...")
    
    # OpenAlex 支持按发表年份分组统计
    params = {
        "filter": f"title_and_abstract.search:{keyword},publication_year:{start_year}-{end_year}",
        "group_by": "publication_year",
        "mailto": "example@test.com" # OpenAlex 建议加上邮箱以便联系
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for group in data['group_by']:
                year = int(group['key'])
                count = group['count']
                if start_year <= year <= end_year:
                    counts[year] = count
        else:
            print(f"❌ Error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")
        
    sleep(0.5) # 礼貌请求，避免被封
    return counts

def compute_distribution_shift():
    # 1. 收集数据
    df_data = []
    years = range(START_YEAR, END_YEAR + 1)
    
    all_counts = {year: {} for year in years}
    
    for kw in KEYWORDS:
        counts = fetch_openalex_counts(kw, START_YEAR, END_YEAR)
        for year in years:
            val = counts.get(year, 0)
            all_counts[year][kw] = val
            df_data.append({"Year": year, "Topic": kw, "Count": val})
            
    df = pd.DataFrame(df_data)
    
    # 2. 计算每年的分布 (归一化)
    # P(Topic | Year)
    distribution_matrix = []
    js_divergences = []
    prev_dist = None
    
    print("\n📊 Calculating Distributions & Drift...")
    
    for year in years:
        year_counts = [all_counts[year][kw] for kw in KEYWORDS]
        total = sum(year_counts)
        
        if total == 0:
            dist = np.array([1.0/len(KEYWORDS)] * len(KEYWORDS)) # 避免除零
        else:
            dist = np.array(year_counts) / total
            
        distribution_matrix.append(dist)
        
        # 计算与上一年的 JS 散度 (Drift Magnitude)
        if prev_dist is not None:
            js_div = jensenshannon(prev_dist, dist)
            js_divergences.append(js_div)
        else:
            js_divergences.append(0.0) # 第一年无偏移
            
        prev_dist = dist

    distribution_matrix = np.array(distribution_matrix).T # 转置以便绘图 (Topics x Years)
    
    return df, distribution_matrix, js_divergences, years

def plot_results(df, dist_matrix, js_divs, years):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # --- 图1: 堆叠面积图 (分布变化) ---
    pal = sns.color_palette("Spectral", len(KEYWORDS))
    
    ax1.stackplot(years, dist_matrix, labels=KEYWORDS, colors=pal, alpha=0.85)
    ax1.set_ylabel("Topic Probability $P(X)$", fontsize=12)
    ax1.set_title(f"Real-world Distribution Shift in AI Research ({START_YEAR}-{END_YEAR})", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Topics")
    ax1.set_xlim(START_YEAR, END_YEAR)
    ax1.set_ylim(0, 1)
    
    # 添加关键事件标注
    ax1.axvline(x=2017, color='white', linestyle='--', alpha=0.5)
    ax1.text(2017.1, 0.5, "Transformer\nPublished", color='white', fontsize=9, fontweight='bold')
    
    ax1.axvline(x=2022, color='white', linestyle='--', alpha=0.5)
    ax1.text(2022.1, 0.8, "ChatGPT\nReleased", color='white', fontsize=9, fontweight='bold')

    # --- 图2: JS 散度 (偏移速率) ---
    sns.lineplot(x=years, y=js_divs, ax=ax2, color='#d62728', linewidth=2.5, marker='o')
    ax2.fill_between(years, js_divs, color='#d62728', alpha=0.1)
    
    ax2.set_ylabel("Drift Magnitude (JS Divergence)", fontsize=12, color='#d62728')
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_title("Rate of Distribution Shift (How fast the field is changing)", fontsize=12)
    
    # 标注高偏移点
    max_drift_idx = np.argmax(js_divs)
    max_drift_year = years[max_drift_idx]
    max_drift_val = js_divs[max_drift_idx]
    
    ax2.annotate(f'Max Drift: {max_drift_year}', 
                 xy=(max_drift_year, max_drift_val), 
                 xytext=(max_drift_year, max_drift_val + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 ha='center')

    plt.tight_layout()
    out_path = "academic_distribution_shift.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {out_path}")
    plt.show()

if __name__ == "__main__":
    # 强制清除代理，防止 API 请求失败
        
    df, dist_matrix, js_divs, years = compute_distribution_shift()
    plot_results(df, dist_matrix, js_divs, years)