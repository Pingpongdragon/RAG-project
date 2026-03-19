"""
Motivation 2: Query-Document Alignment

核心叙事:
  当用户查询话题发生漂移 (Query Shift) 时, 知识库中的文档与新查询不再对齐,
  导致检索和回答性能骤降。通过添加与新查询对齐的文档, 性能可以恢复。

图表布局 (1×2):
  (a) 多 Retriever Scaling Recovery — 4 种检索器在 domain shift 后的恢复曲线对比
  (b) Misalignment Degree (JSD) vs Performance — JSD 越高, Hit@10/Coverage 越低

数据来源:
  - results_scaling.json: SQuAD→HotpotQA domain shift + scaling recovery (4 retrievers)
      生成脚本: legacy/gen_scaling_recovery.py
  - results_misalignment.json: KB分布偏移 (Aligned→Extreme) → Hit@10/Coverage (聚类版)
      生成脚本: gen_misalignment.py
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_style, COLORS, MODEL_STYLES, save_fig

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"


def load_all_results():
    scaling_data = json.loads((DATA_DIR / "results_scaling.json").read_text())
    misalign_data = json.loads((DATA_DIR / "results_misalignment.json").read_text())
    return scaling_data, misalign_data


def plot_figure(scaling_data, misalign_data):
    setup_style()
    plt.rcParams.update({'font.size': 10})

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97, bottom=0.15, top=0.90)

    # ============================
    # (a) Multi-Retriever Timeline
    # ============================
    n_points = len(list(scaling_data["results"].values())[0])
    x_pos = list(range(n_points))

    # 背景区域
    ax_a.axvspan(-0.4, 0.5, alpha=0.06, color='#059669')
    ax_a.axvspan(0.5, 1.5, alpha=0.07, color='#DC2626')
    ax_a.axvspan(1.5, n_points - 0.5, alpha=0.05, color='#2563EB')
    ax_a.axvline(x=0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax_a.axvline(x=1.5, color='gray', ls='--', lw=0.8, alpha=0.5)

    for model_name, accs in scaling_data["results"].items():
        style = MODEL_STYLES.get(model_name, {'color': '#6B7280', 'marker': 'o'})
        ax_a.plot(x_pos, accs, color=style['color'], marker=style['marker'],
                  linewidth=1.8, markersize=6, markeredgecolor='white',
                  markeredgewidth=0.8, label=model_name, zorder=2)

    # Drop & Recovery annotations (BGE-Small 为参考)
    ref_accs = scaling_data["results"]["BGE-Small (Dense)"]
    y_base, y_drop, y_recover = ref_accs[0], ref_accs[1], ref_accs[-1]
    ax_a.annotate('', xy=(1, y_drop), xytext=(1, y_base),
                  arrowprops=dict(arrowstyle='<->', color='#DC2626', lw=1.8))
    ax_a.text(1.3, (y_base + y_drop) / 2, f'−{y_base - y_drop:.0%}\ndrop',
              fontsize=9, fontweight='bold', color='#DC2626',
              bbox=dict(boxstyle='round,pad=0.25', fc='#FEE2E2', ec='#DC2626', alpha=0.9),
              ha='left', va='center')

    mid_x = (2 + n_points - 1) / 2
    ax_a.annotate('', xy=(n_points - 1, y_recover), xytext=(2, ref_accs[2]),
                  arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.5,
                                  ls='--', connectionstyle='arc3,rad=0.15'))
    ax_a.text(mid_x + 0.8, y_recover - 0.03,
              f'+{y_recover - y_drop:.0%} recovery',
              fontsize=9, fontweight='bold', color='#2563EB',
              bbox=dict(boxstyle='round,pad=0.25', fc='#DBEAFE', ec='#2563EB', alpha=0.9),
              ha='center', va='top')

    # 阶段标签
    ax_a.text(0, 0.20, 'Aligned\n(Baseline)', ha='center', fontsize=8,
              fontweight='bold', color='#059669', style='italic')
    ax_a.text(1, 0.20, 'Query\nShift', ha='center', fontsize=8,
              fontweight='bold', color='#DC2626', style='italic')
    ax_a.text(mid_x, 0.20, 'Add Aligned Docs',
              ha='center', fontsize=8, fontweight='bold', color='#2563EB', style='italic')

    x_labels = ['$T_0$\nBase', '$T_1$\nShift',
                '$T_2$\n+10%', '$T_3$\n+30%', '$T_4$\n+50%',
                '$T_5$\n+70%', '$T_6$\n+90%', '$T_7$\n+100%']
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(x_labels, fontsize=7.5)
    ax_a.set_ylabel('End-to-End Accuracy', fontsize=11)
    ax_a.set_xlabel('Timeline', fontsize=11)
    ax_a.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_a.set_ylim(0.17, 0.82)
    ax_a.set_title('(a) Query Shift → Performance Drop → Recovery',
                    fontsize=11, fontweight='bold')
    ax_a.legend(loc='upper left', fontsize=7.5, framealpha=0.95, ncol=1)

    # ============================
    # (b) JSD vs Performance
    # ============================
    jsd_vals = [r["jsd"] for r in misalign_data]
    hit_vals = [r["hit_rate"] for r in misalign_data]
    cov_vals = [r["query_coverage"] for r in misalign_data]
    labels_m = [r["label"] for r in misalign_data]

    ax_b.plot(jsd_vals, hit_vals, 'o-', color='#2563EB', lw=2.2, ms=8,
              markeredgecolor='white', markeredgewidth=1.2, label='Hit@10', zorder=3)
    ax_b.plot(jsd_vals, cov_vals, 's--', color='#D97706', lw=2.0, ms=7,
              markeredgecolor='white', markeredgewidth=1.2, label='Gold Coverage', zorder=3)

    for i, (jsd, hit, cov, lbl) in enumerate(zip(jsd_vals, hit_vals, cov_vals, labels_m)):
        ax_b.annotate(f'{hit:.1%}', xy=(jsd, hit), xytext=(5, 8),
                      textcoords='offset points', fontsize=7.5, fontweight='bold',
                      color='#2563EB')
        oy = -13 if i > 0 else 8
        ax_b.annotate(f'{cov:.1%}', xy=(jsd, cov), xytext=(5, oy),
                      textcoords='offset points', fontsize=7.5, fontweight='bold',
                      color='#D97706')
        ax_b.annotate(lbl, xy=(jsd, 0), xytext=(0, -28), textcoords='offset points',
                      fontsize=7, ha='center', color='#6B7280', rotation=15)

    ax_b.fill_between(jsd_vals, hit_vals, alpha=0.08, color='#2563EB')

    ax_b.set_xlabel('JSD (KB ∥ Query Distribution)', fontsize=11)
    ax_b.set_ylabel('Hit Rate / Coverage', fontsize=11)
    ax_b.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_b.set_title('(b) Higher Misalignment → Lower Performance',
                    fontsize=11, fontweight='bold')
    ax_b.legend(loc='upper right', fontsize=9, framealpha=0.95)

    all_vals = hit_vals + cov_vals
    ax_b.set_ylim(max(0, min(all_vals) - 0.08), max(all_vals) + 0.08)
    ax_b.set_xlim(-0.02, max(jsd_vals) + 0.05)

    # 保存
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(FIG_DIR / "query_doc_alignment")
    save_fig(fig, out_path)
    plt.close()
    print(f"\n✅ Figure saved: {out_path}.png / .pdf")


def print_summary(scaling_data, misalign_data):
    print("=" * 60)
    print("  Motivation 2: Query-Document Alignment Summary")
    print("=" * 60)

    print(f"\n  (a) Multi-Retriever Scaling Recovery:")
    for model, accs in scaling_data["results"].items():
        drop = accs[0] - accs[1]
        recovery = accs[-1] - accs[1]
        print(f"      {model:25s}  Base={accs[0]:.0%}  Shift={accs[1]:.0%}  "
              f"Drop=−{drop:.0%}  Final={accs[-1]:.0%}  Recovery=+{recovery:.0%}")

    print(f"\n  (b) Misalignment Degree:")
    for r in misalign_data:
        print(f"      {r['label']:10s}  JSD={r['jsd']:.3f}  "
              f"Hit@10={r['hit_rate']:.1%}  Coverage={r['query_coverage']:.1%}  "
              f"KB={r['kb_size']:,}")


if __name__ == "__main__":
    scaling_data, misalign_data = load_all_results()
    plot_figure(scaling_data, misalign_data)
    print_summary(scaling_data, misalign_data)
