"""
全局绘图风格配置 — 所有 motivation 图共用
"""
import os
import matplotlib.pyplot as plt
import matplotlib as mpl


def setup_style():
    """设置全局 matplotlib 风格"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.titlesize': 15,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.8,
        'grid.linestyle': '--',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


COLORS = {
    'primary':      '#2563EB',
    'secondary':    '#DC2626',
    'accent1':      '#059669',
    'accent2':      '#D97706',
    'accent3':      '#7C3AED',
    'gray':         '#6B7280',
    'dark':         '#1F2937',
    'light_blue':   '#DBEAFE',
    'light_red':    '#FEE2E2',
    'light_green':  '#D1FAE5',
    'light_yellow': '#FEF3C7',
}

MODEL_STYLES = {
    'MiniLM-L6 (Dense)':   {'color': '#2563EB', 'marker': 'o'},
    'BGE-Small (Dense)':   {'color': '#D97706', 'marker': 's'},
    'BGE-Large (Dense)':   {'color': '#059669', 'marker': '^'},
    'BGE-Small (Hybrid)':  {'color': '#DC2626', 'marker': 'D'},
}


def save_fig(fig, path):
    """保存图片 (png + pdf)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    pdf_path = path.rsplit('.', 1)[0] + '.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ Saved: {path}")
    print(f"✅ Saved: {pdf_path}")