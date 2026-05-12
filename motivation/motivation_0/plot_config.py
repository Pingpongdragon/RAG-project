import matplotlib.pyplot as plt
import matplotlib as mpl

def setup_style():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.linestyle': '--',
    })

def save_fig(fig, path, dpi=200):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    pdf = path.replace('.png', '.pdf')
    fig.savefig(pdf, bbox_inches='tight')
    print(f'Saved {path}')
    print(f'Saved {pdf}')
