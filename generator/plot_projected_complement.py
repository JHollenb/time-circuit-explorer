"""Plot only published scalar comparisons and declared physical work."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from encoder_atoms import read, save, sha


def execute():
    root = Path(__file__).resolve().parent.parent
    parent = root / 'docs/results/sparse-text-kv'
    source = root / 'docs/results/projected-complement'
    out = root / 'docs/results/projected-complement-figure'
    a, b = read(parent / 'report.json'), read(source / 'report.json')
    pa = {x['panel']: x for x in a['queryPanels']}
    pb = {x['panel']: x for x in b['queryPanels'] if x['reference'] == 'family'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': '#172333', 'axes.labelcolor': '#172333', 'figure.facecolor': '#f8fafc', 'axes.facecolor': '#ffffff'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=.08, right=.95, bottom=.24, top=.72, wspace=.4)
    fig.suptitle('Project four live values through the output map', fontsize=18, fontweight='bold', x=.08, ha='left', y=.96)
    fig.text(.08, .86, 'Fixed-query FLUX reader · 24 physical head blocks · original learned maps remain paid', fontsize=11, color='#536273')
    x = np.arange(3)
    for shift, rows, label, color in [(-.19, pa, 'Cycle52: full-grid Wo', '#4678c9'), (.19, pb, 'Cycle53: projected cache', '#d48c36')]:
        values = [rows[p]['relativeRms'] * 100 for p in ['all', 'left-grid', 'right-grid']]
        bars = axes[0].bar(x + shift, values, width=.35, color=color, label=label)
        axes[0].bar_label(bars, fmt='%.2f', padding=4, fontsize=10)
    axes[0].set(xticks=x, xticklabels=['All 1,024', 'Left 512', 'Right 512'], ylabel='Native response error (%)', ylim=(0, 16), title='Response error at every actual image query')
    axes[0].legend(loc='upper left', fontsize=9, frameon=False)
    costs = [a['actorsMACs'] / 2e9, b['actorsMatrixMACs'] / 2e9]
    bars = axes[1].bar(['Cycle52', 'Cycle53'], costs, color=['#4678c9', '#d48c36'], width=.55)
    axes[1].bar_label(bars, fmt='%.3f', padding=5)
    axes[1].set(ylabel='Billion matrix MACs per actor', ylim=(0, 12), title='Recurring full-grid Wo is eliminated')
    fig.text(.08, .16, 'Median query error: 40.03% → 57.20%. Queries losing to zero: 20 → 71.', fontsize=11)
    fig.text(.08, .105, 'Cache: 25.56 → 604.37 MB. New compilation: 9.664B matrix MACs. Scalar work and source production are additional.', fontsize=10)
    fig.text(.08, .05, 'No matched runtime speedup, later-state parity, image inheritance or arbitrary-prompt zero-neural result is established.', fontsize=10, color='#536273')
    out.mkdir(exist_ok=False, parents=True)
    file = out / 'projected-complement.png'
    fig.savefig(file, dpi=160)
    plt.close(fig)
    save(out / 'figure.json', {'parentReportSHA256': sha(parent / 'report.json'), 'reportSHA256': sha(source / 'report.json'),
         'plotSourceSHA256': sha(__file__), 'imageSHA256': sha(file), 'scope': 'Published scalar/operation-count comparison only. Physical halves are not semantic regions; no model or scientific array execution.'})
    return str(file)


if __name__ == '__main__':
    print(execute())
