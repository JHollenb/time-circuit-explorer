"""Plot published scalar errors and norm cells; no activation loads."""
import gzip
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from encoder_atoms import read, save, sha


def execute():
    root = Path(__file__).resolve().parent.parent;p = root / 'docs/results/coherent-norm';old = root / 'docs/results/diagonal-norm'
    r = read(p / 'report.json');rows = [json.loads(x) for x in gzip.decompress((p / 'observations.jsonl.gz').read_bytes()).splitlines()]
    previous = [json.loads(x) for x in gzip.decompress((old / 'observations.jsonl.gz').read_bytes()).splitlines()]
    out = root / 'docs/results/coherent-norm-figure';out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold', 'text.color': '#0f172a', 'axes.labelcolor': '#334155', 'axes.edgecolor': '#cbd5e1'})
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), gridspec_kw={'width_ratios': [1.2, 1.05, .9]});fig.patch.set_facecolor('#f8fafc')
    for a in axes:
        a.set_facecolor('#f8fafc');a.set_axisbelow(True)
    names = ['H', 'A', 'M', 'T', 'nextQProjected', 'nextQ'];xs = list(range(6))
    for shift, key, color, label in [(-.25, 'retainedResponse', '#0d9488', 'Retained P norm'), (0, 'diagonalResponse', '#e05b49', 'Input-channel diagonal'), (.25, 'primaryResponse', '#6366f1', 'Fixed output sketch')]:
        axes[0].bar([x+shift for x in xs], [r['scores'][n][key]['relativeRms'] for n in names], .24, color=color, label=label)
    axes[0].axhline(1, color='#64748b', linestyle='--', linewidth=1);axes[0].text(5.7, 1.12, 'Zero response', ha='right', color='#64748b', fontsize=9)
    axes[0].set_yscale('log');axes[0].set_ylim(.045, 90);axes[0].set_xticks(xs, ['Head H', 'Return A', 'MLP M', 'Exit T', 'Next Q\nprojected', 'Next Q\npositioned'], fontsize=9)
    axes[0].set_title('Better than diagonal; still loses to zero', loc='left', pad=16);axes[0].set_ylabel('RMS error / actual response RMS (log scale)')
    axes[0].legend(frameon=False, fontsize=8.5, loc='upper left');axes[0].grid(axis='y', alpha=.18)
    axes[1].hist([x['normRatio'] for x in previous if x['panel'] == 'image-key-head'], bins=40, range=(0, 1.8), alpha=.55, color='#e05b49', label='Diagonal')
    axes[1].hist([x['normRatio'] for x in rows if x['panel'] == 'image-key-head'], bins=40, range=(0, 1.8), alpha=.65, color='#6366f1', label='Fixed sketch')
    axes[1].axvline(1, color='#64748b', linestyle='--', linewidth=1);axes[1].set_xlim(0, 1.8);axes[1].set_xlabel('Predicted norm / correct smooth Q norm');axes[1].set_ylabel('Image-key / head cells')
    axes[1].set_title('Balanced center, substantial local errors', loc='left', pad=16);axes[1].legend(frameon=False, fontsize=9, loc='upper right');axes[1].grid(axis='y', alpha=.18)
    values = [r['diagnostics']['retainedPNormToQ']['relativeRms'], r['diagnostics']['sketchNormToQ']['relativeRms']]
    axes[2].bar([0, 1], values, color=['#94a3b8', '#6366f1'], width=.5);axes[2].set_yscale('log');axes[2].set_ylim(.001, 1)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1));axes[2].set_xticks([0, 1], ['Actual P→Q\nnorm change', 'Sketch−Q\nnorm error'])
    axes[2].set_ylabel('RMS / correct Q norm RMS (log scale)');axes[2].grid(axis='y', alpha=.18)
    axes[2].set_title('Norm error overwhelms the change', loc='left', pad=16)
    for i, v in enumerate(values):
        axes[2].text(i, 1.12*v, f'{100*v:.2f}%', ha='center', fontsize=11)
    axes[2].text(.05, .96, 'Error / actual norm change: 35.82×', transform=axes[2].transAxes, va='top', fontsize=10)
    fig.suptitle('Preserving cross terms improves energy estimation, but not response fidelity enough', x=.055, ha='left', y=.98, fontsize=18, fontweight='bold')
    fig.text(.055, .055, 'One supplied P/Q pair; fixed r=16 and one seed before outcomes. All 96 next-query heads improve over diagonal; none beats zero or the retained norm.', fontsize=10, color='#475569')
    fig.text(.055, .022, 'Unbiased energy is an ideal ensemble property, not one sketch’s guarantee. Original source states, dual compilation and learned receiving maps remain paid; no new image.', fontsize=10, color='#475569')
    fig.subplots_adjust(left=.055, right=.98, bottom=.22, top=.82, wspace=.42)
    png = out / 'coherent-norm.png';fig.savefig(png, dpi=160, facecolor=fig.get_facecolor());plt.close(fig)
    save(out / 'figure.json', {'imageSHA256': sha(png), 'plotSourceSHA256': sha(__file__), 'reportSHA256': sha(p / 'report.json'),
        'diagonalReportSHA256': sha(old / 'report.json'), 'scope': 'Published scalar observations only; no model/tensor execution.'})
    shutil.copyfile(png, root.parent / 'obsidian/images/zero-neural-cycle60-coherent-norm.png')
    return str(png)


if __name__ == '__main__':
    print(execute())
