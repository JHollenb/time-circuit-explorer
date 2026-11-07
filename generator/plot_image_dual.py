"""A publication figure from immutable scalar rows, without activation loads."""
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
    root = Path(__file__).resolve().parent.parent;p = root / 'docs/results/image-dual'
    report = read(p / 'report.json')
    rows = [json.loads(x) for x in gzip.decompress((p / 'observations.jsonl.gz').read_bytes()).splitlines()]
    out = root / 'docs/results/image-dual-figure';out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.labelcolor': '#334155',
                         'text.color': '#0f172a', 'axes.edgecolor': '#cbd5e1'})
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), gridspec_kw={'width_ratios': [1.1, 1, 1.05]})
    fig.patch.set_facecolor('#f8fafc')
    for a in axes:
        a.set_facecolor('#f8fafc');a.set_axisbelow(True)
    names = ['H', 'A', 'M', 'T', 'nextQProjected', 'nextQ']
    labels = ['Head H', 'Return A', 'MLP M', 'Exit T', 'Next Q\nprojected', 'Next Q\npositioned']
    x = list(range(len(names)))
    axes[0].bar([j-.18 for j in x], [report['scores'][n]['staleResponse']['relativeRms'] for n in names], .36, color='#94a3b8', label='Retain all image keys')
    axes[0].bar([j+.18 for j in x], [report['scores'][n]['primaryResponse']['relativeRms'] for n in names], .36, color='#0d9488', label='Dual scores + retained norm')
    axes[0].set_xticks(x, labels, fontsize=9);axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[0].grid(axis='y', alpha=.2)
    axes[0].set_title('Own-state response fidelity', loc='left', pad=16)
    axes[0].set_ylabel('RMS error / actual context-response RMS')
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    palette = ['#0d9488', '#2563eb', '#7c3aed', '#ea580c']
    for row, color in zip([4, 11, 12, 13], palette):
        points = [x for x in rows if x['field'] == 'nextQ' and x['panel'] == 'query-head' and x['row'] == row]
        axes[1].scatter([x['staleResponse']['relativeRms'] for x in points], [x['primaryResponse']['relativeRms'] for x in points],
                        color=color, s=25, alpha=.75, label='Row ' + str(row))
    axes[1].plot([0, 1.1], [0, 1.1], '--', color='#94a3b8', linewidth=1)
    axes[1].set_xlim(0, 1.1);axes[1].set_ylim(0, 1.1)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1));axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlabel('Retained-key response error');axes[1].set_ylabel('Dual-score response error')
    axes[1].set_title('96 dependent next-query heads', loc='left', pad=16)
    axes[1].text(.04, .96, 'All 96 improve on stale and zero', transform=axes[1].transAxes, va='top', fontsize=10)
    axes[1].legend(frameon=False, fontsize=9, loc='lower right', ncol=2)
    parts = ['denominatorAndContraction', 'keyFinitePolicy', 'attentionFinitePolicy', 'admissionCast', 'total']
    ticks = ['Retained norm +\ncontraction arithmetic', 'Key finite policy', 'Attention finite policy', 'Admission cast', 'Total admitted H error']
    values = [report['headErrorDecomposition'][n]['relativeToActualResponse'] for n in parts]
    axes[2].barh(range(5), values, color=['#0d9488', '#94a3b8', '#94a3b8', '#94a3b8', '#0f172a'], height=.55)
    axes[2].set_yticks(range(5), ticks, fontsize=9);axes[2].invert_yaxis();axes[2].set_xlim(0, .095)
    axes[2].xaxis.set_major_formatter(PercentFormatter(1));axes[2].grid(axis='x', alpha=.2)
    for j, v in enumerate(values):
        axes[2].text(v+.001, j, f'{100*v:.2f}%', va='center', fontsize=9)
    axes[2].set_title('Head-error components', loc='left', pad=16)
    axes[2].set_xlabel('Term RMS / actual H-response RMS')
    fig.suptitle('A factored image-score producer advances its own next query', x=.055, ha='left', fontsize=18, fontweight='bold', y=.98)
    fig.text(.055, .025, 'One supplied P/Q context pair, four selected rows. Signed error terms telescope; their RMS magnitudes do not add. Original maps, compilation and references remain paid.', fontsize=10, color='#475569')
    fig.subplots_adjust(left=.055, right=.98, bottom=.2, top=.82, wspace=.55)
    image = out / 'image-dual.png';fig.savefig(image, dpi=160, facecolor=fig.get_facecolor());plt.close(fig)
    save(out / 'figure.json', {'imageSHA256': sha(image), 'plotSourceSHA256': sha(__file__), 'reportSHA256': sha(p / 'report.json'),
                             'scope': 'Published scalar response errors, all final query-head addresses and signed-telescope term norms; no tensor/model execution.'})
    shutil.copyfile(image, root.parent / 'obsidian/images/zero-neural-cycle58-image-dual.png')
    return str(image)


if __name__ == '__main__':
    print(execute())
