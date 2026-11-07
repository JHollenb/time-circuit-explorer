"""Publication plot from immutable scalar records; no activation loads."""
import gzip
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from encoder_atoms import read, save, sha


def execute():
    root = Path(__file__).resolve().parent.parent;p = root / 'docs/results/diagonal-norm'
    report = read(p / 'report.json');moments_path = root / 'docs/results/diagonal-norm-moments/report.json';moments = read(moments_path)
    rows = [json.loads(x) for x in gzip.decompress((p / 'observations.jsonl.gz').read_bytes()).splitlines()]
    out = root / 'docs/results/diagonal-norm-figure';out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.titleweight': 'bold', 'text.color': '#0f172a',
        'axes.labelcolor': '#334155', 'axes.edgecolor': '#cbd5e1'})
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), gridspec_kw={'width_ratios': [1.05, 1, 1.15]})
    fig.patch.set_facecolor('#f8fafc')
    for a in axes:
        a.set_facecolor('#f8fafc');a.set_axisbelow(True)
    names = ['H', 'A', 'M', 'T', 'nextQProjected', 'nextQ'];x = list(range(6))
    axes[0].bar([j-.19 for j in x], [report['scores'][n]['retainedResponse']['relativeRms'] for n in names], .38, color='#0d9488', label='Retained P norm')
    axes[0].bar([j+.19 for j in x], [report['scores'][n]['primaryResponse']['relativeRms'] for n in names], .38, color='#e05b49', label='Weight-only diagonal norm')
    axes[0].axhline(1, color='#64748b', linestyle='--', linewidth=1)
    axes[0].text(5.6, 1.12, 'Zero response', ha='right', fontsize=9, color='#64748b')
    axes[0].set_yscale('log');axes[0].set_ylim(.045, 90)
    axes[0].set_xticks(x, ['Head H', 'Return A', 'MLP M', 'Exit T', 'Next Q\nprojected', 'Next Q\npositioned'], fontsize=9)
    axes[0].set_title('Own-state response error grows', loc='left', pad=16)
    axes[0].set_ylabel('RMS error / actual response RMS (log scale)');axes[0].grid(axis='y', alpha=.18)
    axes[0].legend(frameon=False, fontsize=8.5, loc='upper left')
    ratios = [x['normRatio'] for x in rows if x['panel'] == 'image-key-head']
    axes[1].hist(ratios, bins=36, range=(0, 1.05), color='#e05b49', alpha=.9)
    axes[1].axvline(1, color='#64748b', linestyle='--', linewidth=1)
    axes[1].set_xlim(0, 1.06);axes[1].set_xlabel('Diagonal norm / correct smooth Q norm');axes[1].set_ylabel('Image-key / head cells')
    axes[1].set_title('All 24,576 norms are too small', loc='left', pad=16)
    axes[1].text(.04, .96, 'Median ratio: 0.476\nRange: 0.152–0.786', transform=axes[1].transAxes, va='top')
    axes[1].grid(axis='y', alpha=.18)
    hs = list(range(24));mr = moments['rows']
    axes[2].bar(hs, [r['meanOmittedEnergy'] for r in mr], color='#6366f1', label='Shared spatial mean')
    axes[2].bar(hs, [r['variationOmittedEnergy'] for r in mr], bottom=[r['meanOmittedEnergy'] for r in mr], color='#d7b679', label='Centered variation')
    axes[2].set_ylim(0, 1.25 * max(r['meanOmittedEnergy'] + r['variationOmittedEnergy'] for r in mr))
    axes[2].set_xticks([0, 5, 10, 15, 20, 23]);axes[2].set_xlabel('Head');axes[2].set_ylabel('Mean omitted squared norm, epsilon excluded')
    axes[2].set_title('Coherence is spread across the field', loc='left', pad=16)
    axes[2].legend(frameon=False, fontsize=9, loc='upper right');axes[2].grid(axis='y', alpha=.18)
    axes[2].text(.02, .98, 'Pooled split\n58.3% mean / 41.7% variation', transform=axes[2].transAxes, va='top', fontsize=9)
    fig.suptitle('A diagonal norm discards channel coherence needed by attention', x=.055, ha='left', fontsize=18, fontweight='bold', y=.98)
    fig.text(.055, .055, 'One supplied P/Q context pair, four selected rows. Five payload controls match; only the norm rule changes. Correct-mean direction still leaves 40.7% of total energy orthogonal.', fontsize=10, color='#475569')
    fig.text(.055, .022, 'Moment split is post-hoc diagnosis, not a serving repair. Original states, learned maps and historical references remain paid; no image or arbitrary-prompt zero-neural result.', fontsize=10, color='#475569')
    fig.subplots_adjust(left=.055, right=.98, bottom=.22, top=.82, wspace=.4)
    target = out / 'diagonal-norm.png';fig.savefig(target, dpi=160, facecolor=fig.get_facecolor());plt.close(fig)
    save(out / 'figure.json', {'imageSHA256': sha(target), 'plotSourceSHA256': sha(__file__),
        'primaryReportSHA256': sha(p / 'report.json'), 'momentsReportSHA256': sha(moments_path),
        'scope': 'Published scalar response errors, complete norm cells and per-head signed mean energies only.'})
    shutil.copyfile(target, root.parent / 'obsidian/images/zero-neural-cycle59-diagonal-norm.png')
    return str(target)


if __name__ == '__main__':
    print(execute())
