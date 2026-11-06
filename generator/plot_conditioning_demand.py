"""Plot published consumer scalar accounts; no state or parameter hydration."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from encoder_atoms import read, save, sha


def plot(source, destination):
    source = Path(source); out = Path(destination); out.mkdir(parents=True, exist_ok=False)
    report = read(source / 'report.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8), constrained_layout=True)
    fig.suptitle('A better local state remains better at the consumer, with reversals on the way', fontweight='bold', fontsize=14)
    for ax, region, title in zip(axes, ['all', 'active', 'padding'], ['All 512 rows', '28 active rows', '484 padding rows']):
        for branch, label, color in [('old', 'Original compact', '#b45309'), ('new', 'New compact policy', '#2563eb')]:
            values = [next(a for a in report['accounts'] if a['port'] == 'H' + str(i) and a['region'] == region) for i in range(2, 28)]
            ax.plot(range(2, 28), [100 * v[branch]['relativeRms'] for v in values], label=label, color=color, linewidth=1.7)
        for tap in (9, 18, 27): ax.axvline(tap, color='#94a3b8', alpha=.25, linewidth=1)
        consumer = next(a for a in report['accounts'] if a['port'] == 'C' and a['region'] == region)
        ax.text(.03, .95, f"Context projection C\n{100*consumer['old']['relativeRms']:.3f}% → {100*consumer['new']['relativeRms']:.3f}%", transform=ax.transAxes,
            va='top', fontsize=9, bbox={'facecolor': 'white', 'edgecolor': '#e2e8f0', 'alpha': .9, 'pad': 6})
        ax.set(title=title, xlabel='Emitted hidden-state index', xticks=[2, 9, 18, 27], xlim=(2, 27), ylabel='Error / matching full-state RMS (%)')
        ax.margins(y=.2); ax.set_ylim(bottom=0); ax.grid(alpha=.15)
    axes[0].legend(fontsize=8, loc='lower right')
    fig.supxlabel('Curves: own states H2–H27 · Three learned taps feed native C · Separate vertical scales · No image-quality claim', fontsize=10)
    path = out / 'conditioning-demand.png'; fig.savefig(path, dpi=160); plt.close(fig)
    save(out / 'figure.json', {'schema': 'tce-conditioning-demand-derived-figure-v1', 'sourceReportSHA256': sha(source / 'report.json'),
        'figureSHA256': sha(path), 'scope': 'Three presealed H2 inputs; 75 paid original blocks and three native context projections. All/active/padding scales differ intentionally; same-context dependent states/rows.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('destination'); a = p.parse_args(); plot(a.source, a.destination)
