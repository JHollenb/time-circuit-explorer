"""Publication plot from paired diagnostic scalar summaries only."""
from pathlib import Path
import math
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from encoder_atoms import read, save, sha


def execute():
    root = Path(__file__).resolve().parent.parent;p = root / 'docs/results/paired-sketch-diagnostic';r = read(p / 'report.json');s = r['pooled']
    out = root / 'docs/results/paired-sketch-figure';out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold', 'text.color': '#0f172a', 'axes.labelcolor': '#334155', 'axes.edgecolor': '#cbd5e1'})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={'width_ratios': [1, 1.4]});fig.patch.set_facecolor('#f8fafc')
    for a in axes:
        a.set_facecolor('#f8fafc');a.set_axisbelow(True)
    vals = [math.sqrt(s['errorMoments']['qErrorMSE']), math.sqrt(s['errorMoments']['differenceErrorMSE']), s['energyResponse']['referenceRms']]
    axes[0].bar(range(3), vals, color=['#94a3b8', '#6366f1', '#0d9488'], width=.58);axes[0].set_yscale('log');axes[0].set_ylim(.04, 20)
    axes[0].set_xticks(range(3), ['Absolute Q\nsketch error', 'Paired P→Q\nsketch error', 'Actual P→Q\nenergy change'])
    axes[0].set_ylabel('Key energy RMS, log scale');axes[0].set_title('Error cancels; the update is small', loc='left', pad=16)
    for j, v in enumerate(vals):
        axes[0].text(j, 1.12*v, f'{v:.3f}', ha='center')
    axes[0].grid(axis='y', alpha=.18)
    errors = [x['anchoredNormResponseValidOnly']['relativeRms'] for x in r['heads']]
    axes[1].bar(range(24), errors, color=['#0d9488' if x < 1 else '#df8b52' for x in errors]);axes[1].axhline(1, color='#64748b', linestyle='--', linewidth=1)
    axes[1].set_ylim(0, 2.05);axes[1].set_xticks([0, 5, 10, 15, 20, 23]);axes[1].set_xlabel('Head');axes[1].set_ylabel('Anchored norm error / actual norm-change RMS')
    axes[1].set_title('11 heads improve on hold; 13 worsen', loc='left', pad=16);axes[1].grid(axis='y', alpha=.18)
    axes[1].text(.02, .96, 'Pooled response error: 96.96%\nHold baseline: 100%', transform=axes[1].transAxes, va='top', fontsize=10)
    fig.suptitle('Shared sketch error can cancel without yielding an accurate state update', x=.065, y=.98, ha='left', fontweight='bold', fontsize=16)
    fig.text(.065, .035, 'Saved-only diagnosis, one supplied pair. Exact P energy is paid assistance; no new consumer, own-state rollout, image or baseline producer.', fontsize=10, color='#475569')
    fig.subplots_adjust(left=.065, right=.98, bottom=.22, top=.8, wspace=.36)
    png = out / 'paired-sketch.png';fig.savefig(png, dpi=160, facecolor=fig.get_facecolor());plt.close(fig)
    save(out / 'figure.json', {'imageSHA256': sha(png), 'plotSourceSHA256': sha(__file__), 'reportSHA256': sha(p / 'report.json'), 'scope': 'Published scalar moments and all24head norm-response errors; no activation/model execution.'})
    shutil.copyfile(png, root.parent / 'obsidian/images/zero-neural-cycle60-paired-sketch.png')
    return str(png)


if __name__ == '__main__':
    print(execute())
