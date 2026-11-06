"""Derived research figure from complete published per-row accounts."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from encoder_atoms import save, sha


def plot(source, destination):
    source = Path(source); out = Path(destination); out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(s) for s in gzip.decompress((source / 'observations.jsonl.gz').read_bytes()).splitlines()]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, arm, title in zip(axes, ('all-negative', 'prefix-negative'),
                              ('All rows negated: exact shortcut, new state geometry',
                               'Prefix negated: unchanged input suffix still moves')):
        for lane, field, label, color, style in (
            ('forecast', 'defectRms', 'Actual distance from signed state', '#1d4ed8', '-'),
            ('forecast', 'bestScaleOrthogonalRms', 'After best per-row scale', '#d97706', '--'),
            ('shortcut', 'defectRms', 'Shortcut distance from signed state', '#08916f', ':')):
            values = sorted([r for r in rows if r['arm'] == arm and r['lane'] == lane], key=lambda r: r['position'])
            ax.plot([r['position'] for r in values], [r[field] for r in values], color=color, linestyle=style,
                    linewidth=1.4, label=label)
        if arm == 'prefix-negative':
            ax.axvspan(256, 511, color='#fef3c7', alpha=.65, zorder=-1)
            ax.axvline(255.5, color='#9ca3af', linewidth=.8)
        ax.set(title=title, xlabel='Query / state row', ylabel='RMS per row', xlim=(0, 511))
        ax.grid(alpha=.16); ax.legend(loc='upper right', fontsize=8)
    fig.suptitle('One exact update does not preserve a reusable sign family', fontsize=16, fontweight='bold')
    path = out / 'composition-boundary.png'; fig.savefig(path, dpi=170); plt.close(fig)
    save(out / 'figure.json', {'schema': 'tce-sign-composition-derived-figure-v1',
         'sourceSHA256': sha(source / 'observations.jsonl.gz'), 'figureSHA256': sha(path),
         'scope': 'Retrospective512-row state geometry, two dependent edits; no next-layer or image prediction. Shortcut exact on all global rows and first256 prefix rows. Best scales fitted after outputs.'})
    return str(path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('destination'); a = p.parse_args()
    print(plot(a.source, a.destination))
