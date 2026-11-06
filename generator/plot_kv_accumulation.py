"""Plot published scalar records only; no tensor or model evaluation."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from encoder_atoms import np, read, save, sha


def plot(source, destination):
    source = Path(source); out = Path(destination); out.mkdir(parents=True, exist_ok=False)
    report = read(source / 'report.json')
    rows = [json.loads(s) for s in gzip.decompress((source / 'observations.jsonl.gz').read_bytes()).splitlines()]
    channels = ['original-compact-minus-original-full', 'alternative-compact-minus-original-full']
    labels = ['Original compact policy', 'K/V-only alternative']; colors = ['#b45309', '#2563eb']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    fig.suptitle('Same omitted K/V rows, less state error under one fixed policy', fontsize=14, fontweight='bold')
    x = np.arange(3)
    for n, (channel, label, color) in enumerate(zip(channels, labels, colors)):
        values = [100 * next(r['relativeRms'] for r in report['accounts'] if r['channel'] == channel and r['port'] == 'Hnext' and r['kind'] == 'endpoint' and r['region'] == region) for region in ['all', 'active', 'padding']]
        bars = axes[0].bar(x + (n - .5) * .35, values, .35, label=label, color=color)
        for bar, value in zip(bars, values): axes[0].text(bar.get_x() + bar.get_width() / 2, value + .008, f'{value:.3f}%', ha='center', fontsize=8)
        selected = sorted([r for r in rows if r['channel'] == channel], key=lambda r: r['position'])
        axes[1].plot([r['position'] for r in selected], [100 * r['blockUpdate']['relativeRms'] for r in selected], linewidth=1, color=color, label=label)
    axes[0].set(xticks=x, xticklabels=['All 512 rows', '28 active', '484 padding'], ylabel='H2 error / original endpoint RMS (%)', ylim=(0, .51), title='Both compared with unchanged original H2')
    axes[0].legend(fontsize=8, loc='upper left')
    axes[1].axvspan(27.5, 511, color='#fef3c7', alpha=.35, zorder=-1, label='Padding queries retained')
    axes[1].set(xlim=(0, 511), xlabel='Query / output-state row', ylabel='Error / original block-update RMS (%)', title='Every output row retained; no refresh or fallback')
    axes[1].legend(fontsize=8, loc='upper right'); axes[1].grid(alpha=.15)
    fig.supxlabel('2.538 B nominal MACs omitted in either compact policy · Full alternative H2 is bit-exact · No measured speedup or image claim', fontsize=9)
    path = out / 'kv-accumulation-state.png'; fig.savefig(path, dpi=160); plt.close(fig)
    save(out / 'figure.json', {'schema': 'tce-kv-accumulation-figure-v1', 'sourceReportSHA256': sha(source / 'report.json'),
        'sourceRowsSHA256': sha(source / 'observations.jsonl.gz'), 'figureSHA256': sha(path),
        'scope': 'One already-scored short context and 512 dependent state rows; numerical policy comparison, no image-quality or latency claim.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('destination'); a = p.parse_args(); plot(a.source, a.destination)
