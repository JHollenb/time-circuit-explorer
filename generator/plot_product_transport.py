"""Scalar-only response comparison and every actual local-exit query."""
import gzip
import json
from pathlib import Path
from encoder_atoms import read, save, sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/product-transport'
    out = root / 'docs/results/product-transport-figure'
    out.mkdir(exist_ok=False, parents=True)
    report = read(p / 'report.json')
    rows = [json.loads(line) for line in gzip.decompress((p / 'queries.jsonl.gz').read_bytes()).splitlines()]
    stages = ['product', 'mlpWrite', 'gatedMLP', 'imageExit']
    labels = ['SiLU × up', 'Down write', 'Clock gate', 'Local exit']
    colors = ['#6949b8', '#d99952', '#b9c3d2']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fafbfe', 'axes.facecolor': '#fafbfe'})
    fig = plt.figure(figsize=(13, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, .95])
    for col, scope in enumerate(['all', 'source-nonlocal']):
        ax = fig.add_subplot(grid[0, col])
        for offset, method, color, label in zip([-.24, 0, .24], ['recipient-baseline-product', 'raw-source-transfer', 'zero-response'], colors, ['Recipient baseline + increment', 'Copy source response', 'Zero response']):
            values = [next(a for a in report['accounts'] if a['stage'] == stage and a['scope'] == scope)['response'][method]['relativeRms'] * 100 for stage in stages]
            ax.bar([i + offset for i in range(4)], values, width=.23, color=color, label=label)
        ax.set_xticks(range(4), labels)
        ax.set_ylabel('Response RMS error / native response (%)')
        ax.set_title('All 1,024 queries' if col == 0 else '944 source-nonlocal queries')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=.15)
        if col == 0:
            ax.legend(fontsize=8, loc='upper right')
    ax = fig.add_subplot(grid[1, :])
    selected = [r for r in rows if r['stage'] == 'imageExit']
    for nonlocal_value, color, label in [(True, '#6949b8', 'Source-nonlocal'), (False, '#d99952', 'Edited region + halo')]:
        points = [r for r in selected if r['sourceNonlocal'] == nonlocal_value]
        ax.scatter([r['response']['raw-source-transfer']['rms'] for r in points], [r['response']['recipient-baseline-product']['rms'] for r in points], s=13, alpha=.65, c=color, label=label)
    highest = max(max(r['response'][method]['rms'] for method in ['raw-source-transfer', 'recipient-baseline-product']) for r in selected)
    ax.plot([0, highest], [0, highest], color='#7e8795', linestyle='--', linewidth=1, label='Equal error')
    ax.set(xlabel='Copy-source local-exit response error RMS', ylabel='Recipient-conditioned error RMS', title='Every actual query: points below the diagonal improve')
    ax.legend(fontsize=8)
    ax.grid(alpha=.15)
    fig.suptitle('A shared increment becomes useful in the recipient’s own nonlinear baseline', fontsize=16, weight='bold')
    fig.supxlabel('One supplied pair • one original down projection • 28.991B nominal MACs • no image or zero-neural claim', fontsize=9)
    path = out / 'recipient-product-transport.png'
    fig.savefig(path, dpi=145)
    plt.close(fig)
    save(out / 'figure.json', {'schema': 'tce-product-transport-figure-v1', 'figureSHA256': sha(path), 'plotterSourceSHA256': sha(__file__), 'reportSHA256': sha(p / 'report.json'), 'scope': 'Published scalar/query records only. Response error denominators are each native recipient response, not source response. Top axes0–110% cover all aggregate values; bottom scatter retains all1024queries on observed linear axes. No matched runtime or image claim.'})
    return str(path)


if __name__ == '__main__':
    print(execute())
