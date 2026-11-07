"""Plot published scalar rows and physical compute; never loads model arrays."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from encoder_atoms import read, sha, save


def plot(source, destination):
    source, out = Path(source), Path(destination)
    report = read(source / 'report.json')
    rows = [json.loads(line) for line in gzip.decompress((source / 'queries.jsonl.gz').read_bytes()).splitlines()]
    grid = np.full((32, 32), np.nan)
    for row in rows:
        value = row['response']['relativeRms']
        grid[row['imageID'][1], row['imageID'][2]] = value * 100 if value is not None else np.nan
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': '#18212e', 'axes.labelcolor': '#18212e', 'figure.facecolor': '#f8fafc', 'axes.facecolor': '#ffffff'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1.1, 1]})
    fig.subplots_adjust(left=.07, right=.95, bottom=.21, top=.73, wspace=.42)
    fig.suptitle('Four changed text rows update an image-attention read', fontsize=17, fontweight='bold', x=.07, ha='left', y=.96)
    fig.text(.07, .865, 'Response error: %.2f%%   ·   Exact source-text reconstruction: %s' %
             (100 * report['response']['relativeRms'], all(x['bitExact'] for x in report['textEndpoints'])), fontsize=12)
    fig.text(.07, .815, 'FLUX.2 Klein 4B · step 2 / joint 1 · one retrospective sphere/cube context', fontsize=10, color='#526171')
    image = axes[0].imshow(grid, cmap='magma', interpolation='nearest')
    axes[0].set(title='Response error at every image query', xlabel='Actual image-grid x', ylabel='Actual image-grid y')
    color = fig.colorbar(image, ax=axes[0], fraction=.045, pad=.04)
    color.set_label('% of native response RMS')
    axes[1].barh([2], [report['compilerMACs'] / 1e9], color='#687c97', height=.6, label='Paid cache compiler')
    for y in [1, 0]:
        left = 0
        for amount, color, label in [(75497472 / 1e9, '#25a8a1', 'Four-row K/V'), (25165824 / 1e9, '#e2ab48', 'Selected QK/AV'), (9663676416 / 1e9, '#4479dc', 'Full-grid Wo')]:
            axes[1].barh([y], [amount], left=left, color=color, height=.6, label=label if y == 1 else None)
            left += amount
    axes[1].set(yticks=[2, 1, 0], yticklabels=['Compiler', 'QQQ sham', 'PQP update'], xlabel='Billion nominal MACs', xlim=(0, 55), title='The baseline and output projection remain paid')
    for y, value in [(2, report['compilerMACs'] / 1e9), (1, report['actorsMACs'] / 2e9), (0, report['actorsMACs'] / 2e9)]:
        axes[1].text(value + .7, y, '%.3f' % value, va='center', fontsize=10)
    axes[1].legend(loc='lower right', fontsize=8, frameon=False)
    fig.text(.07, .10, 'FP64 fixed-query complement: 25.56 MB. Original learned maps and historical source formation remain required.', fontsize=10)
    fig.text(.07, .055, 'No new MLP, image suffix or image. This does not establish arbitrary-prompt zero-neural generation.', fontsize=10, color='#526171')
    out.mkdir(parents=True, exist_ok=False)
    target = out / 'sparse-text-kv.png'
    fig.savefig(target, dpi=160)
    plt.close(fig)
    save(out / 'figure.json', {'sourceReportSHA256': sha(source / 'report.json'), 'sourceRowsSHA256': sha(source / 'queries.jsonl.gz'),
        'plotSourceSHA256': sha(__file__), 'imageSHA256': sha(target), 'scope': 'Published complete query scalars and actual counted MACs only; no model or scientific array computation.'})
    return str(target)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source')
    p.add_argument('destination')
    a = p.parse_args()
    print(plot(a.source, a.destination))
