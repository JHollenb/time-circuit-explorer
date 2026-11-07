"""Render published scalar support/count measurements; no scientific tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(report_path, output, notebook):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    panel = report['panels']['all']
    expr = panel['expressions']
    fold = report['denseStreamFold']
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.9), gridspec_kw={'width_ratios': [1, 1.9]})
    fig.patch.set_facecolor('#f8fafc')
    colors = ['#2563eb', '#64748b', '#e76f36']
    density = [100*expr[name]['coefficientDensity'] for name in ['raw', 'centered', 'combined']]
    bars = axes[0].bar(['Raw r', 'Centered\nr − mean(r)', 'Combined\nr + c'], density, color=colors, width=.65)
    axes[0].bar_label(bars, labels=[f'{x:.3f}%' for x in density], padding=5, fontsize=11)
    axes[0].set_ylim(0, 116)
    axes[0].set_ylabel('Live omitted coefficients (%)')
    axes[0].set_title('The actual combined coefficients are dense', loc='left', fontsize=11, pad=17)
    values = [expr['raw']['idealRowwiseUpDownMACs'], expr['raw']['fixedTileUpDownMACs'],
              expr['centered']['fixedTileUpDownMACs'], expr['combined']['fixedTileUpDownMACs'],
              fold['plusRawRowwiseBranchMACs'], fold['plusRawFixedTileBranchMACs']]
    labels = ['Raw\nrow ideal', 'Raw\n16-row tiles', 'Centered\n16-row tiles', 'Combined\n16-row tiles',
              'Fold + raw\nrow ideal', 'Fold + raw\n16-row tiles']
    bars = axes[1].bar(labels, [x/1e9 for x in values], color=['#2563eb', '#60a5fa', '#64748b', '#e76f36', '#b45354', '#b45354'], width=.68)
    axes[1].bar_label(bars, labels=[f'{x/1e9:.2f}' for x in values], padding=5, fontsize=10)
    axes[1].axhline(panel['denseOmittedUpDownMACs']/1e9, color='#334155', linestyle='--', linewidth=1)
    axes[1].text(2.9, 72, 'Dense omitted baseline: 65.23 B', ha='center', fontsize=9, color='#334155')
    axes[1].set_ylim(0, 220)
    axes[1].set_ylabel('Prospective matrix MACs (billions)')
    axes[1].set_title('Counts are not a sparse kernel or measured speedup', loc='left', fontsize=11, pad=17)
    for axis in axes:
        axis.set_facecolor('#f8fafc')
        axis.grid(axis='y', alpha=.13)
        axis.set_axisbelow(True)
        axis.tick_params(axis='x', length=0)
    fig.suptitle('Raw zeros disappear in the combined rectifier program', x=.06, y=.97, ha='left', fontsize=19, fontweight='bold', color='#0f172a')
    fig.text(.06, .90, 'One saved state · clock 3 / single block 0 · 1,536 rows · 6,912 omitted channels · fixed 16-row tiles', fontsize=11, color='#475569')
    fig.text(.06, .14, 'Raw: 63.97% ideal arithmetic saving becomes 21.11% with fixed tiles. Centered: 0.125% saving. Combined: zero coefficient-based saving.', fontsize=10)
    fig.text(.06, .095, 'Folded c branch: 130.46 B compilation + 14.50 B application before raw work; bias/means/gathers remain excluded, not free.', fontsize=10)
    fig.text(.06, .05, 'Census: 3.46 s CPU lease, zero new learned products, 14.16 M rectifier elements. Paid G/means/history; no U, maps, output or image evaluated.', fontsize=10, color='#475569')
    fig.subplots_adjust(left=.06, right=.985, top=.78, bottom=.27, wspace=.27)
    image = output / 'rectifier-sign-support.png'
    fig.savefig(image, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    notebook = Path(notebook)
    shutil.copyfile(image, notebook)
    receipt = {'schema': 'tce-rectifier-sign-support-figure-v1', 'sourceReportSHA256': sha(report_path),
               'plotSourceSHA256': sha(__file__), 'imageSHA256': sha(image), 'notebookCopy': str(notebook.resolve()),
               'scope': 'Published scalar density/prospective arithmetic visualization only; no new scientific-array or model calculation.'}
    with (output / 'figure.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report')
    parser.add_argument('output')
    parser.add_argument('notebook')
    args = parser.parse_args()
    print(json.dumps(execute(args.report, args.output, args.notebook), indent=2))
