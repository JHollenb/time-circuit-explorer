"""Render published scalar comparisons only; never load scientific arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(report_path, output, notebook):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.patch.set_facecolor('#f8fafc')
    x = np.arange(3)
    cases = [('old82vs73', '82: complete output', '#64748b'),
             ('old83vs73', '83: input-coordinate basis', '#e11d48'),
             ('vs73', '85: current read/product basis', '#2563eb')]
    for ax, key, title in zip(axes,
                             ['errorOverTargetRMS', 'centeredErrorOverCenteredTargetRMS'],
                             ['Complete state increment', 'Changing increment within each stream']):
        maximum = 0.0
        for index, (case, label, color) in enumerate(cases):
            values = [100 * report['panels'][stream]['exitIncrement'][case][key]
                      for stream in ['all', 'text', 'image']]
            maximum = max(maximum, max(values))
            bars = ax.bar(x + (index - 1) * .25, values, .25, color=color, label=label)
            ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=5, fontsize=9)
        ax.set_xticks(x, ['All rows', 'Text', 'Image'])
        ax.set_ylim(0, max(1, maximum) * 1.3)
        ax.set_ylabel('Error / dense73 target RMS (%)')
        ax.set_title(title, loc='left', fontsize=12, pad=14)
        ax.set_facecolor('#f8fafc')
        ax.grid(axis='y', alpha=.13)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.05, .82),
               ncol=3, frameon=False, fontsize=10)
    fig.suptitle('Build the row basis after the nonlinear product', x=.055, y=.97,
                 ha='left', fontsize=19, fontweight='bold', color='#0f172a')
    fig.text(.055, .905,
             'One supplied state · clock 0 / single block 0 · fixed rank 128 per stream · all 1,536 dependent rows',
             fontsize=11, color='#475569')
    counts = report['panels']['all']['exitIncrement']['rowsNewVsOld']['total73vs83']
    primary = 100 * report['panels']['all']['exitIncrement']['vs82']['errorOverTargetRMS']
    fig.text(.055, .18,
             f'Against 83: {counts["improve"]} rows improve / {counts["worsen"]} worsen / {counts["equal"]} tie. '
             f'Difference from primary 82: {primary:.2f}% of its increment RMS.', fontsize=10)
    fig.text(.055, .125,
             'Output path: 15.099 B matrix MACs; diagnostics +0.050 B; QR ≈95.071 M FLOPs additional.', fontsize=10)
    fig.text(.055, .075,
             'Prospective GU + output composition: 34.427 B MACs. Original maps, source state and attention remain paid.',
             fontsize=10, color='#475569')
    fig.text(.055, .035,
             'Local write comparison only. No new image, trajectory, matched runtime saving or arbitrary-prompt zero-neural result.',
             fontsize=10, color='#475569')
    fig.subplots_adjust(left=.065, right=.985, top=.72, bottom=.31, wspace=.22)
    path = output / 'operand-row-output.png'
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    notebook = Path(notebook)
    shutil.copyfile(path, notebook)
    value = {'schema': 'tce-current-operand-row-output-figure-v1',
             'sourceReportSHA256': sha(report_path), 'plotSourceSHA256': sha(__file__),
             'imageSHA256': sha(path), 'notebookCopy': str(notebook.resolve()),
             'scope': 'Published scalar JSON visualization only; no scientific arrays or model evaluation.'}
    with (output / 'figure.json').open('x') as stream:
        json.dump(value, stream, indent=2)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report')
    parser.add_argument('output')
    parser.add_argument('notebook')
    args = parser.parse_args()
    print(json.dumps(execute(args.report, args.output, args.notebook), indent=2))
