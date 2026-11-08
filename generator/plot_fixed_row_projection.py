"""Visualize published scalar component norms; never load scientific arrays."""
import argparse
import json
from pathlib import Path
import shutil
from encoder_atoms import read, save, sha


def execute(source, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    source = Path(source); destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    report = read(source/'report.json'); colors = ['#2563eb', '#d97706', '#0f766e']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.5), gridspec_kw={'width_ratios': [1, 1.08]})
    fig.patch.set_facecolor('#f8fafc')
    panels = [('update', ['projectionResidual', 'finitePolicy', 'admission'], ['Fixed-row projection residual  a', 'Combined finite policy  b', 'Final admission  c']),
        ('exit', ['gatedProjection', 'gatedFinitePolicy', 'gatedAdmission', 'gateRemainder', 'commitRemainder'], ['Gated projection  g·a', 'Gated finite policy  g·b', 'Gated admission  g·c', 'Gate remainder  r_g', 'Commit remainder  r_x'])]
    for ax, (kind, terms, labels) in zip(axes, panels):
        ax.set_facecolor('#f8fafc')
        for index, (stratum, color) in enumerate(zip(['all', 'text', 'image'], colors)):
            values = report['panels'][stratum][kind]['full']['componentNormOverDifference']
            for row, term in enumerate(terms):
                value = 100*values[term]; y = row+(index-1)*.22
                ax.scatter(value, y, color=color, s=42, zorder=3)
                ax.text(value*1.12, y, f'{value:.3f}%', color=color, va='center', fontsize=9)
        ax.set_xscale('log'); ax.set_xlim(.12, 245); ax.set_ylim(len(terms)-.5, -.6)
        ax.set_yticks(range(len(labels)), labels); ax.grid(axis='x', alpha=.17)
        ax.axvline(100, color='#94a3b8', linewidth=1, linestyle=':')
        ax.set_xlabel('Component norm / complete difference norm (%) · log scale', fontsize=10)
        ax.set_title('Complete pre-residual update' if kind == 'update' else 'Actual finite exit difference', loc='left', fontsize=13, pad=20)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle('Projection residual dominates this recorded difference', fontsize=21, fontweight='bold', x=.035, ha='left', y=.98, color='#0f172a')
    fig.legend([Line2D([], [], color=c, marker='o', linestyle='') for c in colors], ['All rows', 'Text rows', 'Image rows'], loc='upper left', bbox_to_anchor=(.03, .925), frameon=False, ncol=3)
    fig.subplots_adjust(left=.20, right=.96, top=.79, bottom=.23, wspace=.93)
    fig.text(.035, .115, 'Norms are not additive causal shares. Complete signed cross terms and all 1,536 dependent rows remain in the evidence.', fontsize=10)
    fig.text(.035, .075, 'Fixed recorded U · one supplied clock 0 / single block 0 state · centered accounts agree in scale · tiny account closures retained separately.', fontsize=10, color='#475569')
    fig.text(.035, .035, 'Saved-only CPU diagnostic: 1.208 B projector MACs + scalar reductions · 9.38 scheduler seconds · no new learned map, continuation or image.', fontsize=10, color='#475569')
    path = destination/'fixed-row-projection.png'; fig.savefig(path, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
    notebook = source.resolve().parents[3]/'obsidian/blog/images/fixed-row-projection-cycle84.png'; shutil.copyfile(path, notebook)
    receipt = {'schema': 'tce-fixed-row-projection-figure-v1', 'sourceReportSHA256': sha(source/'report.json'), 'plotSourceSHA256': sha(__file__), 'imageSHA256': sha(path), 'notebookCopy': str(notebook), 'scope': 'Plot of published scalar JSON only, no scientific arrays/model evaluation.'}
    save(destination/'figure.json', receipt); return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('source'); parser.add_argument('destination'); args = parser.parse_args()
    print(json.dumps(execute(args.source, args.destination), indent=2))
