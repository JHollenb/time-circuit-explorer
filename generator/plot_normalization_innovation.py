"""Scalar-only two-receiver, all/nonlocal response-error comparison."""
from pathlib import Path
from encoder_atoms import read, save, sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/normalization-innovation'
    out = root / 'docs/results/normalization-innovation-figure'
    out.mkdir(exist_ok=False, parents=True)
    report = read(p / 'report.json')
    stages = ['fused', 'product', 'mlpWrite', 'gatedMLP', 'imageExit']
    labels = ['Gate/up', 'SiLU × up', 'Down', 'Clock gate', 'Local exit']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fafbfe', 'axes.facecolor': '#fafbfe'})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), constrained_layout=True)
    ceiling = 1.12 * max(v['relativeRms'] * 100 for row in report['accounts'] for v in row['response'].values())
    for row, receiver in enumerate(['R0', 'R1']):
        methods = ['shared-normalization-innovation', 'zero-response'] if receiver == 'R0' else ['shared-normalization-innovation', 'prior-conditioned-product', 'raw-source-transfer', 'zero-response']
        for col, scope in enumerate(['all', 'source-nonlocal']):
            ax = axes[row, col]
            width = .74 / len(methods)
            for index, method in enumerate(methods):
                color, label = {'shared-normalization-innovation': ('#6949b8', 'Shared normalization innovation'), 'prior-conditioned-product': ('#62a68c', 'Prior: supplied source increment'), 'raw-source-transfer': ('#d99952', 'Copy source response'), 'zero-response': ('#bec6d1', 'Zero response')}[method]
                values = [next(x for x in report['accounts'] if x['receiver'] == receiver and x['stage'] == stage and x['scope'] == scope)['response'][method]['relativeRms'] * 100 for stage in stages]
                offset = (index - (len(methods) - 1) / 2) * width
                ax.bar([i + offset for i in range(5)], values, width=width * .96, color=color, label=label)
            ax.set_xticks(range(5), labels)
            ax.set_ylim(0, ceiling)
            ax.set_title(receiver + (' · all 1,024 queries' if scope == 'all' else ' · 944 source-nonlocal queries'))
            ax.set_ylabel('Response RMS error / native response (%)')
            ax.grid(axis='y', alpha=.15)
            ax.legend(fontsize=7.5, loc='upper left')
    fig.suptitle('Producing both B operands from a shared normalization innovation', fontsize=16, weight='bold')
    fig.supxlabel('Both B fused endpoints omitted • native A baselines and original matrices paid • 4 affines /116.021B nominal MACs', fontsize=9)
    path = out / 'shared-normalization-innovation.png'
    fig.savefig(path, dpi=145)
    plt.close(fig)
    save(out / 'figure.json', {'schema': 'tce-normalization-innovation-figure-v1', 'figureSHA256': sha(path), 'plotterSourceSHA256': sha(__file__), 'reportSHA256': sha(p / 'report.json'), 'scope': 'Published aggregate scalars only; each error normalized by its own recipient native response. Shared linear-axis ceiling covers all aggregate values; per-query heterogeneity retained in lossless streams. No matched speedup or image claim.'})
    return str(path)


if __name__ == '__main__':
    print(execute())
