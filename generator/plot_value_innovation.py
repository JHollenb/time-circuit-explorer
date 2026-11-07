"""Published scalar-only stage progression for one addressed value event."""
from pathlib import Path
from encoder_atoms import read, save, sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/value-innovation'
    out = root / 'docs/results/value-innovation-figure'
    out.mkdir(exist_ok=False, parents=True)
    report = read(p / 'report.json')
    stages = ['gatedCarrier', 'residualCommit', 'fused', 'product', 'mlpWrite', 'gatedMLP', 'imageExit']
    labels = ['Carrier', 'Residual', 'Gate/up', 'Product', 'Down', 'Clock gate', 'Local exit']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fafbfe', 'axes.facecolor': '#fafbfe'})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    ceiling = 1.13 * max([100] + [100 * a['response']['relativeRms'] for a in report['accounts'] if a['response']['relativeRms'] is not None])
    for ax, scope in zip(axes.flat, ['all', 'source-nonlocal', 'event-near', 'event-far']):
        selected = [next(a for a in report['accounts'] if a['scope'] == scope and a['stage'] == stage) for stage in stages]
        values = [100 * a['response']['relativeRms'] if a['response']['relativeRms'] is not None else 0 for a in selected]
        ax.bar(range(7), values, color='#7353ba', width=.64, label='Factored value-event prediction')
        ax.axhline(100, color='#aeb8c7', ls='--', lw=1.5, label='Zero-response control')
        ax.set_xticks(range(7), labels, rotation=18)
        ax.set_ylim(0, ceiling)
        ax.set_title(scope + ' · ' + str(selected[0]['queries']) + ' dependent queries')
        ax.set_ylabel('Response RMS error / native response (%)')
        ax.grid(axis='y', alpha=.15)
        ax.legend(fontsize=8, loc='upper left')
        for i, value in enumerate(values):
            ax.text(i, value + ceiling * .015, f'{value:.1f}', ha='center', fontsize=8)
    fig.suptitle('An addressed value swap produces its own attention and MLP change', fontsize=16, weight='bold')
    fig.supxlabel('Fixed complete Q/K • all 1,536 keys • native A and original maps paid • no new image or zero-neural claim', fontsize=9)
    path = out / 'addressed-value-innovation.png'
    fig.savefig(path, dpi=145)
    plt.close(fig)
    save(out / 'figure.json', {'schema': 'tce-value-innovation-figure-v1', 'figureSHA256': sha(path), 'reportSHA256': sha(p / 'report.json'), 'plotterSourceSHA256': sha(__file__),
                               'scope': 'Published scalar-only response errors. Common linear ceiling covers all aggregates; full dependent row outcomes stay in lossless streams. No new image generation or speed claim.'})
    return str(path)


if __name__ == '__main__':
    print(execute())
