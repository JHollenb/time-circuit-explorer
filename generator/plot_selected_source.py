"""Static research figure from published metrics; no model or tensor execution."""
from pathlib import Path
from encoder_atoms import read, save, sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parent.parent
    source = root / 'docs/results/selected-source'
    parent = root / 'docs/results/projected-complement'
    r, old = read(source / 'report.json'), read(parent / 'report.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#344256', 'text.color': '#172439'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), gridspec_kw={'width_ratios': [1.6, 1]})
    fig.subplots_adjust(left=.075, right=.97, top=.78, bottom=.31, wspace=.32)
    fig.suptitle('Produce four source rows, then consume their own state', x=.075, ha='left', y=.97, fontsize=18, fontweight='bold')
    fig.text(.075, .88, 'Full current K/V retained; only four text queries and their sequential MLPs run.', fontsize=11)
    names = ['A', 'U', 'X', 'M', 'T']
    for offset, role, color in [(-.18, 'P', '#4678c9'), (.18, 'Q', '#d48c36')]:
        errors = [100 * next(x['relativeRms'] for x in r['sourceMetrics'] if x['role'] == role and x['field'] == n) for n in names]
        axes[0].bar([i + offset for i in range(len(names))], errors, width=.33, label=role, color=color)
    axes[0].set(xticks=range(len(names)), xticklabels=['Attention', 'After attention', 'MLP input', 'MLP return', 'Source exit'],
                ylabel='Selected-row endpoint error to native (%)', ylim=(0, .8), title='Small finite differences propagate through the source')
    axes[0].tick_params(axis='x', labelsize=9)
    axes[0].legend(frameon=False)
    errors = [100 * next(x['relativeRms'] for x in z['responses'] if x['reference'] == 'family') for z in [old, r]]
    bars = axes[1].bar(['Supplied source rows', 'Own produced rows'], errors, color=['#7b91ac', '#4678c9'], width=.55)
    axes[1].bar_label(bars, fmt='%.3f%%', padding=5)
    axes[1].set(ylabel='Image-attention response error to native (%)', ylim=(0, 12), title='The following conditional read remains usable')
    axes[1].tick_params(axis='x', labelsize=9)
    fig.text(.075, .19, '9.63% native response error · 57.21% median query error · 71 / 1,024 queries lose to zero', fontsize=11)
    fig.text(.075, .12, '59.744B new matrix MACs · 73.87s instrumented assay · 604MB cache reused · no matched speedup claim', fontsize=10)
    fig.text(.075, .055, 'Original learned maps, native prefixes and receiving context remain paid. No complete state or arbitrary-prompt zero-neural image.', fontsize=9, color='#536273')
    out = root / 'docs/results/selected-source-figure'
    out.mkdir(exist_ok=False, parents=True)
    file = out / 'selected-source.png'
    fig.savefig(file, dpi=160)
    plt.close(fig)
    save(out / 'figure.json', {'reportSHA256': sha(source / 'report.json'), 'parentReportSHA256': sha(parent / 'report.json'),
         'plotSourceSHA256': sha(__file__), 'imageSHA256': sha(file), 'scope': 'Published scalar endpoints and response comparison, retaining paid context and non-image scope.'})
    return str(file)


if __name__ == '__main__':
    print(execute())
