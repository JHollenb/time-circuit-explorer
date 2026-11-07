"""Plot the captured context-effect trajectory using published scalars only."""
from pathlib import Path
from encoder_atoms import read, save, sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/propagated-context'
    r, provenance = read(p / 'report.json'), read(p / 'provenance.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'text.color': '#182b42'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.subplots_adjust(left=.08, right=.97, top=.77, bottom=.31, wspace=.3)
    fig.suptitle('Propagated context → own state → next query', x=.08, y=.97, ha='left', fontsize=19, fontweight='bold')
    fig.text(.08, .87, 'Same four selected P inputs; genuinely changed K/V outside those addresses.', fontsize=11)
    panels = [(['A', 'U', 'X', 'M', 'T'], ['Attention', 'Residual', 'MLP input', 'MLP return', 'Own exit']),
              (['nextLayerNorm', 'nextX', 'nextQProjected', 'nextQNormalized', 'nextQ'], ['Layer norm', 'Modulated', 'Projected', 'Head norm', 'RoPE query'])]
    for ax, (names, labels) in zip(axes, panels):
        values = [100*r['effects'][n]['relativeRms'] for n in names]
        bars = ax.bar(labels, values, color='#3976b7' if ax is axes[0] else '#be7d31', width=.65)
        ax.bar_label(bars, fmt='%.3f%%', padding=4, fontsize=9)
        ax.set_ylabel('Difference / this field’s P-control RMS (%)')
        ax.set_ylim(0, max(values)*1.27 if max(values) else 1)
        ax.tick_params(axis='x', labelsize=9)
    axes[0].set_title('Joint1 uses the changed read through its own MLP')
    axes[1].set_title('The emitted state forms its own joint2 query')
    fig.text(.08, .19, 'Each field has its own endpoint denominator; these bars are not additive causal percentages.', fontsize=11)
    seconds = provenance['schedulerResult']['elapsed_s']
    fig.text(.08, .12, f'58.926B new matrix MACs · {seconds:.2f}s instrumented assay · 17 affines + 2 rectangular attentions', fontsize=10)
    fig.text(.08, .055, 'Native source states and original maps remain paid. No full-width native parity reference, complete diffusion or zero-neural image.', fontsize=9)
    out = root / 'docs/results/propagated-context-figure';out.mkdir(parents=True, exist_ok=False)
    file = out / 'propagated-context.png';fig.savefig(file, dpi=160);plt.close(fig)
    save(out / 'figure.json', {'reportSHA256': sha(p / 'report.json'), 'plotSourceSHA256': sha(__file__), 'imageSHA256': sha(file), 'scope': 'Published control-endpoint-relative context effects; no model execution.'})
    return str(file)


if __name__ == '__main__':
    print(execute())
