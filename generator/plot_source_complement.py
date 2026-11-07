"""Static figure from published scalar results; no tensor or model execution."""
from pathlib import Path
from encoder_atoms import read, save, sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/source-complement'
    r = read(p / 'report.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'text.color': '#182b42'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    fig.subplots_adjust(left=.08, right=.97, top=.76, bottom=.31, wspace=.34)
    fig.suptitle('An unchanged context cannot test context sufficiency', x=.08, ha='left', y=.97, fontsize=18, fontweight='bold')
    fig.text(.08, .87, 'P and Q complementary K/V are byte-identical outside the four selected rows.', fontsize=11)
    axes[0].axis('off')
    for y, label, color in [(.81, 'Shared fresh selected P Q/K/V', '#386eb2'), (.54, 'Q complement = P complement', '#bb7529'), (.27, 'Identical hybrid banks → identical own outputs', '#386eb2')]:
        axes[0].text(.5, y, label, ha='center', va='center', transform=axes[0].transAxes,
                     bbox={'boxstyle': 'round,pad=.8', 'facecolor': color, 'edgecolor': 'none'}, color='white', fontsize=11)
    for y in [.675, .405]:
        axes[0].text(.5, y, '↓', ha='center', va='center', transform=axes[0].transAxes, fontsize=19)
    values = [100 * next(x['relativeRms'] for x in r['responses'] if x['arm'] == a and x['reference'] == 'native') for a in ['primary', 'control']]
    bars = axes[1].bar(['Q complement', 'P complement'], values, color=['#386eb2', '#bb7529'], width=.55)
    axes[1].bar_label(bars, fmt='%.4f%%', padding=5)
    axes[1].set(ylabel='Response error to native (%)', ylim=(0, 12), title='Matched arithmetic; no changed-context contrast')
    fig.text(.08, .20, '9.6285% native response error · 57.33% median query error · 73 / 1,024 rows lose to zero', fontsize=11)
    fig.text(.08, .13, '1.799B new matrix MACs · 111.68s instrumented assay · paid source banks and 604MB receiving cache', fontsize=10)
    fig.text(.08, .06, 'Mechanics complete. Context sufficiency untested. Original learned maps remain; no new image or zero-neural prompt backend.', fontsize=9)
    out = root / 'docs/results/source-complement-figure'
    out.mkdir(parents=True, exist_ok=False)
    file = out / 'source-complement.png'
    fig.savefig(file, dpi=160)
    plt.close(fig)
    save(out / 'figure.json', {'reportSHA256': sha(p / 'report.json'),
         'scalarSummarySHA256': sha(root / 'docs/results/source-complement-evidence/scalar-summary.json'),
         'plotSourceSHA256': sha(__file__), 'imageSHA256': sha(file), 'scope': 'Published scalars and independent complement-byte identity. Context substitution is degenerate.'})
    return str(file)


if __name__ == '__main__':
    print(execute())
