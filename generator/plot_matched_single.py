"""Render published scalar anatomy; no tensor arrays or model execution."""
import json
from pathlib import Path
import shutil
from encoder_atoms import read, save, sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/matched-single-write'
    report = read(p / 'report.json')
    out = root / 'docs/results/matched-single-write-figure'
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.5), facecolor='#fbfcfe')
    names = ['all', 'text', 'image']
    labels = ['All rows', 'Text rows', 'Image rows']
    rows = {x['stratum']: x for x in report['energyRows'] if x['gated']}
    x = list(range(3))
    for ax in axes:
        ax.set_facecolor('#fbfcfe')
        ax.set_xticks(x, labels)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    a, b = axes
    for shift, key, label, color in [(-.18, 'omittedDiagonalEnergy', 'Channel diagonal', '#a2afcf'),
                                    (.18, 'omittedWriteEnergy', 'Signed omitted write', '#bd886c')]:
        values = [100 * rows[n][key] / rows[n]['fullWriteEnergy'] for n in names]
        a.bar([i + shift for i in x], values, width=.32, label=label, color=color)
        for i, value in zip(x, values):
            a.text(i + shift, value + .6, f'{value:.1f}', ha='center', fontsize=9)
    a.set_ylabel('Energy / full gated MLP write energy (%)')
    a.set_title('Omitted channels reinforce', loc='left', pad=16, fontweight='bold')
    a.set_ylim(0, 50)
    a.legend(frameon=False, fontsize=9, loc='upper left')
    for shift, key, label, color in [(-.18, 'actualDeltaRelativeDenseWrite', 'State difference / dense block write', '#6e80ba'),
                                    (.18, 'finiteRemainderRelativeRMS', 'Remainder / state difference', '#5c9c89')]:
        values = [100 * report['decomposition'][n][key] for n in names]
        b.bar([i + shift for i in x], values, width=.32, label=label, color=color)
        for i, value in zip(x, values):
            b.text(i + shift, value + .8, f'{value:.2f}', ha='center', fontsize=9)
    b.set_ylabel('RMS ratio (%) — denominators differ')
    b.set_title('Signed omission explains the local difference', loc='left', pad=16, fontweight='bold')
    b.set_ylim(0, 65)
    b.legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('One matched FLUX single block: quarter channels leave a coherent write',
                 x=.06, ha='left', fontsize=15, fontweight='bold', y=.97)
    fig.text(.06, .09, 'Same incoming state · unchanged attention and retained products · 2,304 / 9,216 channels retained', fontsize=9, color='#596378')
    fig.text(.06, .045, '389.771 billion diagnostic MACs · original learned maps paid · no image, speedup or arbitrary-prompt zero-neural claim', fontsize=8, color='#596378')
    fig.subplots_adjust(left=.075, right=.98, bottom=.23, top=.79, wspace=.32)
    image = out / 'matched-single-write.png'
    fig.savefig(image, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    notebook = root.parent / 'obsidian/images/zero-neural-cycle73-matched-single-write.png'
    shutil.copyfile(image, notebook)
    save(out / 'figure.json', {'schema': 'tce-matched-single-figure-v1',
         'reportSHA256': sha(p / 'report.json'), 'plotSourceSHA256': sha(__file__),
         'imageSHA256': sha(image), 'notebookCopy': str(notebook),
         'scope': 'Published scalar plot only. Gated energy and RMS panels have explicitly different denominators. No tensor or model computation.'})
    return {'image': str(image), 'sha256': sha(image)}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
