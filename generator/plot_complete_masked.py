"""Plot original paired images and published scalar accounts, without models."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from encoder_atoms import read, save, sha


def plot(source, destination):
    source = Path(source); out = Path(destination); out.mkdir(parents=True, exist_ok=False); r = read(source / 'report.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
    fig.suptitle('One reusable K/V dependency rule, two complete prompt-to-image paths', fontsize=15, fontweight='bold')
    originals = {}
    for row, case, relation in [(0, 'A', 'cube left of sphere'), (1, 'B', 'cube right of sphere')]:
        for col, lane in [(0, 'full'), (1, 'candidate')]:
            path = source / (lane + '-' + case + '.png'); axes[row, col].imshow(Image.open(path)); axes[row, col].axis('off')
            axes[row, col].set_title(f'{case}: requested {relation}\n' + ('Original full reference' if lane == 'full' else 'Complete masked-K/V candidate'), fontsize=11)
            originals[path.name] = sha(path)
    ax = axes[0, 2]
    for case, color in [('A', '#b45309'), ('B', '#2563eb')]:
        values = [next(a for a in r['accounts'] if a['case'] == case and a['port'] == 'H' + str(i) and a['region'] == 'all') for i in range(1, 28)]
        ax.plot(range(1, 28), [100*x['relativeRms'] for x in values], color=color, label=case, linewidth=1.8)
    for tap in (9, 18, 27): ax.axvline(tap, color='#94a3b8', alpha=.3)
    ax.set(title='Own-state encoder discrepancy', xlabel='Emitted hidden-state index', ylabel='Error / full state RMS (%)', xticks=[1, 9, 18, 27], xlim=(1, 27)); ax.legend(); ax.grid(alpha=.15)
    maximum = max(100*x['relativeRms'] for x in r['accounts'] if x['port'].startswith('H') and x['region'] == 'all')
    ax.set_ylim(0, max(1., maximum * 1.15))
    if maximum == 0: ax.text(.04, .9, 'A and B coincide at zero error\nAll retained H states match', transform=ax.transAxes, va='top')
    ax = axes[1, 2]; ax.axis('off')
    per_case = {c: next(a for a in r['accounts'] if a['case'] == c and a['port'] == 'pngRGB' and a['region'] == 'all') for c in ['A', 'B']}
    paired = next(a for a in r['pairedPromptResponse'] if a['port'] == 'pngRGB')
    relative = 'undefined (zero reference response)' if paired['relativeRms'] is None else f"{100*paired['relativeRms']:.2f}%"
    ax.text(.02, .97, f"PNG channel errors · scale0–255\n\nA: RMS {per_case['A']['rms']:.3f} / MAE {per_case['A']['mae']:.3f}\n"
        f"B: RMS {per_case['B']['rms']:.3f} / MAE {per_case['B']['mae']:.3f}\n\nPaired B−A response\nError RMS: {paired['rms']:.3f}\n"
        f"Reference response RMS: {paired['referenceRms']:.3f}\nRelative error: {relative}\n\n"
        "16 native denoiser calls +4 VAE decodes\nMost learned computation remains paid\n\n"
        "One fixed noise · two dependent prompts\nOriginal PNGs shown without edits\nVisual inspection is not quality certification",
        transform=ax.transAxes, va='top', linespacing=1.5, fontsize=10.5)
    fig.supxlabel('Candidates advance their own states and seal before full references · All512 query/residual/MLP rows remain · Strict zero-neural goal remains open', fontsize=10)
    path = out / 'complete-masked-image.png'; fig.savefig(path, dpi=160); plt.close(fig)
    save(out / 'figure.json', {'schema': 'tce-complete-masked-image-derived-figure-v1', 'sourceReportSHA256': sha(source / 'report.json'),
        'sourcePNGs': originals, 'figureSHA256': sha(path), 'plotterSourceSHA256': sha(__file__), 'scope': 'Original4PNGs and measured scalar plots; no local state/model hydration or semantic-quality certificate.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('destination'); a = p.parse_args(); plot(a.source, a.destination)
