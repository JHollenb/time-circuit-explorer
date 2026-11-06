"""Plot original PNGs and published state/response measurements; no model work."""
import argparse
from pathlib import Path
from encoder_atoms import read, save, sha


def execute(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    root = Path(source); out = Path(output); out.mkdir(parents=True, exist_ok=False)
    report = read(root / 'report.json'); provenance = read(root / 'provenance.json')
    fig, ax = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for row, case in enumerate(['A', 'B']):
        for col, prefix in enumerate(['full', 'candidate']):
            ax[row, col].imshow(Image.open(root / (prefix + '-' + case + '.png')))
            ax[row, col].set_title(case + ': cube ' + ('left' if case == 'A' else 'right') + ' of sphere\n' +
                ('Previously sealed native reference' if prefix == 'full' else 'Raw prompt → frozen reader → own learned suffix'), fontsize=11)
            ax[row, col].axis('off')
    for case, color in [('A', '#5375ac'), ('B', '#c66f48')]:
        for region, style in [('active', '-'), ('padding', '--')]:
            values = [next(x['relativeRms'] for x in report['accounts'] if x['case'] == case and x['port'] == 'H' + str(i) and x['region'] == region) for i in range(1, 28)]
            ax[0, 2].plot(range(1, 28), [100 * v for v in values], style, color=color, label=case + ' ' + region)
    ax[0, 2].set(title='The original suffix consumes the approximate state', xlabel='Emitted encoder state H', ylabel='Endpoint relative RMS error (%)')
    ax[0, 2].legend(fontsize=9); ax[0, 2].grid(alpha=.2)
    work = report['work']; cost = provenance['schedulerResult']; paired = {x['port']: x for x in report['pairedPromptResponse']}
    lines = ['Signed B−A response error / native response RMS', '']
    for port in ['H1', 'H2', 'H9', 'H18', 'H27', 'E', 'latent4', 'decodedRaw', 'pngRGB']:
        x = paired[port]; ratio = 'undefined' if x['relativeRms'] is None else f"{100*x['relativeRms']:.2f}%"
        lines.append(f'{port:12} {ratio:>10}')
    lines += ['', f"{cost['elapsed_s']:.2f} s entire assay; {cost['peak_vram_mb']:.0f} MB sampled VRAM",
        f"{work['nativeEncoderBlocks']} native encoder blocks + {work['denoiser']} denoisers + {work['VAE']} VAE decodes",
        'Full K/V, selection, moments and envelopes remain paid.',
        'Previously seen targets; one noise; no quality certificate.',
        'Most learned computation remains. No matched speedup.',
        'Arbitrary-prompt zero-neural generation remains open.']
    ax[1, 2].axis('off'); ax[1, 2].text(.02, .98, '\n'.join(lines), transform=ax[1, 2].transAxes,
        va='top', fontsize=10.2, linespacing=1.35, fontfamily='monospace')
    fig.suptitle('A frozen contextual reader reaches pixels through its own state', fontsize=17, weight='bold')
    path = out / 'raw-rank-one-image.png'; fig.savefig(path, dpi=140); plt.close(fig)
    save(out / 'figure.json', {'schema': 'tce-raw-rank-one-derived-figure-v1', 'figureSHA256': sha(path),
        'plotterSourceSHA256': sha(__file__), 'sourceReportSHA256': sha(root / 'report.json'),
        'sourcePNGs': {prefix + '-' + case + '.png': sha(root / (prefix + '-' + case + '.png')) for prefix in ['full', 'candidate'] for case in ['A', 'B']},
        'scope': 'Original image pixels plus published endpoint/signed-pair scalars. No neural evaluation or complete activation hydration.'})
    return str(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('source'); parser.add_argument('output'); a = parser.parse_args(); print(execute(a.source, a.output))
