"""Plot published original images, actual clock bytes and measured work scopes."""
import argparse
from pathlib import Path
from encoder_atoms import np, read, save, sha, floats


def execute(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    root = Path(source); out = Path(output); out.mkdir(parents=True, exist_ok=False)
    report = read(root / 'report.json'); table = read(root / 'portable-table.json'); provenance = read(root / 'provenance.json')
    fig, ax = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for row, case in enumerate(['A', 'B']):
        for col, prefix in enumerate(['full', 'candidate']):
            ax[row, col].imshow(Image.open(root / (prefix + '-' + case + '.png')))
            ax[row, col].set_title(case + ': cube ' + ('left' if case == 'A' else 'right') + ' of sphere\n' + ('Previously sealed native reference' if prefix == 'full' else 'Fresh serving, learned clock tensors absent'), fontsize=11)
            ax[row, col].axis('off')
    records = []
    for row in table['rows']:
        values = []
        for name in ['scheduler', 'pipeline_cast', 'transformer_argument', 'internal']:
            d = row['clockOperands'][name]; a = np.load(root / d['path'], allow_pickle=False)
            value = floats(a).item() if d['logicalDtype'] == 'torch.bfloat16' else a.item()
            values.append(f'{value:.6g}')
        records.append([str(row['clock'] + 1), *values])
    ax[0, 2].axis('off'); ax[0, 2].set_title('The actual finite clock path', fontsize=13)
    tab = ax[0, 2].table(cellText=records, colLabels=['Step', 't FP32', 'Cast BF16', '/1000', '×1000'], loc='center', cellLoc='center')
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.7)
    ax[0, 2].text(.02, .18, 'Clock operands are compiled offline.\nLive latent normalization stays online.\nExact bits, checkpoint, profile and mode are bound.', transform=ax[0, 2].transAxes, fontsize=11)
    work = report['work']; scheduler = provenance['schedulerResult']; endpoints = {x['case']: x for x in report['accounts'] if x['port'] == 'pngRGB'}
    paired = next(x for x in report['pairedPromptResponse'] if x['port'] == 'pngRGB')
    lines = [f"Offline clock weights: {work['removedOriginalParameterBytes']/1e6:,.3f} MB",
        f"Compiled output payload: {work['clockTablePayloadBytes']/1e3:,.3f} KB",
        f"Remaining transformer weights: {work['remainingTransformerParameterBytesLoaded']/1e9:,.3f} GB",
        '', f"Clock learned affines online: {work['onlineClockAffines']}", f"Clock gathers online: {work['onlineClockGathers']}",
        f"Still paid: {work['denoiser']} native denoisers + {work['VAE']} VAE decodes",
        '', f"PNG RMS error A / B: {endpoints['A']['rms']:.6g} / {endpoints['B']['rms']:.6g}",
        f"Paired PNG response error: {paired['rms']:.6g}", f"Reference response RMS: {paired['referenceRms']:.3f} (scale 0–255)",
        '', f"Entire assay: {scheduler['elapsed_s']:.2f} scheduler seconds", f"Peak sampled VRAM: {scheduler['peak_vram_mb']:.1f} MB",
        'Serial setup/capture costs; no matched speedup claim.', '', 'Finite clock partial evaluation. Most learned work remains.', 'Arbitrary-prompt zero-neural images remain open.']
    ax[1, 2].axis('off'); ax[1, 2].text(.02, .97, '\n'.join(lines), transform=ax[1, 2].transAxes, va='top', fontsize=10.5, linespacing=1.35)
    fig.suptitle('Replace repeated learned clock computation with exact deployed constants', fontsize=17, weight='bold')
    path = out / 'compiled-flux-clock.png'; fig.savefig(path, dpi=140); plt.close(fig)
    save(out / 'figure.json', {'schema': 'tce-compiled-clock-derived-figure-v1', 'figureSHA256': sha(path),
        'plotterSourceSHA256': sha(__file__), 'sourceReportSHA256': sha(root / 'report.json'), 'sourcePortableTableSHA256': sha(root / 'portable-table.json'),
        'sourcePNGs': {prefix + '-' + case + '.png': sha(root / (prefix + '-' + case + '.png')) for prefix in ['full', 'candidate'] for case in ['A', 'B']},
        'scope': 'Original image pixels plus measured work/clock scalars. No local model evaluation or complete activation hydration.'})
    return str(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('source'); parser.add_argument('output'); a = parser.parse_args(); print(execute(a.source, a.output))
