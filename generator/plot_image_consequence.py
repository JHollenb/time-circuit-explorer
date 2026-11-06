"""Render original image evidence and equal-scale errors; no model execution."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from encoder_atoms import np, read, save, sha


def plot(source, destination):
    source = Path(source); out = Path(destination); out.mkdir(parents=True, exist_ok=False)
    report = read(source / 'report.json'); images = {}
    for branch in ('full', 'old', 'new'):
        images[branch] = np.asarray(Image.open(source / (branch + '.png')).convert('RGB'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(2, 3, figsize=(12, 9), constrained_layout=True)
    fig.suptitle('The local gain reaches the image, with a different error tradeoff', fontsize=16, fontweight='bold')
    for ax, branch, title in zip(axes[0], ('full', 'old', 'new'), ('Full formation reference', 'Original compact policy', 'New compact policy')):
        ax.imshow(images[branch]); ax.set_title(title, fontweight='bold'); ax.axis('off')
    axes[1, 0].axis('off'); png = report['png']
    axes[1, 0].text(.02, .97,
        f"One prompt · one seed · identical noise\n\nPNG channel scale: 0–255\n\n"
        f"RMS       {png['old']['rms255']:.3f} → {png['new']['rms255']:.3f}\n"
        f"MAE         {png['old']['mae255']:.3f} → {png['new']['mae255']:.3f}\n"
        f"Max |Δ|         {png['old']['maxAbs255']} → {png['new']['maxAbs255']}\n\n"
        f"Pixel squared error versus full:\n{png['pixelSquaredErrorImproved']:,} improve\n"
        f"{png['pixelSquaredErrorWorsened']:,} worsen\n{png['pixelSquaredErrorUnchanged']:,} unchanged\n\n"
        "12 native denoiser calls + 3 VAE decodes\nPaid original learned computation",
        transform=axes[1, 0].transAxes, va='top', fontsize=11, linespacing=1.45)
    for ax, branch in zip(axes[1, 1:], ('old', 'new')):
        delta = images[branch].astype(np.float64) - images['full'].astype(np.float64)
        artist = ax.imshow(np.max(np.abs(delta), axis=2), cmap='magma', vmin=0, vmax=255)
        ax.set_title(('Original' if branch == 'old' else 'New') + ' compact − full\nLargest absolute channel difference per pixel')
        ax.axis('off')
    fig.colorbar(artist, ax=list(axes[1, 1:]), orientation='horizontal', fraction=.05, pad=.03, label='Identical fixed scale 0–255; no error normalization')
    fig.supxlabel('Original native PNGs above · Float decoder/RGB and complete signed differences retained separately · No image-quality certificate', fontsize=10)
    target = out / 'conditioning-image.png'; fig.savefig(target, dpi=160); plt.close(fig)
    save(out / 'figure.json', {'schema': 'tce-conditioning-image-derived-figure-v1', 'sourceReportSHA256': sha(source / 'report.json'),
        'sourcePNGs': {b: sha(source / (b + '.png')) for b in images}, 'figureSHA256': sha(target),
        'scope': 'Original PNGs shown without edits; same0–255 channel-error scale. One dependent prompt/seed, all native image computation paid.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('destination'); a = p.parse_args(); plot(a.source, a.destination)
