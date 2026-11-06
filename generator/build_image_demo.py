"""Project authenticated published evidence into the lazy website demonstration."""
import argparse
from pathlib import Path
import shutil
from encoder_atoms import read, save, sha


def build(source, destination):
    source = Path(source); out = Path(destination)
    inventory = read(source / 'files.json')
    for name, digest in inventory.items():
        if sha(source / name) != digest: raise ValueError('Published evidence changed: ' + name)
    report = read(source / 'report.json'); provenance = read(source / 'provenance.json')
    out.mkdir(parents=True, exist_ok=False)
    for name in ['full.png', 'old.png', 'new.png', 'report.json', 'provenance.json']:
        shutil.copyfile(source / name, out / name)
    save(out / 'summary.json', {'schema': 'tce-conditioning-image-demo-v1', 'jobID': provenance['jobID'], 'report': report,
        'scheduler': {k: provenance['schedulerResult'][k] for k in ['elapsed_s', 'peak_rss_mb', 'peak_vram_mb']},
        'sourceInventorySHA256': sha(source / 'files.json'), 'sourceReportSHA256': sha(source / 'report.json'),
        'builderSourceSHA256': sha(__file__), 'scope': 'Recorded native images and scalar accounts; browser only compares original pixels.'})
    save(out / 'files.json', {p.name: sha(p) for p in out.iterdir()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('destination'); a = p.parse_args(); build(a.source, a.destination)
