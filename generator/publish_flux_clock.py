"""Full-byte verification and compact publication for the finite clock cut."""
import argparse
import gzip
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read, save, sha
from publish_context_statistics import checked_collection
from publish_source_context import REMOTE
from flux_clock_study import OWNED


def checked(root):
    root = Path(root).resolve(); c, j, r = checked_collection(root, current_workers=False)
    if r['ownedSources'] != OWNED: raise ValueError('Owned source declaration differs')
    for name in OWNED:
        if sha(Path(__file__).parent / name) != r['sourceHashes'][name]: raise ValueError('Current executed worker changed: ' + name)
    return root, c, j, r


def verify(root):
    root, c, j, r = checked(root); p = root / 'result'; base = j['result']['diagnostics']['cwd']; inputs = read(p / 'inputs.json')
    table = read(p / 'clock-table.json'); images = read(p / 'images-candidate.json'); targets = read(p / 'targets.json'); items = []
    if images['tableSHA256'] != sha(p / 'clock-table.json'): raise ValueError('Serving table binding differs')
    seal = read(p / 'candidate-seal.json')
    if seal['imagesSHA256'] != sha(p / 'images-candidate.json') or seal['tableSHA256'] != sha(p / 'clock-table.json'): raise ValueError('Candidate seal binding differs')
    for row in table['rows']: items += [*row['clockOperands'].values(), *row['fields'].values()]
    for cohort in [images, targets]:
        for ports in cohort['fields'].values(): items += list(ports.values())
        items += list(cohort['images'].values())
    items += list(read(p / 'differences.json').values()) + list(inputs['E'].values()) + [inputs['noise'][k] for k in ['raw', 'packed', 'ids']]
    items += [{'path': base + '/' + n, 'sha256': sha(p / n)} for n in ['clock-table.json', 'images-candidate.json', 'candidate-seal.json']]
    items += [{'path': inputs['parentRoot'] + '/full-seal.json', 'sha256': inputs['parentFullSealSHA256']},
        {'path': inputs['parentRoot'] + '/images-full.json', 'sha256': targets['parentImagesSHA256']}]
    items += [{'path': base + '/response-result/' + str(f.relative_to(p)), 'sha256': sha(f)} for f in p.rglob('*') if f.is_file()]
    spec = {'allowedRoots': [base, inputs['parentRoot'], inputs['noiseParentRoot']], 'items': items}
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(REMOTE)],
        input=json.dumps(spec), text=True, capture_output=True, check=True, timeout=180)
    value = json.loads(result.stdout); value.update(schema='tce-compiled-clock-retained-bytes-v1', jobID=j['job_id'],
        verifierSourceSHA256=sha(__file__), candidateSealSHA256=sha(p / 'candidate-seal.json'),
        scope='Complete clock constants, candidate/reference image arrays, supplied E/noise, signed differences, compact results and parent seals. Only compact clock operands/PNGs hydrate locally, no full image/encoder arrays.')
    save(root / 'retained-array-verification.json', value)
    return {'files': len(value['files']), 'bytes': value['totalBytesVerified']}


def publish(root, destination):
    root, c, j, r = checked(root); p = root / 'result'; v = read(root / 'retained-array-verification.json')
    if v['candidateSealSHA256'] != sha(p / 'candidate-seal.json') or v['verifierSourceSHA256'] != sha(__file__): raise ValueError('Byte verification identity differs')
    out = Path(destination); out.mkdir(parents=True, exist_ok=False); streams = []
    for f in p.rglob('*'):
        if not f.is_file(): continue
        name = f.relative_to(p); target = out / name; target.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix == '.jsonl':
            data = f.read_bytes(); target = target.with_suffix(target.suffix + '.gz'); target.write_bytes(gzip.compress(data, mtime=0))
            if gzip.decompress(target.read_bytes()) != data: raise ValueError('Lossless stream differs')
            streams.append({'source': str(name), 'sourceSHA256': sha(f), 'sourceBytes': len(data), 'published': str(target.relative_to(out)), 'publishedSHA256': sha(target), 'publishedBytes': target.stat().st_size})
        else:
            if f.suffix not in ['.json', '.png', '.npy', '.md'] or (f.suffix == '.npy' and (name.parts[0] != 'clock-constants' or f.stat().st_size > 65536)):
                raise ValueError('Unexpected compact result member')
            shutil.copyfile(f, target)
    shutil.copyfile(root / 'retained-array-verification.json', out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-compiled-clock-public-v1', 'jobID': j['job_id'], 'requestSHA256': r['requestSHA256'],
        'payloadCustody': c['custody'], 'schedulerResult': c['result'], 'completeCollectedInventory': c['inventory'], 'losslessStreams': streams,
        'scope': 'Finite clock partial evaluation; offline learned compilation and original online live model/VAE computation paid. Not arbitrary-prompt zero-neural images.'})
    save(out / 'files.json', {str(f.relative_to(out)): sha(f) for f in out.rglob('*') if f.is_file()})
    members = [f for f in out.rglob('*') if f.is_file()]
    return {'files': len(members), 'bytes': sum(f.stat().st_size for f in members)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['verify', 'publish']); parser.add_argument('run'); parser.add_argument('destination', nargs='?'); a = parser.parse_args()
    print(json.dumps(verify(a.run) if a.action == 'verify' else publish(a.run, a.destination)))
