"""Verify complete near-compute fields, then publish compact evidence and PNGs."""
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
from complete_masked_study import OWNED


def checked(root):
    root = Path(root).resolve(); c, j, r = checked_collection(root, current_workers=False)
    if r['ownedSources'] != OWNED: raise ValueError('Owned source declaration changed')
    for name in OWNED:
        if sha(Path(__file__).parent / name) != r['sourceHashes'][name]: raise ValueError('Current worker changed: ' + name)
    return root, c, j, r


def verify(root):
    root, c, j, r = checked(root); p = root / 'result'; base = j['result']['diagnostics']['cwd']; source = read(p / 'source.json'); items = []
    for lane in ['candidate', 'full']:
        for name in ['encoder-' + lane + '.json', 'images-' + lane + '.json']:
            phase = read(p / name)
            for ports in phase['fields'].values(): items += list(ports.values())
            items.append({'path': base + '/' + name, 'sha256': sha(p / name)})
        seal = read(p / (lane + '-seal.json'))
        if seal['encoderSHA256'] != sha(p / ('encoder-' + lane + '.json')) or seal['imagesSHA256'] != sha(p / ('images-' + lane + '.json')):
            raise ValueError('Cohort phase binding differs')
        items.append({'path': base + '/' + lane + '-seal.json', 'sha256': sha(p / (lane + '-seal.json'))})
    if read(p / 'encoder-full.json')['referenceReleasedAfterSealSHA256'] != sha(p / 'candidate-seal.json'): raise ValueError('Full reference seal differs')
    items += list(read(p / 'differences.json').values())
    items += read(p / 'lexical-seal.json')['rows']
    items += [source['noise'][n] for n in ['raw', 'packed', 'ids']]
    items += [{'path': base + '/response-result/' + f.name, 'sha256': sha(f)} for f in p.iterdir()]
    proc = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots': [base, source['noiseParentRoot']], 'items': items}), text=True, capture_output=True, check=True, timeout=180)
    result = json.loads(proc.stdout); result.update(schema='tce-complete-masked-image-bytes-v1', jobID=j['job_id'],
        verifierSourceSHA256=sha(__file__), candidateSealSHA256=sha(p / 'candidate-seal.json'), fullSealSHA256=sha(p / 'full-seal.json'),
        scope='Full-byte verification of every retained encoder/image field, signed difference, lexical packet, inherited noise and compact result. No local activation/weight hydration.')
    save(root / 'retained-array-verification.json', result)
    return {'files': len(result['files']), 'bytes': result['totalBytesVerified']}


def publish(root, destination):
    root, c, j, r = checked(root); p = root / 'result'; v = read(root / 'retained-array-verification.json')
    if v['verifierSourceSHA256'] != sha(__file__) or v['candidateSealSHA256'] != sha(p / 'candidate-seal.json') or v['fullSealSHA256'] != sha(p / 'full-seal.json'):
        raise ValueError('Byte verification identity differs')
    out = Path(destination); out.mkdir(parents=True, exist_ok=False); streams = []
    for f in p.iterdir():
        if f.suffix == '.jsonl':
            data = f.read_bytes(); target = out / (f.name + '.gz'); target.write_bytes(gzip.compress(data, mtime=0))
            if gzip.decompress(target.read_bytes()) != data: raise ValueError('Lossless stream differs')
            streams.append({'source': f.name, 'sourceSHA256': sha(f), 'sourceBytes': len(data), 'published': target.name, 'publishedSHA256': sha(target), 'publishedBytes': target.stat().st_size})
        else:
            if f.suffix not in ['.json', '.png']: raise ValueError('Unexpected result member')
            shutil.copyfile(f, out / f.name)
    shutil.copyfile(root / 'retained-array-verification.json', out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-complete-masked-image-public-v1', 'jobID': j['job_id'], 'requestSHA256': r['requestSHA256'],
        'payloadCustody': c['custody'], 'schedulerResult': c['result'], 'completeCollectedInventory': c['inventory'], 'losslessStreams': streams,
        'scope': 'Fresh paired first-prompt reduction; full native images and paid original learned computation. No zero-neural or arbitrary-prompt quality claim.'})
    save(out / 'files.json', {f.name: sha(f) for f in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(f.stat().st_size for f in out.iterdir())}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['verify', 'publish']); p.add_argument('run'); p.add_argument('destination', nargs='?'); a = p.parse_args()
    print(json.dumps(verify(a.run) if a.action == 'verify' else publish(a.run, a.destination)))
