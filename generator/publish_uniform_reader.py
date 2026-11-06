"""Verify uniform/zero/rank-one/full state/image evidence near compute; publish compact files."""
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


def checked(root):
    root = Path(root).resolve(); c, j, r = checked_collection(root, current_workers=False)
    for name in r['ownedSources']:
        if sha(Path(__file__).parent / name) != r['sourceHashes'][name]: raise ValueError('Executed worker changed: ' + name)
    return root, c, j, r


def verify(root):
    root, c, j, r = checked(root); p = root / 'result'; source = read(p / 'inputs.json'); reader = read(p / 'reader-seal.json')
    enc = read(p / 'encoder-candidate.json'); images = read(p / 'images-candidate.json'); targets = read(p / 'targets.json'); table = read(p / 'clock-table.json')
    base = j['result']['diagnostics']['cwd']; seal = read(p / 'candidate-seal.json')
    for key, name in [('readerSHA256', 'reader-seal.json'), ('encoderSHA256', 'encoder-candidate.json'), ('imagesSHA256', 'images-candidate.json'), ('clockTableSHA256', 'clock-table.json')]:
        if seal[key] != sha(p / name): raise ValueError('Candidate seal source differs')
    if enc['readerSealSHA256'] != sha(p / 'reader-seal.json') or images['encoderSHA256'] != sha(p / 'encoder-candidate.json') or images['tableSHA256'] != sha(p / 'clock-table.json'):
        raise ValueError('Actual candidate chain binding differs')
    if enc['lexicalSealSHA256'] != reader['lexicalSealSHA256']: raise ValueError('Reader/encoder lexical seal differs')
    items = list(reader['fields'].values()) + reader['lexicalRows'] + reader['sourceAtomFiles'] + reader['states'] + reader['packets']
    items += [source['noise'][k] for k in ['raw', 'packed', 'ids']]
    for group in [enc['fields'], images['fields'], targets['encoderFields'], targets['fields'], targets['rankEncoderFields'], targets['rankFields'], targets['zeroEncoderFields'], targets['zeroFields']]:
        for ports in group.values(): items += list(ports.values())
    items += [*images['images'].values(), *targets['images'].values(), *targets['rankImages'].values(), *targets['zeroImages'].values(), *read(p / 'differences.json').values()]
    for row in table['rows']: items += [*row['clockOperands'].values(), *row['fields'].values()]
    items += [{'path': base + '/' + n, 'sha256': sha(p / n)} for n in ['reader-seal.json', 'encoder-candidate.json', 'images-candidate.json', 'candidate-seal.json', 'clock-table.json']]
    items += [{'path': base + '/lexical/prediction-seal.json', 'sha256': reader['lexicalSealSHA256']},
        {'path': source['parentRoot'] + '/full-seal.json', 'sha256': source['parentFullSealSHA256']},
        {'path': source['parentRoot'] + '/encoder-full.json', 'sha256': targets['parentEncoderSHA256']},
        {'path': source['parentRoot'] + '/images-full.json', 'sha256': targets['parentImagesSHA256']},
        {'path': source['rankImageParentRoot'] + '/candidate-seal.json', 'sha256': source['rankImageParentSealSHA256']},
        {'path': source['rankImageParentRoot'] + '/encoder-candidate.json', 'sha256': targets['rankEncoderSHA256']},
        {'path': source['rankImageParentRoot'] + '/images-candidate.json', 'sha256': targets['rankImagesSHA256']},
        {'path': source['zeroParentRoot'] + '/candidate-seal.json', 'sha256': source['zeroParentSealSHA256']},
        {'path': source['zeroParentRoot'] + '/encoder-candidate.json', 'sha256': targets['zeroEncoderSHA256']},
        {'path': source['zeroParentRoot'] + '/images-candidate.json', 'sha256': targets['zeroImagesSHA256']}]
    items += [{'path': base + '/response-result/' + f.name, 'sha256': sha(f)} for f in p.iterdir()]
    spec = {'allowedRoots': [base, source['parentRoot'], source['clockParentRoot'], source['rankImageParentRoot'], source['zeroParentRoot'], source['noiseParentRoot'], source['lexicalStore']], 'items': items}
    proc = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(REMOTE)],
        input=json.dumps(spec), text=True, capture_output=True, check=True, timeout=180)
    value = json.loads(proc.stdout); value.update(schema='tce-uniform-reader-image-bytes-v1', jobID=j['job_id'], verifierSourceSHA256=sha(__file__),
        candidateSealSHA256=sha(p / 'candidate-seal.json'), scope='Complete used atom shards, raw lexical/uniform head/state/direct-mean packets, uniform/zero/rank/full encoder/image/target/difference arrays, inherited clocks/noise and seals. No local bulk activation hydration.')
    save(root / 'retained-array-verification.json', value)
    return {'files': len(value['files']), 'bytes': value['totalBytesVerified']}


def publish(root, destination):
    root, c, j, r = checked(root); p = root / 'result'; v = read(root / 'retained-array-verification.json')
    if v['candidateSealSHA256'] != sha(p / 'candidate-seal.json') or v['verifierSourceSHA256'] != sha(__file__): raise ValueError('Byte verification identity differs')
    out = Path(destination); out.mkdir(parents=True, exist_ok=False); streams = []
    for f in p.iterdir():
        if f.suffix == '.jsonl':
            data = f.read_bytes(); target = out / (f.name + '.gz'); target.write_bytes(gzip.compress(data, mtime=0))
            if gzip.decompress(target.read_bytes()) != data: raise ValueError('Lossless stream differs')
            streams.append({'source': f.name, 'sourceSHA256': sha(f), 'sourceBytes': len(data), 'published': target.name, 'publishedSHA256': sha(target), 'publishedBytes': target.stat().st_size})
        else:
            if f.suffix not in ['.json', '.md', '.png']: raise ValueError('Unexpected compact result member')
            shutil.copyfile(f, out / f.name)
    shutil.copyfile(root / 'retained-array-verification.json', out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-uniform-reader-image-public-v1', 'jobID': j['job_id'], 'requestSHA256': r['requestSHA256'],
        'payloadCustody': c['custody'], 'schedulerResult': c['result'], 'completeCollectedInventory': c['inventory'], 'losslessStreams': streams,
        'scope': 'Uniform causal value state from raw strings through own complete learned encoder/image consumer; prior zero/rank/full evidence reused. Remaining original learned work and historical compilation paid, no arbitrary-prompt zero-neural claim.'})
    save(out / 'files.json', {f.name: sha(f) for f in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(f.stat().st_size for f in out.iterdir())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['verify', 'publish']); parser.add_argument('run'); parser.add_argument('destination', nargs='?'); a = parser.parse_args()
    print(json.dumps(verify(a.run) if a.action == 'verify' else publish(a.run, a.destination)))
