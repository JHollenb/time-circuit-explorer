"""Full-byte native image-consumer custody, compact metadata and original PNGs."""
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


def verify(root):
    root = Path(root).resolve(); c, j, r = checked_collection(root)
    p = root / 'result'; source = read(p / 'source.json'); base = j['result']['diagnostics']['cwd']
    fields = read(p / 'retained-fields.json')['fields']
    items = [item for values in fields.values() for item in values.values()] + list(source['E'].values()) + list(source['priorC'].values())
    items += [{'path': base + '/response-result/' + f.name, 'sha256': sha(f)} for f in p.iterdir()]
    proc = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots': [base, source['parentRoot']], 'items': items}), text=True, capture_output=True, check=True, timeout=120)
    v = json.loads(proc.stdout); v.update(schema='tce-image-consequence-retained-bytes-v1', jobID=j['job_id'],
        consumerSealSHA256=sha(p / 'consumer-seal.json'), noiseSealSHA256=sha(p / 'noise-seal.json'), verifierSourceSHA256=sha(__file__),
        scope='Complete native latent/decoder/floatRGB/PNG outputs, signed differences, initial noise/IDs, native context outputs, inherited E/C and reports full-byte verified near compute. Local publication transports PNGs and metadata only.')
    save(root / 'retained-array-verification.json', v)
    return {'files': len(v['files']), 'bytes': v['totalBytesVerified']}


def publish(root, destination):
    root = Path(root).resolve(); c, j, r = checked_collection(root)
    p = root / 'result'; v = read(root / 'retained-array-verification.json')
    if v['consumerSealSHA256'] != sha(p / 'consumer-seal.json') or v['noiseSealSHA256'] != sha(p / 'noise-seal.json') or v['verifierSourceSHA256'] != sha(__file__):
        raise ValueError('Byte verification changed')
    out = Path(destination).resolve(); out.mkdir(parents=True, exist_ok=False); streams = []
    for f in p.iterdir():
        if f.suffix == '.jsonl':
            data = f.read_bytes(); target = out / (f.name + '.gz'); target.write_bytes(gzip.compress(data, mtime=0))
            if gzip.decompress(target.read_bytes()) != data: raise ValueError('Lossless stream differs')
            streams.append({'source': f.name, 'sourceSHA256': sha(f), 'sourceBytes': len(data), 'published': target.name,
                'publishedSHA256': sha(target), 'publishedBytes': target.stat().st_size})
        else:
            if f.suffix not in ('.json', '.png'): raise ValueError('Unexpected result member')
            shutil.copyfile(f, out / f.name)
    shutil.copyfile(root / 'retained-array-verification.json', out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-image-consequence-public-v1', 'jobID': j['job_id'], 'requestSHA256': r['requestSHA256'],
        'payloadCustody': c['custody'], 'schedulerResult': c['result'], 'completeCollectedInventory': c['inventory'], 'losslessStreams': streams,
        'scope': 'Three original PNGs plus compact measured native-consumer evidence. Full/oldcompact/newcompact E, one noise, four-step original native FLUX/VAE. Not zero-neural generation, independent prompt quality or a speedup claim.'})
    save(out / 'files.json', {f.name: sha(f) for f in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(f.stat().st_size for f in out.iterdir())}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['verify', 'publish']); p.add_argument('run'); p.add_argument('destination', nargs='?'); a = p.parse_args()
    print(json.dumps(verify(a.run) if a.action == 'verify' else publish(a.run, a.destination)))
