"""Publish verified reader banks, metadata and the emitted midpoint packet."""
import argparse
import json
from pathlib import Path
import shutil

from response_study import read, save, sha


def publish(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    request, custody, report = [read(run / name) for name in ['request.json', 'collection.json', 'result/report.json']]
    if request['requestSHA256'] != report['requestSHA256']:
        raise ValueError('Request/report identity differs')
    for name, item in custody['inventory']['files'].items():
        path = run / 'result' / name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Collected output changed: ' + name)
    destination.mkdir(parents=True, exist_ok=False)
    names = ['report.json', 'receiver-compile-receipt.json', 'receiver-seal.json', 'native-receipt.json',
             'observations.jsonl', 'packet-map.json', 'serve.json', 'native-spec.json',
             'basis.npy', 'modulation.npz', 'bank-q.npz', 'bank-k.npz', 'bank-v.npz']
    midpoint = next(x for x in read(run / 'result/packet-map.json') if x['alpha'] == .5)
    packet = next(x for x in read(run / 'result/serve.json')['packets'] if x['id'] == midpoint['id'])
    packet_name = Path(packet['path']).name
    names.append(packet_name)
    for name in names:
        shutil.copyfile(run / 'result' / name, destination / name)
    shutil.copyfile(run / 'request.json', destination / 'request.json')
    shutil.copyfile(run / 'payload/compile.json', destination / 'compile.json')
    save(destination / 'sample.json', {'schema': 'tce-state-reader-public-sample-v1',
         'id': packet['id'], 'path': packet_name, 'sha256': packet['sha256'], 'shape': packet['shape'],
         'label': 'Emitted BF16 midpoint state from supplied producer; alpha is explanatory metadata, not a serving input.'})
    save(destination / 'provenance.json', {'schema': 'tce-state-reader-publication-v1',
         'jobID': custody['jobID'], 'requestSHA256': request['requestSHA256'], 'reportSHA256': sha(destination / 'report.json'),
         'payloadCustody': custody['custody'], 'inventory': custody['inventory'],
         'result': {k: v for k, v in custody['result'].items() if k != 'diagnostics'},
         'scope': 'Exact compiled banks and metadata plus one original emitted BF16 packet. Complete primary/native/diagnostic arrays remain hash-bound in the retained full collection. Fixed32row supplied producer family, not arbitrary prompts or full image generation.'})
    save(destination / 'files.json', {p.name: sha(p) for p in sorted(destination.iterdir()) if p.is_file()})
    return {'published': str(destination), 'bytes': sum(p.stat().st_size for p in destination.iterdir())}


def publish_attribution(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    request, custody, report = [read(run / name) for name in ['request.json', 'collection.json', 'result/report.json']]
    if request['requestSHA256'] != report['requestSHA256']:
        raise ValueError('Attribution identity differs')
    for name, item in custody['inventory']['files'].items():
        path = run / 'result' / name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Collected attribution changed')
    destination.mkdir(parents=True, exist_ok=False)
    for name in ['report.json', 'observations.jsonl']:
        shutil.copyfile(run / 'result' / name, destination / name)
    shutil.copyfile(run / 'request.json', destination / 'request.json')
    save(destination / 'provenance.json', {'schema': 'tce-state-reader-attribution-publication-v1',
         'jobID': custody['jobID'], 'requestSHA256': request['requestSHA256'],
         'reportSHA256': sha(destination / 'report.json'), 'payloadCustody': custody['custody'],
         'inventory': custody['inventory'],
         'result': {k: v for k, v in custody['result'].items() if k != 'diagnostics'},
         'scope': 'Complete collected attribution results, original request and custody; parent arrays remain bound in the reader collection. No new model computation.'})
    save(destination / 'files.json', {p.name: sha(p) for p in sorted(destination.iterdir()) if p.is_file()})
    return {'published': str(destination), 'bytes': sum(p.stat().st_size for p in destination.iterdir())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--attribution', action='store_true')
    args = parser.parse_args()
    print(json.dumps((publish_attribution if args.attribution else publish)(args.run, args.destination)))
