"""Publish a custody subset plus the two primary serving inputs."""
import argparse
import json
from pathlib import Path
import shutil

from response_study import read, save, sha


def publish(run, destination, failed=None):
    run, out = Path(run).resolve(), Path(destination).resolve()
    request, custody, report = [read(run / name) for name in ['request.json', 'collection.json', 'result/report.json']]
    if request['requestSHA256'] != report['requestSHA256']:
        raise ValueError('Request/report identity differs')
    for name, item in custody['inventory']['files'].items():
        if sha(run / 'result' / name) != item['sha256']:
            raise ValueError('Collected bytes changed: ' + name)
    out.mkdir(parents=True, exist_ok=False)
    names = ['report.json', 'compile-receipt.json', 'prediction-seal.json', 'control-seal.json',
             'coefficients.npy', 'consumer.npz']
    for name in names:
        shutil.copyfile(run / 'result' / name, out / name)
    shutil.copyfile(run / 'request.json', out / 'request.json')
    for name in ['compile.json', 'evaluation.json']:
        shutil.copyfile(run / 'payload' / name, out / name)
    save(out / 'provenance.json', {
        'schema': 'tce-compiled-mlp-publication-v1', 'jobID': custody['jobID'],
        'requestSHA256': request['requestSHA256'], 'reportSHA256': sha(out / 'report.json'),
        'payloadCustody': custody['custody'], 'inventory': custody['inventory'],
        'result': {k: v for k, v in custody['result'].items() if k != 'diagnostics'},
        'scope': 'Exact collected metadata and primary coefficient/consumer inputs. Complete prediction arrays and native references remain hash-bound in the retained worker and local collection; this is a published subset. No arbitrary-prompt or autonomous rollout claim.'})
    if failed is not None:
        failed = Path(failed).resolve()
        job = read(failed / 'failed-scheduler-job.json')
        save(out / 'failed-run.json', {'jobID': job['job_id'], 'state': job['state'],
             'requestSHA256': read(failed / 'request.json')['requestSHA256'],
             'payloadCustody': job['payload_custody'],
             'result': {k: v for k, v in job['result'].items() if k != 'diagnostics'},
             'scope': 'Compiler receipt was blocked by its own manifest file guard; no serving seal or scored result. Retained as instrument failure, not scientific null. Fixed own-spec permission and added complete synthetic compilation regression.'})
        shutil.copyfile(failed / 'failed-worker.log', out / 'failed-worker.log')
        shutil.copyfile(failed / 'request.json', out / 'failed-request.json')
    with (out / 'observations.jsonl').open('x') as stream:
        for i, row in enumerate(report['rows']):
            metrics = {}
            for field in ['raw', 'rounded', 'state']:
                metrics.update({field + '_' + k: v for k, v in row[field].items()})
                metrics.update({field + '_response_' + k: v for k, v in row['response'][field].items()})
            stream.write(json.dumps({'step': i, 'method': row['method'], 'alpha': row['alpha'],
                  'sourceImageRow': row['sourceImageRow'], 'metrics': metrics,
                  'reportSHA256': sha(out / 'report.json')}, allow_nan=False) + '\n')
    save(out / 'files.json', {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    return {'published': str(out), 'bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--failed', type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.run, args.destination, args.failed)))
