"""Byte-only custody and compact publication of the saved-moment diagnosis."""
import argparse
from pathlib import Path
import json
import shlex
import shutil
import subprocess
from encoder_atoms import read, save, sha
from publish_source_native import checked
from publish_sparse_text_kv import NPY_BYTES
from projected_complement import fields


def execute(root, destination):
    from mrun.client.api import Api
    root, collection, job, request = checked(root);p = root / 'result';s = read(p / 'inputs.json')
    roots = [job['result']['diagnostics']['cwd']]
    for jkey, ckey in [('parentJobID', 'parentCustody'), ('referenceJobID', 'referenceCustody')]:
        parent = Api().json('GET', '/api/jobs/' + s[jkey])
        if parent['state'] != 'succeeded' or parent['payload_custody'] != s[ckey]:
            raise ValueError('Saved-input custody changed')
        roots.append(parent['result']['diagnostics']['cwd'])
    items = list(fields(s)) + list(fields(read(p / 'report.json')))
    items += [{'path': roots[0] + '/response-result/' + n.name, 'sha256': sha(n)} for n in p.iterdir()]
    unique = {}
    for x in items:
        if x['path'] in unique and unique[x['path']]['sha256'] != x['sha256']:
            raise ValueError('Conflicting byte identities')
        unique[x['path']] = x
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(NPY_BYTES)],
        input=json.dumps({'allowedRoots': roots, 'items': list(unique.values())}), text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr)
    receipt = json.loads(result.stdout);receipt.update(schema='tce-diagonal-moments-bytes-v1', jobID=job['job_id'],
        verifierSourceSHA256=sha(__file__), templateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')))
    save(root / 'retained-array-verification.json', receipt)
    out = Path(destination);out.mkdir(parents=True, exist_ok=False)
    for n in p.iterdir():
        shutil.copyfile(n, out / n.name)
    save(out / 'retained-array-verification.json', receipt)
    save(out / 'scheduler.json', {k: job[k] for k in ['job_id', 'state', 'reservation', 'admission', 'payload_custody', 'result']})
    save(out / 'provenance.json', {'schema': 'tce-diagonal-moments-publication-v1', 'jobID': job['job_id'],
        'requestSHA256': request['requestSHA256'], 'payloadCustody': collection['custody'], 'completeCollectedInventory': collection['inventory'],
        'scope': 'Post-hoc saved-only diagnosis. No original map, new model/consumer, fitted replacement, primary feedback or image.'})
    save(out / 'files.json', {n.name: sha(n) for n in out.iterdir()})
    return {'files': len(receipt['files']), 'bytes': receipt['totalBytesVerified'], 'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses'])}


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('run');p.add_argument('destination');a = p.parse_args()
    print(json.dumps(execute(a.run, a.destination), indent=2))
