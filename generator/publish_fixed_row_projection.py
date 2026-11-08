"""Saved-input/output byte custody and lossless scalar publication checks."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read, save, sha, identity
from fixed_row_projection import SOURCES, abis, work, summarize
from fixed_row_projection_study import PARENTS, bindings
from publish_source_native import checked
from publish_sparse_text_kv import NPY_BYTES


def verify(root):
    from mrun.client.api import Api
    root, col, job, r = checked(root); p = root/'result'; s = read(p/'inputs.json'); api = Api()
    for parent in s['parents'].values():
        current = api.json('GET', '/api/jobs/'+parent['jobID'])
        if current['state'] != 'succeeded' or current['payload_custody'] != parent['custody'] or current['result']['diagnostics']['cwd'] != parent['root']:
            raise ValueError('Parent scheduler/custody differs')
    fields = read(p/'fields.json')['fields']; base = job['result']['diagnostics']['cwd']
    items = [*s['fields'].values(), *fields.values()]
    items += [{'path': parent['root']+'/response-result/'+n, 'sha256': h} for parent in s['parents'].values() for n, h in parent['records'].items()]
    items += [{'path': base+'/response-result/'+a.name, 'sha256': sha(a)} for a in p.iterdir()]
    unique = {}
    for item in items:
        if item['path'] in unique and unique[item['path']]['sha256'] != item['sha256']:
            raise ValueError('Conflicting byte identity')
        unique[item['path']] = {**unique.get(item['path'], {}), **item}
    spec = {'items': list(unique.values()), 'allowedRoots': [base, *s['sourceRoots']]}
    proc = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c '+shlex.quote(NPY_BYTES)], input=json.dumps(spec), text=True, capture_output=True, timeout=180)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    v = json.loads(proc.stdout); v.update(schema='tce-fixed-row-projection-bytes-v1', jobID=job['job_id'], reportSHA256=sha(p/'report.json'),
        verifierSourceSHA256=sha(__file__), templateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
        scope='Complete eleven saved inputs,23output arrays, parent/current metadata bytes. SSH hashes and NPY header/payload witnesses only; no independent real-array numerical replay, ideal-projector or accumulator certification.')
    save(root/'retained-array-verification.json', v)
    return {'files': len(v['files']), 'bytes': v['totalBytesVerified'], 'NPYPayloadWitnesses': len(v['npyPayloadWitnesses'])}


def publish(root, destination):
    root, col, job, r = checked(root); p = root/'result'; v = read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256'] != sha(__file__) or v['reportSHA256'] != sha(p/'report.json'):
        raise ValueError('Actual verifier/report identity differs')
    out = Path(destination); out.mkdir(parents=True, exist_ok=False); streams = []
    for a in p.iterdir():
        if a.suffix == '.jsonl':
            raw = a.read_bytes(); target = out/(a.name+'.gz'); target.write_bytes(gzip.compress(raw, mtime=0))
            if gzip.decompress(target.read_bytes()) != raw:
                raise ValueError('Lossless observations differ')
            streams.append({'source': a.name, 'sourceSHA256': sha(a), 'sourceBytes': len(raw), 'published': target.name, 'publishedSHA256': sha(target), 'publishedBytes': target.stat().st_size})
        else:
            shutil.copyfile(a, out/a.name)
    shutil.copyfile(root/'retained-array-verification.json', out/'retained-array-verification.json')
    save(out/'scheduler.json', {k: job[k] for k in ['job_id', 'state', 'reservation', 'admission', 'payload_custody', 'result']})
    save(out/'provenance.json', {'schema': 'tce-fixed-row-projection-publication-v1', 'jobID': job['job_id'], 'requestSHA256': r['requestSHA256'],
        'payloadCustody': col['custody'], 'completeCollectedInventory': col['inventory'], 'losslessStreams': streams,
        'scope': 'Saved-number projection/finite-boundary diagnosis, not a serving correction or new image. All original formation/source/learned work remains paid.'})
    save(out/'files.json', {a.name: sha(a) for a in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(a.stat().st_size for a in out.iterdir())}


def audit(p):
    from publish_row_output import audit as parent_audit
    p = Path(p); root = Path(__file__).resolve().parent.parent
    r, s, report, execution, v = [read(p/n) for n in ['request.json', 'inputs.json', 'report.json', 'execution.json', 'retained-array-verification.json']]
    if identity({k: a for k, a in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request differs')
    for n, digest in read(p/'files.json').items():
        if sha(p/n) != digest:
            raise ValueError('Publication changed: '+n)
    for n, digest in r['sourceHashes'].items():
        source = root/'generator'/n if n in r['ownedSources'] else p/n
        if sha(source) != digest:
            raise ValueError('Executed source or staged metadata changed: '+n)
    parent_audit(root/'docs/results/current-row-output')
    dirs = {tag: root/'artifacts'/folder/'result' for tag, folder in PARENTS.items()}
    if set(s['fields']) != set(SOURCES) or s['fields'] != bindings(dirs['83'], dirs['82']):
        raise ValueError('Exact saved source/common-state ancestry differs')
    for tag, parent in s['parents'].items():
        if any(sha(dirs[tag]/n) != h for n, h in parent['records'].items()):
            raise ValueError('Parent metadata changed')
    if v['verifierSourceSHA256'] != sha(__file__) or v['templateSourceSHA256'] != sha(root/'generator/publish_sparse_text_kv.py') or v['reportSHA256'] != sha(p/'report.json'):
        raise ValueError('Byte verifier/report identity differs')
    outputs = read(p/'fields.json')
    if outputs['requestSHA256'] != r['requestSHA256'] or {n: {'shape': a['shape'], 'dtype': a['logicalDtype']} for n, a in outputs['fields'].items()} != abis():
        raise ValueError('All23 output ABIs differ')
    for item in [*s['fields'].values(), *outputs['fields'].values()]:
        if v['files'][item['path']]['sha256'] != item['sha256'] or v['npyPayloadWitnesses'][item['path']]['payloadSHA256'] != item['payloadSHA256']:
            raise ValueError('Complete array byte witness absent')
    if report['work'] != work() or report['work'] != r['plannedWork'] or report['requestSHA256'] != r['requestSHA256']:
        raise ValueError('Actual/planned work/request differs')
    expected = [('project0', [128, 512], [512, 3072], [128, 3072]), ('reconstruct0', [512, 128], [128, 3072], [512, 3072]),
        ('project1', [128, 1024], [1024, 3072], [128, 3072]), ('reconstruct1', [1024, 128], [128, 3072], [1024, 3072])]
    if [(a['name'], a['leftShape'], a['rightShape'], a['outputShape']) for a in execution['products']] != expected:
        raise ValueError('Four physical contraction geometries differ')
    if sum(a['nominalMACs'] for a in execution['products']) != work()['matrixMACs'] or any(a['dtype'] != 'float64' or a['device'] != 'cpu' or a['operator'] != 'numpy.matmul' for a in execution['products']):
        raise ValueError('Actual contraction policy/work differs')
    for i in [0, 2]:
        if execution['products'][i]['outputPayloadSHA256'] != execution['products'][i+1]['rightPayloadSHA256']:
            raise ValueError('Compressed-row contraction join differs')
    if not execution['sourceClosure'] or execution['loadedNeuralRuntimeRoots']:
        raise ValueError('No-neural runtime/source closure differs')
    stream = read(p/'provenance.json')['losslessStreams'][0]; raw = gzip.decompress((p/stream['published']).read_bytes())
    if hashlib.sha256(raw).hexdigest() != stream['sourceSHA256']:
        raise ValueError('Lossless stream differs')
    rows = [json.loads(line) for line in raw.splitlines()]
    if [a['row'] for a in rows] != list(range(1536)) or any(a['stratum'] != ('text' if a['row'] < 512 else 'image') for a in rows):
        raise ValueError('Complete dependent row coverage differs')
    if report['panels'] != {name: summarize(rows[lo:hi]) for name, lo, hi in [('all', 0, 1536), ('text', 0, 512), ('image', 512, 1536)]}:
        raise ValueError('Published scalar Gram reductions differ')
    return {'schema': 'tce-fixed-row-projection-mechanics-v1', 'sourceInputs': 11, 'outputABIs': 23, 'projectorProducts': 4, 'matrixMACs': work()['matrixMACs'],
        'dependentRows': len(rows), 'fullByteFiles': len(v['files']), 'NPYPayloadWitnesses': len(v['npyPayloadWitnesses']), 'newLearnedMapApplications': 0,
        'scope': 'Source/common-state/array custody, numerical ABI, four-product work and complete signed scalar-row publication joins. Tiny independent references separate; no independent real-array numerical replay or image/serving/ideal-projector certification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['verify', 'publish', 'audit']); parser.add_argument('source'); parser.add_argument('destination', nargs='?'); args = parser.parse_args()
    print(json.dumps(publish(args.source, args.destination) if args.action == 'publish' else globals()[args.action](args.source), indent=2))
