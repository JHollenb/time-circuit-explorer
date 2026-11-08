"""Publish compact authenticated cofactor metadata; no numerical array hydration."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from template_cofactor_program import read, save, sha


def publish(root, mechanics, destination):
    from mrun.client.api import Api
    root, destination = Path(root).resolve(), Path(destination).resolve()
    collection = read(root/'collection.json'); job = read(root/'scheduler-job.json'); request = read(root/'request.json')
    live = Api().json('GET', '/api/jobs/'+job['job_id'])
    if live['state'] != 'succeeded' or live.get('result', {}).get('returncode') != 0 or live['payload_custody'] != job['payload_custody']:
        raise ValueError('Current successful custody differs')
    custody = job['payload_custody']; phases = [custody[k] for k in ('declared', 'sealed', 'executed')]
    if collection['jobID'] != job['job_id'] or collection['custody'] != custody:
        raise ValueError('Collection/scheduler identity differs')
    if not custody['required'] or not job.get('admission') or any((x['sha256'], x['size_bytes']) != (phases[0]['sha256'], phases[0]['size_bytes']) for x in phases):
        raise ValueError('Guarded custody closure missing')
    if job['config']['requestSHA256'] != request['requestSHA256'] or phases[-1]['host'] != 'beast':
        raise ValueError('Request/host binding differs')
    for name, digest in request['sourceHashes'].items():
        if sha(root/'payload'/name) != digest:
            raise ValueError('Frozen payload differs')
    if {p.name for p in (root/'result').iterdir()} != set(collection['inventory']['files']):
        raise ValueError('Collected file coverage differs')
    for name, item in collection['inventory']['files'].items():
        p = root/'result'/name
        if sha(p) != item['sha256'] or p.stat().st_size != item['bytes']:
            raise ValueError('Collected output differs')
    report = read(root/'result/report.json'); child = read(root/'result/receipt.json'); program = read(root/'result/program.json'); seal = read(root/'result/serving-seal.json')
    if read(root/'result/request.json') != request or sha(root/'result/inputs.json') != request['sourceHashes']['inputs.json'] or report['program'] != program:
        raise ValueError('Collected request/input/program authority differs')
    if report['requestSHA256'] != request['requestSHA256'] or seal['requestSHA256'] != request['requestSHA256'] or sha(root/'result/receipt.json') != seal['receiptSHA256'] or sha(root/'result/program.json') != seal['programSHA256'] or sha(root/'result/serving-seal.json') != report['diagnosticsReleasedAfterSealSHA256']:
        raise ValueError('Serving/publication seal binding differs')
    tests = read(mechanics)
    if tests['failures'] or tests['errors'] or tests['testsRun'] != 5:
        raise ValueError('Targeted mechanics incomplete')
    for name, digest in tests['sourceHashes'].items():
        if sha(Path(__file__).parent/name) != digest or request['sourceHashes'].get(name) != digest:
            raise ValueError('Tested source differs')
    if [r['id'] for r in child['rows']] != [r['id'] for r in request['cases']] or len(report['comparisons']) != 28:
        raise ValueError('Complete case/layer coverage differs')
    for row in child['rows']:
        if row['status'] != 'partial' or row['compiledPositions'] != [0, 1] or row['unresolvedPositions'] != [2, 512] or row['completeH1Producer'] or row['imageGenerator'] or row['outputBytes'] != 286720 or row['outputSHA256'] != program['bankSHA256'] or seal['outputs'][row['id']] != row['outputSHA256']:
            raise ValueError('Partial constructor output contract differs')
    if child['loadedNumericalRuntimeRoots'] or not child['wrongPrefix'].get('stateReadRefused') or child['wrongPrefix']['status'] != 'unsupported-prefix':
        raise ValueError('Isolation/refusal mechanics differ')
    for expected_layer, comparison in enumerate(report['comparisons']):
        if comparison['layer'] != expected_layer:
            raise ValueError('Layer order differs')
        observed = child['rows'][0]['layerWordSHA256'][expected_layer] == comparison['nativeBPrefixSHA256']
        if observed != comparison['BPrefixMatchesCompiledWords']:
            raise ValueError('Historical comparison interpretation differs')
    cwd = job['result']['diagnostics']['cwd']
    if cwd != '/mnt/hdd8tb/mrun-work/'+job['job_id']:
        raise ValueError('Unexpected remote source root')
    expected = {cwd+'/compiled/prefix.bf16': program['bankSHA256']}
    expected.update({cwd+'/served/'+row['id']+'.bf16': row['outputSHA256'] for row in child['rows']})
    code = '''import hashlib,json,stat
from pathlib import Path
rows=[]
for name,expected in json.loads(%r).items():
 p=Path(name); s=p.lstat()
 if not stat.S_ISREG(s.st_mode):raise ValueError('Nonregular compiled output')
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h!=expected:raise ValueError('Fresh remote output differs')
 rows.append({'path':name,'bytes':s.st_size,'sha256':h})
print(json.dumps(rows))
''' % json.dumps(expected)
    remote = json.loads(subprocess.run(['ssh', '-T', 'beast', 'python3', '-'], input=code, text=True, capture_output=True, timeout=30, check=True).stdout)
    destination.mkdir(exist_ok=False)
    for p in (root/'result').iterdir():
        shutil.copyfile(p, destination/p.name)
    for source, name in [(root/'collection.json', 'collection.json'), (root/'scheduler-job.json', 'scheduler-job.json'), (Path(mechanics), 'mechanics.json')]:
        shutil.copyfile(source, destination/name)
    publication = {'schema': 'tce-template-cofactor-publication-v1', 'jobID': job['job_id'], 'requestSHA256': request['requestSHA256'],
        'publisherSHA256': sha(__file__), 'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in destination.iterdir()},
        'remoteByteVerification': remote, 'verifiedSourceByteReads': sum(x['bytesRead'] for x in report['sourceReads']+report['diagnosticReads']),
        'coveredRawCases': len(child['rows']), 'historicalComparedLayers': 28,
        'historicalMatchingPrefixLayers': sum(x['BPrefixMatchesCompiledWords'] for x in report['comparisons']),
        'scope': 'All compact results and exact new custody; nine fresh compiled/output byte hashes near compute. Historical comparison remains recorded A/B only. No universal finite parity, full H1 or image claim.'}
    save(destination/'publication.json', publication)
    return publication


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('mechanics'); p.add_argument('destination'); a = p.parse_args()
    print(json.dumps(publish(a.root, a.mechanics, a.destination)))
