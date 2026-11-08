"""Byte/custody/public-scalar QA. Never hydrate the complete score array."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with gzip.open(path, 'rt') as f:return [json.loads(line) for line in f]
def save(path, value):
    with Path(path).open('x') as f:json.dump(value, f, indent=2, allow_nan=False)


def publish(root, mechanics, destination):
    root = Path(root).resolve();local = root/'result';destination = Path(destination).resolve()
    collection = read(root/'collection.json');job = read(root/'scheduler-job.json');request = read(root/'request.json')
    if (read(local/'request.json') != request or job.get('config', {}).get('requestSHA256') != request['requestSHA256'] or
        job.get('payload_custody') != collection['custody']):
        raise ValueError('Collected/submitted/scheduler request or custody join mismatch')
    phases = [collection['custody'][k] for k in ['declared', 'sealed', 'executed']]
    if job['state'] != 'succeeded' or collection['result']['returncode'] != 0 or not collection['custody']['required']:
        raise ValueError('Successful guarded execution missing')
    if any(any(p[k] != phases[0][k] for k in ['sha256', 'size_bytes']) for p in phases) or phases[-1]['host'] != 'beast':
        raise ValueError('Payload custody mismatch')
    if hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != 'requestSHA256'}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() != request['requestSHA256']:
        raise ValueError('Request digest mismatch')
    for name, want in request['sourceHashes'].items():
        if sha(root/'payload'/name) != want:raise ValueError('Staged source mismatch')
        if name in request['ownedSources'] and sha(Path(__file__).resolve().parent/name) != want:
            raise ValueError('Current executed source changed')
        if name not in request['ownedSources']+['ranks.u16.gz'] and (not (local/name).is_file() or sha(local/name) != want):
            raise ValueError('Collected source metadata/input/plan mismatch: '+name)
    measured = read(mechanics)
    for name in ['compiled_mass_program.py', 'compiled_mass_compile.py', 'compiled_mass_study.py', 'test_compiled_mass.py']:
        if measured['sourceHashes'][name] != request['sourceHashes'][name]:raise ValueError('Tested/executed source mismatch')
    actual = {str(p.relative_to(local)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in local.rglob('*') if p.is_file()}
    if actual != collection['inventory']['files'] or any(name.endswith('.f64') for name in actual):
        raise ValueError('Compact collection inventory differs or contains full array')
    manifest = read(local/'result-files.json')
    if any(actual.get(name) != item for name, item in manifest.items()):raise ValueError('Worker result inventory differs')
    report = read(local/'report.json');compiler = read(local/'compile-compiler.json');old_ledger = read(local/'clock-ledger.json')
    if report['requestSHA256'] != request['requestSHA256'] or report['program']['scoreBytes'] != 31395840:
        raise ValueError('Reported request/domain mismatch')
    if compiler['work'] != {'coordinateProducts': 502333440, 'fsumCalls': 3924480}:
        raise ValueError('Compiler work mismatch')
    for clock, (a, b) in enumerate(zip(compiler['clocks'], old_ledger['clocks'])):
        if a['clock'] != b['clock'] or a['clock'] != clock or a['scoreSHA256'] != b['scoreSHA256'] or a['bytes'] != 8*30*clock:
            raise ValueError('Complete historical score-byte qualification mismatch')
    if len(compiler['clocks']) != len(old_ledger['clocks']) or len(compiler['clocks']) != 512:
        raise ValueError('Clock coverage mismatch')
    retained = read(local/'retained-score-file.json')
    expected_path = '/mnt/hdd8tb/mrun-work/'+job['job_id']+'/compiled/scores.f64'
    if retained['host'] != 'beast' or retained['path'] != expected_path or retained['sha256'] != report['program']['scoreSHA256']:
        raise ValueError('Retained score identity mismatch')
    # Remote bytes only, no NumPy/float decoding and no bulk copy to this host.
    code = """import json,hashlib,stat
from pathlib import Path
p=Path(%r);s=p.lstat();ledger=json.loads(%r)
if not stat.S_ISREG(s.st_mode):raise ValueError('Nonregular score asset')
h=hashlib.sha256();total=0
with p.open('rb') as f:
 for row in ledger:
  b=f.read(row['bytes']);total+=len(b);h.update(b)
  if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['scoreSHA256']:raise ValueError('Clock bytes differ')
 if f.read(1):raise ValueError('Trailing score bytes')
print(json.dumps({'path':str(p),'bytes':s.st_size,'sha256':h.hexdigest(),'completeClockSegmentsVerified':len(ledger),'decodedFloatingArrays':False}))
""" % (expected_path, json.dumps(compiler['clocks']))
    remote = json.loads(subprocess.run(['ssh', '-T', 'beast', 'python3', '-'], input=code, text=True, capture_output=True, check=True, timeout=60).stdout)
    if remote['sha256'] != retained['sha256'] or remote['bytes'] != retained['bytes']:
        raise ValueError('Fresh remote full score bytes mismatch')
    serve = read(local/'prediction-receipt.json');seal = read(local/'prediction-seal.json');spec = read(local/'serve-spec.json')
    for name, item in seal['files'].items():
        if actual['prediction-'+name] != item:raise ValueError('Sealed prediction identity mismatch')
    if serve['loadedNumericalRuntimeRoots'] or serve['onlineQKCoordinateProducts'] != 0 or not serve['scalarAttentionComputed']:
        raise ValueError('Serving arithmetic scope mismatch')
    if serve['fileCanaryDenials'] != spec['forbiddenPaths'] or serve['blockedImportCanaries'] != ['numpy', 'torch']:
        raise ValueError('Fresh-process exclusion controls incomplete')
    observations = rows(local/'observations.jsonl.gz')
    domain = {(name, clock) for name in request['cases'] for clock in range(512)}
    if len(observations) != 2560 or {(r['case'], r['clock']) for r in observations} != domain:
        raise ValueError('Observation case/clock domain incomplete or duplicated')
    old = {r['queryPosition']: r for r in rows(local/'old-weights.jsonl.gz')}
    panels = []
    local_defects, local_word_differences = [], 0
    for name in request['cases']:
        predicted = rows(local/('prediction-'+name+'.jsonl.gz'))
        records = [x for x in observations if x['case'] == name]
        packet = read(local/(name+'-tokens.json'))
        if len(predicted) != len(records) or len(records) != 512:raise ValueError('Complete packet rows missing')
        panel = {'case': name, 'referenceRows': 0, 'supportDifferences': 0, 'scoreWordDifferences': 0,
                 'weightWordDifferences': 0, 'maximumAbsWeightDifference': 0., 'historicalRows': 0,
                 'historicalSupportDifferences': 0, 'historicalMaximumAbsWeightDifference': 0.}
        statuses = Counter()
        for clock, (prediction, record) in enumerate(zip(predicted, records)):
            if prediction != record['candidate'] or prediction['clock'] != record['clock'] or clock != record['clock']:
                raise ValueError('Observation/prediction join mismatch')
            statuses[prediction['status']] += 1
            reference, historic = record['sameProfileScalarReference'], record['historical97']
            if packet['mask'][clock]:
                if prediction['status'] != 'active-query-outside-padding-contract' or reference is not None:
                    raise ValueError('Active-query scope mismatch')
            elif name == 'unknown':
                if (prediction['status'] != 'unknown-admitted-token' or prediction['unknownPositions'] != [8] or
                    prediction['weights'] is not None or reference is not None):
                    raise ValueError('Unknown history was skipped or not refused')
            else:
                if prediction['status'] != 'supported' or reference is None:raise ValueError('Known padding reference missing')
                selected, weights = prediction['selectedScores'], prediction['weights']
                if len(selected) != len(weights) or len(weights) != 4 or prediction['paddedWeights'] != weights:
                    raise ValueError('Selected-only normalization extent mismatch')
                if not all(math.isfinite(x) for x in selected+weights):raise ValueError('Nonfinite scalar result')
                maximum = max(selected);exponentials = [math.exp(x-maximum) for x in selected];denominator = math.fsum(exponentials)
                local_weights = [x/denominator for x in exponentials]
                local_defects.extend(abs(a-b) for a, b in zip(weights, local_weights))
                local_word_differences += sum(a.hex() != b.hex() for a, b in zip(weights, local_weights))
                same = prediction['addresses'] == reference['addresses'];panel['referenceRows'] += 1;panel['supportDifferences'] += int(not same)
                if same:
                    panel['scoreWordDifferences'] += sum(a.hex() != b.hex() for a, b in zip(selected, reference['selectedScores']))
                    panel['weightWordDifferences'] += sum(a.hex() != b.hex() for a, b in zip(weights, reference['weights']))
                    differences = [a-b for a, b in zip(weights, reference['weights'])]
                    if differences != reference['signedCandidateMinusReferenceWeights']:raise ValueError('Signed reference differences mismatch')
                    panel['maximumAbsWeightDifference'] = max([panel['maximumAbsWeightDifference']]+[abs(x) for x in differences])
                if name in ['A', 'B']:
                    if historic['addresses'] != old[clock]['support'+name] or historic['weights'] != old[clock]['weights'+name]:
                        raise ValueError('Historical97 scalar identity mismatch')
                    a, b = dict(zip(prediction['addresses'], weights)), dict(zip(historic['addresses'], historic['weights']))
                    union = sorted(set(a)|set(b));differences = [a.get(j, 0.)-b.get(j, 0.) for j in union]
                    if union != historic['unionAddresses'] or differences != historic['signedCandidateMinusHistoricalWeights']:
                        raise ValueError('Historical signed scalar differences mismatch')
                    panel['historicalRows'] += 1;panel['historicalSupportDifferences'] += int(prediction['addresses'] != historic['addresses'])
                    panel['historicalMaximumAbsWeightDifference'] = max([panel['historicalMaximumAbsWeightDifference']]+[abs(x) for x in differences])
        panel['statuses'] = dict(statuses);panels.append(panel)
    if panels != report['panels'] or panels != read(local/'panels.json'):
        raise ValueError('Reconstructed scalar panels mismatch')
    rebuilt_pairs = []
    for left, right in [('A', 'B'), ('C', 'D')]:
        pair_rows = []
        left_rows = rows(local/('prediction-'+left+'.jsonl.gz'))
        right_rows = rows(local/('prediction-'+right+'.jsonl.gz'))
        for a, b in zip(left_rows, right_rows):
            if a['status'] != 'supported' or b['status'] != 'supported':continue
            av, bv = dict(zip(a['addresses'], a['weights'])), dict(zip(b['addresses'], b['weights']))
            addresses = sorted(set(av)|set(bv));delta = [bv.get(j, 0.)-av.get(j, 0.) for j in addresses]
            pair_rows.append({'clock': a['clock'], 'orderedSupportChanged': a['addresses'] != b['addresses'],
                              'unionAddresses': addresses, 'signedWeightChange': delta,
                              'weightL1Change': math.fsum(abs(x) for x in delta)})
        if len(pair_rows) != 477:raise ValueError('Paired observation domain incomplete')
        rebuilt_pairs.append({'pair': [left, right], 'rows': pair_rows,
                              'changedOrderedSupports': sum(x['orderedSupportChanged'] for x in pair_rows),
                              'changedWeightsWithSameOrderedSupport': sum(not x['orderedSupportChanged'] and x['weightL1Change'] > 0 for x in pair_rows),
                              'zeroWeightChanges': sum(x['weightL1Change'] == 0 for x in pair_rows)})
    if rebuilt_pairs != read(local/'pairs.json'):raise ValueError('Reconstructed complete paired scalar observations mismatch')
    expected_reference = {'coordinateProducts': 8547840, 'scoreFsumCalls': 66780, 'normalizerExpCalls': 7632, 'normalizerFsumCalls': 1908}
    if report['diagnosticWork'] != expected_reference:raise ValueError('Separate direct reference work mismatch')
    mass = {k: sum(c['massWork'][k] for c in serve['cases']) for k in serve['cases'][0]['massWork']}
    if any(mass[k] != 7632 for k in ['scoreLookups', 'finiteScoreChecks', 'subtractions', 'expCalls', 'divisions']) or mass['fsumCalls'] != 1908:
        raise ValueError('Complete serving operation counts mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    for source in local.iterdir():shutil.copyfile(source, destination/source.name)
    shutil.copyfile(mechanics, destination/'mechanics.json')
    shutil.copyfile(root/'collection.json', destination/'collection.json');shutil.copyfile(root/'scheduler-job.json', destination/'scheduler-job.json')
    audit = {'schema': 'tce-compiled-mass-scalar-byte-audit-v1', 'panels': panels, 'massWork': mass,
             'pairedRowsIndependentlyRebuilt': 954, 'completeObservationRows': 2560,
             'sameProfileReferenceWork': expected_reference, 'freshRemoteScoreByteVerification': remote,
             'localMathLibmMaxAbsDifference': max(local_defects, default=0.), 'localMathDifferentWords': local_word_differences,
             'localMathScope': 'Public selected scalars only, host libm may differ; this observation is not a numerical quality gate or native-reader proof.',
             'scope': 'Independent custody/hash/public-scalar joins and aggregate reconstruction. Full source QK references are worker-source-audited, not locally recomputed. No complete score/operand array hydration.'}
    save(destination/'scalar-audit.json', audit)
    save(destination/'publication.json', {'schema': 'tce-compiled-mass-publication-v1', 'jobID': job['job_id'],
         'requestSHA256': request['requestSHA256'], 'publisherSHA256': sha(__file__), 'payloadCustody': collection['custody'],
         'rawFiles': len(actual), 'rawBytes': sum(x['bytes'] for x in actual.values()),
         'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in destination.iterdir()},
         'scope': 'Byte-exact compact evidence plus independent public-scalar audit. Large compiled scores remain near compute. Original learned preparation and scalar attention paid; arbitrary-prompt zero-neural generation unestablished.'})
    return {'destination': str(destination), 'rawFiles': len(actual), 'rawBytes': sum(x['bytes'] for x in actual.values()),
            'remoteScoreBytesFreshVerified': remote['bytes'], 'localScalarMaxAbsDifference': audit['localMathLibmMaxAbsDifference']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root');parser.add_argument('mechanics');parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(publish(args.root, args.mechanics, args.destination)))
