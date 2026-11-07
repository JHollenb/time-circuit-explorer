"""Summarize published scalar observations and array metadata; no activation loads."""
import gzip
import json
import math
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read, save, sha
from projected_complement import fields
from publish_image_dual import audit


def execute():
    root = Path(__file__).resolve().parent.parent;p = root / 'docs/results/image-dual';e = root / 'docs/results/image-dual-evidence'
    mechanics = audit(p)
    if (e / 'mechanics.json').exists() and read(e / 'mechanics.json') != mechanics:
        raise ValueError('Earlier mechanics receipt differs')
    report, receipt, provenance = [read(p / n) for n in ['report.json', 'retained-array-verification.json', 'provenance.json']]
    rows = [json.loads(line) for line in gzip.decompress((p / 'observations.jsonl.gz').read_bytes()).splitlines()]
    heads = [x for x in rows if x['field'] == 'nextQ' and x['panel'] == 'query-head']
    if len(heads) != 96 or len({(x['row'], x['head']) for x in heads}) != 96:
        raise ValueError('Complete unique final query-head observations absent')
    def distribution(name):
        values = [x[name]['relativeRms'] for x in heads if x[name]['relativeRms'] is not None]
        return {'headCount': len(heads), 'min': min(values), 'median': statistics.median(values), 'max': max(values),
                'betterThanZeroResponse': sum(x < 1 for x in values), 'worseThanZeroResponse': sum(x > 1 for x in values),
                'negativeResponseCosines': sum(x[name]['cosine'] is not None and x[name]['cosine'] < 0 for x in heads),
                'undefinedRelativeErrors': len(heads) - len(values)}
    arrays = {x['path']: x for n in ['compiler.json', 'actor.json', 'reference.json', 'report.json']
              for x in fields(read(p / n)) if provenance['jobID'] + '/arrays/' in x['path']}
    new_bytes = sum(x['bytes'] for x in arrays.values())
    dual = read(p / 'compiler.json')['compile']['fields']['duals']
    value = {'schema': 'tce-image-dual-scalars-v1', 'reportSHA256': sha(p / 'report.json'),
        'queryHeadResponseTrends': {n: distribution(n) for n in ['primaryResponse', 'staleResponse']},
        'headsBetterThanStale': sum(x['primaryResponse']['relativeRms'] < x['staleResponse']['relativeRms'] for x in heads),
        'headsWorseThanStale': sum(x['primaryResponse']['relativeRms'] > x['staleResponse']['relativeRms'] for x in heads),
        'nextQueryRows': [{k: x[k] for k in ['row', 'primaryResponse', 'staleResponse']} for x in rows if x['field'] == 'nextQ' and x['panel'] == 'whole-row'],
        'newArrays': len(arrays), 'newArrayBytes': new_bytes,
        'dualNPYBytes': dual['bytes'], 'dualFP64PayloadBytes': math.prod(dual['shape']) * 8,
        'fullByteFiles': len(receipt['files']), 'fullByteCount': receipt['totalBytesVerified'],
        'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
        'NPYPayloadBytesReadAgain': sum(x['payloadBytes'] for x in receipt['npyPayloadWitnesses'].values()),
        'localFreeBytesObserved': shutil.disk_usage(root).free,
        'scope': 'Published scalar/address rows and complete array metadata only. Dependent heads of one paid native source pair; no tensor arithmetic/model execution or generalization certification.'}
    if not (e / 'mechanics.json').exists():
        save(e / 'mechanics.json', mechanics)
    save(e / 'scalar-summary.json', value)
    job = read(root / 'artifacts/image-dual-v1/scheduler-job.json')
    save(e / 'scheduler.json', {k: job[k] for k in ['job_id', 'state', 'reservation', 'admission', 'payload_custody', 'result']})
    return value


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
