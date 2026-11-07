"""Publish complete scalar trends and metadata, without activation loads."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read, save, sha
from projected_complement import fields
from publish_coherent_norm import audit


def execute():
    root = Path(__file__).resolve().parent.parent;p = root / 'docs/results/coherent-norm';e = root / 'docs/results/coherent-norm-evidence'
    mechanics = audit(p)
    rows = [json.loads(x) for x in gzip.decompress((p / 'observations.jsonl.gz').read_bytes()).splitlines()]
    heads = [x for x in rows if x['field'] == 'nextQ' and x['panel'] == 'query-head']
    norms = [x for x in rows if x['panel'] == 'image-key-head']
    if len(heads) != 96 or len(norms) != 24576:
        raise ValueError('Complete observations required')
    def distribution(name):
        values = [x[name]['relativeRms'] for x in heads if x[name]['relativeRms'] is not None]
        return {'count': len(values), 'min': min(values), 'median': statistics.median(values), 'max': max(values),
            'betterThanZero': sum(x < 1 for x in values), 'worseThanZero': sum(x > 1 for x in values),
            'negativeCosines': sum(x[name]['cosine'] is not None and x[name]['cosine'] < 0 for x in heads)}
    provenance, receipt = [read(p / n) for n in ['provenance.json', 'retained-array-verification.json']]
    arrays = {x['path']: x for n in ['compiler.json', 'actor.json', 'report.json'] for x in fields(read(p / n))
        if provenance['jobID'] + '/arrays/' in x['path']}
    result = {'schema': 'tce-coherent-norm-scalars-v1', 'reportSHA256': sha(p / 'report.json'),
        'queryHeadTrends': {n: distribution(n) for n in ['primaryResponse', 'retainedResponse', 'staleResponse', 'diagonalResponse']},
        'headsWorseThanRetained': sum(x['primaryResponse']['relativeRms'] > x['retainedResponse']['relativeRms'] for x in heads),
        'headsBetterThanDiagonal': sum(x['primaryResponse']['relativeRms'] < x['diagonalResponse']['relativeRms'] for x in heads),
        'headsBetterThanStale': sum(x['primaryResponse']['relativeRms'] < x['staleResponse']['relativeRms'] for x in heads),
        'nextQueryRows': [{k: x[k] for k in ['row', 'primaryResponse', 'retainedResponse', 'diagonalResponse']} for x in rows if x['field'] == 'nextQ' and x['panel'] == 'whole-row'],
        'normCells': len(norms), 'underestimatedNormCells': sum(x['normRatio'] < 1 for x in norms),
        'overestimatedNormCells': sum(x['normRatio'] > 1 for x in norms),
        'positiveEnergyErrorCells': sum(x['referenceMinusSketchEnergy'] > 0 for x in norms),
        'newArrays': len(arrays), 'newArrayBytes': sum(x['bytes'] for x in arrays.values()),
        'fullByteFiles': len(receipt['files']), 'fullByteCount': receipt['totalBytesVerified'],
        'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
        'localFreeBytesObserved': shutil.disk_usage(root).free,
        'scope': 'Published scalar/address rows and metadata only; one supplied pair, dependent heads. No effect-size gate or generalization/image/zero-neural certification.'}
    save(e / 'mechanics.json', mechanics);save(e / 'scalar-summary.json', result)
    job = read(root / 'artifacts/coherent-norm-v1/scheduler-job.json')
    save(e / 'scheduler.json', {k: job[k] for k in ['job_id', 'state', 'reservation', 'admission', 'payload_custody', 'result']})
    return result


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
