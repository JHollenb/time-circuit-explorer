"""Compact audited trends and footprint from existing published scalar evidence."""
import gzip
import json
from pathlib import Path
import shutil
from encoder_atoms import read, save, sha
from publish_sparse_text_kv import audit
from sparse_text_kv_worker import fields


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/sparse-text-kv'
    out = root / 'docs/results/sparse-text-kv-evidence'
    out.mkdir(parents=True, exist_ok=False)
    mechanics = audit(p)
    save(out / 'mechanics.json', mechanics)
    report, provenance, receipt = [read(p / n) for n in ['report.json', 'provenance.json', 'retained-array-verification.json']]
    base = provenance['schedulerResult']['diagnostics']['cwd'] + '/arrays/'
    arrays = {x['path']: x for name in ['compiler.json', 'primary.json'] for x in fields(read(p / name)) if x['path'].startswith(base)}
    rows = [json.loads(line) for line in gzip.decompress((p / 'queries.jsonl.gz').read_bytes()).splitlines()]
    save(out / 'scalar-summary.json', {'schema': 'tce-sparse-text-kv-scalars-v1', 'reportSHA256': sha(p / 'report.json'),
         'compilerSealSHA256': sha(p / 'compiler-seal.json'), 'primarySealSHA256': sha(p / 'primary-seal.json'),
         'newArrays': len(arrays), 'newArrayBytes': sum(x['bytes'] for x in arrays.values()),
         'fullByteFiles': len(receipt['files']), 'fullByteCount': receipt['totalBytesVerified'],
         'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
         'NPYPayloadBytesReadAgain': sum(x['payloadBytes'] for x in receipt['npyPayloadWitnesses'].values()),
         'queryPanels': report['queryPanels'], 'negativeResponseCosines': sum(x['response']['cosine'] is not None and x['response']['cosine'] < 0 for x in rows),
         'textEndpointBF16WordsExact': mechanics['textEndpointBF16WordsExact'], 'localFreeBytesObserved': shutil.disk_usage(root).free,
         'scope': 'Already scored dependent rows, complete byte custody and artifact sizes only. Extra payload rereads are not new storage; free space is an observation, not cleanup attribution.'})
    job = read(root / 'artifacts/sparse-text-kv-v1/scheduler-job.json')
    save(out / 'scheduler.json', {'jobID': job['job_id'], 'state': job['state'], 'reservation': job['reservation'], 'result': job['result'], 'executedPayload': job['payload_custody']['executed']})
    save(out / 'report-qualifications.json', {'schema': 'tce-sparse-text-kv-qualifications-v1', 'immutableReportSHA256': sha(p / 'report.json'),
         'textEquality': {'originalLabel': 'bitExact', 'originalComputation': 'numpy.array_equal on decoded float64 values',
                          'qualification': 'Numerical equality alone does not distinguish signed zero. Independent original NPY payload hashes now confirm complete BF16 word identity for both endpoints.',
                          'BF16WordsExact': mechanics['textEndpointBF16WordsExact']},
         'compilerRelease': {'qualification': 'Actor/scorer bind compiler seal identity and individual admitted array hashes. Independent publication audit additionally verifies the complete compiler.json payload-to-seal join.',
                             'compilerSHA256': sha(p / 'compiler.json')},
         'layout': 'Actor Wo receives a fresh contiguous BF16 head field rather than the native concatenated attention slice; FP64 scores/AV and compact four-row GEMMs also alter finite arithmetic.',
         'developmentDiagnostics': [{'stage': 'local arithmetic fixture', 'issue': 'Extreme-score fixture accidentally mixed float32 K with float64 Q; corrected fixture dtype before staging; no scientific run.'},
             {'stage': 'byte publisher first attempt', 'issue': 'Generic field walker treated external pretrained retrieval recipes as retained per-run parameter copies. Unpublished verifier corrected to enumerate activation/source fields explicitly; frozen scientific payload unchanged.'},
             {'stage': 'quarantine test fixture', 'issue': 'macOS /var versus /private/var spelling made the expected access-log path differ. Fixture root normalized; scientific guard and frozen worker unchanged.'}],
         'scope': 'Additive audit qualifications. Original scientific report, executed sources and all predictions remain unchanged.'})
    return {'arrays': len(arrays), 'bytes': sum(x['bytes'] for x in arrays.values()), 'negativeCosines': sum(x['response']['cosine'] is not None and x['response']['cosine'] < 0 for x in rows)}


if __name__ == '__main__':
    print(execute())
