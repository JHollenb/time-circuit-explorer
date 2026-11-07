"""Summaries of complete published response observations and custody metadata."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read, save, sha
from projected_complement import fields
from publish_stale_key_context import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/stale-key-context';e=root/'docs/results/stale-key-context-evidence';e.mkdir(parents=True,exist_ok=False)
    save(e/'mechanics.json',audit(p));r,receipt,provenance=[read(p/n) for n in ['report.json','retained-array-verification.json','provenance.json']]
    prefix=provenance['schedulerResult']['diagnostics']['cwd']+'/arrays/'
    arrays={x['path']:x for n in ['primary.json','diagnostic.json','report.json'] for x in fields(read(p/n)) if x['path'].startswith(prefix)}
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()]
    trends={}
    for role in ['primaryResponse','diagnosticResponse']:
        heads=[x[role] for x in rows if x['field']=='nextQ' and x['panel']=='query-head'];valid=[x['relativeRms'] for x in heads if x['relativeRms'] is not None]
        trends[role]={'headCount':len(heads),'undefinedRelativeErrors':len(heads)-len(valid),'median':statistics.median(valid) if valid else None,
            'min':min(valid) if valid else None,'max':max(valid) if valid else None,'betterThanZeroResponse':sum(x<1 for x in valid),
            'worseThanZeroResponse':sum(x>1 for x in valid),'negativeResponseCosines':sum(x['cosine'] is not None and x['cosine']<0 for x in heads)}
    summary={'schema':'tce-stale-key-context-scalars-v1','reportSHA256':sha(p/'report.json'),'newArrays':len(arrays),
        'newArrayBytes':sum(x['bytes'] for x in arrays.values()),'fullByteFiles':len(receipt['files']),'fullByteCount':receipt['totalBytesVerified'],
        'NPYPayloadWitnesses':len(receipt['npyPayloadWitnesses']),'NPYPayloadBytesReadAgain':sum(x['payloadBytes'] for x in receipt['npyPayloadWitnesses'].values()),
        'spliceBytesReadAgain':receipt['spliceBytesReadAgain'],'queryHeadResponseTrends':trends,'localFreeBytesObserved':shutil.disk_usage(root).free,
        'scope':'Published scalars/address rows and complete metadata. Response denominator is actualQQ-PP; endpoint denominators remain separate. No activation/model execution.'}
    save(e/'scalar-summary.json',summary);job=read(root/'artifacts/stale-key-context-v1/scheduler-job.json')
    save(e/'scheduler.json',{'jobID':job['job_id'],'state':job['state'],'reservation':job['reservation'],'result':job['result'],'executedPayload':job['payload_custody']['executed']})
    return summary


if __name__=='__main__':print(json.dumps(execute(),indent=2))
