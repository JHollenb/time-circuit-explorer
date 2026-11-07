"""Complete scalar/head trends and metadata from the published value assay."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read, save, sha
from projected_complement import fields
from publish_deferred_value import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/deferred-value';e=root/'docs/results/deferred-value-evidence'
    e.mkdir(parents=True,exist_ok=True);mechanics=audit(p)
    r=read(p/'report.json');rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()]
    heads=[x for x in rows if x['field']=='nextQ' and x['panel']=='query-head']
    def distribution(name):
        values=[x[name]['relativeRms'] for x in heads]
        return {'count':len(values),'min':min(values),'median':statistics.median(values),'max':max(values),
            'betterThanZero':sum(x<1 for x in values),'negativeCosines':sum(x[name]['cosine'] is not None and x[name]['cosine']<0 for x in heads)}
    job=read(p/'provenance.json')['jobID']
    arrays={x['path']:x for n in ['actor.json','report.json'] for x in fields(read(p/n)) if job+'/arrays/' in x['path']}
    v=read(p/'retained-array-verification.json');execution=read(p/'actor-execution.json')
    result={'schema':'tce-deferred-value-scalars-v1','reportSHA256':sha(p/'report.json'),
        'queryHeadTrends':{n:distribution(n) for n in ['primaryResponse','retainedResponse','staleResponse']},
        'headsBetterThanRetained':sum(x['primaryResponse']['relativeRms']<x['retainedResponse']['relativeRms'] for x in heads),
        'headsWorseThanRetained':sum(x['primaryResponse']['relativeRms']>x['retainedResponse']['relativeRms'] for x in heads),
        'headsBetterThanStale':sum(x['primaryResponse']['relativeRms']<x['staleResponse']['relativeRms'] for x in heads),
        'stageResponseErrors':{n:{k:r['scores'][n][k]['relativeRms'] for k in ['primaryResponse','retainedResponse','staleResponse']}
            for n in ['H','A','M','T','nextQ']},'newArrays':len(arrays),'newArrayBytes':sum(x['bytes'] for x in arrays.values()),
        'fullByteFiles':len(v['files']),'fullByteCount':v['totalBytesVerified'],'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),
        'valueArithmeticSeconds':next(x['arithmeticSecondsBeforeCapture'] for x in execution['arithmeticTimings'] if x['stage']=='value'),
        'valueMatrixMACs':sum(x['nominalMACs'] for x in execution['operations'] if x['stage']=='value'),
        'fullNativeValueFormationMACs':14495514624,'previousBankAVMACs':18874368,
        'newActorMACs':execution['nominalMatrixMACs'],'oldActorPlusFullValueFormationMACs':1346371584+14495514624,
        'localFreeBytesObserved':shutil.disk_usage(root).free,
        'scope':'Published scalar fields only. Nominal production account includes formerly supplied V formation; operation counts are not measured speedup. No global improvement/image/generalization/zero-neural certification.'}
    save(e/'mechanics.json',mechanics);save(e/'scalar-summary.json',result)
    return {k:result[k] for k in ['headsBetterThanRetained','headsWorseThanRetained','headsBetterThanStale','newArrays','newArrayBytes','valueMatrixMACs']}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
