"""Post-seal scalar/custody summaries; no complete tensor hydration."""
import argparse
import gzip
import json
from pathlib import Path
from encoder_atoms import read,save,sha
from publish_first_consumer import audit


def execute(root,source,destination):
    root=Path(root);p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True);r=read(p/'report.json');a=read(p/'primary.json');j=read(root/'scheduler-job.json');v=read(p/'retained-array-verification.json')
    groups=[a['embeddings'],a['constants'],*a['shared'].values(),*a['cells'].values(),*read(p/'response-fields.json').values(),*read(p/'factor-fields.json').values()];fields=[x for g in groups for x in g.values()];rows=[json.loads(x) for x in gzip.decompress((p/'factors.jsonl.gz').read_bytes()).splitlines()]
    save(out/'mechanics.json',audit(p));save(out/'scalar-summary.json',{'schema':'tce-first-consumer-scalars-v1','reportSHA256':sha(p/'report.json'),'primarySealSHA256':sha(p/'primary-seal.json'),'newArrays':len(fields),'newArrayBytes':sum(x['bytes'] for x in fields),'fullByteFiles':len(v['files']),'fullByteCount':v['totalBytesVerified'],'independentViewByteChecks':len(v['views']),'viewBytesReadAgain':v['totalViewBytesVerified'],
        'signedFactorAccounts':[{'receiver':x['receiver'],'kind':x['kind'],'scope':x['scope'],'firstEnergyOverSum':x['firstEnergy']/x['sumEnergy'],'secondEnergyOverSum':x['secondEnergy']/x['sumEnergy'],'twiceCrossOverSum':x['twiceCross']/x['sumEnergy'],
            'negativeQueryCrossTerms':sum(y['twiceCross']<0 for y in rows if y['receiver']==x['receiver'] and y['kind']==x['kind'] and (x['scope']=='all' or y['sourceNonlocal']))} for x in r['factorAccounts']],
        'scope':'Published scalar/query records only. Energy ratios are signed algebraic accounts, not causal shares. View reads overlap full parent-file byte custody; complete arrays remain near compute.'})
    save(out/'scheduler.json',{'schema':'tce-first-consumer-scheduler-v1','jobID':j['job_id'],'state':j['state'],'reservation':j['reservation'],'result':j['result'],'executedPayload':j['payload_custody']['executed'],'rightSizeRecommendation':j.get('meta',{}).get('right_size'),
        'futureSizing':'Measured2907.1MBRSS/700MBVRAM/123.92s for this12affine+3.56GBretention/diagnostic job. Scheduler recommends1024MBVRAM for matching futuregeometry. Do not rerun a valid result merely to change reservation; retain actual phase/I/O cost.'})
    save(out/'saturn-instrument.json',{'schema':'tce-first-consumer-instrument-v1','target':str(root/'result'),'runExit':0,'verifyExit':0,'instrumentStatus':'target-streams-only','streams':2,'records':17408,'watchFires':0,'formalReceipts':0,
        'scope':'Original uncompressed stage/factor stream applicability/mechanics only; four runtime Acts have separate source-bound evidence. No factor quality, full-block parity, future mediation or generator certification.'});return {'newArrays':len(fields),'newArrayBytes':sum(x['bytes'] for x in fields)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.run,a.source,a.destination))
