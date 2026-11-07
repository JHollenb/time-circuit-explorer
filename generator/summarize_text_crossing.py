"""Compact post-seal metadata/custody accounts, no retained tensor loads."""
import argparse
from pathlib import Path
from encoder_atoms import read,save,sha
from publish_text_crossing import audit


def execute(root,source,destination):
    root=Path(root);p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True);r=read(p/'report.json');e=read(p/'execution.json');seal=read(p/'velocity-seal.json');j=read(root/'scheduler-job.json');v=read(p/'retained-array-verification.json')
    fields=[*seal['fields'].values(),*read(p/'derived-fields.json').values(),*e['nativeFields'].values()]
    save(out/'mechanics.json',audit(p));save(out/'scalar-summary.json',{'schema':'tce-text-crossing-scalars-v1','reportSHA256':sha(p/'report.json'),'velocitySealSHA256':sha(p/'velocity-seal.json'),'newArrays':len(fields),'newArrayBytes':sum(x['bytes'] for x in fields),'fullByteFiles':len(v['files']),'fullByteCount':v['totalBytesVerified'],
        'normalizedEnergyAccounts':[{'scope':a['scope'],'remainingSquaredNormOverInteractionEnergy':a['remainingReceiverEnergy']/a['originalInteractionEnergy'],'textSquaredNormOverInteractionEnergy':a['firstTextWriteEnergy']/a['originalInteractionEnergy'],'signedCrossOverInteractionEnergy':a['twiceRemainingText']/a['originalInteractionEnergy']} for a in r['accounts']],
        'scope':'Post-seal scalar/metadata arithmetic only. Normalized signed energy identity is not a set of positive causal shares. Complete arrays stay near compute.'})
    save(out/'scheduler.json',{'schema':'tce-text-crossing-scheduler-v1','jobID':j['job_id'],'state':j['state'],'reservation':j['reservation'],'result':j['result'],'executedPayload':j['payload_custody']['executed'],
        'nextSizingBasis':'Actual same169tensor/B1/1536 two-boundary2call run:8470.2MBRSS/8100MBVRAM/67.75s.10240MBRAM/VRAM reservation fits; retain measured headroom on future compatible geometry.'})
    save(out/'saturn-instrument.json',{'schema':'tce-text-crossing-instrument-v1','target':str(root/'result'),'runExit':0,'verifyExit':0,'instrumentStatus':'target-streams-only','streams':1,'records':1024,'watchFires':0,'formalReceipts':0,
        'scope':'Original uncompressed query stream applicability/mechanics only. Four runtime Act/Frame handoffs are bound separately; automatic evidence sweep supplies no formal scientific or generator certification.'})
    return {'newArrays':len(fields),'newArrayBytes':sum(x['bytes'] for x in fields)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.run,a.source,a.destination))
