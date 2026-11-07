"""Compact post-seal scalar and custody accounts; no retained tensor loading."""
import argparse
from pathlib import Path
from encoder_atoms import read,save,sha
from publish_carrier_crossing import audit


def execute(root,source,destination):
    root=Path(root);p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True)
    r=read(p/'report.json');e=read(p/'execution.json');s=read(p/'inputs.json');seal=read(p/'velocity-seal.json');j=read(root/'scheduler-job.json');v=read(p/'retained-array-verification.json')
    save(out/'mechanics.json',audit(p));accounts=[]
    for a in r['accounts']:
        e0=a['scalars']*a['referenceRms']**2;e1=a['scalars']*a['candidateRms']**2;ei=a['scalars']*a['rms']**2;twice=2*a['cosine']*(e0*e1)**.5
        accounts.append({'scope':a['scope'],'queries':a['queries'],'responseR0Rms':a['referenceRms'],'responseR1Rms':a['candidateRms'],'interactionRms':a['rms'],
            'interactionOverResponseR0':a['relativeRms'],'responseAmplitudeRatio':a['candidateRms']/a['referenceRms'],'responseCosine':a['cosine'],
            'responseR0Energy':e0,'responseR1Energy':e1,'interactionEnergy':ei,'twiceResponseInnerProduct':twice,'scalarEnergyClosure':ei-e0-e1+twice})
    fields=[*seal['fields'].values(),*read(p/'derived-fields.json').values(),*[x for n in ['nativeH0Fields','textHeadFields','textWriteFields'] for x in e[n].values()]]
    save(out/'scalar-summary.json',{'schema':'tce-carrier-crossing-scalars-v1','sourceReportSHA256':sha(p/'report.json'),'velocitySealSHA256':sha(p/'velocity-seal.json'),'accounts':accounts,
        'nativeH0EqualsDonor':{k:{d:x['payloadSHA256']==h['payloadSHA256'] for d,h in s['donors'].items()} for k,x in e['nativeH0Fields'].items()},
        'newArrays':len(fields),'newArrayBytes':sum(x['bytes'] for x in fields),'fullByteFiles':len(v['files']),'fullByteCount':v['totalBytesVerified'],
        'scope':'Post-seal arithmetic over published scalar records and metadata only. Energies/inner products reconstructed from RMS/cosine summaries; arrays remain near compute. Coincident R0/H_A bytes are an observation, not broad natural parity.'})
    save(out/'scheduler.json',{'schema':'tce-carrier-crossing-scheduler-v1','jobID':j['job_id'],'state':j['state'],'reservation':j['reservation'],'result':j['result'],'executedPayload':j['payload_custody']['executed'],
        'nextSizingBasis':'Actual same-geometry4call/noVAE run:7546.3MBRSS,8120MBVRAM,55.38s. Reconsider measured resource exception on later geometry; generic alias B64 preview inapplicable.'})
    save(out/'saturn-instrument.json',{'schema':'tce-carrier-crossing-instrument-v1','target':str(root/'result'),'runExit':0,'verifyExit':0,'instrumentStatus':'target-streams-only','streams':1,'records':1024,'watchFires':0,'formalReceipts':0,
        'scope':'Original uncompressed query stream applicability/mechanics only. Runtime has4typed Act/Frame handoffs; automatic evidence sweep does not interpret those as formal scientific claims or certify the interaction/generator.'})
    return {'accounts':accounts,'newArrays':len(fields),'newArrayBytes':sum(x['bytes'] for x in fields)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.run,a.source,a.destination))
