"""Publish an immutable assay subset and a separately labelled post-hoc contrast."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
def summarize(run,dest):
    run=Path(run).resolve();dest=Path(dest).resolve();raw=run/'result'
    request=read(run/'request.json');report=read(raw/'report.json');seal=read(raw/'prediction-seal.json')
    collection=read(run/'collection.json');job=read(run/'scheduler-job.json')
    if report['requestSHA256']!=request['requestSHA256'] or sha(raw/'prediction-seal.json')!=report['predictionSealSHA256']:raise ValueError('Report identity differs')
    for path,digest in seal['images'].items():
        if sha(raw/'images'/path)!=digest:raise ValueError('Sealed prediction differs')
    if sha(raw/'chart.npz')!=seal['bundleSHA256'] or sha(raw/'serve-receipt.json')!=seal['servingReceiptSHA256']:raise ValueError('Serving evidence differs')
    # Keep published implementation identical to the executed payload.
    for name in ['response_chart.py','response_study.py']:
        if sha(Path(__file__).parent/name)!=request['sourceHashes'][name]:raise ValueError('Executed source differs: '+name)
    dest.mkdir(parents=True,exist_ok=False)
    for name in ['report.json','prediction-seal.json','compile-receipt.json','serve-receipt.json','chart.npz']:shutil.copyfile(raw/name,dest/name)
    shutil.copyfile(run/'request.json',dest/'request.json')
    images=dest/'images';images.mkdir()
    for row in report['serving']['rows']:
        if row['method']=='piecewise' and row['alpha']==request['novelAlpha']:shutil.copyfile(raw/'images'/row['path'],images/row['path'])
    write(dest/'example-request.json',{'alphas':[request['novelAlpha']]})
    write(dest/'provenance.json',{'jobID':job['job_id'],'state':job['state'],'payloadCustody':collection['custody'],
        'requestSHA256':request['requestSHA256'],'sourceHashes':request['sourceHashes'],'collectedInventory':collection['inventory'],
        'schedulerResult':{k:v for k,v in job['result'].items() if k!='diagnostics'},
        'scope':'Authenticated before/after worker inventories matched local result bytes. Public subset omits held PNG predictions; their hashes and scores remain in the raw report.'})
    pairs=report['scores']['pairedEffects'];interactions=[]
    for row in pairs:
        copy=next(x for x in pairs if x['method']=='copy' and x['seed']==row['seed'] and x['alpha']==row['alpha'])
        item={k:row[k] for k in ['seed','alpha','method']}
        for region in ['full','eye']:
            native=copy[region]['rms'];error=row[region]['rms']
            item[region]={'nativeInteractionRms':native,'errorRms':error,'relativeRms':error/native if native else None}
        interactions.append(item)
    methods=['piecewise','quadratic','copy','nearest','endpoint','transfer'];groups=[]
    for method in methods:
        for alpha in [.05,.1,.5]:
            rows=[x for x in report['scores']['rows'] if x['method']==method and x['alpha']==alpha]
            pr=[x for x in pairs if x['method']==method and x['alpha']==alpha]
            ir=[x for x in interactions if x['method']==method and x['alpha']==alpha]
            mean=lambda xs:statistics.mean(xs)
            cosines=[x['edit']['cosine'] for x in rows if x['edit']['cosine'] is not None]
            groups.append({'method':method,'alpha':alpha,'curves':len(rows),'seeds':len(pr),
                'meanEditRelativeRms':mean([x['edit']['relativeRms'] for x in rows]),
                'meanEditCosine':mean(cosines) if cosines else None,'editCosineValidCurves':len(cosines),
                'meanPairedRelativeRms':mean([x['full']['relativeRms'] for x in pr]),
                'meanInteractionRelativeRms':mean([x['full']['relativeRms'] for x in ir]),
                'meanEyeInteractionRelativeRms':mean([x['eye']['relativeRms'] for x in ir])})
    primary=[x for x in report['serving']['rows'] if x['method']=='piecewise']
    analysis={'schema':'tce-zero-network-analysis-v1','rawReportSHA256':sha(raw/'report.json'),'primary':'piecewise',
        'postHocContrast':'J(alpha) = [Y_I1(alpha)-Y_I0(alpha)] - [Y_I1(0)-Y_I0(0)]. Copy control error is exactly native J RMS. Re-normalize each existing paired-error numerator by that copy error. No new fitting, pixel access, or model execution.',
        'interactionRows':interactions,'groups':groups,
        'timing':{'primaryImages':len(primary),'medianComputeSeconds':statistics.median(x['computeSeconds'] for x in primary),
                  'medianCpuSeconds':statistics.median(x['cpuSeconds'] for x in primary),'medianComputeAndSaveSeconds':statistics.median(x['computeAndSaveSeconds'] for x in primary)},
        'mechanics_status':'complete','instrument_status':'raw scalar re-expression','trend_status':'Local broad response predicted; fine interaction weak at small dose; transfer counter-trend.',
        'terminal_status':'unassessed','dependence':'Four curves from two known seeds; no independent-seed generalization estimate.'}
    write(dest/'analysis.json',analysis)
    with (dest/'observations.jsonl').open('x') as f:
        for i,row in enumerate(groups):f.write(json.dumps({'schema':'tce-response-observation-v1','step':i,'metrics':row,'reportSHA256':analysis['rawReportSHA256']},allow_nan=False)+'\n')
    write(dest/'files.json',{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()})
    print(json.dumps({'published':str(dest),'groups':groups,'timing':analysis['timing']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();summarize(a.run,a.destination)
