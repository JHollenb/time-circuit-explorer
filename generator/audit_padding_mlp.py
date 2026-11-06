"""Check published custody, actual consumer joins, and lossless observation rows."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from encoder_atoms import read,sha


def audit(primary,remainder,failure):
    p,q,f=map(Path,[primary,remainder,failure]);manifests={}
    for directory in [p,q,f]:
        manifest=read(directory/'files.json')
        for n,h in manifest.items():
            if sha(directory/n)!=h:raise ValueError('Published member differs: '+n)
        manifests[directory.name]=len(manifest)
    r=read(p/'request.json');enc=read(p/'encoder-candidate.json');im=read(p/'images-candidate.json');reader=read(p/'reader-seal.json');diag=read(p/'native-write-diagnostic.json');seal=read(p/'candidate-seal.json');report=read(p/'report.json')
    ids=[x['id'] for x in r['prompts']];noise=read(p/'noise-seal.json');clocks=read(p/'clock-table.json')
    if len(ids)!=4 or len(set(ids))!=4:raise ValueError('Four unique own candidates required')
    for name,key in [('compile-receipt.json','compileReceiptSHA256'),('reader-seal.json','readerSHA256'),('encoder-candidate.json','encoderSHA256'),('images-candidate.json','imagesSHA256')]:
        if seal[key]!=sha(p/name):raise ValueError('Candidate release differs')
    if diag['releasedAfterCandidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Diagnostic release differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed source differs')
    handoffs=0;joins=0
    for c in ids:
        if enc['fields'][c]['H0']['payloadSHA256']!=reader['h0Fields'][c]['payloadSHA256']:raise ValueError('Wrong lexical H0')
        for i in range(27):
            h=next(x for x in enc['handoffs'] if x['case']==c and x['layer']==i)
            if h['actualInputPayloadSHA256']!=enc['fields'][c]['H'+str(i)]['payloadSHA256'] or h['outputPayloadSHA256']!=enc['fields'][c]['H'+str(i+1)]['payloadSHA256']:raise ValueError('Own state handoff differs')
            handoffs+=1
        if im['actualEFields'][c]!=enc['fields'][c]['E']:raise ValueError('Wrong conditioning field')
        for step in range(4):
            a=next(x for x in im['actual'] if x['case']==c and x['kind']=='denoiser-input' and x['step']==step)
            latent=noise['packed'] if step==0 else im['fields'][c]['latent'+str(step)]
            if a['EPayloadSHA256']!=enc['fields'][c]['E']['payloadSHA256'] or a['latentPayloadSHA256']!=latent['payloadSHA256'] or a['clockArgumentPayloadSHA256']!=clocks['rows'][step]['clockOperands']['transformer_argument']['payloadSHA256']:raise ValueError('Denoiser input join differs')
            joins+=1
        if sha(p/(c+'.png'))!=im['images'][c]['sha256']:raise ValueError('Original candidate PNG differs')
    target=read(p/'targets.json')
    for role in ['top4','full']:
        for c in ['00','10']:
            if sha(p/(role+'-'+c+'.png'))!=target[role+'Images']['images'][c]['sha256']:raise ValueError('Historical PNG differs')
    for c in ['00','10']:
        a,b=[enc['firstMLPFields'][policy+'-'+c] for policy in ['constant','tangent']]
        for n in ['residual','normalized','activeWrite','mask']:
            if a[n]['payloadSHA256']!=b[n]['payloadSHA256']:raise ValueError('Matched active/receiver input differs')
        d=next(x for x in diag['actualInputs'] if x['case']==c)
        if d['actualNormalizedPayloadSHA256']!=a['normalized']['payloadSHA256']:raise ValueError('Native diagnostic received another input')
    if len(enc['sharedActiveWriteReads'])!=2 or len(enc['tangentCalls'])!=2:raise ValueError('Active reuse/tangent census differs')
    counts={}
    for directory in [p,q]:
        provenance=read(directory/'provenance.json');key='losslessStreams' if directory==p else 'losslessStreamTransports'
        for stream in provenance[key]:
            data=gzip.decompress((directory/stream['published']).read_bytes())
            import hashlib
            if hashlib.sha256(data).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless stream differs')
        rows=[json.loads(line) for line in gzip.decompress((directory/'observations.jsonl.gz').read_bytes()).splitlines()]
        if len(rows)!=read(directory/'report.json')['observations']:raise ValueError('Observation count differs')
        counts[directory.name]=len(rows)
        if directory==p:
            if Counter(x['id'] for x in rows)!=Counter({c:30720 for c in ids}):raise ValueError('Candidate row coverage differs')
        else:
            if Counter((x['case'],x['policy']) for x in rows)!=Counter({(c,s):475 for c in ['00','10'] for s in ['constant','tangent']}):raise ValueError('Padding decomposition coverage differs')
    if read(q/'report.json')['candidateSealSHA256']!=sha(p/'candidate-seal.json') or read(f/'failure.json')['newImages']!=0:raise ValueError('Addon/failure scope differs')
    return {'schema':'tce-padding-mlp-mechanics-v1','manifestMembers':manifests,'ownedSourcePins':len(r['ownedSources']),'ownEncoderHandoffs':handoffs,'actualDenoiserInputJoins':joins,
        'candidateFields':sum(len(v) for g in [enc['fields'],im['fields']] for v in g.values()),'firstMLPFields':sum(map(len,enc['firstMLPFields'].values())),
        'nativeDiagnosticFields':sum(map(len,diag['fields'].values())),'sharedActiveWriteReuses':2,'originalNewPNGs':4,'historicalPNGs':4,'observations':counts,
        'candidateSealSHA256':sha(p/'candidate-seal.json'),'reportSHA256':sha(p/'report.json'),'remainderReportSHA256':sha(q/'report.json'),
        'scope':'Integrity and actual input/state/release/custody coverage, not semantic certification or zero-neural completion.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('primary');p.add_argument('remainder');p.add_argument('failure');a=p.parse_args();print(json.dumps(audit(a.primary,a.remainder,a.failure),indent=2))
