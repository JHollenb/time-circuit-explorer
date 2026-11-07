"""Audit actual routing, full-state handoffs, sliced work and publication custody."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from encoder_atoms import read,sha,identity
from rotating_writes import image_rows


def audit(source):
    p=Path(source);r=read(p/'request.json');s=read(p/'inputs.json');im=read(p/'images.json');t=read(p/'targets.json');report=read(p/'report.json');seal=read(p/'candidate-seal.json')
    v=read(p/'retained-array-verification.json');prov=read(p/'provenance.json')
    if im['schema']!='tce-rotating-write-images-v1':raise ValueError('Unexpected image schema')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Published member differs')
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed source differs')
    h=sha(p/'candidate-seal.json')
    if report['candidateSealSHA256']!=h or v['candidateSealSHA256']!=h or seal['imagesSHA256']!=sha(p/'images.json') or seal['fields']!=im['fields']:raise ValueError('Complete release differs')
    def custody(item):
        if v['files'][item['path']]['sha256']!=item['sha256']:raise ValueError('Complete field custody absent')
    for c in ['00','10']:
        fs=im['fields'][c];source=s['prefix'][c];native=t['HN']['fields'][c]
        if fs['latent3']!=source['latent3'] or native['latent3']!=source['latent3']:raise ValueError('Same incoming state differs')
        a=next(x for x in im['actual'] if x['case']==c and x['kind']=='routed-input')
        expected={'hidden_states':source['latent3'],'encoder_hidden_states':source['E'],'img_ids':s['ids'],
            'txt_ids':im['common'][c]['txtIDs'],'timestep':s['clockTable']['rows'][3]['clockOperands']['transformer_argument']}
        for n,item in expected.items():
            if a[n+'PayloadSHA256']!=item['payloadSHA256']:raise ValueError('Actual complete input differs')
        final=next(x for x in im['actual'] if x['case']==c and x['kind']=='final-commit')
        if final['incomingPayloadSHA256']!=source['latent3']['payloadSHA256'] or final['velocityPayloadSHA256']!=fs['velocity3']['payloadSHA256'] or final['outgoingPayloadSHA256']!=fs['latent4']['payloadSHA256'] or final['finalStepIndex']!=4:raise ValueError('Actual final commit differs')
        routes=[x for x in im['routes'] if x['case']==c]
        if [x['blockIndex'] for x in routes]!=list(range(25)):raise ValueError('Global block sequence differs')
        for i,row in enumerate(routes):
            selected=np.concatenate([np.arange(512),512+image_rows(i)]).astype(np.int64)
            if row['selectedImageRows']!=image_rows(i).tolist() or row['selectedGlobalRowsPayloadSHA256']!=hashlib.sha256(selected.tobytes()).hexdigest():raise ValueError('Actual selected phase differs')
            if row['unchangedInputPayloadSHA256']!=row['unchangedOutputPayloadSHA256']:raise ValueError('Unselected row bits differ')
            if i and routes[i-1]['fullOutputStatePayloadSHA256']!=row['fullInputStatePayloadSHA256']:raise ValueError('Own-state block handoff differs')
        calls=[x for x in im['attentionCalls'] if x['case']==c]
        if [x['blockIndex'] for x in calls]!=list(range(25)):raise ValueError('Dispatched block sequence differs')
        for x in calls:
            if x['queryShape']!=[1,768,24,128] or x['keyShape']!=[1,1536,24,128] or x['valueShape']!=x['keyShape'] or x['mask'] is not None or any(x[n+'Dtype']!='torch.bfloat16' for n in ['query','key','value']):raise ValueError('Actual rectangular Q/K/V differs')
        linears=[x for x in im['linears'] if x['case']==c and x['parameter'].startswith('transformer.')]
        if len(linears)!=143 or sum(x['nominalMACs'] for x in linears)!=2730793893888:raise ValueError('Paid affine work differs')
        for i in range(5,25):
            ops=[x for x in linears if x.get('blockIndex')==i and x['parameter'].endswith('to_qkv_mlp_proj.weight')]
            actual=[(x['inputShape'],x['weightShape'],x['weightStorageOffsetElements']) for x in ops]
            expected=[([1,768,3072],[3072,3072],0),([1,1536,3072],[6144,3072],9437184),([1,768,3072],[18432,3072],28311552)]
            if actual!=expected or any(x['weightStride']!=[3072,1] for x in ops):raise ValueError('Physical fused projection slices differ')
        if im['counts'][c]!={'denoiser':1,'context':1,'joint':5,'single':20,'VAE':1}:raise ValueError('Physical consumer count differs')
        if sha(p/('routed-'+c+'.png'))!=im['images'][c]['sha256']:raise ValueError('Original new PNG differs')
        for group in [source,fs,im['common'][c],read(p/'diagnostic-fields.json')[c]]:
            for item in group.values():custody(item)
    for ref in ['HN','HH','NN','NH','full']:
        for c,item in t[ref]['images'].items():
            if sha(p/(ref+'-'+c+'.png'))!=item['sha256']:raise ValueError('Inherited image differs')
    if len(im['VAEOperations'])!=80 or report['work']['routedNominalAffineMACs']!=5461587787776 or report['work']['routedAttentionCalls']!=50:raise ValueError('Reported physical work differs')
    for stream in prov['losslessStreams']:
        if hashlib.sha256(gzip.decompress((p/stream['published']).read_bytes())).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless stream differs')
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()];counts=Counter((x['case'],x['port']) for x in rows)
    for c in ['00','10']:
        for port,n in [('velocity3',1024),('latent4',1024),('pngRGB',512)]:
            if counts[c,port]!=n:raise ValueError('Dependent row coverage differs')
    if len(rows)!=report['observations']:raise ValueError('Observation total differs')
    return {'schema':'tce-rotating-write-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'actualSameStateNativeComparisons':2,'actualFinalCommits':2,
        'actualRectangularAttentionCalls':50,'blockHandoffJoins':48,'unchangedRowWitnesses':50,'physicallySlicedSingleInputCalls':120,
        'routedTransformerLinears':286,'routedNominalAffineMACs':5461587787776,'VAEConvLinearCalls':80,'originalNewPNGs':2,'inheritedPNGs':10,
        'observations':len(rows),'candidateSealSHA256':h,'reportSHA256':sha(p/'report.json'),
        'scope':'Source/release/actualconsumer/ownstatehandoff/routing/physicalwork/fullbyte/losslessrow mechanics; no quality,speedup or arbitrarypromptzero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');a=p.parse_args();print(json.dumps(audit(a.source),indent=2))
