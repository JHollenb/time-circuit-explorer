"""Audit actual coarse operands, dispatched key mass, final commits and custody."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from encoder_atoms import read,sha,identity


def audit(source):
    p=Path(source);r=read(p/'request.json');s=read(p/'inputs.json');im=read(p/'images.json');t=read(p/'targets.json');report=read(p/'report.json');seal=read(p/'candidate-seal.json')
    v=read(p/'retained-array-verification.json');prov=read(p/'provenance.json')
    if im['schema']!='tce-coarse-global-images-v2':raise ValueError('Expected explicit requested/consumed bias contract')
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
        a=next(x for x in im['actual'] if x['case']==c and x['kind']=='coarse-input')
        for n,port in [('hidden_states','pooledLatent'),('img_ids','coarseIDs')]:
            if a[n+'PayloadSHA256']!=fs[port]['payloadSHA256']:raise ValueError('Actual coarse operand differs')
        if a['encoder_hidden_statesPayloadSHA256']!=source['E']['payloadSHA256'] or a['maskPayloadSHA256']!=fs['keyLogweights']['payloadSHA256'] or a['timestepPayloadSHA256']!=s['clockTable']['rows'][3]['clockOperands']['transformer_argument']['payloadSHA256'] or a['txt_idsPayloadSHA256']!=im['common'][c]['txtIDs']['payloadSHA256']:raise ValueError('Actual full conditioning/clock/position/mass differs')
        final=next(x for x in im['actual'] if x['case']==c and x['kind']=='final-commit')
        if final['incomingPayloadSHA256']!=source['latent3']['payloadSHA256'] or final['velocityPayloadSHA256']!=fs['velocity3']['payloadSHA256'] or final['outgoingPayloadSHA256']!=fs['latent4']['payloadSHA256'] or final['finalStepIndex']!=4:raise ValueError('Actual final commit differs')
        calls=[x for x in im['attentionCalls'] if x['case']==c]
        if len(calls)!=25 or any(x['queryShape']!=[1,768,24,128] or x['keyShape']!=[1,768,24,128] or x['maskPayloadSHA256']!=fs['actualKeyLogweights']['payloadSHA256'] or x['requestedMaskPayloadSHA256']!=fs['keyLogweights']['payloadSHA256'] or x['consumedMaskDtype']!='torch.bfloat16' for x in calls):raise ValueError('Actual dispatched geometry/key mass differs')
        if im['counts'][c]!={'denoiser':1,'context':1,'joint':5,'single':20,'VAE':1}:raise ValueError('Physical call census differs')
        if sha(p/('coarse-'+c+'.png'))!=im['images'][c]['sha256']:raise ValueError('Original new image differs')
        for group in [source,fs,im['common'][c],read(p/'diagnostic-fields.json')[c]]:
            for item in group.values():custody(item)
    for item in read(p/'diagnostic-fields.json')['color'].values():custody(item)
    for ref in ['HN','HH','NN','NH','full']:
        for c,item in t[ref]['images'].items():
            if sha(p/(ref+'-'+c+'.png'))!=item['sha256']:raise ValueError('Inherited image differs')
    linears=[x for x in im['linears'] if x['parameter'].startswith('transformer.')]
    if len(linears)!=206 or sum(x['nominalMACs'] for x in linears)!=4735604097024 or len(im['VAEOperations'])!=80:raise ValueError('Paid affine/VAE work differs')
    if report['work']['coarseNominalAffineMACs']!=4735604097024 or report['work']['coarseAttentionCalls']!=50:raise ValueError('Reported work differs')
    for stream in prov['losslessStreams']:
        if hashlib.sha256(gzip.decompress((p/stream['published']).read_bytes())).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless stream differs')
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()];counts=Counter((x['case'],x['port']) for x in rows)
    for c in ['00','10']:
        for port,n in [('velocity3',1024),('latent4',1024),('pngRGB',512)]:
            if counts[c,port]!=n:raise ValueError('Dependent row coverage differs')
    if len(rows)!=report['observations']:raise ValueError('Observation total differs')
    return {'schema':'tce-coarse-global-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'actualSameStateNativeComparisons':2,'actualFinalCommits':2,
        'actualCoarseAttentionDispatchesWithFixedMass':50,'coarseTransformerLinears':206,'coarseNominalAffineMACs':4735604097024,'VAEConvLinearCalls':80,
        'originalNewPNGs':2,'inheritedPNGs':10,'observations':len(rows),'candidateSealSHA256':h,'reportSHA256':sha(p/'report.json'),
        'scope':'Identity/release/actualconsumer/completebyte/physicalwork/losslessrow mechanics, not quality, speedup or arbitrarypromptzero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');a=p.parse_args();print(json.dumps(audit(a.source),indent=2))
