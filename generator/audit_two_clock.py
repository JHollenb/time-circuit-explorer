"""Public metadata mechanics audit; complete source/operand/byte identities."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from encoder_atoms import read,sha,identity


def audit(source):
    p=Path(source);r=read(p/'request.json');s=read(p/'inputs.json');im=read(p/'images.json');t=read(p/'targets.json')
    seal=read(p/'candidate-seal.json');report=read(p/'report.json');v=read(p/'retained-array-verification.json');prov=read(p/'provenance.json');bindings=read(p/'inherited-bindings.json');manifest=read(p/'files.json')
    for n,h in manifest.items():
        if sha(p/n)!=h:raise ValueError('Published member differs: '+n)
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Owned source differs')
    seal_sha=sha(p/'candidate-seal.json')
    if report['candidateSealSHA256']!=seal_sha or v['candidateSealSHA256']!=seal_sha or seal['imagesSHA256']!=sha(p/'images.json') or seal['fields']!=im['fields'] or seal['common']!=im['common']:
        raise ValueError('Complete release chain differs')
    def custody(item):
        if v['files'][item['path']]['sha256']!=item['sha256']:raise ValueError('Complete byte custody missing')
    joins=0
    for c in ['00','10']:
        b=next(x for x in bindings['joins'] if x['case']==c)
        if b['Z2']!=s['prefix'][c]['latent2'] or b['v1']!=s['prefix'][c]['velocity1'] or b['E']!=s['prefix'][c]['E'] or b['NNZ3']!=t['NN']['fields'][c]['latent3'] or b['NHZ4']!=t['NH']['fields'][c]['latent4']:
            raise ValueError('Inherited interface differs')
        shared=next(x for x in im['actual'] if x['case']==c and x['kind']=='shared-commit')
        z3=im['common'][c]['latent3']
        if shared['incomingPayloadSHA256']!=b['Z2']['payloadSHA256'] or shared['velocityPayloadSHA256']!=b['v1']['payloadSHA256'] or shared['outgoingPayloadSHA256']!=z3['payloadSHA256'] or shared['finalStepIndex']!=3:
            raise ValueError('Actual shared commit differs')
        native=next(x for x in im['actual'] if x['case']==c and x['kind']=='native-input')
        if native['latentPayloadSHA256']!=z3['payloadSHA256'] or native['EPayloadSHA256']!=b['E']['payloadSHA256'] or native['imgIDsPayloadSHA256']!=s['ids']['payloadSHA256'] or native['txtIDsPayloadSHA256']!=im['common'][c]['txtIDs']['payloadSHA256'] or native['clockArgumentPayloadSHA256']!=s['clockTable']['rows'][3]['clockOperands']['transformer_argument']['payloadSHA256']:
            raise ValueError('Actual native read differs')
        for policy in ['HN','HH']:
            key=policy+'-'+c;fs=im['fields'][key];actual=next(x for x in im['actual'] if x['case']==c and x['kind']=='final-commit' and x['policy']==policy)
            if fs['latent3']!=z3 or actual['incomingPayloadSHA256']!=z3['payloadSHA256'] or actual['velocityPayloadSHA256']!=fs['velocity3']['payloadSHA256'] or actual['outgoingPayloadSHA256']!=fs['latent4']['payloadSHA256'] or actual['finalStepIndex']!=4:
                raise ValueError('Actual final commit differs')
            if policy=='HH' and fs['velocity3']['payloadSHA256']!=b['v1']['payloadSHA256']:raise ValueError('HH did not retain v1')
            if sha(p/(key+'.png'))!=im['images'][key]['sha256']:raise ValueError('New original image differs')
            for item in fs.values():custody(item)
            joins+=1
        for fs in [s['prefix'][c],im['common'][c],read(p/'interaction-fields.json')[c],read(p/'finite-fields.json')[c]]:
            for item in fs.values():custody(item)
        for ref in ['NN','NH','full']:
            if sha(p/(ref+'-'+c+'.png'))!=t[ref]['images'][c]['sha256']:raise ValueError('Inherited PNG differs')
    if len(im['clockCalls'])!=10:raise ValueError('Compiled clock call count differs')
    for call in im['clockCalls']:
        if call['clock']!=3 or call['outputPayloadSHA256']!=s['clockTable']['rows'][3]['fields'][call['port']]['payloadSHA256']:raise ValueError('Actual compiled operand differs')
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()]
    counts=Counter((x['policy'],x['case'],x['port']) for x in rows)
    for policy in ['HH','HN']:
        for c in ['00','10']:
            for port,n in [('velocity3',1024),('latent4',1024),('pngRGB',512)]:
                if counts[policy,c,port]!=n:raise ValueError('Dependent observation coverage differs')
    if len(rows)!=report['observations']:raise ValueError('Observation total differs')
    for stream in prov['losslessStreams']:
        if hashlib.sha256(gzip.decompress((p/stream['published']).read_bytes())).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless stream differs')
    for replay in report['nativeSchedulerReplays']:
        c=replay['case']
        if not replay['bitExact'] or replay['nativeZ3PayloadSHA256']!=t['NN']['fields'][c]['latent3']['payloadSHA256'] or replay['holdZ4PayloadSHA256']!=t['NH']['fields'][c]['latent4']['payloadSHA256']:raise ValueError('Finite primitive replay differs')
    linears=[x for x in im['linears'] if x['parameter'].startswith('transformer.')]
    if len(linears)!=206 or sum(x['nominalMACs'] for x in linears)!=report['work']['nativeTransformerNominalAffineMACs'] or len(im['VAEOperations'])!=160 or sum(x['nominalMACs'] for x in im['VAEOperations'])!=report['work']['VAENominalConvLinearMACs']:
        raise ValueError('Measured work differs')
    for c in ['00','10']:
        if im['counts'][c]!={'VAE':2,'context':1,'denoiser':1,'joint':5,'single':20}:raise ValueError('Actual native call census differs')
    return {'schema':'tce-two-clock-mechanics-v1','manifestMembers':len(manifest),'ownedSourcePins':len(r['ownedSources']),
        'sharedActualCommits':2,'actualOwnStateNativeReads':2,'actualFinalCommits':joins,'compiledClockReads':10,
        'originalNewPNGs':4,'inheritedPNGs':6,'nativePrimitiveParityCommits':4,'observations':len(rows),
        'transformerLinears':len(linears),'VAEConvLinearCalls':160,'candidateSealSHA256':seal_sha,'reportSHA256':sha(p/'report.json'),
        'scope':'Actual input/commit/source/release/custody/work and lossless row audit; not a semantic, contraction or arbitrary-prompt zero-neural certificate.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');a=p.parse_args();print(json.dumps(audit(a.source),indent=2))
