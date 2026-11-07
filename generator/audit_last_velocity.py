"""Local audit of public source, actual inputs, release and lossless custody."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from encoder_atoms import read,sha,identity


def audit(source):
    p=Path(source);r=read(p/'request.json');s=read(p/'inputs.json');f=read(p/'forecast-seal.json')
    im=read(p/'images.json');seal=read(p/'candidate-seal.json');t=read(p/'targets.json');report=read(p/'report.json')
    manifest=read(p/'files.json');v=read(p/'retained-array-verification.json');prov=read(p/'provenance.json')
    for n,h in manifest.items():
        if sha(p/n)!=h:raise ValueError('Published member differs: '+n)
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:
        raise ValueError('Request differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Owned source differs')
    if seal['forecastSealSHA256']!=sha(p/'forecast-seal.json') or seal['imagesSHA256']!=sha(p/'images.json') or report['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):
        raise ValueError('Release/custody chain differs')
    if f['prefix']!=s['prefix'] or im['forecastSealSHA256']!=sha(p/'forecast-seal.json'):
        raise ValueError('Prefix or forecast consumer differs')
    joins=0;fields=0
    for policy in ['secant','hold']:
        for case in ['00','10']:
            key=policy+'-'+case;a=next(x for x in im['actual'] if x['policy']==policy and x['case']==case)
            if a['Z3PayloadSHA256']!=s['prefix'][case]['latent3']['payloadSHA256'] or a['velocityPayloadSHA256']!=f['fields'][key]['velocity3']['payloadSHA256'] or a['Z4PayloadSHA256']!=im['fields'][key]['latent4']['payloadSHA256'] or a['finalStepIndex']!=4:
                raise ValueError('Actual finite input/output join differs')
            joins+=1
            for item in [*f['fields'][key].values(),*im['fields'][key].values(),*read(p/'diagnostic-fields.json')[key].values()]:
                if v['files'][item['path']]['sha256']!=item['sha256']:raise ValueError('Complete array custody absent')
                fields+=1
            if sha(p/(key+'.png'))!=im['images'][key]['sha256']:raise ValueError('New original PNG differs')
    for ref in ['own','full']:
        for case,item in t[ref]['images'].items():
            if sha(p/(ref+'-'+case+'.png'))!=item['sha256']:raise ValueError('Historical original PNG differs')
    for item in prov['losslessStreams']:
        raw=gzip.decompress((p/item['published']).read_bytes())
        if hashlib.sha256(raw).hexdigest()!=item['sourceSHA256']:raise ValueError('Lossless stream differs')
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()]
    if len(rows)!=report['observations']:raise ValueError('Observation count differs')
    counts=Counter((x['policy'],x['case'],x['port']) for x in rows)
    for policy in ['secant','hold']:
        for case in ['00','10']:
            for port,n in [('velocity3',1024),('latent4',1024),('pngRGB',512)]:
                if counts[policy,case,port]!=n:raise ValueError('Dependent row coverage differs')
    for row in report['nativeSchedulerReplays']:
        if not row['bitExact'] or row['replayedZ4PayloadSHA256']!=t['own']['fields'][row['case']]['latent4']['payloadSHA256']:
            raise ValueError('Native finite parity differs')
    op_counts=Counter(x['policy']+'-'+x['case'] for x in im['operations'])
    if set(op_counts)!=set(im['fields']) or set(op_counts.values())!={40} or len(im['operations'])!=160:
        raise ValueError('VAE instrumentation differs')
    if sum(x['nominalMACs'] for x in im['operations'])!=report['work']['VAENominalConvLinearMACs']:
        raise ValueError('Paid VAE work differs')
    return {'schema':'tce-last-velocity-mechanics-v1','manifestMembers':len(manifest),'ownedSourcePins':len(r['ownedSources']),
        'actualFiniteCommitJoins':joins,'completeNewArrayFields':fields,'originalNewPNGs':4,'historicalPNGs':4,
        'nativeBitExactSchedulerReplays':2,'observations':len(rows),'VAEConvLinearCalls':160,
        'VAENominalConvLinearMACs':report['work']['VAENominalConvLinearMACs'],'candidateSealSHA256':sha(p/'candidate-seal.json'),
        'reportSHA256':sha(p/'report.json'),'scope':'Source/request/release, complete field custody, actual finite inputs, original PNGs, native scheduler parity and lossless dependent observations. No semantic or zero-neural certificate.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');a=p.parse_args();print(json.dumps(audit(a.source),indent=2))
