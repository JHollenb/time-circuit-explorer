"""Audit source-native reader/writer custody and actual consumer handoffs."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from encoder_atoms import read,sha,identity


def audit(source):
    p=Path(source);r=read(p/'request.json');s=read(p/'inputs.json');im=read(p/'images.json');t=read(p/'targets.json');report=read(p/'report.json')
    cs=read(p/'coefficient-seal.json');served=read(p/'serving.json');fit=read(p/'fit.json');corp=read(p/'corpus.json');v=read(p/'retained-array-verification.json');prov=read(p/'provenance.json')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Published member differs')
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed source differs: '+n)
    seal=read(p/'candidate-seal.json');h=sha(p/'candidate-seal.json')
    if report['candidateSealSHA256']!=h or v['candidateSealSHA256']!=h or seal['imagesSHA256']!=sha(p/'images.json') or seal['fields']!=im['fields']:raise ValueError('Release differs')
    if im['coefficientSealSHA256']!=sha(p/'coefficient-seal.json') or served['coefficientSealSHA256']!=sha(p/'coefficient-seal.json') or im['fitReceiptSHA256']!=sha(p/'fit.json') or cs['fitReceiptSHA256']!=sha(p/'fit.json'):raise ValueError('Coefficient release differs')
    def custody(item):
        if v['files'][item['path']]['sha256']!=item['sha256']:raise ValueError('Complete field custody absent')
    reads={x['id']:x for x in im['reads']}
    if len(reads)!=46 or Counter(x['population'] for x in reads.values())!={'discovery':18,'feedback':18,'evaluation':8,'main':2}:raise ValueError('Read coverage differs')
    for row in reads.values():
        for n in ['z','previous','H0']:custody(row[n])
        frame=row['frame'];out=row['outputFrame'];ev=row['evidence'];slot=frame['slots']['Z']
        if slot['kind']!='tensor' or slot['abi']['shape']!=row['z']['shape'] or slot['fingerprint']!=row['z']['payloadSHA256']:raise ValueError('Tensor source ABI differs')
        if ev['input_frame_id']!=frame['frame_id'] or ev['output_frame_id']!=out['frame_id'] or out['parent_frame_id']!=frame['frame_id'] or out['slots']['H0']['kind']!='tensor':raise ValueError('Typed native read transition differs')
        ops=[x for x in im['linears'] if x['id']==row['id']];att=[x for x in im['attentionCalls'] if x['id']==row['id']]
        if len(ops)!=7 or sum(x['nominalMACs'] for x in ops)!=51136954368 or len(att)!=1:raise ValueError('Actual original affine work differs')
        a=att[0]
        if a['queryShape']!=[1,1024,24,128] or a['keyShape']!=[1,1536,24,128] or a['valueShape']!=a['keyShape'] or a['mask'] is not None or a['outputPayloadSHA256']!=row['H0']['payloadSHA256']:raise ValueError('Global read geometry/output differs')
    if len(served['rows'])!=30 or len(im['writerBindings'])!=30:raise ValueError('Writer coverage differs')
    for row in served['rows']:
        if row['inputFields']!={k:reads[row['id']][k] for k in ['z','previous','H0']}:raise ValueError('Writer bypassed its actual inputs')
        for n in ['velocity','rawVelocity']:custody(row[n])
        binding=next(x for x in im['writerBindings'] if (x['id'],x['method'])==(row['id'],row['method']))
        if binding['outputFrame']['parent_frame_id']!=binding['inputFrame']['frame_id'] or binding['evidence']['metrics']['additionalWriterProducts']!=0:raise ValueError('Typed writer handoff differs')
    if cs['restoredFrame']!=cs['sourceFrame'] or cs['gain']!=1 or fit['discoveryRows']!=18432 or fit['pairedRows']!=1888 or fit['features']!=3328 or fit['ridge']!=.01:raise ValueError('Discovery proposal/immutable restoration differs')
    for actor in [fit,served]:
        if actor['frameworksLoaded']!=[] or len(actor['fileQuarantine']['deniedCanaries'])!=3:raise ValueError('Actor quarantine receipt absent')
    custody(cs['baselineWriter']);custody(cs['discoveryPacket'])
    for item in fit['fields'].values():custody(item)
    for item in corp['sourceFiles'].values():custody(item)
    for c in ['00','10']:
        fs=im['fields'][c];prefix=s['prefix'][c]
        if fs['latent3']!=prefix['latent3'] or fs['latent3']!=t['HN']['fields'][c]['latent3'] or reads[c]['previous']['payloadSHA256']!=prefix['velocity1']['payloadSHA256']:raise ValueError('Main history/state differs')
        if reads[c]['z']['payloadSHA256']!=prefix['latent3']['payloadSHA256'] or reads[c]['frame']['slots']['E']['fingerprint']!=prefix['E']['payloadSHA256']:raise ValueError('Actual main read inputs differ')
        writer=next(x for x in served['rows'] if x['id']==c and x['method']=='candidate')
        if fs['velocity3']!=writer['velocity']:raise ValueError('Native consumer used another velocity')
        final=next(x for x in im['actual'] if x['case']==c)
        if [final[k] for k in ['incomingPayloadSHA256','velocityPayloadSHA256','outgoingPayloadSHA256']]!=[fs[k]['payloadSHA256'] for k in ['latent3','velocity3','latent4']] or final['finalStepIndex']!=4:raise ValueError('Actual final commit differs')
        for item in fs.values():custody(item)
        if sha(p/('reader-'+c+'.png'))!=im['images'][c]['sha256']:raise ValueError('Original new PNG differs')
    for ref in ['HN','HH','NN','NH','full']:
        for c,item in t[ref]['images'].items():
            if sha(p/(ref+'-'+c+'.png'))!=item['sha256']:raise ValueError('Inherited image differs')
    for stream in prov['losslessStreams']:
        if hashlib.sha256(gzip.decompress((p/stream['published']).read_bytes())).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless rows differ')
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()];counts=Counter((x['id'],x['method'],x['port']) for x in rows)
    for row in served['rows']:
        if counts[row['id'],row['method'],'velocity3']!=1024:raise ValueError('Velocity row coverage differs')
    for c in ['00','10']:
        if counts[c,'candidate','latent4']!=1024 or counts[c,'candidate','pngRGB']!=512:raise ValueError('Native consumer row coverage differs')
    if len(rows)!=report['observations'] or len(im['VAEOperations'])!=80:raise ValueError('Observation/consumer work differs')
    return {'schema':'tce-source-native-mechanics-v1','sourcePins':len(r['ownedSources']),'sourceNativeReads':46,'originalAffines':322,'typedWriterHandoffs':30,
        'nativeFinalCommits':2,'originalVAEDecodes':2,'originalNewPNGs':2,'observations':len(rows),'historicalSourceFiles':len(corp['sourceFiles']),
        'candidateSealSHA256':h,'reportSHA256':sha(p/'report.json'),
        'scope':'Source/quarantine/typed transition/cost/full-byte/original-consumer/lossless-row mechanics only; no image quality, speedup or arbitraryprompt zero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');a=p.parse_args();print(json.dumps(audit(a.source),indent=2))
