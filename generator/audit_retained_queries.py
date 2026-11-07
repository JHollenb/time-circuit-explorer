"""Publication custody, closure witnesses, own-state joins and lossless rows."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from encoder_atoms import read,sha
from retained_queries import np,selection


def audit(source):
    p=Path(source);r=read(p/'request.json');e=read(p/'encoder-candidate.json');im=read(p/'images-candidate.json');lex=read(p/'lexical-seal.json');seal=read(p/'candidate-seal.json');target=read(p/'targets.json');report=read(p/'report.json');prov=read(p/'provenance.json');manifest=read(p/'files.json')
    for n,h in manifest.items():
        if sha(p/n)!=h:raise ValueError('Published bytes differ')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed source differs')
    for n,key in [('encoder-candidate.json','encoderSHA256'),('images-candidate.json','imagesSHA256'),('lexical-seal.json','lexicalSealSHA256'),('clock-table.json','clockTableSHA256'),('noise-seal.json','noiseSealSHA256')]:
        if sha(p/n)!=seal[key]:raise ValueError('Release chain differs')
    if report['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Score release differs')
    clocks=read(p/'clock-table.json');noise=read(p/'noise-seal.json');handoffs=0;joins=0
    for row in e['selectionRows']:
        c=row['id'];setup=e['setupFields'][c];selected=np.array(row['selectedOriginalIndices']);expansion=np.array(row['expansionIndices']);mask=np.arange(512)<row['activeRows'];expected,expand=selection(mask,r['paddingQueries'])
        if not np.array_equal(selected,expected) or not np.array_equal(expansion,expand):raise ValueError('Frozen position rule differs from declared prefix mask')
        if hashlib.sha256(mask.tobytes()).hexdigest()!=setup['activeMask']['payloadSHA256']:raise ValueError('Mask bits differ from declared contiguous active support')
        if hashlib.sha256(selected.astype(np.int64).tobytes()).hexdigest()!=setup['selected']['payloadSHA256'] or hashlib.sha256(expansion.astype(np.int64).tobytes()).hexdigest()!=setup['expansion']['payloadSHA256']:raise ValueError('Actual selector witness differs')
        for name,array in [('fullPositions',np.arange(512,dtype=np.int64)[None,:]),('compactPositions',selected.astype(np.int64)[None,:])]:
            if hashlib.sha256(array.tobytes()).hexdigest()!=setup[name]['payloadSHA256']:raise ValueError('Original position witness differs')
        full_visibility=(np.tri(512,dtype=bool)&mask[None,:])[None,None,:,:];small_visibility=full_visibility[:,:,selected][:,:,:,selected]
        for name,array in [('fullVisibility',full_visibility),('compactVisibility',small_visibility)]:
            if hashlib.sha256(array.tobytes()).hexdigest()!=setup[name]['payloadSHA256']:raise ValueError('Actual native mask witness differs')
        if e['compactFields'][c]['H0']['shape']!=[1,len(selected),2560]:raise ValueError('Compact own H0 geometry differs')
        for layer in range(27):
            h=next(x for x in e['handoffs'] if x['case']==c and x['layer']==layer)
            if h['actualInputPayloadSHA256']!=e['compactFields'][c]['H'+str(layer)]['payloadSHA256'] or h['outputPayloadSHA256']!=e['compactFields'][c]['H'+str(layer+1)]['payloadSHA256']:raise ValueError('Compact own-state handoff differs')
            call=next(x for x in e['primitives'] if x['case']==c and x['layer']==layer and x['kind']=='SDPA')
            if call['actualMaskPayloadSHA256']!=setup['compactVisibility']['payloadSHA256']:raise ValueError('Actual attention consumed another mask')
            handoffs+=1
        if im['actualEFields'][c]!=e['fields'][c]['E']:raise ValueError('Image consumed another E')
        for step in range(4):
            a=next(x for x in im['actual'] if x['case']==c and x['step']==step and x['kind']=='denoiser-input');latent=noise['packed'] if step==0 else im['fields'][c]['latent'+str(step)]
            if a['latentPayloadSHA256']!=latent['payloadSHA256'] or a['EPayloadSHA256']!=e['fields'][c]['E']['payloadSHA256'] or a['clockArgumentPayloadSHA256']!=clocks['rows'][step]['clockOperands']['transformer_argument']['payloadSHA256']:raise ValueError('Actual image input join differs')
            joins+=1
        if sha(p/('candidate-'+c+'.png'))!=im['images'][c]['sha256']:raise ValueError('Original candidate PNG differs')
        for ref in ['full','top4']:
            if sha(p/(ref+'-'+c+'.png'))!=target[ref+'Images']['images'][c]['sha256']:raise ValueError('Historical PNG differs')
    streams=prov['losslessStreams']
    for item in streams:
        data=gzip.decompress((p/item['published']).read_bytes())
        if hashlib.sha256(data).hexdigest()!=item['sourceSHA256']:raise ValueError('Lossless observations differ')
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()]
    if len(rows)!=report['observations']:raise ValueError('Observation count differs')
    counts=Counter(x['case'] for x in rows)
    if set(counts)!={'00','10'} or counts['00']!=counts['10']:raise ValueError('Case observation coverage differs')
    distributions=[]
    for ref in ['full','top4']:
        for attended,label in [(True,'active'),(False,'padding')]:
            for c in ['00','10']:
                data=[x['relativeRms'] for x in rows if x.get('reference')==ref and x['port']=='E' and x['case']==c and x['attended']==attended];valid=[x for x in data if x is not None]
                distributions.append({'case':c,'reference':ref,'port':'E','region':label,'rows':len(data),'definedRelativeErrors':len(valid),'medianRowRelativeRms':float(np.median(valid)) if valid else None,'p90RowRelativeRms':float(np.quantile(valid,.9)) if valid else None,'maxRowRelativeRms':max(valid) if valid else None})
    return {'schema':'tce-retained-query-publication-qa-v1','manifestMembers':len(manifest),'ownedSourcePins':len(r['ownedSources']),'originalVisibilityPositionWitnesses':'Exact bit hashes recomputed from declared current source positions/mask; selected indices and native consumed mask joined. RoPE fields independently full-byte verified near compute.',
        'ownEncoderHandoffs':handoffs,'actualDenoiserInputJoins':joins,'compactStateFields':sum(map(len,e['compactFields'].values())),'expandedEncoderFields':sum(map(len,e['fields'].values())),'setupWitnessFields':sum(map(len,e['setupFields'].values())),'imageFields':sum(map(len,im['fields'].values())),'originalNewPNGs':2,'historicalPNGs':4,
        'observations':len(rows),'caseObservationCounts':dict(counts),'conditioningRowDistributions':distributions,'candidateSealSHA256':sha(p/'candidate-seal.json'),'reportSHA256':sha(p/'report.json'),'scope':'Mechanics, source/custody/actual inputs and complete observations. No semantic certification or zero-neural completion.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');a=p.parse_args();print(json.dumps(audit(a.source),indent=2))
