"""Verify complete direct-reader bytes and explicitly inherited output evidence."""
import argparse
import gzip
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read, save, sha
from publish_context_statistics import checked_collection
from publish_source_context import REMOTE


def checked(root):
    root=Path(root).resolve();c,j,r=checked_collection(root,current_workers=False)
    for name in r['ownedSources']:
        if sha(Path(__file__).parent/name)!=r['sourceHashes'][name]:raise ValueError('Executed source changed: '+name)
    return root,c,j,r


def verify(root):
    root,c,j,r=checked(root);p=root/'result';source=read(p/'inputs.json');target=read(p/'targets.json');reader=read(p/'reader-seal.json');compile=read(p/'compile-receipt.json');seal=read(p/'head-seal.json');report=read(p/'report.json');base=j['result']['diagnostics']['cwd']
    if seal['readerSealSHA256']!=sha(p/'reader-seal.json') or report['headSealSHA256']!=sha(p/'head-seal.json'):raise ValueError('Own head seal differs')
    if compile['centers']!=reader['centers'] or compile['originalGeometry']!=source['originalGeometry']:raise ValueError('Centers compilation binding differs')
    items=[source['originalGeometry'],compile['centers'],*reader['fields'].values(),*reader['h0Fields'].values(),*reader['lexicalRows'],*reader['packets'],*reader['states'],*reader['sourceAtomFiles']]
    items+=read(p/'comparison-fields.json')['fields']+target['reader']['packets']+target['reader']['lexicalRows']
    items+=list(target['reader']['fields'].values())+[fs['H0'] for fs in target['encoder']['fields'].values()]
    items+=list(target['sourceFiles'].values())
    items += [{'path':base+'/'+n,'sha256':sha(p/n)} for n in ['reader-seal.json','head-seal.json','compile-receipt.json']]
    items += [{'path':base+'/reader/prediction-seal.json','sha256':reader['readerPredictionSealSHA256']},
        {'path':base+'/serve-source.json','sha256':reader['reader']['specSHA256']},
        {'path':base+'/lexical/prediction-seal.json','sha256':reader['lexicalSealSHA256']}]
    if report['outcome']=='reuse-qualified':
        inherited=read(p/'inherited-outputs.json')
        if inherited['inputIdentitySHA256']!=sha(p/'input-identity.json') or inherited['parentSealSHA256']!=source['parentSealSHA256']:raise ValueError('Inherited output proof differs')
        for group in [inherited['encoderFields'],inherited['imageFields']]:
            for fields in group.values():items+=list(fields.values())
        items+=list(inherited['images'].values())
        for row in target['clockTable']['rows']:items += [*row['clockOperands'].values(),*row['fields'].values()]
        items += [target['parentSource']['noise'][k] for k in ['raw','packed','ids']]
    items += [{'path':base+'/response-result/'+f.name,'sha256':sha(f)} for f in p.iterdir()]
    s=target['parentSource'];roots=sorted(set([base,source['parentRoot'],str(Path(source['originalGeometry']['path']).parent.parent),source['lexicalStore'],s['clockParentRoot'],s['noiseParentRoot']]))
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],input=json.dumps({'allowedRoots':roots,'items':items}),text=True,capture_output=True,check=True,timeout=180)
    value=json.loads(proc.stdout);value.update(schema='tce-direct-reader-bytes-v1',jobID=j['job_id'],verifierSourceSHA256=sha(__file__),headSealSHA256=sha(p/'head-seal.json'),
        scope='Complete used atoms, original geometry and centers extraction, current raw/BF16/H0/state/diagnostic packets, parent comparison inputs, and exact-input-inherited complete encoder/image outputs/clocks/noise where qualified. No bulk activation hydration.')
    save(root/'retained-array-verification.json',value);return {'files':len(value['files']),'bytes':value['totalBytesVerified']}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['headSealSHA256']!=sha(p/'head-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Independent byte verification differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for f in p.iterdir():
        if f.suffix=='.jsonl':
            data=f.read_bytes();target=out/(f.name+'.gz');target.write_bytes(gzip.compress(data,mtime=0))
            if gzip.decompress(target.read_bytes())!=data:raise ValueError('Lossless stream differs')
            streams.append({'source':f.name,'sourceSHA256':sha(f),'sourceBytes':len(data),'published':target.name,'publishedSHA256':sha(target),'publishedBytes':target.stat().st_size})
        else:
            if f.suffix not in ['.json','.md','.png']:raise ValueError('Unexpected compact member')
            shutil.copyfile(f,out/f.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-direct-reader-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],
        'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreams':streams,
        'scope':'New CPU source extraction/raw reader and complete input-identity proof. Qualified parent encoder/image evidence is inherited explicitly; no new images/dense consumers, arbitrary-prompt certificate or speedup claim.'})
    save(out/'files.json',{f.name:sha(f) for f in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(f.stat().st_size for f in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(verify(a.run) if a.action=='verify' else publish(a.run,a.destination)))
