"""Authenticate complete fresh square fields near compute; publish compact evidence."""
import argparse
import gzip
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read,save,sha
from publish_context_statistics import checked_collection
from publish_source_context import REMOTE


def checked(root):
    root=Path(root).resolve();c,j,r=checked_collection(root,current_workers=False)
    for name in r['ownedSources']:
        if sha(Path(__file__).parent/name)!=r['sourceHashes'][name]:raise ValueError('Executed source changed: '+name)
    return root,c,j,r


def verify(root):
    root,c,j,r=checked(root);p=root/'result';source=read(p/'inputs.json');reader=read(p/'reader-seal.json');base=j['result']['diagnostics']['cwd'];report=read(p/'report.json');table=read(p/'clock-table.json')
    items=[*reader['fields'].values(),*reader['h0Fields'].values(),*reader['lexicalRows'],*reader['packets'],*reader['states'],*reader['sourceAtomFiles']]
    for lane in ['candidate','full']:
        seal=read(p/(lane+'-seal.json'));enc=read(p/('encoder-'+lane+'.json'));im=read(p/('images-'+lane+'.json'))
        for key,name in [('readerSHA256','reader-seal.json'),('encoderSHA256','encoder-'+lane+'.json'),('imagesSHA256','images-'+lane+'.json'),('noiseSealSHA256','noise-seal.json'),('clockTableSHA256','clock-table.json')]:
            if seal[key]!=sha(p/name):raise ValueError('Consumer seal binding differs')
        if enc['readerSealSHA256']!=sha(p/'reader-seal.json') or im['encoderSHA256']!=sha(p/('encoder-'+lane+'.json')):raise ValueError('Actual reader/encoder/image join differs')
        if lane=='full' and (enc['referenceReleasedAfterSealSHA256']!=sha(p/'candidate-seal.json') or im['referenceReleasedAfterSealSHA256']!=sha(p/'candidate-seal.json')):raise ValueError('Reference release differs')
        for group in [enc['fields'],im['fields']]:
            for fs in group.values():items+=list(fs.values())
        items+=list(im['images'].values())
    if report['candidateSealSHA256']!=sha(p/'candidate-seal.json') or report['fullSealSHA256']!=sha(p/'full-seal.json'):raise ValueError('Report seal differs')
    noise=read(p/'noise-seal.json');items += [noise[k] for k in ['raw','packed','ids']]+list(read(p/'responses.json')['fields'].values())
    for row in table['rows']:items += [*row['clockOperands'].values(),*row['fields'].values()]
    items += [{'path':base+'/'+n,'sha256':sha(p/n)} for n in ['candidate-seal.json','full-seal.json','reader-seal.json','encoder-candidate.json','images-candidate.json','encoder-full.json','images-full.json','noise-seal.json','clock-table.json']]
    items += [{'path':base+'/reader/prediction-seal.json','sha256':reader['readerPredictionSealSHA256']},
        {'path':base+'/lexical/prediction-seal.json','sha256':reader['lexicalSealSHA256']},
        {'path':base+'/serve-source.json','sha256':reader['reader']['specSHA256']}]
    items += [{'path':base+'/response-result/'+f.name,'sha256':sha(f)} for f in p.iterdir()]
    spec={'allowedRoots':[base,source['lexicalStore'],source['clockParentRoot']],'items':items}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],input=json.dumps(spec),text=True,capture_output=True,check=True,timeout=180)
    value=json.loads(proc.stdout);value.update(schema='tce-fresh-square-bytes-v1',jobID=j['job_id'],verifierSourceSHA256=sha(__file__),candidateSealSHA256=sha(p/'candidate-seal.json'),fullSealSHA256=sha(p/'full-seal.json'),
        scope='Complete used lexical atoms, frozen own reader packets/states/heads/H0, all4candidate and4fresh full-attention encoder/image trajectories, original PNGs, complete signed edge/mixed response arrays, clocks and fresh noise. No bulk local hydration.')
    save(root/'retained-array-verification.json',value);return {'files':len(value['files']),'bytes':value['totalBytesVerified']}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['fullSealSHA256']!=sha(p/'full-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Independent verification identity differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for f in p.iterdir():
        if f.suffix=='.jsonl':
            data=f.read_bytes();target=out/(f.name+'.gz');target.write_bytes(gzip.compress(data,mtime=0))
            if gzip.decompress(target.read_bytes())!=data:raise ValueError('Lossless stream differs')
            streams.append({'source':f.name,'sourceSHA256':sha(f),'sourceBytes':len(data),'published':target.name,'publishedSHA256':sha(target),'publishedBytes':target.stat().st_size})
        else:
            if f.suffix not in ['.json','.png','.md']:raise ValueError('Unexpected compact member')
            shutil.copyfile(f,out/f.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-fresh-square-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],
        'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreams':streams,
        'scope':'Fresh fixed prompt square with own four candidate images sealed before four new full-attention references. Full learned encoder/FLUX/VAE and exact compiled clocks remain paid; no arbitrary-prompt or zero-neural certificate.'})
    save(out/'files.json',{f.name:sha(f) for f in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(f.stat().st_size for f in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args();print(json.dumps(verify(a.run) if a.action=='verify' else publish(a.run,a.destination)))
