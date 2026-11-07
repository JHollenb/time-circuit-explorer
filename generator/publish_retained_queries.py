"""Complete byte custody and lossless compact retained-query evidence."""
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
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed source changed')
    return root,c,j,r


def verify(root):
    root,c,j,r=checked(root);p=root/'result';base=j['result']['diagnostics']['cwd'];s=read(p/'inputs.json');e=read(p/'encoder-candidate.json');im=read(p/'images-candidate.json');lex=read(p/'lexical-seal.json');seal=read(p/'candidate-seal.json');target=read(p/'targets.json')
    for k,n in [('encoderSHA256','encoder-candidate.json'),('imagesSHA256','images-candidate.json'),('lexicalSealSHA256','lexical-seal.json'),('clockTableSHA256','clock-table.json'),('noiseSealSHA256','noise-seal.json')]:
        if seal[k]!=sha(p/n):raise ValueError('Complete release chain differs')
    if read(p/'report.json')['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Score release differs')
    items=[*lex['rows'],*lex['sourceAtomFiles'],*im['images'].values(),*target['sourceFiles'].values()]
    for group in [e['fields'],e['compactFields'],e['setupFields'],im['fields'],read(p/'diagnostic-fields.json')]:
        for fs in group.values():items+=list(fs.values())
    for role in ['fullEncoder','top4Encoder','fullImages','top4Images']:
        for fs in target[role]['fields'].values():items+=list(fs.values())
        if role.endswith('Images'):items+=list(target[role]['images'].values())
    noise=read(p/'noise-seal.json');items += [noise[k] for k in ['raw','packed','ids']]
    for row in read(p/'clock-table.json')['rows']:items += [*row['fields'].values(),*row['clockOperands'].values()]
    items += [{'path':base+'/lexical/prediction-seal.json','sha256':sha(p/'lexical-seal.json')}]
    items += [{'path':base+'/'+n,'sha256':sha(p/n)} for n in ['encoder-candidate.json','images-candidate.json','candidate-seal.json','clock-table.json','noise-seal.json']]
    items += [{'path':base+'/response-result/'+f.name,'sha256':sha(f)} for f in p.iterdir()]
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots':[base,s['parentRoot'],s['clockParentRoot'],s['lexicalStore']],'items':items}),text=True,capture_output=True,check=True,timeout=180)
    v=json.loads(proc.stdout);v.update(schema='tce-retained-query-byte-verification-v1',jobID=j['job_id'],candidateSealSHA256=sha(p/'candidate-seal.json'),verifierSourceSHA256=sha(__file__),
        scope='Complete own compactH0..H27/E, original/sliced visibility-position-RoPE witnesses, expanded conditioning, nativeimage trajectories, all signed diagnostic arrays and original PNGs, historical full/top4 comparators, source atoms/noise/clocks and compact result bytes. Arrays stay near compute.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Byte verification differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for f in p.iterdir():
        if f.suffix=='.jsonl':
            data=f.read_bytes();to=out/(f.name+'.gz');to.write_bytes(gzip.compress(data,mtime=0))
            if gzip.decompress(to.read_bytes())!=data:raise ValueError('Observation transport differs')
            streams.append({'source':f.name,'sourceSHA256':sha(f),'sourceBytes':len(data),'published':to.name,'publishedSHA256':sha(to),'publishedBytes':to.stat().st_size})
        else:
            if f.suffix not in ['.json','.png','.md']:raise ValueError('Unexpected compact artifact')
            shutil.copyfile(f,out/f.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-retained-query-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],'schedulerResult':c['result'],
        'completeCollectedInventory':c['inventory'],'losslessStreams':streams,'scope':'Two known raw strings, native nonlinear compact queries and full demanded conditioning/image consumer. Original weights/work remain paid; no new reference campaign, fit, matched-speedup or strictzero-neural claim.'})
    save(out/'files.json',{f.name:sha(f) for f in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(f.stat().st_size for f in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args();print(json.dumps(verify(a.run) if a.action=='verify' else publish(a.run,a.destination)))
