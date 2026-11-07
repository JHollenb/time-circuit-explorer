"""Full-byte verification and compact lossless publication for last-velocity replay."""
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
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:
            raise ValueError('Executed scientific source changed')
    return root,c,j,r


def verify(root):
    root,c,j,r=checked(root);p=root/'result';s=read(p/'inputs.json');f=read(p/'forecast-seal.json');im=read(p/'images.json');t=read(p/'targets.json')
    seal=read(p/'candidate-seal.json')
    if seal['forecastSealSHA256']!=sha(p/'forecast-seal.json') or seal['imagesSHA256']!=sha(p/'images.json') or read(p/'report.json')['candidateSealSHA256']!=sha(p/'candidate-seal.json'):
        raise ValueError('Complete release chain differs')
    items=[s['ids'],*im['images'].values(),*t['sourceFiles'].values(),*im['assets']]
    for group in [s['prefix'],f['fields'],im['fields'],read(p/'diagnostic-fields.json')]:
        for fs in group.values():items+=list(fs.values())
    for ref in ['own','full']:
        for fs in t[ref]['fields'].values():
            for port in (['velocity3','latent4','vaeInput','decodedRaw','rgbFloat','pngRGB'] if ref=='own' else ['latent4','pngRGB']):
                items.append(fs[port])
        items+=list(t[ref]['images'].values())
    base=j['result']['diagnostics']['cwd']
    items += [{'path':base+'/'+n,'sha256':sha(p/n)} for n in ['forecast-seal.json','images.json','candidate-seal.json']]
    items += [{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()]
    roots={base,s['parentRoot'],s['modelRoot']}
    roots.update(str(Path(x['path']).parents[2]) for fs in t['full']['fields'].values() for x in fs.values())
    roots.update(str(Path(x['path']).parent) for x in t['full']['images'].values())
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots':sorted(roots),'items':items}),text=True,capture_output=True,check=True,timeout=180)
    v=json.loads(proc.stdout);v.update(schema='tce-last-velocity-byte-verification-v1',jobID=j['job_id'],
        candidateSealSHA256=sha(p/'candidate-seal.json'),verifierSourceSHA256=sha(__file__),
        scope='Complete allowed own-prefix operands, raw/BF16 forecasts, candidate scheduler/VAE/image fields, signed finite defects, consumed historical comparisons/PNGs, decoding assets and compact result bytes. Arrays remain near compute.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['verifierSourceSHA256']!=sha(__file__):
        raise ValueError('Byte verification differs')
    out=Path(destination);out.mkdir(exist_ok=False,parents=True);streams=[]
    for x in p.iterdir():
        if x.suffix=='.jsonl':
            data=x.read_bytes();to=out/(x.name+'.gz');to.write_bytes(gzip.compress(data,mtime=0))
            if gzip.decompress(to.read_bytes())!=data:raise ValueError('Lossless stream differs')
            streams.append({'source':x.name,'sourceSHA256':sha(x),'sourceBytes':len(data),'published':to.name,
                'publishedSHA256':sha(to),'publishedBytes':to.stat().st_size})
        else:shutil.copyfile(x,out/x.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-last-velocity-publication-v1','jobID':j['job_id'],
        'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],'schedulerResult':c['result'],
        'completeCollectedInventory':c['inventory'],'losslessStreams':streams,
        'scope':'Own-prefix component replay; six original denoisers remain charged. Four new paid VAE decodes. No fresh raw-prompt or strictzero-neural generator.'})
    files={x.name:sha(x) for x in out.iterdir()};save(out/'files.json',files)
    return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(verify(a.run) if a.action=='verify' else publish(a.run,a.destination)))
