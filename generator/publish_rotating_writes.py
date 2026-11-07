"""Byte custody and lossless compact rotating-write publication."""
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
    root,c,j,r=checked(root);p=root/'result';s=read(p/'inputs.json');im=read(p/'images.json');t=read(p/'targets.json')
    if read(p/'report.json')['candidateSealSHA256']!=sha(p/'candidate-seal.json') or read(p/'candidate-seal.json')['imagesSHA256']!=sha(p/'images.json'):raise ValueError('Candidate release differs')
    items=[s['ids'],*im['images'].values(),*im['assets'],*t['sourceFiles'].values()]
    for group in [s['prefix'],im['common'],im['fields'],read(p/'diagnostic-fields.json')]:
        for fs in group.values():items+=list(fs.values())
    for row in s['clockTable']['rows']:items += [*row['fields'].values(),*row['clockOperands'].values()]
    for ref in ['HN','HH','NN','NH','full']:
        for fs in t[ref]['fields'].values():items += [fs[n] for n in ['velocity3','latent4','pngRGB']]
        items+=list(t[ref]['images'].values())
    for fs in t['HN']['fields'].values():items.append(fs['latent3'])
    base=j['result']['diagnostics']['cwd'];items += [{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()]
    roots=[base,s['routingParentRoot'],s['nativeParentRoot'],s['holdParentRoot'],s['parentRoot'],s['clockParentRoot'],s['modelRoot']]
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],input=json.dumps({'allowedRoots':roots,'items':items}),text=True,capture_output=True,check=True,timeout=180)
    v=json.loads(proc.stdout);v.update(schema='tce-rotating-write-bytes-v1',jobID=j['job_id'],candidateSealSHA256=sha(p/'candidate-seal.json'),verifierSourceSHA256=sha(__file__),
        scope='Complete ownZ3H/E/IDs, actualfullconditioning/velocity/imagefields, fullsignedvelocity/latent/finitevectors, consumedhistoricaltargets/PNGs, clocks and originalresidentimageweights. Fullarrays/weights remainnearcompute; blockhandoffs aredigestwitnesses.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Byte verification differs')
    out=Path(destination);out.mkdir(exist_ok=False,parents=True);streams=[]
    for x in p.iterdir():
        if x.suffix=='.jsonl':
            raw=x.read_bytes();to=out/(x.name+'.gz');to.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(to.read_bytes())!=raw:raise ValueError('Lossless stream differs')
            streams.append({'source':x.name,'sourceSHA256':sha(x),'sourceBytes':len(raw),'published':to.name,'publishedSHA256':sha(to),'publishedBytes':to.stat().st_size})
        else:shutil.copyfile(x,out/x.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-rotating-write-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],'schedulerResult':c['result'],
        'completeCollectedInventory':c['inventory'],'losslessStreams':streams,'scope':'Twoowncandidateimages/fullreadablestate/fixedrotatingwrites,2paidrouteddenoisers/2VAE+4historicalprefixdenoisers/encoding. No arbitrarypromptzero-neural,heldoutquality or matchedspeedupclaim.'})
    save(out/'files.json',{x.name:sha(x) for x in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args();print(json.dumps(verify(a.run) if a.action=='verify' else publish(a.run,a.destination)))
