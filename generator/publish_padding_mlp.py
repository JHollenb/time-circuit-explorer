"""Complete byte custody for the source-pad tangent and own image consumers."""
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
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed source changed: '+n)
    return root,c,j,r


def verify(root):
    root,c,j,r=checked(root);p=root/'result';source=read(p/'inputs.json');reader=read(p/'reader-seal.json');enc=read(p/'encoder-candidate.json');im=read(p/'images-candidate.json');target=read(p/'targets.json');compiled=read(p/'compile-receipt.json');diag=read(p/'native-write-diagnostic.json');seal=read(p/'candidate-seal.json');base=j['result']['diagnostics']['cwd']
    for key,n in [('compileReceiptSHA256','compile-receipt.json'),('readerSHA256','reader-seal.json'),('encoderSHA256','encoder-candidate.json'),('imagesSHA256','images-candidate.json'),('clockTableSHA256','clock-table.json'),('noiseSealSHA256','noise-seal.json')]:
        if seal[key]!=sha(p/n):raise ValueError('Seal chain differs')
    if diag['releasedAfterCandidateSealSHA256']!=sha(p/'candidate-seal.json') or enc['compileReceiptSHA256']!=sha(p/'compile-receipt.json'):raise ValueError('Compiler/diagnostic release differs')
    items=[*reader['fields'].values(),*reader['h0Fields'].values(),*reader['lexicalRows'],*reader['packets'],*reader['states'],*reader['sourceAtomFiles'],*compiled['fields'].values(),*compiled['sourceAtomFiles']]
    for group in [enc['fields'],im['fields'],enc['firstMLPFields'],diag['fields']]:
        for fs in group.values():items+=list(fs.values())
    for role in ['top4Encoder','fullEncoder','top4Images','fullImages']:
        for c in ['00','10']:items+=list(target[role]['fields'][c].values())
        if role.endswith('Images'):items += [target[role]['images'][c] for c in ['00','10']]
    items += [*im['images'].values(),*read(p/'differences.json').values(),*target['sourceFiles'].values()]
    noise=read(p/'noise-seal.json');items += [noise[k] for k in ['raw','packed','ids']]
    for row in read(p/'clock-table.json')['rows']:items += [*row['clockOperands'].values(),*row['fields'].values()]
    items += [{'path':base+'/'+n,'sha256':sha(p/n)} for n in ['compile-receipt.json','reader-seal.json','encoder-candidate.json','images-candidate.json','candidate-seal.json','native-write-diagnostic.json','clock-table.json','noise-seal.json']]
    items += [{'path':base+'/lexical/prediction-seal.json','sha256':reader['lexicalSealSHA256']},{'path':base+'/reader/prediction-seal.json','sha256':reader['readerPredictionSealSHA256']},{'path':base+'/serve-source.json','sha256':reader['reader']['specSHA256']}]
    items += [{'path':base+'/response-result/'+f.name,'sha256':sha(f)} for f in p.iterdir()]
    spec={'allowedRoots':[base,source['parentRoot'],source['clockParentRoot'],source['lexicalStore']],'items':items}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],input=json.dumps(spec),text=True,capture_output=True,check=True,timeout=180)
    v=json.loads(proc.stdout);v.update(schema='tce-padding-mlp-bytes-v1',jobID=j['job_id'],verifierSourceSHA256=sha(__file__),candidateSealSHA256=sha(p/'candidate-seal.json'),
        scope='Full compiledJ/anchor/emission-error arrays, used atoms, own raw reader/firstMLP/H/E/image arrays and original PNGs, paid postseal native write diagnostic, saved top4/full comparison trajectories, differences/clocks/noise/source records. Full arrays stay near compute.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Byte verification differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for f in p.iterdir():
        if f.suffix=='.jsonl':
            data=f.read_bytes();to=out/(f.name+'.gz');to.write_bytes(gzip.compress(data,mtime=0))
            if gzip.decompress(to.read_bytes())!=data:raise ValueError('Lossless stream differs')
            streams.append({'source':f.name,'sourceSHA256':sha(f),'sourceBytes':len(data),'published':to.name,'publishedSHA256':sha(to),'publishedBytes':to.stat().st_size})
        else:
            if f.suffix not in ['.json','.png','.md']:raise ValueError('Unexpected compact artifact')
            shutil.copyfile(f,out/f.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-padding-mlp-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreams':streams,
        'scope':'Source-pad tangent/constant firstMLP on two known rawstrings, four new own images. Original activeMLP/later encoder/denoiser/VAE and learnedJ remain paid. Original diagnostic and saved comparison targets open only after complete candidate seal. No strictzero-neural or speedup claim.'})
    save(out/'files.json',{f.name:sha(f) for f in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(f.stat().st_size for f in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args();print(json.dumps(verify(a.run) if a.action=='verify' else publish(a.run,a.destination)))
