"""Frozen retained-query encoder→two own images, postseal full-target diagnosis."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import np,read,save,sha,identity
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['retained_query_study.py','retained_query_encoder.py','retained_query_images.py','retained_queries.py','encoder_atoms.py','paired_fields.py',
    'padding_mlp.py','masked_kv_study.py','conditioning_demand.py','complete_masked_study.py','flux_clock.py','flux_clock_study.py']
ENCODER_PORTS=['H9','H18','H27','E']
IMAGE_PORTS=['C'+str(i) for i in range(4)]+['velocity'+str(i) for i in range(4)]+['latent'+str(i) for i in range(1,5)]+['vaeInput','decodedRaw','rgbFloat','pngRGB']


def install_guard(root):
    from flux_clock import deny_targets
    root=Path(root);s=read(root/'inputs.json');deny_targets([root/'targets.json',*s['forbiddenTargets']])
    try:open(root/'targets.json','rb')
    except PermissionError:return
    raise RuntimeError('Target value guard inactive')


def inherited_clock(root,source):
    p=Path(root)/'clock-table.json'
    if sha(p)!=source['clockTableSHA256']:raise ValueError('Clock differs')
    value=read(p)
    if value['requestSHA256']!=source['clockParentRequestSHA256']:raise ValueError('Clock parent differs')
    return value


def stage(root):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    parent=HERE.parent/'artifacts/fresh-prompt-square-v1';c,j,old=checked_collection(parent,current_workers=False);api=Api();live=api.json('GET','/api/jobs/'+j['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody']:raise ValueError('Historical square differs')
    f=parent/'result';source=read(f/'inputs.json');root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for n in OWNED:
        if n in old['sourceHashes'] and sha(HERE/n)!=old['sourceHashes'][n]:raise ValueError('Qualified shared source changed')
        shutil.copyfile(HERE/n,p/n)
    for n in ['flux_functional_qwen_prefix.py','qwen_h18_state_history.py']:
        if sha(parent/'payload'/n)!=old['sourceHashes'][n]:raise ValueError('Qualified graph differs')
        shutil.copyfile(parent/'payload'/n,p/n)
    for n in ['noise-seal.json','clock-table.json']:shutil.copyfile(f/n,p/n)
    shutil.copyfile(HERE.parent/'docs/plans/retained-padding-queries.md',p/'frozen-plan.md')
    source.update(parentRoot=j['result']['diagnostics']['cwd'],parentRequestSHA256=old['requestSHA256'],noiseSealSHA256=sha(f/'noise-seal.json'),noiseParentRequestSHA256=old['requestSHA256'])
    target={'sourceFiles':{n:{'path':source['parentRoot']+'/'+n,'sha256':sha(f/n)} for n in ['encoder-full.json','images-full.json','full-seal.json','encoder-candidate.json','images-candidate.json','candidate-seal.json']}}
    for lane,label in [('full','full'),('candidate','top4')]:
        for kind in ['encoder','images']:
            original=read(f/(kind+'-'+lane+'.json'));target[label+kind.title()]={'requestSHA256':original['requestSHA256'],'fields':{k:v for k,v in original['fields'].items() if k in ['00','10']}}
            source['forbiddenTargets'] += [x['path'] for fs in original['fields'].values() for x in fs.values()]
            if kind=='images':
                target[label+kind.title()]['images']={k:v for k,v in original['images'].items() if k in ['00','10']};source['forbiddenTargets'] += [x['path'] for x in original['images'].values()]
    source['forbiddenTargets']=sorted(set(source['forbiddenTargets']));save(p/'inputs.json',source);save(p/'targets.json',target)
    save(p/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredParentJobID':j['job_id'],'measuredParentResult':c['result'],'measuredRecentJobID':'job-922839f9980e',
        'exception':'Generic diffusionB64/fallback plan inapplicable. Inherited serialB1 CUDA envelope14336RAM/13000VRAM, measured8850MBVRAM. CPU1encoder/4images separatechildren; no co-resident models. 53queryrows,2images,150sexpected600scap,noretries.'})
    r={'schema':'tce-retained-query-request-v1','prompts':[q for q in old['prompts'] if q['id'] in ['00','10']],'seed':old['seed'],'paddingQueries':16,'manifestSHA256':old['manifestSHA256'],
        'selection':'Allactive + P[floor(j*(len(P)-1)/15)] j0..15, allpaddingif<=16. Nearest original padding position, ties earlier; no embedding scaling.',
        'parentRequestSHA256':old['requestSHA256'],'ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'model':'flux2-klein-4b-distilled','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'steps':4,'timeoutSeconds':600,
        'protocol':'Own compiled lexicalH0; native attention/MLP all27blocks at retained originalpositions/mask/RoPE. Expand onlyH9/H18/H27/E. Both images seal before savedfull/top4target diagnosis. No fit/sweep/newnoise/refimages/refresh.',
        'scope':'Structural querytrajectory reduction with original learned computation on retained rows. Copying preserves rowcount/multiplicity, not attention mass. No speedup or strictzero-neural claim.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r);save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from run import command
    from mrun.client.submit import launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted')
    cmd=command()[:-5]+['python','-u','retained_query_study.py','execute','.'];i=cmd.index('uv');cmd[i:i]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=launch(cmd,experiment='tce-retained-query-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs={'cuda':True},config={k:r[k] for k in ['schema','requestSHA256','model','backend','device','dtype','batch_size','seq_len','height','width']},
        ram_mb=14336,vram_mb=13000,cpu_threads=4,disk_gb=2,est_wall_s=150,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='00/10 all37active+16fixedpadding=53originalqueries. Nativeall27blocks/378linears/54SDPA on ownstates, originalmask/RoPE/positions. Nearest expand H9/H18/H27/E->2originalimages/8denoisers/2VAE. Bothseal before savedfull/top4 scores and signed finite/reconstruction telescope. No newrefs/noise/fit/rescue. CPU1encoder/4image serialchildren B1 measured14336RAM13000VRAM exception vsbadB64plan.150sexpected600scap,noretries. Learnedworkstillpaid.')
    v={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',v);return v


def lexical(root):
    from encoder_atoms import Prefix,guard
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json');install_guard(root);guard();start=time.perf_counter();prefix=Prefix(s['lexicalStore'],r['manifestSHA256']);out=root/'lexical';out.mkdir();rows=[]
    for q in r['prompts']:
        tick=time.perf_counter();text,ids,mask=prefix.tokenize(q['prompt']);atoms=prefix.store.gather(ids);e=atoms[:,:prefix.m['hiddenSize']].copy();path=out/(q['id']+'.npz');np.savez(path,ids=ids,mask=mask,positions=np.arange(len(ids)),e=e)
        rows.append({'id':q['id'],'prompt':q['prompt'],'path':str(path),'sha256':sha(path),'activeRows':int(mask.sum()),'renderedTextSHA256':hashlib.sha256(text.encode()).hexdigest(),'wallSeconds':time.perf_counter()-tick})
    m=prefix.store.manifest;files=[{'path':str(Path(s['lexicalStore'])/'manifest.json'),'sha256':r['manifestSHA256']}]
    for k in ['coverage','rowHashes','tokenizer','template','rope']:files.append({'path':str(Path(s['lexicalStore'])/m[k]['path']),'sha256':m[k]['sha256']})
    for j in sorted(prefix.store.shards):files.append({'path':str(Path(s['lexicalStore'])/m['shards'][j]['path']),'sha256':m['shards'][j]['sha256']})
    save(out/'prediction-seal.json',{'requestSHA256':r['requestSHA256'],'manifestSHA256':r['manifestSHA256'],'rows':rows,'sourceAtomFiles':files,'coverageCount':len(prefix.store.ids),'uniqueRowsRead':prefix.store.rowsRead,'compiledRowPayloadBytesRead':prefix.store.rowBytesRead,
        'frameworkCalls':0,'originalDenseProjections':0,'workerSecondsAfterImports':time.perf_counter()-start,'scope':'Raw tokenize+complete learned atom-row gather, retain embedding only. No contextual read or QKV arithmetic. Source table stores additional unused lexical operands; bytes remain charged.'})


def seal(root):
    from paired_fields import load
    root=Path(root);r=read(root/'request.json');enc=read(root/'encoder-candidate.json');im=read(root/'images-candidate.json');ids=[q['id'] for q in r['prompts']]
    if set(enc['fields'])!=set(ids) or set(im['fields'])!=set(ids):raise ValueError('Candidate coverage differs')
    for c in ids:
        if set(enc['fields'][c])!=set(ENCODER_PORTS) or set(im['fields'][c])!=set(IMAGE_PORTS) or set(enc['compactFields'][c])!={'H'+str(i) for i in range(28)}|{'E'}:raise ValueError('Candidate boundary coverage differs')
        for group in [enc['fields'],enc['compactFields'],enc['setupFields'],im['fields']]:
            for item in group[c].values():load(item)
        if im['actualEFields'][c]!=enc['fields'][c]['E'] or sha(im['images'][c]['path'])!=im['images'][c]['sha256']:raise ValueError('Actual image input/output differs')
    save(root/'candidate-seal.json',{'schema':'tce-retained-query-seal-v1','requestSHA256':r['requestSHA256'],'encoderSHA256':sha(root/'encoder-candidate.json'),'imagesSHA256':sha(root/'images-candidate.json'),
        'lexicalSealSHA256':sha(root/'lexical/prediction-seal.json'),'clockTableSHA256':sha(root/'clock-table.json'),'noiseSealSHA256':sha(root/'noise-seal.json'),'encoderFields':enc['fields'],'compactFields':enc['compactFields'],'imageFields':im['fields'],'images':im['images']})


def score(root,out,phases,started):
    import json
    from paired_fields import load,retain,metrics,paired
    from retained_queries import error_terms,energy_account
    r=read(root/'request.json');e=read(root/'encoder-candidate.json');im=read(root/'images-candidate.json');target=read(root/'targets.json');seal_sha=sha(root/'candidate-seal.json');accounts=[];responses=[];decompositions=[];terms={};retained={};observations=0
    for item in target['sourceFiles'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Historical source record differs')
    with (out/'observations.jsonl').open('x') as stream:
        for q in r['prompts']:
            c=q['id'];setup=e['setupFields'][c];selected=load(setup['selected']).astype(np.int64);expansion=load(setup['expansion']).astype(np.int64);mask=load(setup['activeMask']).astype(bool);regions={'all':np.ones(512,dtype=bool),'active':mask,'padding':~mask}
            for port,item in e['compactFields'][c].items():
                a=load(item).reshape(len(selected),-1);b=load(target['fullEncoder']['fields'][c][port]).reshape(512,-1)[selected]
                for region,use in [('retained',np.ones(len(selected),dtype=bool)),('active',mask[selected]),('retained-padding',~mask[selected])]:accounts.append({'case':c,'port':port,'region':region,'reference':'full-selected',**metrics(a[use],b[use])})
                for j,(ar,br) in enumerate(zip(a,b)):stream.write(json.dumps({'case':c,'port':port,'position':int(selected[j]),'reference':'full-selected','attended':bool(mask[selected[j]]),**metrics(ar,br)},allow_nan=False)+'\n');observations+=1
            for port in ENCODER_PORTS+IMAGE_PORTS:
                is_enc=port in ENCODER_PORTS;a=load((e if is_enc else im)['fields'][c][port])
                for ref in ['full','top4']:
                    b=load(target[ref+('Encoder' if is_enc else 'Images')]['fields'][c][port]);accounts.append({'case':c,'port':port,'region':'all','reference':ref,**metrics(a,b)})
                    if is_enc:
                        for region in ['active','padding']:accounts.append({'case':c,'port':port,'region':region,'reference':ref,**metrics(a.reshape(512,-1)[regions[region]],b.reshape(512,-1)[regions[region]])})
                    if is_enc or port in ['pngRGB','rgbFloat']:
                        for j,(ar,br) in enumerate(zip(a.reshape(512,-1),b.reshape(512,-1))):stream.write(json.dumps({'case':c,'port':port,'position':j,'reference':ref,'attended':bool(mask[j]) if is_enc else None,**metrics(ar,br)},allow_nan=False)+'\n');observations+=1
            small=load(e['compactFields'][c]['E']).reshape(len(selected),-1);full=load(target['fullEncoder']['fields'][c]['E']).reshape(512,-1);f,g,total=error_terms(small,full,selected,expansion);terms[c]=(f,g,total)
            if not np.array_equal(total,load(e['fields'][c]['E']).reshape(512,-1)-full):raise ValueError('Telescope received another candidate E')
            retained[c]={n:retain(root/'arrays/diagnostic'/c,n,v) for n,v in [('finiteExecution',f),('reconstruction',g),('totalError',total)]}
            for region,use in regions.items():decompositions.append({'case':c,'region':region,**energy_account(f[use],g[use],total[use])})
            for j in range(512):stream.write(json.dumps({'case':c,'port':'E-decomposition','position':j,'attended':bool(mask[j]),**energy_account(f[j],g[j],total[j])},allow_nan=False)+'\n');observations+=1
        for port in ENCODER_PORTS+['velocity0','latent4','pngRGB']:
            is_enc=port in ENCODER_PORTS;fs=(e if is_enc else im)['fields'];a,b=[load(fs[c][port]) for c in ['00','10']]
            for ref in ['full','top4']:
                targets=target[ref+('Encoder' if is_enc else 'Images')]['fields'];ra,rb=[load(targets[c][port]) for c in ['00','10']]
                responses.append({'port':port,'reference':ref,'region':'all',**paired(a,b,ra,rb)})
                if is_enc:
                    for region,use in [('active',mask),('padding',~mask)]:responses.append({'port':port,'reference':ref,'region':region,**paired(a.reshape(512,-1)[use],b.reshape(512,-1)[use],ra.reshape(512,-1)[use],rb.reshape(512,-1)[use])})
    paired_terms=[b-a for a,b in zip(terms['00'],terms['10'])];color_account=energy_account(*paired_terms)
    if sha(root/'candidate-seal.json')!=seal_sha:raise ValueError('Candidate seal changed')
    for c in ['00','10']:
        shutil.copyfile(im['images'][c]['path'],out/('candidate-'+c+'.png'))
        for ref in ['full','top4']:
            item=target[ref+'Images']['images'][c]
            if sha(item['path'])!=item['sha256']:raise ValueError('Historical PNG differs')
            shutil.copyfile(item['path'],out/(ref+'-'+c+'.png'))
    for n in ['request.json','inputs.json','targets.json','resource-basis.json','frozen-plan.md','clock-table.json','noise-seal.json','encoder-candidate.json','images-candidate.json','candidate-seal.json']:shutil.copyfile(root/n,out/n)
    shutil.copyfile(root/'lexical/prediction-seal.json',out/'lexical-seal.json');save(out/'diagnostic-fields.json',retained);calls=e['primitives'];linears=im['linears']
    save(out/'report.json',{'schema':'tce-retained-query-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal_sha,'accounts':accounts,'pairedColorResponse':responses,'conditioningDecomposition':decompositions,'pairedConditioningDecomposition':color_account,
        'observations':observations,'phases':phases,'workerSecondsAfterImports':time.perf_counter()-started,'work':{'nativeEncoderBlocks':54,'encoderSDPA':sum(x['kind']=='SDPA' for x in calls),'encoderLinears':sum(x['kind']=='linear' for x in calls),
        'encoderNominalLinearMACs':sum(x['nominalMACs'] for x in calls if x['kind']=='linear'),'encoderAttentionNominalQKAVMACs':sum(2*x['shapes'][0][1]*x['shapes'][0][2]*x['shapes'][1][2]*x['shapes'][0][3] for x in calls if x['kind']=='SDPA'),
        'nativeDenoisers':sum(x['denoiser'] for x in im['counts'].values()),'nativeVAE':sum(x['VAE'] for x in im['counts'].values()),'nativeTransformerLinears':sum(x['parameter'].startswith('transformer.') for x in linears),
        'nativeTransformerNominalLinearMACs':sum(x['nominalMACs'] for x in linears if x['parameter'].startswith('transformer.')),'newImages':2,'newReferenceImages':0,'newNoiseDraws':0,'postsealOriginalModelCalls':0},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed','limitations':['Two known prompts/one noise; no independent arbitrary-prompt quality or speedup certificate.','Original learned weights/native nonlinear work remain on selected queries and complete FLUX/VAE. Not strict zero-neural.','Finite compact GEMM/SDPA and propagated effects measured separately from fixed demanded-row reconstruction. Rowcopy preserves multiplicity, not softmax mass.']})


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);started=time.perf_counter();out=root/'response-result';out.mkdir();phases=[]
    for action in ['lexical','encoder','images']:
        tick=time.perf_counter();subprocess.run([sys.executable,'-u',str(root/'retained_query_study.py'),action,str(root)],check=True,timeout=480);phases.append({'action':action,'wallSecondsIncludingFreshProcess':time.perf_counter()-tick})
    seal(root);print('RETAINED_QUERY_CANDIDATES_SEALED',sha(root/'candidate-seal.json'),flush=True);score(root,out,phases,started);print('RETAINED_QUERY_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','lexical','encoder','images','collect']);p.add_argument('run');a=p.parse_args()
    if a.action in ['encoder','images']:v=__import__('retained_query_'+a.action).execute(a.run)
    elif a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(__import__('json').dumps(v))
