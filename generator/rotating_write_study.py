"""Stage/run/collect one source-fixed complete-state rotating-write assay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import np,read,save,sha,identity
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['rotating_write_study.py','rotating_write_images.py','rotating_writes.py','two_clock.py','two_clock_carry.py',
    'last_velocity_decode.py','encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py',
    'complete_masked_study.py','masked_kv_study.py','retained_queries.py']
PORTS=['velocity3','latent4','vaeInput','decodedRaw','rgbFloat','pngRGB']


def stage(root):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    parent=HERE.parent/'artifacts/two-clock-v1';c,j,old=checked_collection(parent,current_workers=False);api=Api()
    live=api.json('GET','/api/jobs/'+j['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody']:raise ValueError('Own-state parent differs')
    p0=parent/'result';s=read(p0/'inputs.json');im=read(p0/'images.json');t=read(p0/'targets.json');report=read(p0/'report.json')
    if report['candidateSealSHA256']!=sha(p0/'candidate-seal.json'):raise ValueError('Parent release differs')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for n in OWNED:
        if n in old['ownedSources'] and sha(HERE/n)!=old['sourceHashes'][n]:raise ValueError('Qualified helper changed')
        shutil.copyfile(HERE/n,p/n)
    shutil.copyfile(HERE.parent/'docs/plans/rotating-residual-writes.md',p/'frozen-plan.md')
    prefix={case:{'latent3':im['common'][case]['latent3'],'E':s['prefix'][case]['E']} for case in ['00','10']}
    targets={ref:{'fields':{case:im['fields'][ref+'-'+case] for case in ['00','10']},'images':{case:im['images'][ref+'-'+case] for case in ['00','10']}} for ref in ['HN','HH']}
    targets.update({ref:t[ref] for ref in ['NN','NH','full']})
    base=j['result']['diagnostics']['cwd'];targets['sourceFiles']={n:{'path':base+'/response-result/'+n,'sha256':sha(p0/n)} for n in ['images.json','targets.json','candidate-seal.json','report.json']}
    forbidden=set(s['forbiddenTargets'])
    for ref in ['HN','HH','NN','NH','full']:
        forbidden.update(x['path'] for fs in targets[ref]['fields'].values() for x in fs.values());forbidden.update(x['path'] for x in targets[ref]['images'].values())
    allowed={x['path'] for fs in prefix.values() for x in fs.values()}|{s['ids']['path']}
    source=dict(s);source.update(prefix=prefix,routingParentRoot=base,parentRequestSHA256=old['requestSHA256'],parentReportSHA256=sha(p0/'report.json'),
        parentSealSHA256=sha(p0/'candidate-seal.json'),targetRoots=sorted(set(s['targetRoots'])|{base}),forbiddenTargets=sorted(forbidden-allowed),guardCanary=im['fields']['HN-00']['velocity3']['path'])
    save(p/'inputs.json',source);save(p/'targets.json',targets)
    from fractions import Fraction
    full=[x for x in im['linears'] if x['case']=='00' and x['parameter'].startswith('transformer.')]
    routed=Fraction(0)
    for x in full:
        n=x['parameter'];factor=Fraction(1)
        if '.single_transformer_blocks.' in n:factor=Fraction(11,18) if n.endswith('to_qkv_mlp_proj.weight') else Fraction(1,2)
        elif '.transformer_blocks.' in n and any(n.endswith(t) for t in ['attn.to_q.weight','attn.to_out.0.weight','ff.linear_in.weight','ff.linear_out.weight']):factor=Fraction(1,4)
        routed+=x['nominalMACs']*factor
    if len(full)!=103 or routed.denominator!=1 or routed!=2730793893888:raise ValueError('Source routed affine census differs')
    routed=int(routed)
    save(p/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredCompatibleJobID':j['job_id'],'measuredCompatibleResult':c['result'],'measuredCompatibleReservation':j['reservation'],
        'routedNominalAffineMACsPerCall':routed,'fullNominalAffineMACsPerCall':sum(x['nominalMACs'] for x in full),
        'exception':'Generic fallback does not describe this serialB1 native-shape consumer. Same originalresidentparameters/compiledclocks/fullresolutionVAE as successful fullqueryparent peak8684MBVRAM. Reuse14336MBRAM/13000MBVRAM CPU4,strictguardfit. Full1536state/KV,768Q/writes; physically sliced originalmaps. No encoder/referencecampaign/fit/retries.'})
    r={'schema':'tce-rotating-write-request-v1','parentJobID':j['job_id'],'parentRequestSHA256':old['requestSHA256'],'parentReportSHA256':sha(p0/'report.json'),
        'prompts':old['prompts'],'seed':old['seed'],'ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'timeoutSeconds':600,
        'protocol':'OwnZ3H/E/clock3 ->full1024image512textstate/allcurrentKV,originalpositions ->alltext+imagephase(bmod4)Q/attention/MLPwrites,globalb0..24,carryotherimagerowsunchanged ->fullvelocityprojection/originalfinitecommit/VAE. Single fusedweights sliced beforeevaluation. Twoimagessealbefore existingHN/HH/NN/NH/full developmenttargets. No pooling/fit/gain/nativehiddenrefresh/newprefix/noise/reference.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from run import command
    from mrun.client.submit import launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same job')
    cmd=command()[:-5]+['python','-u','rotating_write_study.py','execute','.'];i=cmd.index('uv');cmd[i:i]=['OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=launch(cmd,experiment='tce-rotating-write-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs={'cuda':True},
        config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width']},
        ram_mb=14336,vram_mb=13000,cpu_threads=4,disk_gb=1,est_wall_s=120,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='OwnZ3H/E full1536state/globalKV;768rotatingQ/attention/MLPwrites eachof25blocks,carryotherimagebits. Jointsequential/singleparallelMLP; slicedfusedweightsphysically.2routedcalls/286affines/2VAE;sealbothbefore existingHN/HHdevelopmenttargets.4historicalprefixcalls/encodingpaid. No fit/newreference/noise. CPU4B1/1536,14336RAM13000VRAMmeasurednativeparentexception,120sexpected600scap,noretries.')
    v={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',v);return v


def seal(root):
    from paired_fields import load
    from rotating_writes import image_rows
    root=Path(root);r=read(root/'request.json');s=read(root/'inputs.json');im=read(root/'images.json')
    if im['requestSHA256']!=r['requestSHA256'] or set(im['fields'])!={'00','10'} or set(im['images'])!={'00','10'} or len(im['routes'])!=50:raise ValueError('Candidate coverage differs')
    for c,fs in im['fields'].items():
        if set(fs)!=set(PORTS)|{'C3','latent3'}:raise ValueError('Complete image fields absent')
        if fs['latent3']!=s['prefix'][c]['latent3']:raise ValueError('Candidate used another state')
        for x in fs.values():load(x)
        if sha(im['images'][c]['path'])!=im['images'][c]['sha256']:raise ValueError('Original candidate PNG differs')
        actual=next(x for x in im['actual'] if x['case']==c and x['kind']=='routed-input')
        if actual['hidden_statesPayloadSHA256']!=fs['latent3']['payloadSHA256'] or actual['encoder_hidden_statesPayloadSHA256']!=s['prefix'][c]['E']['payloadSHA256'] or actual['img_idsPayloadSHA256']!=s['ids']['payloadSHA256']:raise ValueError('Actual full input differs')
        routes=[x for x in im['routes'] if x['case']==c]
        if [x['blockIndex'] for x in routes]!=list(range(25)):raise ValueError('Global block routing sequence differs')
        for index,row in enumerate(routes):
            if row['unchangedInputPayloadSHA256']!=row['unchangedOutputPayloadSHA256']:raise ValueError('Carried row bits differ')
            if row['selectedImageRows']!=image_rows(index).tolist() or row['blockKind']!=('joint' if index<5 else 'single'):raise ValueError('Actual selected phase differs')
            if index and routes[index-1]['fullOutputStatePayloadSHA256']!=row['fullInputStatePayloadSHA256']:raise ValueError('Own-state block handoff differs')
        final=next(x for x in im['actual'] if x['case']==c and x['kind']=='final-commit')
        if final['incomingPayloadSHA256']!=fs['latent3']['payloadSHA256'] or final['velocityPayloadSHA256']!=fs['velocity3']['payloadSHA256'] or final['outgoingPayloadSHA256']!=fs['latent4']['payloadSHA256'] or final['finalStepIndex']!=4:raise ValueError('Actual final commit differs')
    save(root/'candidate-seal.json',{'schema':'tce-rotating-write-seal-v1','requestSHA256':r['requestSHA256'],'imagesSHA256':sha(root/'images.json'),'fields':im['fields'],'images':im['images']})


def score(root,out,phases,start):
    from paired_fields import load,retain,metrics,paired
    from two_clock_carry import energy_account
    root=Path(root);r=read(root/'request.json');s=read(root/'inputs.json');im=read(root/'images.json');seal_sha=sha(root/'candidate-seal.json');t=read(root/'targets.json')
    for x in t['sourceFiles'].values():
        if sha(x['path'])!=x['sha256']:raise ValueError('Historical released record differs')
    values={c:{p:load(fs[p]) for p in PORTS} for c,fs in im['fields'].items()};accounts=[];responses=[];holds=[];fields={};commits=[];observations=0
    gap=s['schedule']['sigmas'][4]-s['schedule']['sigmas'][3]
    with (out/'observations.jsonl').open('x') as stream:
        for c,fs in im['fields'].items():
            if fs['latent3']!=t['HN']['fields'][c]['latent3']:raise ValueError('Native comparator state differs')
            ev=values[c]['velocity3']-load(t['HN']['fields'][c]['velocity3'])
            ez=values[c]['latent4']-load(t['HN']['fields'][c]['latent4'])
            terms={'scaledVelocityError':gap*ev,'finiteCommitDifference':ez-gap*ev}
            fields[c]={n:retain(root/'arrays/diagnostic'/c,n,a) for n,a in dict(velocityError=ev,latentError=ez,**terms).items()}
            commits.append({'case':c,'gap':gap,**energy_account(ez,terms)})
            for port in ['velocity3','latent4','pngRGB']:
                a=values[c][port]
                for ref in ['HN','HH','NN','full']:accounts.append({'case':c,'port':port,'reference':ref,**metrics(a,load(t[ref]['fields'][c][port]))})
                native=load(t['HN']['fields'][c][port]);holds.append({'case':c,'port':port,'reference':'HN',**metrics(load(t['HH']['fields'][c][port]),native)})
                aa=a.reshape(512 if port=='pngRGB' else 1024,-1);bb=native.reshape(aa.shape)
                for pos,(ar,br) in enumerate(zip(aa,bb)):
                    stream.write(json.dumps({'case':c,'port':port,'position':pos,**metrics(ar,br)},allow_nan=False)+'\n');observations+=1
        for port in ['velocity3','latent4','pngRGB']:
            for ref in ['HN','HH','NN','full']:responses.append({'port':port,'reference':ref,**paired(values['00'][port],values['10'][port],load(t[ref]['fields']['00'][port]),load(t[ref]['fields']['10'][port]))})
    if sha(root/'candidate-seal.json')!=seal_sha:raise ValueError('Candidate seal changed')
    for c,x in im['images'].items():shutil.copyfile(x['path'],out/('routed-'+c+'.png'))
    for ref in ['HN','HH','NN','NH','full']:
        for c,x in t[ref]['images'].items():
            if sha(x['path'])!=x['sha256']:raise ValueError('Historical image changed')
            shutil.copyfile(x['path'],out/(ref+'-'+c+'.png'))
    for n in ['request.json','inputs.json','targets.json','resource-basis.json','frozen-plan.md','images.json','candidate-seal.json']:shutil.copyfile(root/n,out/n)
    save(out/'diagnostic-fields.json',fields)
    save(out/'report.json',{'schema':'tce-rotating-write-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal_sha,'accounts':accounts,'holdAccounts':holds,'pairedColorResponse':responses,
        'finiteCommitAccounts':commits,'observations':observations,'phases':phases,'workerSecondsAfterImports':time.perf_counter()-start,
        'work':{'routedDenoisers':2,'routedTransformerLinears':sum(x['parameter'].startswith('transformer.') for x in im['linears']),
            'routedNominalAffineMACs':sum(x['nominalMACs'] for x in im['linears'] if x['parameter'].startswith('transformer.')),
            'routedAttentionCalls':len(im['attentionCalls']),'routedAttentionNominalQKAVMACs':sum(2*x['queryShape'][0]*x['queryShape'][1]*x['keyShape'][1]*x['queryShape'][2]*x['queryShape'][3] for x in im['attentionCalls']),
            'nativeVAEDecodes':2,'VAEConvLinearCalls':len(im['VAEOperations']),'VAENominalConvLinearMACs':sum(x['nominalMACs'] for x in im['VAEOperations']),
            'chargedHistoricalPrefixDenoisers':4,'chargedHistoricalPrefixNominalAffineMACs':18895708618752,'newImages':2,'newEncoderEvaluations':0,'newReferenceImages':0,'newNoiseDraws':0,'newFits':0},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limitations':['Known00/10ownZ3H/fullE/clock and originalweights/prefixes/encoding/VAE paid; no arbitrarypromptzero-neural serving.',
            'Fixed rotatingimagewrites approximate originalcomputation; changed slicedGEMM geometry carriesfinite differences.',
            'Existing developmenttargets seenhistorically butdenieduntil bothnewimagesseal. No matchedspeedup/globalquality/heldoutclaim; denoisertimingincludesfullstatehashwitnesses.']})


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);start=time.perf_counter();out=root/'response-result';out.mkdir();tick=time.perf_counter()
    subprocess.run([sys.executable,'-u',str(root/'rotating_write_study.py'),'images',str(root)],check=True,timeout=480)
    phases=[{'action':'images','wallSecondsIncludingFreshProcess':time.perf_counter()-tick}];seal(root);print('ROTATING_WRITE_CANDIDATES_SEALED',sha(root/'candidate-seal.json'),flush=True)
    score(root,out,phases,start);print('ROTATING_WRITE_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','images','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='images':
        from rotating_write_images import execute
        v=execute(a.run)
    elif a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(json.dumps(v))
