"""One finite source-native read -> sealed ridge writer -> native image assay."""
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
OWNED=['source_native_study.py','source_native_worker.py','source_native_reader.py','source_native_writer.py','native_reader_corpus.py',
    'last_velocity_decode.py','encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']
PORTS=['velocity3','latent4','vaeInput','decodedRaw','rgbFloat','pngRGB']


def stage(root):
    from publish_context_statistics import checked_collection
    from native_reader_corpus import inventory
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    parent=HERE.parent/'artifacts/two-clock-v1';c,j,old=checked_collection(parent,current_workers=False);api=Api()
    live=api.json('GET','/api/jobs/'+j['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody']:raise ValueError('Own-state parent differs')
    p0=parent/'result';s=read(p0/'inputs.json');im=read(p0/'images.json');t=read(p0/'targets.json');report=read(p0/'report.json')
    if report['candidateSealSHA256']!=sha(p0/'candidate-seal.json'):raise ValueError('Parent release differs')
    corpus=inventory(HERE.parent.parent,s['imageAssets'])
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for n in OWNED:
        if n in old['ownedSources'] and sha(HERE/n)!=old['sourceHashes'][n]:raise ValueError('Qualified helper changed: '+n)
        shutil.copyfile(HERE/n,p/n)
    wheel_root=HERE.parent.parent/'artifacts/flux-global-reader-20261106/run-01/payload'
    for n in WHEELS:shutil.copyfile(wheel_root/n,p/n)
    shutil.copyfile(HERE.parent/'docs/plans/source-native-global-reader.md',p/'frozen-plan.md')
    prefix={case:{'latent3':im['common'][case]['latent3'],'velocity1':s['prefix'][case]['velocity1'],'E':s['prefix'][case]['E']} for case in ['00','10']}
    targets={ref:{'fields':{case:im['fields'][ref+'-'+case] for case in ['00','10']},'images':{case:im['images'][ref+'-'+case] for case in ['00','10']}} for ref in ['HN','HH']}
    targets.update({ref:t[ref] for ref in ['NN','NH','full']})
    base=j['result']['diagnostics']['cwd'];targets['sourceFiles']={n:{'path':base+'/response-result/'+n,'sha256':sha(p0/n)} for n in ['images.json','targets.json','candidate-seal.json','report.json']}
    forbidden=set(s['forbiddenTargets'])
    for ref in ['HN','HH','NN','NH','full']:
        forbidden.update(x['path'] for fs in targets[ref]['fields'].values() for x in fs.values());forbidden.update(x['path'] for x in targets[ref]['images'].values())
    allowed={x['path'] for fs in prefix.values() for x in fs.values()}|{s['ids']['path']}
    source=dict(s);source.update(prefix=prefix,nativeReaderParentRoot=base,parentRequestSHA256=old['requestSHA256'],parentReportSHA256=sha(p0/'report.json'),
        targetRoots=sorted(set(s['targetRoots'])|{base}),forbiddenTargets=sorted(forbidden-allowed),guardCanary=im['fields']['HN-00']['velocity3']['path'])
    save(p/'inputs.json',source);save(p/'targets.json',targets);save(p/'corpus.json',corpus)
    save(p/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredCompatibleJobID':j['job_id'],'measuredCompatibleResult':c['result'],'measuredCompatibleReservation':j['reservation'],
        'exception':'Generic diffusion fallback is inapplicable to serial B1/512 selective read. Reuse measured successful full-native-parent14336RAM/13000VRAM CPU4 envelope; new model closure is only142344960B plus one VAE and FP64 ridge workspace. Host-local immutable source cache requires beast. Expected180-300s,900s cap, no retries.'})
    contract=one_lease_scheduling_contract(logical_branches=30,logical_operations=46+30+2,logical_checkpoint_replays=0,
        conditioner_policy='Reuse source-bound historical conditioning; no new encoding.',
        extra={'modelLoadScope':'One resident composite selective-reader/VAE consumer; fit and readout-only CPU actors under same lease, no per-branch jobs.'})
    r={'schema':'tce-source-native-reader-request-v1','parentJobID':j['job_id'],'parentRequestSHA256':old['requestSHA256'],
        'prompts':old['prompts'],'seed':old['seed'],'ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},
        'scheduling_contract':contract,'needs':{'cuda':True},'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16',
        'batch_size':1,'seq_len':1536,'height':512,'width':512,'timeoutSeconds':900,
        'protocol':'18discovery live H0 reads -> isolated fixed .01 FP64 paired ridge -> coefficient seal ->28own serving reads -> isolated readout-only actor30outputs ->2native scheduler/VAEimages ->allcandidate seal ->18feedback/8evaluation/main historical scoring. Seven original affines/read and fitted learned writer paid. No arbitraryprompt/zero-neural claim.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256'],'corpus':corpus['counts']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same job')
    cmd=command()[:-5]
    for n in WHEELS:cmd+=['--with','./'+n]
    cmd+=['python','-u','source_native_study.py','execute','.']
    i=cmd.index('uv');cmd[i:i]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-source-native-reader-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width','scheduling_contract']},
        ram_mb=14336,vram_mb=13000,cpu_threads=4,disk_gb=2,est_wall_s=300,timeout_s=900,detach=True,retry_on_kill=False,preflight=True,
        note='Single resident selective reader/VAE;46firstjoint image reads/322originalaffines, discovery18 ->fixed FP64ridge noepochs ->coeffseal ->28ownH0/30readouts ->2VAE. Existingforeign labels/prefixes paid. Fresh isolated fit/readout actors, targetallowlists, allcandidates seal before scoring. No encoder/fullnative/referencecampaign/newnoise.14336RAM13000VRAM measuredfullparentexception,CPU4B1,180-300sexpected900scap,noretries;beastlocal sources.')
    v={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',v);return v


def seal(root):
    from paired_fields import load
    root=Path(root);r=read(root/'request.json');s=read(root/'inputs.json');im=read(root/'images.json');served=read(root/'serving.json');fit=read(root/'fit/fit.json')
    if im['requestSHA256']!=r['requestSHA256'] or set(im['fields'])!={'00','10'} or len(im['reads'])!=46 or len(im['writerBindings'])!=30 or len(served['rows'])!=30:raise ValueError('Candidate coverage differs')
    for file,key in [('coefficient-seal.json','coefficientSealSHA256'),('serving.json','servingSHA256'),('fit/fit.json','fitReceiptSHA256')]:
        if sha(root/file)!=im[key]:raise ValueError('Candidate receipt differs')
    for x in fit['fields'].values():load(x)
    read_map={row['id']:row for row in im['reads']}
    for row in served['rows']:
        if row['inputFields']!={k:read_map[row['id']][k] for k in ['z','previous','H0']}:raise ValueError('Writer bypassed actual read')
        for item in [*row['inputFields'].values(),row['velocity'],row['rawVelocity']]:load(item)
        binding=next(x for x in im['writerBindings'] if (x['id'],x['method'])==(row['id'],row['method']))
        if binding['inputFrame']['slots']['coefficients']['kind']!='receipt' or binding['evidence']['input_frame_id']!=binding['inputFrame']['frame_id']:raise ValueError('Typed writer binding differs')
    for c,fs in im['fields'].items():
        if set(fs)!=set(PORTS)|{'latent3','rawVelocity'} or fs['latent3']!=s['prefix'][c]['latent3']:raise ValueError('Image field/input coverage differs')
        for item in fs.values():load(item)
        row=next(x for x in served['rows'] if x['id']==c and x['method']=='candidate')
        if fs['velocity3']!=row['velocity']:raise ValueError('Native consumer used another writer')
        final=next(x for x in im['actual'] if x['case']==c)
        if [final[k] for k in ['incomingPayloadSHA256','velocityPayloadSHA256','outgoingPayloadSHA256']]!=[fs[k]['payloadSHA256'] for k in ['latent3','velocity3','latent4']] or final['finalStepIndex']!=4:raise ValueError('Actual final commit differs')
        if sha(im['images'][c]['path'])!=im['images'][c]['sha256']:raise ValueError('Original candidate PNG differs')
    save(root/'candidate-seal.json',{'schema':'tce-source-native-candidate-seal-v1','requestSHA256':r['requestSHA256'],'imagesSHA256':sha(root/'images.json'),
        'servingSHA256':sha(root/'serving.json'),'coefficientSealSHA256':sha(root/'coefficient-seal.json'),'fields':im['fields'],'images':im['images']})


def score(root,out,phases,start):
    from paired_fields import load,retain,metrics,paired
    from native_reader_corpus import load as historical
    root=Path(root);out=Path(out);r=read(root/'request.json');s=read(root/'inputs.json');im=read(root/'images.json');served=read(root/'serving.json');fit=read(root/'fit/fit.json')
    seal_sha=sha(root/'candidate-seal.json');t=read(root/'targets.json');corpus=read(root/'corpus.json')
    for item in t['sourceFiles'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Historical record changed')
    targets={x['id']:x for x in corpus['records'] if x['population']!='discovery'};values={};native={};holds={};accounts=[];responses=[];diagnostics={};observations=0
    with (out/'observations.jsonl').open('x') as stream:
        for row in served['rows']:
            candidate=load(row['velocity']);key=row['id']+':'+row['method'];values[key]=candidate
            target=(load(t['HN']['fields'][row['id']]['velocity3']) if row['population']=='main' else historical(targets[row['id']]['target']).double().numpy())
            native[row['id']]=target;hold=load(row['inputFields']['previous']);holds[row['id']]=hold
            accounts.append({'id':row['id'],'population':row['population'],'method':row['method'],'port':'velocity3',**metrics(candidate,target)})
            if row['method']=='candidate':accounts.append({'id':row['id'],'population':row['population'],'method':'hold','port':'velocity3',**metrics(hold,target)})
            diagnostics[key]={'velocityError':retain(root/'arrays/diagnostic'/key.replace(':','_'),'velocityError',candidate-target)}
            for index,(a,b) in enumerate(zip(candidate.reshape(1024,128),target.reshape(1024,128))):
                stream.write(json.dumps({'id':row['id'],'population':row['population'],'method':row['method'],'port':'velocity3','position':index,**metrics(a,b)},allow_nan=False)+'\n');observations+=1
        for c,fs in im['fields'].items():
            for port in ['latent4','pngRGB']:
                a=load(fs[port]);b=load(t['HN']['fields'][c][port])
                for method,v in [('candidate',a),('hold',load(t['HH']['fields'][c][port]))]:accounts.append({'id':c,'population':'main','method':method,'port':port,**metrics(v,b)})
                for index,(ar,br) in enumerate(zip(a.reshape(512 if port=='pngRGB' else 1024,-1),b.reshape(512 if port=='pngRGB' else 1024,-1))):
                    stream.write(json.dumps({'id':c,'population':'main','method':'candidate','port':port,'position':index,**metrics(ar,br)},allow_nan=False)+'\n');observations+=1
        mask=np.ones((32,32),bool);mask[:8,:8]=False;mask[8,:8]=False;mask[:8,8]=False;mask=mask.ravel()
        for a,b in corpus['evaluationPairs']:
            for method,bank in [('candidate',{k:values[k+':candidate'] for k in targets}),('hold',holds)]:
                for scope,select in [('all',np.ones(1024,bool)),('unchanged-query',mask)]:responses.append({'population':'evaluation','pair':[a,b],'method':method,'scope':scope,'port':'velocity3',**paired(bank[a][:,select],bank[b][:,select],native[a][:,select],native[b][:,select])})
        for port in ['velocity3','latent4','pngRGB']:
            for method in ['candidate','hold']:
                bank=im['fields'] if method=='candidate' else t['HH']['fields']
                responses.append({'population':'main','method':method,'port':port,**paired(load(bank['00'][port]),load(bank['10'][port]),load(t['HN']['fields']['00'][port]),load(t['HN']['fields']['10'][port]))})
        responses.append({'population':'main','method':'zeroRead','port':'velocity3',**paired(values['00:zeroRead'],values['10:zeroRead'],native['00'],native['10'])})
    if sha(root/'candidate-seal.json')!=seal_sha:raise ValueError('Candidate seal changed during scoring')
    for c,x in im['images'].items():shutil.copyfile(x['path'],out/('reader-'+c+'.png'))
    for ref in ['HN','HH','NN','NH','full']:
        for c,x in t[ref]['images'].items():
            if sha(x['path'])!=x['sha256']:raise ValueError('Historical image changed')
            shutil.copyfile(x['path'],out/(ref+'-'+c+'.png'))
    for n in ['request.json','inputs.json','targets.json','corpus.json','resource-basis.json','frozen-plan.md','images.json','candidate-seal.json','coefficient-seal.json','serving.json']:shutil.copyfile(root/n,out/n)
    shutil.copyfile(root/'fit/fit.json',out/'fit.json');save(out/'diagnostic-fields.json',diagnostics)
    save(out/'report.json',{'schema':'tce-source-native-reader-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal_sha,
        'accounts':accounts,'pairedResponses':responses,'observations':observations,'phases':phases,'workerSecondsAfterImports':time.perf_counter()-start,
        'work':{'sourceNativeReads':46,'discoveryReads':18,'servedReads':28,'originalAffineCalls':len(im['linears']),'originalAffineMACs':sum(x['nominalMACs'] for x in im['linears']),
            'attentionQKAVMACs':sum(2*x['queryShape'][0]*x['queryShape'][1]*x['keyShape'][1]*x['queryShape'][2]*x['queryShape'][3] for x in im['attentionCalls']),
            'fittedWriterProducts':len(served['rows']),'fittedWriterMACs':sum(x['writerNominalMACs'] for x in served['rows']),
            'trainingPostFitWriterMACs':fit['postFitTrainingWriterNominalMACs'],'nativeVAEDecodes':2,'VAEConvLinearCalls':len(im['VAEOperations']),
            'fitGramMACs':fit['gramNominalMACs'],'fitRhsMACs':fit['rhsNominalMACs'],'fitBiasMACs':fit['biasNominalMACs'],'fitEstimatedSolveFLOPs':fit['solveEstimatedDenseLUFLOPs'],
            'VAENominalConvLinearMACs':sum(x['nominalMACs'] for x in im['VAEOperations']),'originalParameterBytes':sum(x['bytes'] for x in im['loadedParameters']),
            'fittedCoefficientBytes':fit['coefficientBytes'],'chargedHistoricalMainPrefixDenoisers':4,'newFullDenoisers':0,'newEncoders':0,'newReferenceImages':0,'newNoiseDraws':0,'closedFormFits':1},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limitations':['Seven original learned affines/read, native RMS/attention, fitted writer and native VAE remain paid; not zero-neural or arbitraryprompt generation.',
            'Discovery18 fields, odd-seed feedback18 and old remote evaluation8 are historical; delayed file access does not make them prospective.',
            'Main previous velocity is held v1; canonical discovery generally uses v2, so main history transfer does not isolate reader sufficiency.',
            'Foreign labels qualify source velocity only; their CPU VAE normalization is not mixed with current native GPU pixel comparisons.',
            'Image-only query SDPA geometry may differ finitely from full native attention. No bit-parity or matched runtime speedup claim.',
            'zeroRead sets raw H0=0 before standardization with the same fitted writer; no matched refit or extra VAE decode.']})


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);start=time.perf_counter();out=root/'response-result';out.mkdir();tick=time.perf_counter()
    subprocess.run([sys.executable,'-u',str(root/'source_native_study.py'),'images',str(root)],check=True,timeout=720)
    phases=[{'action':'images','wallSecondsIncludingFreshProcess':time.perf_counter()-tick}];seal(root);print('SOURCE_NATIVE_CANDIDATES_SEALED',sha(root/'candidate-seal.json'),flush=True)
    score(root,out,phases,start);print('SOURCE_NATIVE_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','images','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='images':
        from source_native_worker import execute as images
        v=images(a.run)
    elif a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(json.dumps(v))
