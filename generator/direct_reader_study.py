"""CPU direct-score reader and complete input-bound downstream reuse proof."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import np, read, save, sha, identity, guard, floats

HERE = Path(__file__).resolve().parent
OWNED = ['direct_reader_study.py', 'direct_reader.py', 'encoder_atoms.py', 'paired_fields.py', 'masked_kv_study.py', 'flux_clock.py']


def verify(root, request):
    if identity({k:v for k,v in request.items() if k!='requestSHA256'}) != request['requestSHA256']: raise ValueError('Request identity differs')
    for name, digest in request['sourceHashes'].items():
        if sha(Path(root)/name) != digest: raise ValueError('Frozen source differs: '+name)


def consumer_contract(source, parent_request):
    keys = ['lexicalSource', 'qwenShards', 'encoderNumericalProfile', 'encoderVersions', 'sdpaSourceSHA256',
        'imageAssets', 'nativeSources', 'numericalProfile', 'versions', 'attentionProcessors', 'schedule', 'noise', 'clockTableSHA256']
    code = ['selected_reader_encoder.py', 'selected_reader_images.py', 'flux_functional_qwen_prefix.py',
        'qwen_h18_state_history.py', 'conditioning_demand.py', 'complete_masked_study.py', 'flux_clock.py', 'flux_clock_study.py']
    return {'source': {k:source[k] for k in keys}, 'consumerSourceHashes': {k:parent_request['sourceHashes'][k] for k in code},
        'geometry': {'rows':512, 'heads':32, 'headWidth':128, 'hiddenWidth':2560}, 'steps':4, 'size':[512,512], 'batch':1}


def stage(root):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    parent = HERE.parent/'artifacts/selected-reader-image-v1'; c,j,old = checked_collection(parent,current_workers=False)
    live = Api().json('GET','/api/jobs/'+j['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody'] or live['config']['requestSHA256']!=old['requestSHA256']: raise ValueError('Qualified parent differs')
    f=parent/'result'; source=read(f/'inputs.json'); reader=read(f/'reader-seal.json'); enc=read(f/'encoder-candidate.json'); img=read(f/'images-candidate.json'); seal=read(f/'candidate-seal.json')
    for key,name in [('readerSHA256','reader-seal.json'),('encoderSHA256','encoder-candidate.json'),('imagesSHA256','images-candidate.json')]:
        if seal[key]!=sha(f/name): raise ValueError('Parent output seal differs')
    contract=consumer_contract(source,old); root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for name in OWNED:
        if name in old['sourceHashes'] and sha(HERE/name)!=old['sourceHashes'][name]: raise ValueError('Qualified shared source differs: '+name)
        shutil.copyfile(HERE/name,p/name)
    shutil.copyfile(HERE.parent/'docs/plans/direct-selected-reader.md',p/'frozen-plan.md')
    forbidden=source['forbiddenTargets']+[x['path'] for x in reader['lexicalRows']+reader['packets']+reader['states']]
    forbidden += [x['path'] for x in reader['fields'].values()]
    forbidden += [x['path'] for group in [enc['fields'],img['fields']] for fields in group.values() for x in fields.values()]
    forbidden += [x['path'] for x in img['images'].values()]
    inputs={'lexicalStore':source['lexicalStore'],'originalGeometry':source['readerAsset'],'readerGeometry':source['readerGeometry'],
        'forbiddenTargets':sorted(set(forbidden)),'consumerContract':contract,'parentRoot':j['result']['diagnostics']['cwd'],
        'parentRequestSHA256':old['requestSHA256'],'parentSealSHA256':sha(f/'candidate-seal.json'),
        'parentByteVerificationSHA256':sha(parent/'retained-array-verification.json')}
    save(p/'inputs.json',inputs)
    save(p/'targets.json',{'reader':reader,'encoder':enc,'images':img,'seal':seal,'parentRequest':old,'parentSource':source,
        'clockTable':read(f/'clock-table.json'),
        'parentReport':read(f/'report.json'),'parentReportSHA256':sha(f/'report.json'),'consumerContract':contract,
        'sourceFiles':{n:{'path':inputs['parentRoot']+'/'+n,'sha256':sha(f/n)} for n in ['reader-seal.json','encoder-candidate.json','images-candidate.json','candidate-seal.json','clock-table.json']},
        'scope':'Parent head/H0 ports open only after new complete head seal. Output inheritance requires exact demanded BF16 input identity and source/numerical closure.'})
    r={'schema':'tce-direct-reader-request-v1','prompts':old['prompts'],'manifestSHA256':old['manifestSHA256'],
        'readerPolicy':'current-top4-direct-score-reuse-fp64-fp32-bf16-v1','ownedSources':OWNED,'parentRequestSHA256':old['requestSHA256'],
        'consumerContractSHA256':identity(contract),'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':300,
        'scope':'Centers-only source -> raw strings -> direct current-score top4 -> complete raw/BF16 heads/H0 -> exact input-bound reuse proof, otherwise own learned continuation required. No arbitrary-prompt zero-neural claim.',
        'protocol':'Separate centers compiler and fresh geometry-denied serving child. Seal complete heads/H0 before parent values. Preserve raw finite changes. No native call, fit, precision sweep or new reference. Full KV and score scan paid.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from response_study import command
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted')
    cmd=command();cmd[-3]='direct_reader_study.py';cmd[cmd.index('python') : cmd.index('python')]=['--with','tokenizers>=0.21','--with','jinja2>=3.1']
    job=launch(cmd,experiment='tce-direct-reader-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'device':'cpu','backend':'numpy','cpu_threads':1,'rows':512,'query_heads':32,'kv_groups':8,'head_dim':128},
        timeout_s=300,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 rawA/B->centers-only direct current-score top4->sealed BF16heads/H0->parent inputidentity proof. No anchors/metrics/cache/tail or neural runtime/weights. Full causalKV/scan paid. Source-local atoms/parent.0new dense consumers/images in this lease; differinginputs require ownCUDAcontinuation later. Exactinputmatch inherits qualified prior outputs. Intent-first sizing,no guessedRAM/GPU.60sexpected300scap,noretries.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def npz(item):
    if sha(item['path'])!=item['sha256']:raise ValueError('Retained packet differs')
    with np.load(item['path'],allow_pickle=False) as f:return {k:f[k] for k in f.files}


def bits(item):
    from paired_fields import payload
    if sha(item['path'])!=item['sha256']:raise ValueError('Retained input bytes differ')
    value=np.load(item['path'],allow_pickle=False)
    if list(value.shape)!=item['shape'] or value.dtype!=np.uint16 or payload(value)!=item['payloadSHA256'] or item['logicalDtype']!='torch.bfloat16':raise ValueError('Typed BF16 input differs')
    if not np.isfinite(floats(value)).all():raise ValueError('Nonfinite BF16 input')
    return value


def qualify_inputs(own, target, contract, parent_contract):
    if identity(contract)!=identity(parent_contract):raise ValueError('Consumer contract differs; no reuse authority')
    rows=[]
    for case in ['A','B']:
        a,b=own['fields'][case],target['reader']['fields'][case]; ah,bh=own['h0Fields'][case],target['encoder']['fields'][case]['H0']
        if a['shape']!=[1,512,4096] or b['shape']!=a['shape'] or ah['shape']!=[1,512,2560] or bh['shape']!=ah['shape']:raise ValueError('Demanded input geometry differs')
        head_mismatch=int(np.count_nonzero(bits(a)!=bits(b)));h0_mismatch=int(np.count_nonzero(bits(ah)!=bits(bh)))
        own_lex=npz(next(x for x in own['lexicalRows'] if x['id']==case));parent_lex=npz(next(x for x in target['reader']['lexicalRows'] if x['id']==case))
        controls={}
        for key in ['ids','mask','positions']:
            if own_lex[key].shape!=parent_lex[key].shape or own_lex[key].dtype!=parent_lex[key].dtype:raise ValueError('Lexical control schema differs')
            controls[key]=int(np.count_nonzero(own_lex[key]!=parent_lex[key]))
        if not np.array_equal(own_lex['e'][None],bits(ah)):raise ValueError('Candidate H0 is not its own lexical state')
        actual=next(x for x in target['encoder']['actualHeadReads'] if x['case']==case)
        handoff=next(x for x in target['encoder']['handoffs'] if x['case']==case and x['layer']==0)
        if actual['actualHeadPayloadSHA256']!=b['payloadSHA256'] or actual['readerFieldSHA256']!=b['sha256'] or handoff['actualInputPayloadSHA256']!=bh['payloadSHA256'] or handoff['inputFieldSHA256']!=bh['sha256']:raise ValueError('Parent actual consumer input binding differs')
        rows.append({'id':case,'headBF16WordMismatches':head_mismatch,'H0BF16WordMismatches':h0_mismatch,'lexicalControlMismatches':controls,'headScalars':int(np.prod(a['shape'])),'H0Scalars':int(np.prod(ah['shape'])),
            'candidateHead':a,'parentActualHead':b,'candidateH0':ah,'parentActualH0':bh})
    # This is a complete demanded-input identity condition for deterministic
    # result reuse, not an approximation-accuracy gate or a scientific verdict.
    identical=sum(x['headBF16WordMismatches']+x['H0BF16WordMismatches']+sum(x['lexicalControlMismatches'].values()) for x in rows)==0
    return {'schema':'tce-direct-reader-input-identity-v1','rows':rows,'consumerContractSHA256':identity(contract),
        'completeDemandedInputsIdentical':identical,'outcome':'reuse-qualified' if identical else 'own-consumer-required',
        'scope':'Only exact typed BF16 inputs plus unchanged source/numerical closure authorize deterministic parent-output inheritance. Raw differences remain observations.'}


def score(root,out,phases,started):
    from paired_fields import metrics,retain
    from direct_reader import POLICY
    r=read(root/'request.json');source=read(root/'inputs.json');target=read(root/'targets.json');reader=read(root/'reader-seal.json');seal_sha=sha(root/'head-seal.json')
    if reader['policy']!=POLICY or identity(source['consumerContract'])!=r['consumerContractSHA256']:raise ValueError('Declared computation differs')
    actual_contract=consumer_contract(target['parentSource'],target['parentRequest'])
    for item in target['sourceFiles'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Frozen parent source record differs')
    enc,img=target['encoder'],target['images']
    if enc['lexicalSealSHA256']!=target['reader']['lexicalSealSHA256'] or enc['numericalProfile']!=actual_contract['source']['encoderNumericalProfile'] or img['numericalProfile']!=actual_contract['source']['numericalProfile']:raise ValueError('Parent lexical/numerical consumer closure differs')
    for case in ['A','B']:
        if img['actualEFields'][case]!=enc['fields'][case]['E'] or img['schedules'][case]!=actual_contract['source']['schedule']:raise ValueError('Parent actual conditioning/schedule differs')
    proof=qualify_inputs(reader,target,source['consumerContract'],actual_contract);comparisons=[];differences=[];observations=0
    with (out/'observations.jsonl').open('x') as stream:
        for case in ['A','B']:
            own=npz(next(x for x in reader['packets'] if x['id']==case));old=npz(next(x for x in target['reader']['packets'] if x['id']==case))
            current=npz(next(x for x in reader['lexicalRows'] if x['id']==case));prior=npz(next(x for x in target['reader']['lexicalRows'] if x['id']==case))
            raw=own['raw']-old['raw'];differences.append(retain(root/'arrays/comparison',case+'-raw-difference',raw))
            finite=np.isfinite(old['selectedLogits']);logits=own['selectedLogits']-old['selectedLogits'];weights=own['selectedWeights']-old['selectedWeights']
            comparisons.append({'id':case,'raw':metrics(own['raw'],old['raw']),'BF16WordMismatches':int(np.count_nonzero(own['rounded']!=old['rounded'])),
                'supportPositionMismatches':int(np.count_nonzero(own['supportPositions']!=old['supportPositions'])),
                'selectedLogitMaxAbsDifference':float(np.max(np.abs(logits),where=finite,initial=0.)),
                'selectedWeightMaxAbsDifference':float(np.max(np.abs(weights))),
                'lexicalMismatches':{k:int(np.count_nonzero(current[k]!=prior[k])) for k in ['ids','mask','positions','e','qpos','kpos','v']}})
            for i in range(512):
                for h in range(32):
                    a,b=own['raw'][i,h*128:(h+1)*128],old['raw'][i,h*128:(h+1)*128]
                    stream.write(json.dumps({'kind':'head-reassociation','case':case,'position':i,'queryHead':h,'attended':bool(current['mask'][i]),
                        'headBF16WordMismatches':int(np.count_nonzero(own['rounded'][i,h*128:(h+1)*128]!=old['rounded'][i,h*128:(h+1)*128])),
                        **metrics(a,b)},allow_nan=False)+'\n');observations+=1
    save(out/'input-identity.json',proof);save(out/'comparison-fields.json',{'fields':differences})
    inherited=None
    if proof['completeDemandedInputsIdentical']:
        enc,img,seal=target['encoder'],target['images'],target['seal']
        if enc['readerSealSHA256']!=seal['readerSHA256'] or img['encoderSHA256']!=seal['encoderSHA256'] or img['tableSHA256']!=seal['clockTableSHA256']:raise ValueError('Inherited downstream source chain differs')
        inherited={'schema':'tce-direct-reader-inherited-output-v1','inputIdentitySHA256':sha(out/'input-identity.json'),'parentSealSHA256':source['parentSealSHA256'],
            'consumerContractSHA256':r['consumerContractSHA256'],'encoderFields':enc['fields'],'imageFields':img['fields'],'images':img['images'],'parentSourceFiles':target['sourceFiles'],
            'parentReportSHA256':target['parentReportSHA256'],'scope':'Already generated parent outputs inherited by complete demanded input identity; no new learned encoder/image execution or new PNG generation.'}
        save(out/'inherited-outputs.json',inherited)
        for case,item in img['images'].items():
            if sha(item['path'])!=item['sha256']:raise ValueError('Inherited original PNG differs')
            shutil.copyfile(item['path'],out/('inherited-'+case+'.png'))
    if sha(root/'head-seal.json')!=seal_sha:raise ValueError('Sealed candidate changed')
    for n in ['request.json','inputs.json','targets.json','frozen-plan.md','compile-receipt.json','reader-seal.json','head-seal.json']:shutil.copyfile(root/n,out/n)
    parent=target['parentReport'];group_work=[g for c in reader['work'] for g in c['groups']]
    save(out/'report.json',{'schema':'tce-direct-reader-result-v1','requestSHA256':r['requestSHA256'],'scope':r['scope'],'headSealSHA256':seal_sha,'outcome':proof['outcome'],
        'comparisons':comparisons,'readerWork':reader['work'],'observations':observations,'phases':phases,'workerSecondsAfterImports':time.perf_counter()-started,
        'work':{'newOriginalParameterReads':0,'newOriginalProjections':0,'newOriginalEncoderBlocks':0,'newDenoisers':0,'newVAEDecodes':0,'newImages':0,
            'inheritedImages':2 if inherited else 0,'modelDerivedContextualReads':2,'originalGeometryLoadedByServing':False,'runtimeCentersPayloadBytes':8192,
            'allocatedNumericalStateBytesEach':sum(g['allocatedIndexPayloadBytes']+g['countAndClockBytes'] for g in reader['work'][0]['groups']),
            'currentQueryScoreProducts':sum(g['selectorScoreProducts'] for g in group_work),'selectedValueProducts':sum(g['selectedValueProducts'] for g in group_work),
            'anchorAdmissionMACs':0,'anchorScoreCacheBytes':0,'routingCalls':0,'selectedDeltaProducts':0,'tailMomentEnvelopeBytes':0},
        'inheritedConsumerWork':parent['work'] if inherited else None,
        'inheritedPairedPromptResponse':[{**x,'branch':'direct-input-equivalent','evidenceOrigin':'inherited-same-input'} for x in parent['pairedPromptResponse'] if x['branch']=='selected'] if inherited else [],
        'inheritedEndpointAccounts':[{**x,'branch':'direct-input-equivalent','evidenceOrigin':'inherited-same-input'} for x in parent['accounts'] if x['branch']=='selected'] if inherited else [],
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limitations':['Known raw pair/one inherited noise. Exact-port downstream result reuse is not arbitrary-prompt zero-neural generation.',
            'Direct source-centered score scan still retains full K/V and performs model-derived attention arithmetic. Complete learned lexical compilation and buffers/diagnostics/IO remain paid.',
            'Most learned encoder/FLUX/VAE work is historical inherited evidence only when all demanded BF16 inputs match; changed inputs require their own original consumer. No matched speedup claim.']})
    print('DIRECT_READER_DONE',proof['outcome'],flush=True)


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);started=time.perf_counter();out=root/'response-result';out.mkdir();phases=[]
    for action in ['compile','reader']:
        tick=time.perf_counter();subprocess.run([sys.executable,'-u',str(root/'direct_reader_study.py'),action,str(root)],check=True,timeout=240)
        phases.append({'action':action,'wallSecondsIncludingFreshProcess':time.perf_counter()-tick})
    guard();reader=read(root/'reader-seal.json')
    for item in [*reader['fields'].values(),*reader['h0Fields'].values()]:bits(item)
    for item in reader['lexicalRows']+reader['packets']+reader['states']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Candidate packet changed before seal')
    save(root/'head-seal.json',{'schema':'tce-direct-reader-head-seal-v1','requestSHA256':r['requestSHA256'],'readerSealSHA256':sha(root/'reader-seal.json'),
        'fields':reader['fields'],'h0Fields':reader['h0Fields'],'scope':'Complete own raw/BF16 head/H0 candidates authenticated before parent values open.'})
    print('DIRECT_HEADS_GLOBALLY_SEALED',sha(root/'head-seal.json'),flush=True);score(root,out,phases,started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','compile','reader','collect']);p.add_argument('run');a=p.parse_args()
    if a.action in ['compile','reader']:
        import direct_reader
        value=direct_reader.compile_centers(a.run) if a.action=='compile' else direct_reader.execute(a.run)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
