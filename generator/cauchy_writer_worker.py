"""One inspectable constant update; actual finite consumer closes feedback."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import numpy as np
from contextual_writer_program import features,route,finite_add_array,read,save,sha,restore_program
from contextual_writer_worker import floats,raw,metric,serve,emit_program
from coupled_normalization_worker import configure
from coupled_normalization import SOURCE_POLICY
from joint_writer_math import objective
from cauchy_writer_math import TERMS,POLICY,ETA,propose,admit


def execute(root):
    from cauchy_writer_study import verify,WORK
    import torch
    from saturn.titan import Frame,ActSpec,ActRegistry
    from saturn.tensor_hash import tensor_bytes_sha256
    root=Path(root).resolve();req=read(root/'request.json');verify(root,req);start=time.perf_counter();cpu=time.process_time();profile=configure(torch,req)
    if req['policy']!=POLICY or req['eta']!=ETA or req['physicalWork']!=WORK:raise ValueError('Frozen construction differs')
    out=root/'response-result';out.mkdir();folder=root/'arrays';folder.mkdir();manifest={};authenticated={};released={'transfer':False}
    prior=read(root/'parent-array-manifest.json');source=read(root/'parent-parent-parent-inputs.json');prior_serving=read(root/'parent-final-serving.json');packets={r['id']:r['packet'] for r in prior_serving['rows']}
    forbidden={x['path'] for k,x in prior.items() if any('-'+case+'-' in k for case in req['transfer'])}
    forbidden.update(source['cases'][case][p]['path'] for case in req['transfer'] for p in ('H0','H1'))
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and not released['transfer'] and str(Path(os.fsdecode(args[0])).resolve()) in forbidden:raise PermissionError('Known-transfer numeric fields quarantined before final seal')
    sys.addaudithook(audit)
    def authenticate(item):
        path=Path(item['path'])
        if sha(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:raise ValueError('Retained input bytes differ')
        authenticated[str(path)]={k:item[k] for k in ('sha256','bytes')}
    def load(item):
        authenticate(item);a=np.load(item['path'],allow_pickle=False) if item['format']=='npy' else np.fromfile(item['path'],item['storedDtype']).reshape(item['shape'])
        if list(a.shape)!=item['shape'] or a.dtype.str!=item['storedDtype'] or hashlib.sha256(a.tobytes()).hexdigest()!=item['payloadSHA256']:raise ValueError('Retained field ABI/payload differs')
        return a
    def decode(item):return floats(load(item))[0].astype(np.float64)
    def retain(name,value):
        t=value.detach().cpu().contiguous();bf16=t.dtype==torch.bfloat16;a=t.view(torch.uint16).numpy() if bf16 else t.numpy();path=folder/(name+('.bf16' if bf16 else '.npy'))
        if bf16:path.write_bytes(a.tobytes())
        else:np.save(path,a,allow_pickle=False)
        item={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'payloadSHA256':tensor_bytes_sha256(t),'shape':list(t.shape),'dtype':str(t.dtype),'storedDtype':a.dtype.str,'format':'raw' if bf16 else 'npy'};manifest[name]=item;return item
    for path,digest in req['servingAssets'].items():authenticate({'path':path,'sha256':digest,'bytes':req['remoteInputs'][path]['bytes']})
    (root/'lexical').symlink_to(req['lexicalDirectory'],target_is_directory=True)
    keys=req['discovery'];data={k:load(req['fields'][k]) for k in ('h','z','target','zt','features','weights','edges','pairWeights','gamma','scaleVector','gradient')};n,width=data['h'].shape
    if (n,width)!=(2040,2560):raise ValueError('Discovery shape differs')
    parent_dir=Path(req['parent']['program']['directory']);p=read(parent_dir/'program.json');atoms=[np.fromfile(parent_dir/d['path'],'<f4').reshape(d['atoms'],width) for d in p['dictionaries']]
    if p!=req['parent']['program']['program'] or sha(parent_dir/'program.json')!=req['parent']['program']['programSHA256'] or [len(a) for a in atoms]!=WORK['atomCounts']:raise ValueError('Fixed parent differs')
    labels=[np.asarray([route(tree,row)[0] for row in data['features']],np.int64) for tree in p['trees']];prefix=(parent_dir/'prefix.bf16').read_bytes();h0={}
    for at,key in enumerate(keys):
        authenticate(req['ownH0'][key]);h0[key]=raw({**req['ownH0'][key],'shape':[512,width]}).astype(np.float64)
        if not np.array_equal(np.asarray(features(packets[key],h0[key],p['literals'])[2:],np.float64),data['features'][at*510:(at+1)*510]):raise ValueError('Actual own feature domain differs')
        for port,name in [('H','h'),('Z','z')]:
            if not np.array_equal(decode(req['fields']['parent-'+key+'-'+port])[2:],data[name][at*510:(at+1)*510]):raise ValueError('Actual parent/saved input join differs')
    gamma=torch.from_numpy(load(req['fields']['normGamma']).copy()).view(torch.bfloat16).cuda();gamma_hash=tensor_bytes_sha256(gamma)
    if not np.array_equal(gamma.float().cpu().numpy().astype(np.float64),data['gamma']):raise ValueError('Saved ideal/source gamma differs')
    tensors={k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in data.items()};tensors.update({'labels'+str(s):torch.from_numpy(x) for s,x in enumerate(labels)})
    before={k:tensor_bytes_sha256(v) for k,v in tensors.items()};frame=Frame.from_values(tensors,program_fingerprint=req['requestSHA256']);captured={}
    abis={'r0':{'shape':[14,width],'dtype':'float64'},'r1':{'shape':[9,width],'dtype':'float64'},'q':{'shape':[n,width],'dtype':'float64'},'Jq':{'shape':[n,width],'dtype':'float64'}}
    spec=ActSpec('cauchy_writer.joint_constants','v1',reads=tuple(tensors),writes=tuple(abis),write_abis=abis,numerical_contract=POLICY)
    def construct(values):
        source_data={k:values[k].numpy() for k in data};r,q,jq,info=propose(source_data,[values['labels'+str(s)].numpy() for s in (0,1)],WORK['atomCounts'],req['rmsNormEpsilon']);captured.update(info)
        return {k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in {'r0':r[0],'r1':r[1],'q':q,'Jq':jq}.items()},{'candidateRays':1,'routeFits':0,'originalLinears':0,'fullForwardJacobianApplications':1}
    registry=ActRegistry();registry.register(spec,'FP64-fixed-route-full-J',construct);tick=time.perf_counter();result,evidence=registry.run(spec.identity,'FP64-fixed-route-full-J',frame);proposal_seconds=time.perf_counter()-tick
    if before!={k:tensor_bytes_sha256(v) for k,v in tensors.items()}:raise ValueError('Proposal input mutated')
    for k in abis:retain(k,result.values[k])
    new_atoms=admit(atoms,[result.values['r'+str(s)].numpy() for s in (0,1)],captured['alpha']);candidate_dir=root/'candidate'
    if captured['alpha']==0:restore_program(parent_dir,candidate_dir)
    else:emit_program(root,'candidate',p['trees'],new_atoms,p['literals'],prefix,p['leafOccupancy'])
    candidate_program=read(candidate_dir/'program.json');candidate_sha=sha(candidate_dir/'program.json')
    if any(candidate_program[k]!=p[k] for k in p if k!='dictionaries'):raise ValueError('Candidate changed frozen program schema/routes')
    norm_parent=Frame.from_values({'normWeight':gamma},program_fingerprint=req['requestSHA256']);norm_before=norm_parent.fingerprint;typed_norm=[];candidate_states={};norm_fields={}
    normspec=ActSpec('cauchy_writer.source_operand','v1',reads=('H1','normWeight'),writes=('Z',),numerical_contract=SOURCE_POLICY,write_abis={'Z':{'shape':[1,512,width],'dtype':'bfloat16'}})
    def normalize(values):
        h=values['H1'];work=h.float();work=work*torch.rsqrt(work.pow(2).mean(-1,keepdim=True)+req['rmsNormEpsilon']);z=values['normWeight']*work.to(h.dtype)
        if z.dtype!=torch.bfloat16 or list(z.shape)!=[1,512,width] or not z.is_contiguous() or not torch.isfinite(z).all():raise ValueError('Source-policy output differs')
        return {'Z':z},{'normalizationUnits':1,'originalLinears':0,'nominalMatrixMACs':0}
    norms=ActRegistry();norms.register(normspec,'original-normalization-only',normalize)
    def consume(key,value):
        words=(value.astype('<f4').view('<u4')>>16).astype('<u2')
        if not np.array_equal(floats(words).astype(np.float64),value):raise ValueError('Expected actual BF16 candidate')
        h=torch.from_numpy(words[None].copy()).view(torch.bfloat16).cuda();hf=retain('candidate-'+key+'-H',h)
        if hf['payloadSHA256']==req['fields']['parent-'+key+'-H']['payloadSHA256']:
            z=decode(req['fields']['parent-'+key+'-Z']);norm_fields[key]={'H':hf,'Z':req['fields']['parent-'+key+'-Z'],'alias':'exact-parent-H'};candidate_states[key]=(value,z);return z
        f=norm_parent.with_updates({'H1':h},act='cauchy_writer.install:v1',implementation='actual-two-BF16-adds',metadata={'case':key,'programSHA256':candidate_sha});hashes={k:tensor_bytes_sha256(v) for k,v in f.values.items()};torch.cuda.synchronize();tick=time.perf_counter()
        with torch.inference_mode():nf,ne=norms.run(normspec.identity,'original-normalization-only',f)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        if hashes!={k:tensor_bytes_sha256(v) for k,v in f.values.items()}:raise ValueError('Normalization input mutated')
        zf=retain('candidate-'+key+'-Z',nf.values['Z']);z=floats(nf.values['Z'].cpu().view(torch.uint16).numpy())[0].astype(np.float64)
        typed_norm.append({'case':key,'spec':normspec.to_dict(),'input':f.to_dict(),'output':nf.to_dict(),'evidence':ne.to_dict(),'inputPayloadBefore':hashes,'inputPayloadAfter':hashes,'inputField':hf,'outputField':zf,'seconds':seconds})
        norm_fields[key]={'H':hf,'Z':zf};candidate_states[key]=(value,z);return z
    middle_parent=[];middle_candidate=[]
    for at,key in enumerate(keys):
        old=h0[key][2:];new=old
        for slot in (0,1):
            a=labels[slot][at*510:(at+1)*510];old=finite_add_array(old,atoms[slot][a]);new=finite_add_array(new,new_atoms[slot][a])
            if slot==0:middle_parent.append(old.astype(np.float64));middle_candidate.append(new.astype(np.float64))
        if not np.array_equal(old.astype(np.float64),data['h'][at*510:(at+1)*510]):raise ValueError('Two-add actual parent replay differs')
        consume(key,np.concatenate([floats(np.frombuffer(prefix,'<u2').reshape(2,width)).astype(np.float64),new.astype(np.float64)]))
    rows=lambda values,at:np.concatenate([values[k][at][2:] for k in keys])
    w,edges,v=[data[k] for k in ('weights','edges','pairWeights')];scales=dict(zip(TERMS,data['scaleVector'].tolist()));finite_score=objective(rows(candidate_states,0),rows(candidate_states,1),data['target'],data['zt'],w,edges,v,scales)
    selected='candidate' if finite_score['score']<captured['parent']['score'] else 'parent';selected_dir=candidate_dir if selected=='candidate' else parent_dir;restored=restore_program(selected_dir,root/'accepted')
    selection={'parent':captured['parent'],'candidate':finite_score,'selected':selected,'parentProgramSHA256':sha(parent_dir/'program.json'),'candidateProgramSHA256':candidate_sha,'acceptedProgramSHA256':sha(root/'accepted/program.json'),'restoredFiles':restored,'actualReduction':captured['parent']['score']-finite_score['score'],'policy':'Strict lower discovery finite score;parent wins ties. No transfer selection.'}
    save(out/'selection.json',selection);print('DISCOVERY_SELECTED',selected,finite_score['score'],flush=True)
    source['forbiddenServingPaths']=sorted(set(source['forbiddenServingPaths'])|{item['path'] for item in [*req['fields'].values(),*req['ownH0'].values(),*manifest.values()]})
    fresh,serving_seconds=serve(root,source,req['cases'],{'parent':parent_dir,'candidate':candidate_dir,'accepted':root/'accepted'},'final')
    for row in fresh['rows']:
        key=row['id'];pred={x['method']:x for x in row['predictions']}
        if row['packet']!=packets[key]:raise ValueError('Fresh raw packet differs from historical own input')
        if key in keys and (pred['candidate']['sha256']!=norm_fields[key]['H']['sha256'] or pred['parent']['sha256']!=req['fields']['parent-'+key+'-H']['sha256']):raise ValueError('Fresh serving/discovery numerical H differs')
        if pred['accepted']['sha256']!=pred[selected]['sha256']:raise ValueError('Fresh accepted alias differs')
    save(out/'release-seal.json',{'selectionSHA256':sha(out/'selection.json'),'servingSealSHA256':sha(root/'final-seal.json'),'candidateProgramSHA256':candidate_sha,'acceptedProgramSHA256':sha(root/'accepted/program.json'),'scope':'Selection and all owned serving outputs fixed before known-transfer numeric release.'});seal=sha(out/'release-seal.json');released['transfer']=True;print('TRANSFER_RELEASE',seal,flush=True)
    for row in fresh['rows']:
        if row['id'] in req['transfer']:
            pred=next(p for p in row['predictions'] if p['method']=='candidate');consume(row['id'],raw({**pred,'shape':[512,width]}).astype(np.float64))
    if len(typed_norm)>6 or tensor_bytes_sha256(gamma)!=gamma_hash or norm_parent.fingerprint!=norm_before:raise ValueError('Bounded work/immutable gamma differs')
    summaries=[];pairs=[];observations=[]
    for key in keys+req['transfer']:
        mask=np.asarray(packets[key]['mask'],bool);learned=np.arange(512)>=2;target={port:decode(req['fields']['target-'+key+'-'+port]) for port in ('H','Z')}
        for method in ('parent','candidate'):
            for port,at in [('H',0),('Z',1)]:
                pred=decode(req['fields']['parent-'+key+'-'+port]) if method=='parent' else candidate_states[key][at]
                for label,take in [('all',np.ones(512,bool)),('learned',learned),('active',learned&mask),('padding',learned&~mask)]:summaries.append({'case':key,'method':method,'port':port,'stratum':label,'rows':int(take.sum()),**metric(pred[take],target[port][take])})
                for position in range(512):observations.append({'case':key,'method':method,'port':port,'position':position,'active':bool(mask[position]),**metric(pred[position],target[port][position])})
    for left,right in [('A','B'),('00','10'),('01','11')]:
        same=np.asarray(packets[left]['ids'])==packets[right]['ids'];mask=np.asarray(packets[left]['mask'],bool);learned=np.arange(512)>=2
        for method in ('parent','candidate'):
            for port,at in [('H',0),('Z',1)]:
                def state(key):return decode(req['fields']['parent-'+key+'-'+port]) if method=='parent' else candidate_states[key][at]
                delta=state(right)-state(left);target=decode(req['fields']['target-'+right+'-'+port])-decode(req['fields']['target-'+left+'-'+port])
                for label,take in [('all',np.ones(512,bool)),('unchangedLearned',same&learned),('unchangedActive',same&learned&mask),('unchangedPadding',same&learned&~mask)]:pairs.append({'pair':[left,right],'method':method,'port':port,'stratum':label,'rows':int(take.sum()),**metric(delta[take],target[take])})
                for position in range(512):observations.append({'pair':[left,right],'method':method,'port':port,'position':position,'active':bool(mask[position]),'sameToken':bool(same[position]),**metric(delta[position],target[position])})
    actual=rows(candidate_states,0)-data['h'];atom_change=sum(((new_atoms[s].astype(np.float64)-atoms[s].astype(np.float64))[labels[s]] for s in (0,1)),np.zeros_like(actual));ideal=captured['alpha']*result.values['q'].numpy();mid=np.concatenate(middle_candidate)-np.concatenate(middle_parent);radius=np.sqrt(np.mean(data['h']**2,axis=1)+req['rmsNormEpsilon'])
    diagnostic=np.column_stack([np.sum(data['gradient']*x,axis=1)/2 for x in (ideal,atom_change,actual,atom_change-ideal,actual-atom_change)]+[np.sum(x*x,axis=1) for x in (ideal,atom_change,actual,mid)]+[np.sqrt(np.mean(actual*actual,axis=1))/radius]);retain('admission-diagnostics',torch.from_numpy(diagnostic))
    diagnostic_columns=['slope.ideal','slope.atomAdmitted','slope.stateAdmitted','slope.atomRounding','slope.twoStateAdmissions','squaredNorm.ideal','squaredNorm.atomAdmitted','squaredNorm.stateAdmitted','squaredNorm.middleChange','actualRelativeStep']
    diagnostic_rows=[{'case':keys[i//510],'position':i%510+2,'active':bool(packets[keys[i//510]]['mask'][i%510+2]),'values':row.tolist()} for i,row in enumerate(diagnostic)]
    with gzip.open(out/'observations.jsonl.gz','xt') as f:
        for row in observations:f.write(json.dumps(row,allow_nan=False)+'\n')
    save(out/'admission.json',{'columns':diagnostic_columns,'rows':diagnostic_rows,'globalSums':dict(zip(diagnostic_columns,np.sum(diagnostic,axis=0).tolist())),'actualMaximumRelativeStep':float(diagnostic[:,-1].max()),'idealMaximumRelativeStep':captured['alpha']*captured['rowRelativeTau']})
    save(out/'typed-proposal.json',{'spec':spec.to_dict(),'input':frame.to_dict(),'output':result.to_dict(),'evidence':evidence.to_dict(),'inputPayloadBefore':before,'inputPayloadAfter':before,'inputABIs':{k:{'shape':list(t.shape),'dtype':str(t.dtype),'payloadSHA256':before[k]} for k,t in tensors.items()},'seconds':proposal_seconds})
    save(out/'typed-normalization.json',{'rows':typed_norm,'weightParent':norm_parent.to_dict(),'parameterPayloadBefore':gamma_hash,'parameterPayloadAfter':tensor_bytes_sha256(gamma),'fields':norm_fields})
    def bundle(directory):return {'directory':str(directory),'program':read(directory/'program.json'),'programSHA256':sha(directory/'program.json'),'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in directory.iterdir()}}
    save(out/'programs.json',{k:bundle(p) for k,p in [('parent',parent_dir),('candidate',candidate_dir),('accepted',root/'accepted')]});save(out/'array-manifest.json',manifest)
    shutil.copyfile(root/'final/receipt.json',out/'final-serving.json');shutil.copyfile(root/'final-seal.json',out/'final-seal.json')
    for name in req['retainedMetadata']:shutil.copyfile(root/name,out/name)
    members=['selection.json','release-seal.json','observations.jsonl.gz','admission.json','typed-proposal.json','typed-normalization.json','programs.json','array-manifest.json','final-serving.json','final-seal.json'];save(out/'analysis-seal.json',{n:sha(out/n) for n in members})
    save(out/'report.json',{'schema':req['schema'],'requestSHA256':req['requestSHA256'],'policy':POLICY,'physicalWork':WORK,'proposal':captured,'selection':selection,'summaries':summaries,'pairs':pairs,'observationRows':len(observations),'normalizationUnits':len(typed_norm),'normalizationAliases':{k:v['alias'] for k,v in norm_fields.items() if 'alias' in v},'originalLinears':0,'nominalMatrixMACs':0,'images':0,'sourcePolicy':SOURCE_POLICY,'numericalProfile':profile,'torchVersion':torch.__version__,'numpyVersion':np.__version__,'authenticatedInputs':authenticated,'proposalActSeconds':proposal_seconds,'freshServingSeconds':serving_seconds,'releaseSealSHA256':seal,'analysisSealSHA256':sha(out/'analysis-seal.json'),'workerWallSecondsAfterImportsAndVerification':time.perf_counter()-start,'cpuSeconds':time.process_time()-cpu,'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'retainedNumericFields':len(manifest),'retainedNumericFileBytes':sum(x['bytes'] for x in manifest.values()),'mechanics_status':'completed','instrument_status':'actual-finite-normalization-feedback-with-reused-known-transfer','trend_status':'retain-terms-strata-admissions-and-rejected-future','terminal_status':'arbitrary-prompt-zero-neural-images-unachieved','limitations':['One joint constant step,not general convergence.','Targets,gradient,gamma and original source-policy normalization paid offline.','Fixed routes cannot separate identical H0/two-leaf collisions.','Known transfer reused,not fresh heldout;novel prompt unscored.','No QKV,attention,MLP,fullencoder,denoiser,VAE,image or runtime-saving claim.']})
    print('CAUCHY_WRITER_COMPLETE',flush=True)
