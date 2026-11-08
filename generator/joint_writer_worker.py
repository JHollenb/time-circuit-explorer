"""Bounded two-slot futures, actual finite normalization, fresh owned serving."""
import collections
import copy
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
from contextual_writer_program import BUDGET,FINITE,features,route,finite_add_array,read,save,sha,restore_program
from contextual_writer_worker import native,raw,floats,metric,serve,emit_program
from contrast_writer_fit import fit_contrast_tree
from coupled_normalization_worker import configure,gamma_only
from coupled_normalization import SOURCE_POLICY
from joint_writer_math import POLICY,DOSES,objective,propose,dose_atoms,choose,candidate_add,CandidateAdmissionError

SCALE_KEYS=('Hendpoint','Hresponse','Zendpoint','Zresponse')


def panel(keys,pairs,packets):
    weights=[];indices=[];edge_weights=[];offset={k:i*510 for i,k in enumerate(keys)}
    for key in keys:
        mask=np.asarray(packets[key]['mask'][2:],bool)
        if not mask.any() or mask.all():raise ValueError('Expected both active and padding discovery strata')
        weights.extend(np.where(mask,1/(2*len(keys)*int(mask.sum())),1/(2*len(keys)*int((~mask).sum()))))
    for left,right in pairs:
        if packets[left]['mask']!=packets[right]['mask']:raise ValueError('Paired masks differ')
        same=np.asarray(packets[left]['ids'][2:])==packets[right]['ids'][2:];mask=np.asarray(packets[left]['mask'][2:],bool)
        for take in (same&mask,same&~mask):
            positions=np.flatnonzero(take)
            if not len(positions):raise ValueError('Paired stratum empty')
            for pos in positions:indices.append((offset[left]+int(pos),offset[right]+int(pos)));edge_weights.append(1/(2*len(pairs)*len(positions)))
    return np.asarray(weights,np.float64),np.asarray(indices,np.int64).reshape(-1,2),np.asarray(edge_weights,np.float64)


def execute(root):
    from joint_writer_study import verify,identity,WORK
    import torch
    from saturn.titan import Frame,ActSpec,ActRegistry
    from saturn.tensor_hash import tensor_bytes_sha256
    root=Path(root).resolve();req=read(root/'request.json');verify(root,req);start=time.perf_counter();cpu=time.process_time();profile=configure(torch,req)
    if req['physicalWork']!=WORK or req['budget']!=BUDGET or req['objectivePolicy']!=POLICY or req['finitePolicy']!=FINITE:raise ValueError('Frozen construction differs')
    out=root/'response-result';out.mkdir();folder=root/'arrays';folder.mkdir();source=read(root/'parent-parent-inputs.json');released={'transfer':False}
    forbidden={source['cases'][k][n]['path'] for k in req['transfer'] for n in ('H0','H1')}
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and not released['transfer'] and str(Path(os.fsdecode(args[0])).resolve()) in forbidden:raise PermissionError('Transfer numeric fields quarantined before final seal')
    sys.addaudithook(audit);(root/'lexical').symlink_to(req['lexicalDirectory'],target_is_directory=True);authenticated=[]
    for path,digest in req['servingAssets'].items():
        if sha(path)!=digest:raise ValueError('Selected serving bytes changed')
        authenticated.append({'path':path,'sha256':digest,'bytes':Path(path).stat().st_size})
    source['forbiddenServingPaths']=sorted(set(source['forbiddenServingPaths'])|{p['path'] for r in read(root/'parent-final-serving.json')['rows'] for p in [r['H0'],*r['predictions']]})
    base=Path(req['parent']['program']['directory']);initial,initial_seconds=serve(root,source,[r for r in req['cases'] if r['id'] in req['discovery']],{'parent':base},'initializer')
    packets={r['id']:r['packet'] for r in initial['rows']};h0={};targets={};parent_h={};prefix=(base/'prefix.bf16').read_bytes();literals=read(base/'program.json')['literals']
    actual_literals=sorted({t for row in initial['rows'] for t,m in zip(row['packet']['ids'],row['packet']['mask']) if m})[:64]
    if actual_literals!=literals:raise ValueError('Frozen literal feature set differs')
    for row in initial['rows']:
        key=row['id'];h0[key]=raw({**row['H0'],'shape':[512,2560]}).astype(np.float64);targets[key]=native(source['cases'][key]['H1']).astype(np.float64)
        if not np.array_equal(h0[key],native(source['cases'][key]['H0'])):raise ValueError('Fresh H0 differs from compiled lexical authority')
        for label,expected in [('ids',source['cases'][key]['tokenIDsSHA256']),('mask',source['cases'][key]['maskSHA256'])]:
            if hashlib.sha256(np.asarray(row['packet'][label],'<i8').tobytes()).hexdigest()!=expected:raise ValueError('Fresh packet differs')
        parent_h[key]=raw({**row['predictions'][0],'shape':[512,2560]}).astype(np.float64)
    manifest={};typed_norm=[];outputs={};norm_fields={};fit_records=[];futures=[];catalog={};selection=[]
    def retain(name,value,raw_bf16=False):
        value=value.detach().cpu().contiguous();data=value.view(torch.uint16).numpy() if raw_bf16 else value.numpy();path=folder/(name+('.bf16' if raw_bf16 else '.npy'))
        if raw_bf16:path.write_bytes(data.tobytes())
        else:np.save(path,data,allow_pickle=False)
        item={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'payloadSHA256':tensor_bytes_sha256(value),'shape':list(value.shape),'dtype':str(value.dtype),'storedDtype':data.dtype.str,'format':'raw' if raw_bf16 else 'npy'}
        if hashlib.sha256(data.tobytes()).hexdigest()!=item['payloadSHA256']:raise ValueError('Retained tensor payload differs')
        manifest[name]=item;return item
    def bf16(value):
        words=(np.asarray(value,'<f4').view('<u4')>>16).astype('<u2')
        if not np.array_equal(floats(words).astype(np.float64),np.asarray(value,np.float64)):raise ValueError('Expected actual BF16 state')
        return torch.from_numpy(words.copy()).view(torch.bfloat16)
    gamma,load=gamma_only(req,torch);gamma_hash=tensor_bytes_sha256(gamma);retain('gamma',gamma,True)
    weight_parent=Frame.from_values({'normWeight':gamma},program_fingerprint=req['requestSHA256']);weight_id=weight_parent.fingerprint
    normspec=ActSpec('joint_writer.source_operand','v1',reads=('H1','normWeight'),writes=('Z',),numerical_contract=SOURCE_POLICY,write_abis={'Z':{'shape':[1,512,2560],'dtype':'bfloat16'}})
    def normalize(values):
        h=values['H1'];work=h.float();work=work*torch.rsqrt(work.pow(2).mean(-1,keepdim=True)+req['rmsNormEpsilon']);norm=values['normWeight']*work.to(h.dtype)
        if norm.dtype!=torch.bfloat16 or list(norm.shape)!=[1,512,2560] or not norm.is_contiguous() or not torch.isfinite(norm).all():raise ValueError('Source operand ABI differs')
        return {'Z':norm},{'normalizationUnits':1,'originalLinears':0,'nominalMatrixMACs':0,'inputShape':list(h.shape),'inputStride':list(h.stride()),'outputStride':list(norm.stride())}
    registry=ActRegistry();registry.register(normspec,'original-normalization-only',normalize)
    def consume(name,value,program=None):
        h=bf16(value[None]).cuda();input_field=retain(name+'-H',h,True);frame=weight_parent.with_updates({'H1':h},act='joint_writer.H1-install:v1',implementation='actual-finite',metadata={'state':name,'programSHA256':program})
        before={k:tensor_bytes_sha256(v) for k,v in frame.values.items()};torch.cuda.synchronize();tick=time.perf_counter()
        with torch.inference_mode():result,evidence=registry.run(normspec.identity,'original-normalization-only',frame)
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick;after={k:tensor_bytes_sha256(v) for k,v in frame.values.items()}
        if before!=after or before['normWeight']!=gamma_hash:raise ValueError('Actual immutable normalization inputs changed')
        output_field=retain(name+'-Z',result.values['Z'],True);z=floats(result.values['Z'].cpu().view(torch.uint16).numpy())[0].astype(np.float64)
        typed_norm.append({'state':name,'spec':normspec.to_dict(),'input':frame.to_dict(),'output':result.to_dict(),'evidence':evidence.to_dict(),'inputPayloadBefore':before,'inputPayloadAfter':after,'secondsSynchronizedRegistryOnly':elapsed,'inputField':input_field,'outputField':output_field,'programSHA256':program})
        outputs[name]=(value,z);norm_fields[name]={'H':input_field,'Z':output_field};return z
    for key in req['discovery']:consume('target-'+key,targets[key]);consume('parent-'+key,parent_h[key],req['parent']['program']['programSHA256'])
    keys=req['discovery'];w,edges,v=panel(keys,[('A','B'),('00','10')],packets);x=np.asarray([f for key in keys for f in features(packets[key],h0[key],literals)[2:]],np.float64)
    rows=lambda data:np.concatenate([data[k][2:] for k in keys],axis=0)
    target=rows(targets);zt=rows({k:outputs['target-'+k][1] for k in keys});scales=objective(target,zt,target,zt,w,edges,v)['nativeEnergies']
    def bundle(path):
        p=read(path/'program.json');return {'directory':str(path),'program':p,'programSHA256':sha(path/'program.json'),'files':{'program.json':sha(path/'program.json'),'prefix.bf16':sha(path/'prefix.bf16'),**{d['path']:sha(path/d['path']) for d in p['dictionaries']}}}
    catalog['parent']=bundle(base);current='parent';current_path=base;current_h=parent_h;current_z={k:outputs['parent-'+k][1] for k in keys};initial_score=objective(rows(current_h),rows(current_z),target,zt,w,edges,v,scales)
    program_frame=Frame.from_values({'programSHA256':catalog['parent']['programSHA256']},program_fingerprint=req['requestSHA256'])
    for slot in (0,1):
        p=read(current_path/'program.json');atoms=[np.fromfile(current_path/d['path'],'<f4').reshape(d['atoms'],2560) for d in p['dictionaries']]
        labels=np.asarray([route(p['trees'][slot],row)[0] for row in x]);old_slot=atoms[slot][labels].astype(np.float64);inputs={'h':rows(current_h),'z':rows(current_z),'target':target,'zt':zt,'oldSlot':old_slot,'features':x,'weights':w,'edges':edges,'pairWeights':v,
            'gamma':gamma.float().cpu().numpy().astype(np.float64),'scaleVector':np.asarray([scales[k] for k in SCALE_KEYS],np.float64)}
        tensors={k:torch.from_numpy(np.ascontiguousarray(a)) for k,a in inputs.items()};fields={k:retain('fit'+str(slot)+'-'+k,t) for k,t in tensors.items()};before={k:tensor_bytes_sha256(t) for k,t in tensors.items()}
        frame=Frame.from_values(tensors,program_fingerprint=req['requestSHA256'],metadata={'slot':slot,'programSHA256':catalog[current]['programSHA256'],'inputProgram':current})
        spec=ActSpec('joint_writer.propose_slot'+str(slot),'v1',reads=tuple(tensors),writes=('gradient','step','fullCandidateSHA','halfCandidateSHA'),numerical_contract=POLICY);captured={};made={};unavailable={}
        def construct(values):
            data={k:t.numpy() for k,t in values.items()};bound_scales=dict(zip(SCALE_KEYS,data['scaleVector'].tolist()))
            gradient,step,diagnostic=propose(data['h'],data['z'],data['target'],data['zt'],data['weights'],data['edges'],data['pairWeights'],data['gamma'],req['rmsNormEpsilon'],bound_scales)
            try:
                tree,_,occupancy,fit=fit_contrast_tree(data['features'],data['oldSlot']+step,data['weights'],data['edges'],data['pairWeights'],bound_scales['Hendpoint'],bound_scales['Hresponse'])
            except ValueError as error:
                if str(error)!='Nonfinite FP32 atom admission':raise
                fit={'status':'construction-incomplete','reason':str(error)};projection=None;dictionaries=[None,None]
            else:
                new_labels=np.asarray([route(tree,row)[0] for row in data['features']],np.int64);dictionaries,projection=dose_atoms(new_labels,len(occupancy),data['oldSlot'],step,data['weights'],data['edges'],data['pairWeights'],bound_scales)
            for at,(dose,dictionary) in enumerate(zip(DOSES,dictionaries)):
                name='slot'+str(slot)+'-dose'+('1' if at==0 else 'half')
                if dictionary is None:
                    unavailable[name]={'status':'construction-incomplete' if projection is None else 'numerically-inadmissible','stage':'tree-fit' if projection is None else 'FP32-dictionary-admission','reason':fit['reason'] if projection is None else projection['admissions'][at]['reason']};continue
                trees=copy.deepcopy(p['trees']);trees[slot]=tree;writes=[a.copy() for a in atoms];writes[slot]=dictionary
                occupied=copy.deepcopy(p['leafOccupancy']);occupied[slot]=[{**{k:v for k,v in item.items() if k!='rows'},'histories':dict(collections.Counter(keys[i//510] for i in item['rows']))} for item in occupancy]
                made[name]=emit_program(root,name,trees,writes,literals,prefix,occupied);catalog[name]=bundle(made[name])
            captured.update({'proposal':diagnostic,'fit':fit,'projection':projection,'parent':current,'slot':slot,'sourceProgramSHA256':catalog[current]['programSHA256']})
            return {'gradient':torch.from_numpy(np.ascontiguousarray(gradient)),'step':torch.from_numpy(np.ascontiguousarray(step)),
                    'fullCandidateSHA':catalog.get('slot'+str(slot)+'-dose1',{}).get('programSHA256'),'halfCandidateSHA':catalog.get('slot'+str(slot)+'-dosehalf',{}).get('programSHA256')},{'slot':slot,'treeFitAttempts':1,'candidatePrograms':len(made),'newOriginalLinears':0}
        actor=ActRegistry();actor.register(spec,'paired-proposal-and-H-graph-projection',construct);tick=time.perf_counter();result,evidence=actor.run(spec.identity,'paired-proposal-and-H-graph-projection',frame);elapsed=time.perf_counter()-tick
        if before!={k:tensor_bytes_sha256(t) for k,t in tensors.items()}:raise ValueError('Proposal inputs mutated')
        outputs_fields={k:retain('fit'+str(slot)+'-'+k,result.values[k]) for k in ('gradient','step')}
        fit_records.append({'slot':slot,'input':frame.to_dict(),'output':result.to_dict(),'spec':spec.to_dict(),'evidence':evidence.to_dict(),'inputPayloadBefore':before,'inputPayloadAfter':before,'inputFields':fields,'outputFields':outputs_fields,'candidateSHAs':{k:result.values[k] for k in ('fullCandidateSHA','halfCandidateSHA')},'seconds':elapsed,**captured})
        parent_row={'id':current,**objective(rows(current_h),rows(current_z),target,zt,w,edges,v,scales)};candidates=[];candidate_states={};candidate_frames={}
        for at,dose in enumerate(DOSES):
            name='slot'+str(slot)+'-dose'+('1' if at==0 else 'half');common={'id':name,'slot':slot,'dose':dose,'sourceProgramSHA256':catalog[current]['programSHA256']}
            if name in unavailable:
                futures.append({**common,**unavailable[name],'completedHistories':[],'score':None,'programSHA256':None});continue
            path=made[name]
            state={};zstate={};cp=read(path/'program.json');cd=[np.fromfile(path/d['path'],'<f4').reshape(d['atoms'],2560) for d in cp['dictionaries']]
            try:
                for key in keys:
                    value=h0[key][2:]
                    for tree,dictionary in zip(cp['trees'],cd):value=candidate_add(value,dictionary[np.asarray([route(tree,row)[0] for row in features(packets[key],h0[key],literals)[2:]])])
                    full=np.concatenate([floats(np.frombuffer(prefix,'<u2').reshape(2,2560)).astype(np.float64),value.astype(np.float64)]);state[key]=full;zstate[key]=consume(name+'-'+key,full,catalog[name]['programSHA256'])
            except CandidateAdmissionError as error:
                futures.append({**common,'status':'numerically-inadmissible','stage':'BF16-state-admission','reason':str(error),'completedHistories':list(zstate),'score':None,'programSHA256':catalog[name]['programSHA256']});continue
            item={**common,'status':'complete','completedHistories':list(zstate),'programSHA256':catalog[name]['programSHA256'],**objective(rows(state),rows(zstate),target,zt,w,edges,v,scales)}
            futures.append(item);candidates.append(item);candidate_states[name]=(state,zstate);candidate_frames[name]=program_frame.with_updates({'programSHA256':catalog[name]['programSHA256']},act='joint_writer.candidate:v1',implementation='bounded-proposal',metadata={'slot':slot,'candidate':name})
        selected=choose(parent_row,candidates);selected_path=current_path if selected['id']==current else made[selected['id']];restored=root/('selected-slot'+str(slot));restored_files=restore_program(selected_path,restored)
        selection.append({'slot':slot,'parent':parent_row,'candidates':candidates,'selected':selected['id'],'parentFrame':program_frame.to_dict(),'candidateFrames':{k:f.to_dict() for k,f in candidate_frames.items()},'selectedFrame':(program_frame if selected['id']==current else candidate_frames[selected['id']]).to_dict(),'restoredFiles':restored_files,'restoredDirectory':str(restored),'programSHA256':sha(restored/'program.json')})
        if selected['id']!=current:current_h,current_z=candidate_states[selected['id']];program_frame=candidate_frames[selected['id']]
        current=selected['id'];current_path=restored;print('SLOT_SELECTED',slot,current,selected['score'],flush=True)
    restore_program(current_path,root/'accepted');catalog['accepted']=bundle(root/'accepted')
    final,final_seconds=serve(root,source,req['cases'],{'parent':base,'accepted':root/'accepted'},'final');seal=sha(root/'final-seal.json')
    for row in final['rows']:
        key=row['id'];pred={p['method']:p for p in row['predictions']};packets[key]=row['packet']
        if key in keys:
            for method,name in [('parent','parent'),('accepted',current)]:
                expected=norm_fields[name+'-'+key]['H']
                if pred[method]['sha256']!=expected['sha256']:raise ValueError('Fresh serving differs from actual finite candidate H')
    released['transfer']=True;print('TRANSFER_TARGET_RELEASE',seal,flush=True)
    for row in final['rows']:
        key=row['id']
        if key not in req['transfer']:continue
        if hashlib.sha256(np.asarray(row['packet']['ids'],'<i8').tobytes()).hexdigest()!=source['cases'][key]['tokenIDsSHA256'] or hashlib.sha256(np.asarray(row['packet']['mask'],'<i8').tobytes()).hexdigest()!=source['cases'][key]['maskSHA256']:raise ValueError('Transfer packet differs')
        targets[key]=native(source['cases'][key]['H1']).astype(np.float64);consume('target-'+key,targets[key])
        for pred in row['predictions']:consume(pred['method']+'-'+key,raw({**pred,'shape':[512,2560]}).astype(np.float64),catalog[pred['method']]['programSHA256'])
    if not 14<=len(typed_norm)<=WORK['maximumNormalizationUnits'] or len(futures)!=4 or tensor_bytes_sha256(gamma)!=gamma_hash or weight_parent.fingerprint!=weight_id:raise ValueError('Actual work/gamma closure differs')
    summaries=[];pairs=[];count=0;all_rows=np.ones(512,bool);learned=np.arange(512)>=2
    def actual(case,method):return outputs[('parent' if method=='parent' else current)+'-'+case] if case in keys else outputs[method+'-'+case]
    with gzip.open(out/'observations.jsonl.gz','xt') as stream:
        for key in keys+req['transfer']:
            mask=np.asarray(packets[key]['mask'],bool)
            for method in ('parent','accepted'):
                for port,at in [('H',0),('Z',1)]:
                    pred=actual(key,method)[at];target_value=outputs['target-'+key][at]
                    for label,take in [('all',all_rows),('learned',learned),('active',mask),('padding',~mask)]:summaries.append({'case':key,'method':method,'port':port,'selection':label,'rows':int(take.sum()),**metric(pred[take],target_value[take])})
                    for position in range(512):stream.write(json.dumps({'case':key,'method':method,'port':port,'position':position,'active':bool(mask[position]),**metric(pred[position],target_value[position])},allow_nan=False)+'\n');count+=1
        for left,right in [('A','B'),('00','10'),('01','11')]:
            same=np.asarray(packets[left]['ids'])==packets[right]['ids'];mask=np.asarray(packets[left]['mask'],bool)
            for method in ('parent','accepted'):
                for port,at in [('H',0),('Z',1)]:
                    pred=actual(right,method)[at]-actual(left,method)[at];target_value=outputs['target-'+right][at]-outputs['target-'+left][at]
                    for label,take in [('all',all_rows),('unchangedToken',same),('unchangedLearned',same&learned),('unchangedActive',same&learned&mask),('unchangedPadding',same&learned&~mask)]:pairs.append({'pair':[left,right],'method':method,'port':port,'selection':label,'rows':int(take.sum()),**metric(pred[take],target_value[take])})
                    for position in range(512):stream.write(json.dumps({'pair':[left,right],'method':method,'port':port,'position':position,'sameToken':bool(same[position]),'active':bool(mask[position]),**metric(pred[position],target_value[position])},allow_nan=False)+'\n');count+=1
    save(out/'typed-normalization.json',{'rows':typed_norm,'weightParent':weight_parent.to_dict(),'parameterPayloadBefore':gamma_hash,'parameterPayloadAfter':tensor_bytes_sha256(gamma)})
    save(out/'typed-proposals.json',fit_records);save(out/'selection.json',selection);save(out/'futures.json',futures);save(out/'programs.json',catalog);save(out/'array-manifest.json',manifest)
    for phase in ('initializer','final'):
        shutil.copyfile(root/(phase+'-seal.json'),out/(phase+'-seal.json'));shutil.copyfile(root/phase/'receipt.json',out/(phase+'-serving.json'))
    for name in req['retainedMetadata']:shutil.copyfile(root/name,out/name)
    members=['typed-normalization.json','typed-proposals.json','selection.json','futures.json','programs.json','array-manifest.json','observations.jsonl.gz','initializer-seal.json','final-seal.json','initializer-serving.json','final-serving.json']
    save(out/'analysis-seal.json',{n:sha(out/n) for n in members})
    save(out/'report.json',{'schema':req['schema'],'requestSHA256':req['requestSHA256'],'physicalWork':WORK,'objectivePolicy':POLICY,'sourcePolicy':SOURCE_POLICY,'budget':BUDGET,'initialObjective':initial_score,'selected':current,'selectedDiscoveryObjective':objective(rows(current_h),rows(current_z),target,zt,w,edges,v,scales),
        'summaries':summaries,'pairs':pairs,'observationRows':count,'normalizationUnits':len(typed_norm),'proposalFitAttempts':len(fit_records),'candidatePrograms':len(catalog)-2,'candidateFuturesAttempted':len(futures),'completeCandidateFutures':sum(f['status']=='complete' for f in futures),'analysisSealSHA256':sha(out/'analysis-seal.json'),'transferReleasedAfterSealSHA256':seal,
        'authenticatedServingAssets':authenticated,'gammaLoad':load,'initialFreshServingSeconds':initial_seconds,'finalFreshServingSeconds':final_seconds,'workerWallSecondsAfterImportsAndRequestVerification':time.perf_counter()-start,'cpuSeconds':time.process_time()-cpu,
        'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'numericalProfile':profile,'torchVersion':torch.__version__,'numpyVersion':np.__version__,'retainedNumericFileBytes':sum(v['bytes'] for v in manifest.values()),
        'mechanics_status':'completed','instrument_status':'inspect-whole-discovery-transfer-endpoints-responses-and-finite-feedback','trend_status':'retain-both-doses-parents-all-strata-and-regressions','terminal_status':'arbitrary-prompt-zero-neural-images-unachieved',
        'limitations':['One approximate alternating sweep,not global joint optimization.','Ideal Jacobian and isotropic surrogate propose only;Hgraph projection differs from full surrogate.','Native targets and original gamma/normalization paid offline;all serving uses own lexical/history features and two finite additions.','Known transfer is not newly held out;novel raw prompt unscored.','No QKV,attention,MLP,fullencoder,denoiser,VAE,image or matched speedup.']})
    print('JOINT_WRITER_COMPLETE',flush=True)
