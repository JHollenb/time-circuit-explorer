"""Consume authenticated saved vectors on Beast; emit compact signed accounts."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import numpy as np
from template_cofactor_program import read,save,sha
from contextual_writer_program import route
from writer_direction_math import TERMS,DIRECTIONS,GRADIENTS,POLICY,components,analyze,columns


def floats(words):return (np.asarray(words,'<u2').astype('<u4')<<16).view('<f4').astype(np.float64)


def execute(root):
    from writer_direction_study import verify,WORK,FITS,CASES,CANDIDATES
    import torch
    from saturn.titan import Frame,ActSpec,ActRegistry
    from saturn.tensor_hash import tensor_bytes_sha256
    root=Path(root).resolve();req=read(root/'request.json');verify(root,req);start=time.perf_counter();cpu=time.process_time();torch.set_num_threads(1)
    if req['physicalWork']!=WORK or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Declared work/thread policy differs')
    out=root/'response-result';out.mkdir();folder=root/'arrays';folder.mkdir();authenticated=[]
    for path,item in req['remoteInputs'].items():
        p=Path(path)
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Saved input changed: '+path)
        authenticated.append({'path':path,**item})
    catalog=read(root/'parent-programs.json');original_arrays=read(root/'parent-array-manifest.json');futures=read(root/'parent-futures.json');prior=read(root/'parent-report.json');fits=read(root/'parent-typed-proposals.json')
    if any(original_arrays[k]!=v for k,v in req['fields'].items()) or prior['selected']!='parent' or [x['id'] for x in futures]!=list(CANDIDATES):raise ValueError('Saved four-future field binding differs')
    cache={}
    def load(item):
        key=(item['sha256'],tuple(item['shape']),item['storedDtype'],item['format'])
        if key not in cache:
            p=Path(item['path']);a=np.load(p,allow_pickle=False) if item['format']=='npy' else np.fromfile(p,dtype=item['storedDtype']).reshape(item['shape'])
            if a.dtype.str!=item['storedDtype'] or list(a.shape)!=item['shape'] or hashlib.sha256(a.tobytes()).hexdigest()!=item['payloadSHA256'] or not np.isfinite(a).all():raise ValueError('Saved numerical ABI/payload differs')
            cache[key]=a
        return cache[key]
    data={k:load(req['fields']['fit0-'+k]) for k in FITS if k!='oldSlot'}
    for k in FITS:
        if k!='oldSlot' and any(req['fields']['fit0-'+k][n]!=req['fields']['fit1-'+k][n] for n in ('payloadSHA256','shape','storedDtype')):raise ValueError('Two actual fit aliases differ')
    n,width=data['h'].shape
    if (n,width)!=(2040,2560) or data['features'].shape!=(2040,155) or data['edges'].shape!=(1017,2):raise ValueError('Exact110 discovery geometry differs')
    packets={r['id']:r['packet'] for r in read(root/'parent-final-serving.json')['rows']};positions=[]
    for case in CASES:
        for position in range(2,512):positions.append({'case':case,'position':position,'active':bool(packets[case]['mask'][position])})
    if not np.array_equal(data['features'][:,0],[r['position'] for r in positions]) or not np.array_equal(data['features'][:,1],[r['active'] for r in positions]):raise ValueError('Feature/history ordering differs')
    expected_edges=[]
    for at,(left,right) in enumerate((('A','B'),('00','10'))):
        same=np.asarray(packets[left]['ids'][2:])==packets[right]['ids'][2:];mask=np.asarray(packets[left]['mask'][2:],bool)
        for take in (same&mask,same&~mask):
            for pos in np.flatnonzero(take):expected_edges.append((2*at*510+int(pos),(2*at+1)*510+int(pos)))
    if not np.array_equal(data['edges'],expected_edges):raise ValueError('Original oriented unchanged-token edge order differs')
    paired=set(data['edges'].ravel().tolist())
    for at,row in enumerate(positions):row['paired']=at in paired
    for port,name in [('H','h'),('Z','z')]:
        actual=np.concatenate([floats(load(req['fields']['parent-'+case+'-'+port]))[0,2:] for case in CASES])
        if not np.array_equal(actual,data[name]):raise ValueError('Typed fit parent does not match actual finite parent '+port)
    del actual
    tensors={k:torch.from_numpy(np.ascontiguousarray(a)) for k,a in data.items()};route_work=0;writer_joins=[]
    parent=catalog['parent'];p=parent['program'];base=Path(parent['directory'])
    if read(base/'program.json')!=p:raise ValueError('Parent program metadata differs')
    parent_atoms=[np.fromfile(base/d['path'],'<f4').reshape(d['atoms'],width) for d in p['dictionaries']]
    for slot in (0,1):
        old=load(req['fields']['fit'+str(slot)+'-oldSlot']);routed=[route(p['trees'][slot],row) for row in data['features']];labels=np.asarray([x[0] for x in routed]);route_work+=sum(x[1] for x in routed)
        if not np.array_equal(parent_atoms[slot][labels].astype(np.float64),old) or fits[slot]['sourceProgramSHA256']!=parent['programSHA256']:raise ValueError('Actual parent slot gather differs')
        tensors['oldSlot'+str(slot)]=torch.from_numpy(np.ascontiguousarray(old))
    for future in futures:
        name=future['id'];slot=future['slot'];program=catalog[name];cp=program['program'];base=Path(program['directory']);other=1-slot
        if future['sourceProgramSHA256']!=parent['programSHA256'] or future['programSHA256']!=program['programSHA256'] or read(base/'program.json')!=cp or any(cp[k]!=p[k] for k in ('prefixSHA256','budget','literals')) or cp['trees'][other]!=p['trees'][other] or cp['dictionaries'][other]['sha256']!=p['dictionaries'][other]['sha256']:raise ValueError('Changed-slot/unchanged-slot closure differs')
        d=cp['dictionaries'][slot];atoms=np.fromfile(base/d['path'],'<f4').reshape(d['atoms'],width);routed=[route(cp['trees'][slot],row) for row in data['features']];labels=np.asarray([x[0] for x in routed]);route_work+=sum(x[1] for x in routed)
        if not np.isfinite(atoms).all():raise ValueError('Nonfinite actual candidate atoms')
        tensors[name+'-write']=torch.from_numpy(np.ascontiguousarray(atoms[labels].astype(np.float64)))
        words=np.concatenate([load(req['fields'][name+'-'+case+'-H'])[0,2:] for case in CASES])
        tensors[name+'-H']=torch.from_numpy(np.ascontiguousarray(words)).view(torch.bfloat16)
        writer_joins.append({'candidate':name,'slot':slot,'dose':future['dose'],'parentProgramSHA256':parent['programSHA256'],'candidateProgramSHA256':program['programSHA256'],'changedDictionary':d,'unchangedDictionarySHA256':cp['dictionaries'][other]['sha256']})
    before={k:tensor_bytes_sha256(v) for k,v in tensors.items()};frame=Frame.from_values(tensors,program_fingerprint=req['requestSHA256'],metadata={'caseOrder':list(CASES),'rowsPerCase':510,'savedInputs':req['fields'],'writerJoins':writer_joins})
    writes=tuple(name+'-products' for name in CANDIDATES)+('gradient-reconstruction',)
    spec=ActSpec('writer_direction.saved_futures','v1',reads=tuple(tensors),writes=writes,numerical_contract=POLICY,
        write_abis={k:{'shape':[2040,2 if k=='gradient-reconstruction' else 52],'dtype':'float64'} for k in writes})
    captured={};registry=ActRegistry()
    def act(values):
        source={k:values[k].numpy() for k in data};parts=components(source,req['rmsNormEpsilon']);g=source['gradient'];combined=sum(parts.values(),np.zeros_like(g));difference=combined-g
        gradient_rows=np.column_stack([np.sum(difference*difference,axis=1,dtype=np.float64),np.max(np.abs(difference),axis=1)])
        result={'gradient-reconstruction':torch.from_numpy(np.ascontiguousarray(gradient_rows))};work=0
        for future in futures:
            name=future['id'];u=values[name+'-write'].numpy()-values['oldSlot'+str(future['slot'])].numpy();v=floats(values[name+'-H'].view(torch.uint16).numpy())-source['h']
            rows,diagnostic=analyze(g,parts,float(future['dose'])*source['step'],u,v);work+=diagnostic['coordinateProducts'];result[name+'-products']=torch.from_numpy(rows)
        captured.update({'directionCoordinateProducts':work,'gradientReconstructionNormProducts':difference.size,'idealPullbackCalls':2,'idealRadiusReconstructions':2,'gradientReconstructionRMS':float(np.sqrt(gradient_rows[:,0].sum()/difference.size)),'gradientReconstructionMaximumAbsolute':float(gradient_rows[:,1].max())})
        return result,{'newNeuralEvaluations':0,'originalNormalizationInvocations':0,**captured}
    registry.register(spec,'saved-input-FP64-explanation',act);tick=time.perf_counter();result,evidence=registry.run(spec.identity,'saved-input-FP64-explanation',frame);act_seconds=time.perf_counter()-tick
    after={k:tensor_bytes_sha256(v) for k,v in tensors.items()}
    if before!=after or captured['directionCoordinateProducts']!=WORK['directionCoordinateProducts']:raise ValueError('Input immutability/work account differs')
    manifest={}
    for name in writes:
        tensor=result.values[name]
        a=tensor.numpy();path=folder/(name+'.npy');np.save(path,a,allow_pickle=False)
        manifest[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'payloadSHA256':tensor_bytes_sha256(tensor),'shape':list(a.shape),'storedDtype':a.dtype.str,'dtype':str(tensor.dtype),'format':'npy'}
    groups={'all':list(range(n)),'active':[i for i,r in enumerate(positions) if r['active']],'padding':[i for i,r in enumerate(positions) if not r['active']]}
    for case in CASES:
        for label,active in [('all',None),('active',True),('padding',False)]:groups[case+':'+label]=[i for i,r in enumerate(positions) if r['case']==case and (active is None or r['active']==active)]
    summaries=[];count=0
    with gzip.open(out/'observations.jsonl.gz','xt') as stream:
        for future in futures:
            name=future['id'];a=result.values[name+'-products'].numpy()
            for i,row in enumerate(positions):stream.write(json.dumps({'candidate':name,**row,'values':a[i].tolist()},allow_nan=False)+'\n');count+=1
            for group,indices in groups.items():
                block=a[indices];sums=np.sum(block,axis=0,dtype=np.float64);sums[-1]=np.max(block[:,-1]);summary=dict(zip(columns(),sums.tolist()))
                summaries.append({'candidate':name,'group':group,'rows':len(indices),'values':summary,
                    'localModelSlopes':{g+'.'+d:summary[g+'.'+d]/2 for g in GRADIENTS for d in DIRECTIONS}})
    objectives=[{'candidate':f['id'],'slot':f['slot'],'dose':f['dose'],'finiteScoreChange':f['score']-prior['initialObjective']['score'],
                 'finiteTermChanges':{k:f['terms'][k]-prior['initialObjective']['terms'][k] for k in TERMS},'parent':prior['initialObjective'],'candidateObjective':{k:f[k] for k in ('score','terms','errorEnergies','nativeEnergies')}} for f in futures]
    save(out/'analysis.json',{'columns':columns(),'groups':groups,'positions':positions,'summaries':summaries,'objectives':objectives,'gradientReconstruction':captured,'gradientReconstructionColumns':['squaredNorm','maximumAbsolute'],'scope':POLICY})
    save(out/'typed-analysis.json',{'spec':spec.to_dict(),'input':frame.to_dict(),'output':result.to_dict(),'evidence':evidence.to_dict(),'inputPayloadBefore':before,'inputPayloadAfter':after,
        'inputABIs':{k:{'shape':list(t.shape),'dtype':str(t.dtype),'payloadSHA256':before[k]} for k,t in tensors.items()},'writerJoins':writer_joins})
    save(out/'array-manifest.json',manifest)
    for name in req['retainedMetadata']:shutil.copyfile(root/name,out/name)
    members=['analysis.json','typed-analysis.json','array-manifest.json','observations.jsonl.gz'];save(out/'analysis-seal.json',{n:sha(out/n) for n in members})
    save(out/'report.json',{'schema':req['schema'],'requestSHA256':req['requestSHA256'],'physicalWork':WORK,'policy':POLICY,'actualWork':captured,'routeDecisions':route_work,'analysisActSeconds':act_seconds,'observationRows':count,
        'authenticatedInputs':authenticated,'uniqueNumericInputsLoaded':len(cache),'uniqueNumericInputPayloadBytes':sum(a.nbytes for a in cache.values()),'retainedNumericFiles':len(manifest),'retainedNumericFileBytes':sum(x['bytes'] for x in manifest.values()),
        'workerWallSecondsAfterImportsAndRequestVerification':time.perf_counter()-start,'cpuSeconds':time.process_time()-cpu,'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'torchVersion':torch.__version__,'numpyVersion':np.__version__,'analysisSealSHA256':sha(out/'analysis-seal.json'),'mechanics_status':'completed','instrument_status':'saved-fields-owned-signed-local-model-account','trend_status':'retain-all-directions-strata-and-finite-disagreement','terminal_status':'arbitrary-prompt-zero-neural-images-unachieved',
        'limitations':['Ideal pullback/radius/gamma arithmetic remains paid offline;zero new source-policy normalization invocation is not zero normalization-related arithmetic.','Saved native targets/states and historical original work remain paid.','No finite BF16 derivative,causal110 attribution from scalar signs alone,capacity theorem,full encoder or image capability.','No tokenizer,lexical table,H0,checkpoint shard,transfer target,new fit,dose,model or image is read/executed.']})
    print('WRITER_DIRECTION_COMPLETE',flush=True)
