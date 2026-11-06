"""Chronological causal value means from compiled lexical operands; no Q/K read."""
from pathlib import Path
import hashlib
import resource
import sys
import time
from encoder_atoms import np, Prefix, guard, read, save, sha, floats, words
from paired_fields import retain

POLICY = 'uniform-causal-values-fp64-fp32-bf16-v1'
CONVERSION = 'Chronological FP64 sum/division, then FP64->FP32->BF16 RNE via encoder_atoms.words'


def stream(values, mask, positions, state=None, *, groups=8, heads=32, dim=128, source_identity='mechanics'):
    values=np.asarray(values);mask=np.asarray(mask);positions=np.asarray(positions)
    if groups<1 or dim<1 or heads<groups or heads%groups:raise ValueError('Invalid grouped head geometry')
    if values.ndim!=2 or values.shape[1]!=groups*dim or values.dtype!=np.uint16:raise ValueError('BF16 value geometry differs')
    if mask.shape!=(len(values),) or positions.shape!=mask.shape or not np.issubdtype(positions.dtype,np.integer) or np.any((mask!=0)&(mask!=1)):raise ValueError('Mask/position contract differs')
    if not isinstance(source_identity,str):raise ValueError('String source identity required')
    tag={'policy':POLICY,'sourceIdentity':source_identity,'geometry':[groups,heads,dim]}
    if state is None:state={**tag,'sums':np.zeros((groups,dim),dtype=np.float64),'count':np.asarray(0,dtype=np.int64),'clock':np.asarray(0,dtype=np.int64)}
    else:state={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in state.items()}
    for key in ['policy','sourceIdentity']:
        value=np.asarray(state[key])
        if value.shape!=() or value.item()!=tag[key]:raise ValueError('State law/source differs')
    if not np.array_equal(np.asarray(state['geometry']),tag['geometry']):raise ValueError('State grouped-head geometry differs')
    state.update(tag)
    if state['sums'].shape!=(groups,dim) or state['sums'].dtype!=np.float64 or not np.isfinite(state['sums']).all():raise ValueError('Finite FP64 sum state required')
    for name in ['count','clock']:
        if state[name].shape!=() or state[name].dtype!=np.int64:raise ValueError('Integer count/clock required')
    count=int(state['count']);clock=int(state['clock'])
    if not 0<=count<=clock or not np.array_equal(positions,np.arange(clock,clock+len(values))):raise ValueError('State causal clock differs')
    if count==0 and np.any(state['sums']):raise ValueError('Empty summary must be zero')
    decoded=floats(values).astype(np.float64).reshape(len(values),groups,dim)
    if not np.isfinite(decoded).all():raise ValueError('Nonfinite compiled value')
    raw=np.zeros((len(values),heads*dim),dtype=np.float64);counts=np.empty(len(values),dtype=np.int64);nonempty=0
    for i in range(len(values)):
        if mask[i]:state['sums']+=decoded[i];count+=1
        counts[i]=count
        if count:raw[i]=np.repeat(state['sums']/count,heads//groups,axis=0).reshape(-1);nonempty+=1
    state['count']=np.asarray(count,dtype=np.int64);state['clock']=np.asarray(clock+len(values),dtype=np.int64)
    return {'raw':raw,'rounded':words(raw),'prefixCounts':counts},state,{
        'admittedKeys':int(mask.sum()),'emittedQueryRows':len(values),'valueSumAdditions':int(mask.sum())*groups*dim,
        'meanDivisions':nonempty*groups*dim,'expandedQueryScalarWrites':len(values)*heads*dim,
        'sumPayloadBytes':state['sums'].nbytes,'countPayloadBytes':state['count'].nbytes,'clockPayloadBytes':state['clock'].nbytes,
        'coreStatePayloadBytes':sum(state[k].nbytes for k in ['sums','count','clock']),
        'decodedInputPayloadBytes':decoded.nbytes,'rawOutputPayloadBytes':raw.nbytes,'roundedOutputPayloadBytes':raw.size*2,
        'prefixCountDiagnosticBytes':counts.nbytes,'queryKeyScoreProducts':0,'retrievalIndexBytes':0,'momentEnvelopeBytes':0,'fallbackCalls':0}


def direct_means(values, mask, groups=8, heads=32, dim=128):
    # Independent prefix reconstruction; np.add.accumulate preserves chronological FP64 addition.
    v=floats(values).astype(np.float64).reshape(len(values),groups,dim);out=np.zeros((len(v),heads*dim))
    for i in range(len(v)):
        prefix=v[:i+1][np.asarray(mask[:i+1],dtype=bool)]
        if len(prefix):out[i]=np.repeat(np.add.accumulate(prefix,axis=0)[-1]/len(prefix),heads//groups,axis=0).reshape(-1)
    return out


def execute(root):
    from uniform_reader_study import verify
    from flux_clock import deny_targets
    root=Path(root).resolve();request=read(root/'request.json');verify(root,request);source=read(root/'inputs.json')
    if request['readerPolicy']!=POLICY:raise ValueError('Frozen uniform numerical policy differs')
    guard();deny_targets([root/'targets.json',*source['forbiddenTargets']])
    started=time.perf_counter();cpu=time.process_time();prefix=Prefix(source['lexicalStore'],request['manifestSHA256'])
    if len(prefix.store.ids)!=151936 or [prefix.m[k] for k in ['headDim','qWidth','kvWidth','sequenceLength']]!=[128,4096,1024,512]:raise ValueError('Qualified complete lexical geometry required')
    lexical=root/'lexical';lexical.mkdir();rows=[];fields={};states=[];packets=[];work=[]
    for prompt in request['prompts']:
        tick=time.perf_counter();values,text=prefix.forward(prompt['prompt']);seconds=time.perf_counter()-tick
        path=lexical/(prompt['id']+'.npz');np.savez(path,**values)
        rows.append({'id':prompt['id'],'prompt':prompt['prompt'],'path':str(path),'sha256':sha(path),'activeTokens':int(values['mask'].sum()),
            'firstArithmeticSeconds':seconds,'renderedTextSHA256':hashlib.sha256(text.encode()).hexdigest(),
            'idsPayloadSHA256':hashlib.sha256(values['ids'].tobytes()).hexdigest(),'maskPayloadSHA256':hashlib.sha256(values['mask'].tobytes()).hexdigest()})
        tick=time.perf_counter();packet,state,counts=stream(values['v'],values['mask'],values['positions'],source_identity=request['manifestSHA256']);arithmetic=time.perf_counter()-tick
        tick=time.perf_counter();reference=direct_means(values['v'],values['mask']);diagnostic=time.perf_counter()-tick
        packet.update(directRaw=reference,directRounded=words(reference))
        target=root/'reader';target.mkdir(exist_ok=True);out=target/(prompt['id']+'-mean.npz');np.savez(out,**packet)
        state_path=target/(prompt['id']+'-state.npz');np.savez(state_path,**state)
        packets.append({'id':prompt['id'],'path':str(out),'sha256':sha(out)})
        states.append({'id':prompt['id'],'path':str(state_path),'sha256':sha(state_path),'count':int(state['count']),'clock':int(state['clock'])})
        f=retain(root/'arrays/reader',prompt['id']+'-head',packet['rounded'][None,:,:]);f['logicalDtype']='torch.bfloat16';fields[prompt['id']]=f
        work.append({'id':prompt['id'],**counts,'arithmeticSeconds':arithmetic,'independentDiagnosticSeconds':diagnostic,
            'directRawMaxAbsDifference':float(np.max(np.abs(packet['raw']-reference))),
            'directBF16WordMismatches':int(np.count_nonzero(packet['rounded']!=packet['directRounded']))})
    save(lexical/'prediction-seal.json',{'manifestSHA256':request['manifestSHA256'],'rows':rows,'coverageCount':len(prefix.store.ids),
        'uniqueRowsRead':prefix.store.rowsRead,'compiledRowPayloadBytesRead':prefix.store.rowBytesRead,
        'frameworkCalls':0,'originalDenseProjections':0,'scope':'Same bundled compiled E/QKV/position path; only V used by uniform contextual reader.'})
    m=prefix.store.manifest;atom_files=[{'path':str(Path(source['lexicalStore'])/'manifest.json'),'sha256':request['manifestSHA256']}]
    atom_files += [{'path':str(Path(source['lexicalStore'])/m[k]['path']),'sha256':m[k]['sha256']} for k in ['tokenizer','template','coverage','rowHashes','rope']]
    atom_files += [{'path':str(Path(source['lexicalStore'])/m['shards'][j]['path']),'sha256':m['shards'][j]['sha256']} for j in sorted(prefix.store.shards)]
    if {'rank_one_tail','attention_bank','selector_objective','torch','transformers','diffusers'}.intersection(sys.modules):raise ValueError('Disallowed reader/runtime imported')
    save(root/'reader-seal.json',{'schema':'tce-uniform-reader-seal-v1','requestSHA256':request['requestSHA256'],'policy':POLICY,'conversion':CONVERSION,
        'lexicalSealSHA256':sha(lexical/'prediction-seal.json'),'fields':fields,'lexicalRows':rows,'sourceAtomFiles':atom_files,'states':states,'packets':packets,'work':work,
        'workerSecondsAfterImports':time.perf_counter()-started,'cpuSecondsAfterImports':time.process_time()-cpu,'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'numpyVersion':np.__version__,'frameworkCalls':0,'originalDenseProjections':0,
        'scope':'Uniform live causal V summaries; noQK/anchors/top4/KVindex/quadraticmoments/envelopes. Core state is not total working set; bundled lexical/input/output/diagnostics paid.'})
    print('UNIFORM_HEADS_AND_STATE_SEALED',sha(root/'reader-seal.json'),flush=True)
