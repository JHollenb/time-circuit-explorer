"""State-fed selected-head attention with an explicit frozen context remainder."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

from state_receiver import np, read, sha, write, audit_reads, quantize, peak_mib, read_state, receive


def tensor(ref, verified):
    path = Path(ref['path'])
    if str(path) not in verified:
        if sha(path) != ref['sha256']:
            raise ValueError('Context tensor file changed')
        verified[str(path)]=ref['sha256']
    elif verified[str(path)]!=ref['sha256']:
        raise ValueError('Conflicting context file identity')
    with path.open('rb') as stream:
        length = int.from_bytes(stream.read(8), 'little')
        if not 2 <= length <= 16 << 20:
            raise ValueError('Invalid context header')
        header = json.loads(stream.read(length))[ref['tensor']]
        first, last = header['data_offsets']
        stream.seek(8 + length + first)
        raw = stream.read(last-first)
    if header['shape'] != ref['shape'] or header['dtype'] != ref['dtype'] or hashlib.sha256(raw).hexdigest() != ref['tensorSHA256']:
        raise ValueError('Context tensor payload/shape/dtype differs')
    if header['dtype'] == 'BF16':
        value = (np.frombuffer(raw, '<u2').astype(np.uint32) << 16).view(np.float32)
    elif header['dtype'] == 'F32':
        value = np.frombuffer(raw, '<f4').copy()
    else:
        raise ValueError('Unsupported context scalar type')
    value = value.reshape(ref['shape'])
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite context')
    return value


def norm_rope(value, gamma, cos, sin, eps):
    wide = value.astype(np.float64)
    normalized = quantize(wide / np.sqrt(np.mean(wide*wide, axis=-1, keepdims=True)+eps) * gamma)
    rotated = np.empty_like(normalized)
    rotated[:, 0::2], rotated[:, 1::2] = -normalized[:, 1::2], normalized[:, 0::2]
    # Native reference: separate FP32 multiplications and addition, then BF16.
    return quantize(np.add(np.multiply(normalized, cos, dtype=np.float32),
                           np.multiply(rotated, sin, dtype=np.float32), dtype=np.float32))


def prepare(raw, reader, eps):
    return {'q': norm_rope(raw['q'], reader['gamma_q'], reader['cos'], reader['sin'], eps),
            'k': norm_rope(raw['k'], reader['gamma_k'], reader['cos'], reader['sin'], eps), 'v': raw['v']}


def full_attention(fields, chunk=64):
    q, k, v = [fields[name].astype(np.float64) for name in ['q','k','v']]
    output = np.empty_like(q)
    for first in range(0, len(q), chunk):
        score = q[first:first+chunk] @ k.T / np.sqrt(float(q.shape[-1]))
        score -= score.max(axis=-1, keepdims=True)
        probability = np.exp(score)
        probability /= probability.sum(axis=-1, keepdims=True)
        output[first:first+chunk] = probability @ v
    return output


def fixed_cache(fixed, changed, chunk=64):
    unchanged = np.setdiff1d(np.arange(len(fixed['q'])), changed)
    q, k, v = [fixed[name][unchanged].astype(np.float64) for name in ['q','k','v']]
    numerator, partition, maximum = np.empty_like(q), np.empty((len(q),1)), np.empty((len(q),1))
    for first in range(0,len(q),chunk):
        score = q[first:first+chunk] @ k.T / np.sqrt(float(q.shape[-1]))
        peak = score.max(axis=-1,keepdims=True)
        exp = np.exp(score-peak)
        numerator[first:first+chunk] = exp @ v
        partition[first:first+chunk] = exp.sum(axis=-1,keepdims=True)
        maximum[first:first+chunk] = peak
    return {'unchanged':unchanged, 'changed':changed, 'numerator':numerator, 'partition':partition, 'maximum':maximum}


def merge(fixed, changed, fields):
    result = {name: value.copy() for name,value in fixed.items()}
    for name in ['q','k','v']:
        result[name][changed] = fields[name]
    return result


def incremental(fields, fixed, cache):
    changed, unchanged = cache['changed'], cache['unchanged']
    q, k, v = [fields[name].astype(np.float64) for name in ['q','k','v']]
    score = fixed['q'][unchanged].astype(np.float64) @ k.T / np.sqrt(float(q.shape[-1]))
    maximum = np.maximum(cache['maximum'], score.max(axis=-1,keepdims=True))
    rescale = np.exp(cache['maximum']-maximum)
    exp = np.exp(score-maximum)
    numerator = cache['numerator']*rescale + exp @ v
    partition = cache['partition']*rescale + exp.sum(axis=-1,keepdims=True)
    output = np.empty_like(fixed['q'],dtype=np.float64)
    output[unchanged] = numerator/partition
    # Changed queries must recompute their interactions with ALL sources.
    complete = merge(fixed, changed, fields)
    score = q @ complete['k'].astype(np.float64).T / np.sqrt(float(q.shape[-1]))
    score -= score.max(axis=-1,keepdims=True)
    probability = np.exp(score)
    probability /= probability.sum(axis=-1,keepdims=True)
    output[changed] = probability @ complete['v'].astype(np.float64)
    return output


def load_bundle(ref):
    if sha(ref['path']) != ref['sha256']:
        raise ValueError('Compiled global-consumer input changed')
    with np.load(ref['path'],allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def compile_head(spec_path, output):
    started,cpu = time.perf_counter(),time.process_time()
    spec,output = read(spec_path),Path(output)
    allowed = [spec_path,spec['checkpointFile']]+[r['path'] for r in spec['operands'].values()]
    allowed += [spec['basis']['path']]+[r['path'] for r in spec['banks'].values()]
    opened = audit_reads(allowed,spec['sourceRoots'])
    verified={}
    operands = {k:tensor(ref,verified) for k,ref in spec['operands'].items()}
    rows,head,dim = spec['selectedRows'],spec['head'],spec['headDim']
    ntext,nimage,width = operands['text'].shape[1],operands['image'].shape[1],operands['image'].shape[-1]
    changed = np.asarray(rows,dtype=np.int64)+ntext
    fixed_image_rows = np.setdiff1d(np.arange(nimage),rows)
    normalized = {}
    for stream,indices in [('image',fixed_image_rows),('text',np.arange(ntext))]:
        state = operands[stream][:,indices].astype(np.float64)
        centered = state-state.mean(axis=-1,keepdims=True)
        value = quantize(centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+spec['eps']))
        shift,scale,_,_,_,_ = np.split(operands[stream+'Mod'].reshape(1,1,-1),6,axis=-1)
        normalized[stream] = quantize(quantize(quantize(1+scale)*value)+shift).astype(np.float64)
    import retrieve_weights as retrieval
    manifest = retrieval.validate_manifest(spec['parameters'])
    fields = {stream:{} for stream in ['image','text']}
    gammas,parameter_bytes = {},0
    for entry,source,size in retrieval._source_entries(manifest,spec['checkpointRoot']):
        raw = b''.join(retrieval._artifact_chunks(entry,source,size))
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise ValueError('Fixed-context parameter changed')
        word = np.load(io.BytesIO(raw),allow_pickle=False)
        weight = (word.astype(np.uint32)<<16).view(np.float32)
        name = entry['source']['tensor'].split('.attn.')[1]
        parameter_bytes += size
        if name.startswith('norm_'):
            gammas[name.removesuffix('.weight')] = weight.copy()
        else:
            stream = 'text' if name.startswith('add_') else 'image'
            field = name.split('_')[1] if stream=='text' else name.split('_')[1].split('.')[0]
            selected_weight = weight[head*dim:(head+1)*dim].astype(np.float64)
            fields[stream][field] = quantize(normalized[stream] @ selected_weight.T)[0]
        del raw,word,weight
    fixed = {name:np.zeros((ntext+nimage,dim),np.float32) for name in ['q','k','v']}
    for stream,indices,offset in [('image',fixed_image_rows,ntext),('text',np.arange(ntext),0)]:
        for name in ['q','k','v']:
            value = fields[stream][name]
            if name!='v':
                norm = 'norm_'+('added_' if stream=='text' else '')+name
                value = norm_rope(value,gammas[norm],operands['cos'][indices+offset],operands['sin'][indices+offset],spec['eps'])
            fixed[name][indices+offset] = value
    if any(np.any(a[changed]) for a in fixed.values()):
        raise ValueError('Selected rows must not contain native refresh values')
    reader = {'gamma_q':gammas['norm_q'],'gamma_k':gammas['norm_k'],
              'cos':operands['cos'][changed], 'sin':operands['sin'][changed]}
    if sha(spec['basis']['path'])!=spec['basis']['sha256']:
        raise ValueError('Producer basis changed')
    reader['basis'] = np.load(spec['basis']['path'],allow_pickle=False)
    for name,ref in spec['banks'].items():
        bank = load_bundle(ref)
        reader['bank_'+name] = bank['bank'][:,:,head*dim:(head+1)*dim]
        reader['offset_'+name] = bank['offset'][:,:,head*dim:(head+1)*dim]
    cache = fixed_cache(fixed,changed)
    for name,values in [('reader-head.npz',reader),('fixed-head.npz',fixed),('attention-cache.npz',cache)]:
        np.savez(output/name,**values)
    write(output/'compile-receipt.json', {'schema':'tce-global-head-compile-v1','specSHA256':sha(spec_path),
          'bundles':{name:{'path':name,'sha256':sha(output/name)} for name in ['reader-head.npz','fixed-head.npz','attention-cache.npz']},
          'bundlePayloadBytes':{'reader':sum(a.nbytes for a in reader.values()),'fixed':sum(a.nbytes for a in fixed.values()),'cache':sum(a.nbytes for a in cache.values())},
          'head':head,'headDim':dim,'imageRows':nimage,'textRows':ntext,'selectedRows':rows,
          'fullCheckpointBytesHashed':Path(spec['checkpointFile']).stat().st_size,'originalParameterPayloadBytes':parameter_bytes,
          'originalHeadProjectionMACs':3*(nimage-len(rows)+ntext)*width*dim,
          'fixedCacheAttentionMACs':2*(nimage+ntext-len(rows))**2*dim,
          'originalHeadProjectionEvaluations':6,'learnedHeadNormsInFixedPreparation':4,
          'wallSeconds':time.perf_counter()-started,'cpuSeconds':time.process_time()-cpu,'peakRSSMiB':peak_mib(),
          'opened':sorted(opened),'nativeFrameworkCalls':0,'offlineOriginalDenseLearnedWeightMath':True,'scope':spec['scope']})


def from_state(state,reader,eps):
    result = receive(state,reader['basis'],{n:reader['bank_'+n] for n in ['q','k','v']},
                     {n:reader['offset_'+n] for n in ['q','k','v']},eps)
    raw = {n:result[n][0] for n in ['q','k','v']}
    return raw,prepare(raw,reader,eps)


def serve(spec_path,output):
    started,cpu = time.perf_counter(),time.process_time()
    spec,output = read(spec_path),Path(output)
    if set(spec)!={'reader','fixed','cache','packets','eps','repeats','scope','sourceRoots'} or len(spec['packets'])!=2:
        raise ValueError('Global consumer accepts two opaque state packets and fixed context only')
    controls = ['unchanged_replay','keys_only','values_only','queries_only','keys_values','wrong_address']
    names = [('all',p['id']) for p in spec['packets']]+[(method,spec['packets'][1]['id']) for method in controls]
    paths = [spec_path,output/'prediction-seal.json']+[spec[k]['path'] for k in ['reader','fixed','cache']]
    paths += [p['path'] for p in spec['packets']]+[output/(method+'-'+pid+'.npz') for method,pid in names]
    opened = audit_reads(paths,spec['sourceRoots'],serving=True)
    reader,fixed,cache = [load_bundle(spec[k]) for k in ['reader','fixed','cache']]
    records,base_fields = [],None

    def retain(method,packet,raw,rounded,fields,samples,cpu_samples):
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite global head')
        path=output/(method+'-'+packet['id']+'.npz')
        np.savez(path,raw=raw,rounded=rounded,**{'changed_'+k:v for k,v in fields.items()})
        records.append({'method':method,'id':packet['id'],'path':path.name,'sha256':sha(path),
                        'medianArithmeticSeconds':float(np.median(samples)), 'medianCPUSeconds':float(np.median(cpu_samples)),
                        'arithmeticSeconds':samples})

    for packet in spec['packets']:
        state = read_state(packet)
        _,fields = from_state(state,reader,spec['eps'])
        quantize(incremental(fields,fixed,cache))
        samples,cpu_samples=[],[]
        for _ in range(spec['repeats']):
            wall0,cpu0=time.perf_counter(),time.process_time()
            raw_fields,fields=from_state(state,reader,spec['eps'])
            head=incremental(fields,fixed,cache)
            rounded=quantize(head)
            samples.append(time.perf_counter()-wall0);cpu_samples.append(time.process_time()-cpu0)
        retain('all',packet,head,rounded,fields,samples,cpu_samples)
        if base_fields is None:
            base_fields={k:v.copy() for k,v in fields.items()}
    for method in controls:
        if method=='wrong_address':
            chosen=prepare({k:v[np.roll(np.arange(len(v)),1)] for k,v in raw_fields.items()},reader,spec['eps'])
        else:
            active={'keys_only':{'k'},'values_only':{'v'},'queries_only':{'q'},'keys_values':{'k','v'},'unchanged_replay':set()}[method]
            chosen={k:fields[k] if k in active else base_fields[k] for k in ['q','k','v']}
        wall0,cpu0=time.perf_counter(),time.process_time();head=incremental(chosen,fixed,cache);rounded=quantize(head)
        retain(method,packet,head,rounded,chosen,[time.perf_counter()-wall0],[time.process_time()-cpu0])
    n,m,dim=len(fixed['q']),len(cache['changed']),fixed['q'].shape[1]
    write(output/'prediction-seal.json',{'schema':'tce-global-head-seal-v1','specSHA256':sha(spec_path),'rows':records,
          'opened':sorted(opened),'wallSecondsAfterImports':time.perf_counter()-started,'cpuSecondsAfterImports':time.process_time()-cpu,
          'peakRSSMiB':peak_mib(),'doseInput':False,'originalDenseProjections':0,'neuralFrameworkCalls':0,
          'learnedReducedMapsPerPrimaryEvaluation':3,'learnedHeadScaleApplicationsPerPrimaryEvaluation':2,
          'attentionMACsPerEvaluation':2*((n-m)*m+m*n)*dim,'fullAttentionMACsPerEvaluation':2*n*n*dim,
          'timingRepeats':spec['repeats'],'primaryWarmupsPerPacket':1,'controlsTimedOnceWithoutReader':True,
          'scope':spec['scope']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['compile','serve']);parser.add_argument('spec',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();(compile_head if args.action=='compile' else serve)(args.spec,args.output)
