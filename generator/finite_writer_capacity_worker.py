"""One CPU target-assisted capacity assay on retained, authenticated states."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import numpy as np
from contextual_writer_program import features, finite_add_array, read, save, sha, route, validate
from contextual_writer_worker import floats, native, raw, metric
from finite_writer_capacity import enumerate_capacity, SCORING


def words(value):
    return (np.asarray(value, '<f4').view('<u4') >> 16).astype('<u2')


def execute(root):
    from finite_writer_capacity_study import verify
    import torch
    from saturn.titan import Frame, ActSpec, ActRegistry
    from saturn.tensor_hash import tensor_bytes_sha256
    root = Path(root).resolve(); req = read(root/'request.json'); verify(root, req)
    started = time.perf_counter(); cpu0 = time.process_time(); torch.set_num_threads(1)
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')): raise ValueError('Expected one numerical CPU thread')
    out = root/'response-result'; out.mkdir(); arrays = root/'arrays'; arrays.mkdir()
    catalog = read(root/'parent-programs.json'); child = read(root/'parent-final-serving.json'); source = read(root/'parent-parent-inputs.json')
    program = catalog['accepted']; p = program['program']; parent_fit = read(root/'parent-fit.json')
    if program['programSHA256'] != catalog['candidate']['programSHA256'] or [x['atoms'] for x in p['dictionaries']] != [14,9]: raise ValueError('Frozen candidate family differs')
    authenticated = []
    for path, expected in req['remoteInputs'].items():
        q = Path(path)
        if q.stat().st_size != expected['bytes'] or sha(q) != expected['sha256']: raise ValueError('Retained input changed: '+path)
        authenticated.append({'path': path, **expected})
    folder = Path(program['directory'])
    if read(folder/'program.json') != p: raise ValueError('Actual program metadata differs')
    dictionaries = [np.fromfile(folder/x['path'], dtype='<f4').reshape(x['atoms'],2560) for x in p['dictionaries']]
    if any(not np.isfinite(x).all() for x in dictionaries): raise ValueError('Nonfinite frozen atoms')
    prefix = (folder/'prefix.bf16').read_bytes()
    if len(prefix) != 10240: raise ValueError('Fixed cofactor geometry differs')
    prepared = {}; owned = {}; targets = {}; packets = {}; initial = {}; input_reconstruction = []
    prep_tick = time.perf_counter()
    for key in req['cases']:
        row = next(x for x in child['rows'] if x['id'] == key); packet = row['packet']; validate(packet); item = source['cases'][key]
        for label, expected in [('ids',item['tokenIDsSHA256']),('mask',item['maskSHA256'])]:
            if hashlib.sha256(np.asarray(packet[label],'<i8').tobytes()).hexdigest() != expected: raise ValueError('Owned packet/native binding differs')
        if row['H0']['sha256'] != item['H0']['payloadSHA256']: raise ValueError('Owned H0/native identity differs')
        h0 = raw({**row['H0'],'shape':[512,2560]}); target = native(item['H1'])
        prediction = next(x for x in row['predictions'] if x['method'] == 'candidate')
        current = raw({**prediction,'shape':[512,2560]})
        x = features(packet,h0,p['literals']); routes = np.asarray([[route(tree,v)[0] for tree in p['trees']] for v in x[2:]],'<i8')
        if routes.tolist() != prediction['leaves']: raise ValueError('Owned source routes differ from sealed104 routes')
        stage0 = finite_add_array(h0[2:], dictionaries[0][routes[:,0]])
        reconstructed = finite_add_array(stage0,dictionaries[1][routes[:,1]])
        if words(current[:2]).tobytes() != prefix or words(target[:2]).tobytes() != prefix or not np.array_equal(reconstructed,current[2:]): raise ValueError('Actual finite candidate reconstruction differs')
        rebuilt_hash = hashlib.sha256(prefix+words(reconstructed).tobytes()).hexdigest()
        if rebuilt_hash != prediction['sha256']: raise ValueError('Reconstructed complete candidate bytes differ')
        prepared[key] = (h0[2:],stage0,target[2:],current[2:],routes[:,1]); owned[key] = current.astype(np.float64); targets[key] = target.astype(np.float64); packets[key] = packet; initial[key] = h0
        input_reconstruction.append({'case':key,'H0':row['H0'],'target':item['H1'],'current':prediction,
            'slot0PayloadSHA256':hashlib.sha256(words(stage0).tobytes()).hexdigest(),'currentReconstructedPayloadSHA256':rebuilt_hash})
    preparation_seconds = time.perf_counter()-prep_tick
    manifest = {}; typed = []; groups = []; capacity = {}; case_fields = []
    def tensor_file(name, values, logical=None):
        data = np.ascontiguousarray(values); path = arrays/(name+'.bin'); data.tofile(path)
        tensor = torch.from_numpy(data.copy()).view(torch.bfloat16) if logical == 'bfloat16' else torch.from_numpy(data.copy())
        item = {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'payloadSHA256':tensor_bytes_sha256(tensor),
                'shape':list(tensor.shape),'dtype':str(tensor.dtype),'storedDtype':data.dtype.str}
        if item['sha256'] != item['payloadSHA256']: raise ValueError('Actual typed tensor/file payload differs')
        manifest[name] = item; return tensor
    with gzip.open(out/'components.jsonl.gz','xt') as stream:
        for group_name, cases, pair_cases, frozen in [('discovery',req['discovery'],[('A','B'),('00','10')],True),('reusedTransfer',req['transfer'],[('01','11')],False)]:
            h0,stage0,y,current = [np.concatenate([prepared[k][j] for k in cases]).astype(np.float64) for j in range(4)]
            labels = np.concatenate([prepared[k][4] for k in cases]); keys = [(k,i) for k in cases for i in range(2,512)]; offsets = {k:i*510 for i,k in enumerate(cases)}
            weights = []; edges = []; pair_weights = []
            for key in cases:
                mask = np.asarray(packets[key]['mask'][2:],bool)
                if not mask.any() or mask.all(): raise ValueError('Endpoint strata empty')
                weights.extend(np.where(mask,.5/int(mask.sum()),.5/int((~mask).sum()))/len(cases))
            for a,b in pair_cases:
                if packets[a]['mask'] != packets[b]['mask']: raise ValueError('Paired masks differ')
                same = np.asarray(packets[a]['ids'][2:]) == np.asarray(packets[b]['ids'][2:]); mask = np.asarray(packets[a]['mask'][2:],bool)
                for take in (same & mask,same & ~mask):
                    positions = np.flatnonzero(take)
                    if not len(positions): raise ValueError('Paired strata empty')
                    for pos in positions:
                        edges.append((offsets[a]+int(pos),offsets[b]+int(pos))); pair_weights.append(1/(2*len(pair_cases)*len(positions)))
            w = np.asarray(weights,'<f8'); v = np.asarray(pair_weights,'<f8'); w /= w.sum(); v /= v.sum(); edge_array = np.asarray(edges,'<i8')
            frozen_energy = np.asarray([parent_fit['endpointEnergy'],parent_fit['responseEnergy']] if frozen else [0.,0.],'<f8')
            values = {}
            for name,value in [('H0',h0),('S',stage0),('Y',y),('current',current)]: values[name] = tensor_file(group_name+'-'+name,words(value),'bfloat16')
            for name,value in [('atoms',dictionaries[1]),('labels',labels.astype('<i8')),('weights',w),('edges',edge_array),('pairWeights',v),('frozenEnergies',frozen_energy)]: values[name] = tensor_file(group_name+'-'+name,value)
            frame = Frame.from_values(values,program_fingerprint=req['requestSHA256'],metadata={'group':group_name,'targetAssisted':True,'casePositionOrder':keys,'frozenDiscoveryEnergies':frozen})
            before = {k:tensor_bytes_sha256(t) for k,t in values.items()}; captured = {}
            spec = ActSpec('finite_writer.capacity','v1',reads=tuple(values),writes=('capacityH1','capacityLabels'),
                numerical_contract='target-assisted-frozen-dictionary-finite-admission-and-evaluated-component-enumeration')
            def act(inputs):
                def array(t): return floats(t.view(torch.uint16).numpy()).astype(np.float64) if t.dtype == torch.bfloat16 else t.numpy()
                args = [array(inputs[n]) for n in ('H0','S','Y','atoms','current','labels','weights','edges','pairWeights')]
                def emit(row): stream.write(json.dumps({'group':group_name,**row},allow_nan=False)+'\n')
                result,choices,diagnostic = enumerate_capacity(*args,array(inputs['frozenEnergies']) if frozen else None,emit)
                captured.update(diagnostic)
                return {'capacityH1':torch.from_numpy(words(result)).view(torch.bfloat16),'capacityLabels':torch.from_numpy(choices)}, {
                    'targetAssisted':True,'newNeuralEvaluations':0,'evaluatedChoices':diagnostic['evaluatedChoices'],
                    'finiteOptionCoordinateAdditions':diagnostic['finiteOptionCoordinateAdditions']}
            registry = ActRegistry(); registry.register(spec,'exhaustive-target-assisted',act); tick = time.perf_counter()
            after,evidence = registry.run(spec.identity,'exhaustive-target-assisted',frame); elapsed = time.perf_counter()-tick
            actual_after = {k:tensor_bytes_sha256(t) for k,t in values.items()}
            if actual_after != before: raise ValueError('Immutable capacity input payload changed')
            result_words = after.values['capacityH1'].view(torch.uint16).numpy(); result = floats(result_words).astype(np.float64)
            output_fields = {}
            for name in ('capacityH1','capacityLabels'):
                tensor = after.values[name]; data = tensor.view(torch.uint16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
                recorded = tensor_file(group_name+'-'+name,data,'bfloat16' if tensor.dtype == torch.bfloat16 else None)
                if tensor_bytes_sha256(recorded) != tensor_bytes_sha256(tensor): raise ValueError('Output custody differs')
                output_fields[name] = manifest[group_name+'-'+name]
            typed.append({'group':group_name,'spec':spec.to_dict(),'input':frame.to_dict(),'output':after.to_dict(),'evidence':evidence.to_dict(),
                'inputPayloadBefore':before,'inputPayloadAfter':actual_after,'fields':output_fields})
            groups.append({'group':group_name,'cases':cases,'pairCases':pair_cases,'casePositionOrder':keys,'enumerationActWallSeconds':elapsed,**captured})
            for index,key in enumerate(cases):
                full = np.concatenate([floats(np.frombuffer(prefix,'<u2')).reshape(2,2560),result[index*510:(index+1)*510]])
                capacity[key] = full; tensor_file('capacity-full-'+key,words(full),'bfloat16'); case_fields.append({'case':key,'field':manifest['capacity-full-'+key]})
            print('FINITE_CAPACITY_GROUP',group_name,'choices',captured['evaluatedChoices'],flush=True)
    summaries = []; pairs = []; row_count = 0
    with gzip.open(out/'observations.jsonl.gz','xt') as stream:
        for key in req['cases']:
            mask = np.asarray(packets[key]['mask'],bool); h0 = initial[key]
            for method,pred in [('current',owned[key]),('capacity',capacity[key])]:
                scale = targets[key]-h0
                for label,take in [('all',np.ones(512,bool)),('learned',np.arange(512)>=2),('active',mask),('padding',~mask)]:
                    summaries.append({'case':key,'method':method,'port':'H1','stratum':label,'rows':int(take.sum()),**metric(pred[take],targets[key][take],scale[take])})
                for i in range(512): stream.write(json.dumps({'case':key,'method':method,'port':'H1','position':i,'active':bool(mask[i]),**metric(pred[i],targets[key][i],scale[i])},allow_nan=False)+'\n'); row_count+=1
        for a,b in [('A','B'),('00','10'),('01','11')]:
            same = np.asarray(packets[a]['ids'])==np.asarray(packets[b]['ids']); mask = np.asarray(packets[a]['mask'],bool); learned = np.arange(512)>=2
            target = targets[b]-targets[a]
            for method,states in [('current',owned),('capacity',capacity)]:
                pred = states[b]-states[a]
                for label,take in [('all',np.ones(512,bool)),('unchangedToken',same),('unchangedLearned',same & learned),('unchangedActive',same & learned & mask),('unchangedPadding',same & learned & ~mask)]:
                    pairs.append({'A':a,'B':b,'method':method,'port':'H1','selection':label,'rows':int(take.sum()),**metric(pred[take],target[take])})
                for i in range(512): stream.write(json.dumps({'pair':[a,b],'method':method,'port':'H1','position':i,'sameToken':bool(same[i]),'active':bool(mask[i]),**metric(pred[i],target[i])},allow_nan=False)+'\n'); row_count+=1
    work = {'newNeuralEvaluations':0,'newOriginalLinears':0,'newImages':0,'groups':len(groups),
        **{k:sum(g[k] for g in groups) for k in ('rows','pairedComponents','unpairedComponents','evaluatedChoices','finiteOptionStates','finiteOptionCoordinateAdditions','endpointNormProducts','pairedResponseNormProducts')}}
    if work != req['physicalWork']: raise ValueError('Actual enumeration geometry differs')
    save(out/'array-manifest.json',manifest); save(out/'typed-capacity.json',typed)
    save(out/'input-reconstruction.json',{'rows':input_reconstruction,'authenticatedRemoteInputs':authenticated,'caseOutputs':case_fields})
    for name in req['retainedMetadata']: shutil.copyfile(root/name,out/name)
    seal = {name:sha(out/name) for name in ('components.jsonl.gz','observations.jsonl.gz','array-manifest.json','typed-capacity.json','input-reconstruction.json')}; save(out/'analysis-seal.json',seal)
    save(out/'report.json',{'schema':req['schema'],'requestSHA256':req['requestSHA256'],'groups':groups,'summaries':summaries,'pairs':pairs,'observationRows':row_count,
        'analysisSealSHA256':sha(out/'analysis-seal.json'),'workerWallSecondsAfterWorkerImport':time.perf_counter()-started,'cpuSeconds':time.process_time()-cpu0,
        'preparationSeconds':preparation_seconds,'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'numpyVersion':np.__version__,'torchVersion':torch.__version__,
        'device':'cpu','threads':torch.get_num_threads(),'physicalWork':work,'finitePolicy':p['finitePolicy'],'scoring':SCORING,
        'mechanics_status':'completed','instrument_status':'target-assisted-fixed-dictionary-H1-capacity','trend_status':'inspect-all-endpoint-response-row-tradeoffs',
        'terminal_status':'arbitrary-prompt-zero-neural-images-unachieved','limitations':['All six histories are previously public; transfer is target-assisted diagnostic,not prediction.','No fit,source-input resolver,native QKV optimum,full encoder,denoiser,VAE or image.','Combinatorial optimum of declared component scores,not exact-real or old global reassociation.','Historical learned preparation remains paid. TorchCPU is used for canonical typed custody only; no learned operator or CUDA computation.']})
    print('FINITE_WRITER_CAPACITY_COMPLETE',flush=True)
