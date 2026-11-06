"""Matched unselected-tail deletion through own original encoder/image consumers."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import np, read, save, sha, identity
from complete_masked_study import PROMPTS
from flux_clock_study import verify

HERE = Path(__file__).resolve().parent
OWNED = ['selected_reader_study.py', 'selected_reader.py', 'selected_reader_encoder.py', 'selected_reader_images.py',
    'encoder_atoms.py', 'paired_fields.py', 'masked_kv_study.py', 'conditioning_demand.py',
    'complete_masked_study.py', 'flux_clock.py', 'flux_clock_study.py']


def inherited_clock(root, source):
    path = Path(root) / 'clock-table.json'
    if sha(path) != source['clockTableSHA256']: raise ValueError('Inherited clock bytes changed')
    table = read(path)
    if table['requestSHA256'] != source['clockParentRequestSHA256']: raise ValueError('Inherited clock request differs')
    if table['contract']['checkpointSHA256'] != source['imageAssets']['transformer/diffusion_pytorch_model.safetensors']:
        raise ValueError('Inherited clock checkpoint differs')
    return table


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from publish_context_statistics import checked_collection
    parent=HERE.parent/'artifacts/uniform-reader-image-v1'; c,job,old=checked_collection(parent,current_workers=False)
    rank_parent=HERE.parent/'artifacts/raw-rank-one-image-v2'; rc,rj,rr=checked_collection(rank_parent,current_workers=False)
    api=Api()
    for collection,j,request in [(c,job,old),(rc,rj,rr)]:
        live=api.json('GET','/api/jobs/'+j['job_id'])
        if live['state']!='succeeded' or live['payload_custody']!=collection['custody'] or live['config']['requestSHA256']!=request['requestSHA256'] or request['prompts']!=PROMPTS:raise ValueError('Qualified ancestor differs')
    f=parent/'result'; source=read(f/'inputs.json'); uniform_enc=read(f/'encoder-candidate.json'); uniform_img=read(f/'images-candidate.json'); targets=read(f/'targets.json'); parent_seal=read(f/'candidate-seal.json')
    rank_source=read(rank_parent/'result/inputs.json'); rank_reader=read(rank_parent/'result/reader-seal.json')
    for key,name in [('encoderSHA256','encoder-candidate.json'),('imagesSHA256','images-candidate.json')]:
        if parent_seal[key]!=sha(f/name):raise ValueError('Uniform parent seal differs')
    if source['rankImageParentRoot']!=rj['result']['diagnostics']['cwd'] or source['rankImageParentRequestSHA256']!=rr['requestSHA256']:raise ValueError('Rank ancestor identity differs')
    preview=_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for name in OWNED:
        if name in old['sourceHashes'] and sha(HERE/name)!=old['sourceHashes'][name]:raise ValueError('Qualified shared source changed: '+name)
        shutil.copyfile(HERE/name,p/name)
    for name in ['flux_functional_qwen_prefix.py','qwen_h18_state_history.py']:
        path=parent/'payload'/name
        if sha(path)!=old['sourceHashes'][name]:raise ValueError('Qualified graph/history differs')
        shutil.copyfile(path,p/name)
    shutil.copyfile(f/'clock-table.json',p/'clock-table.json');shutil.copyfile(HERE.parent/'docs/plans/selected-only-reader.md',p/'frozen-plan.md')
    source['forbiddenTargets'] += [x['path'] for group in [uniform_enc['fields'],uniform_img['fields']] for fields in group.values() for x in fields.values()]
    source['forbiddenTargets'] += [x['path'] for x in uniform_img['images'].values()]
    source['forbiddenTargets'] += [x['path'] for x in rank_reader['reader']['rows']+rank_reader['reader']['states']+rank_reader['lexicalRows']]
    source['forbiddenTargets'] += [rank_source['readerBasis']['path']]
    source.update(uniformParentJobID=job['job_id'],uniformParentRoot=job['result']['diagnostics']['cwd'],uniformParentRequestSHA256=old['requestSHA256'],
        uniformParentSealSHA256=sha(f/'candidate-seal.json'),uniformParentByteVerificationSHA256=sha(parent/'retained-array-verification.json'),
        geometryParentRoot=rank_source['geometryParentRoot'],readerAsset=rank_source['readerAsset'],readerGeometry=rank_source['readerGeometry'])
    save(p/'inputs.json',source)
    save(p/'targets.json',{**targets,'uniformEncoderFields':uniform_enc['fields'],'uniformFields':uniform_img['fields'],'uniformImages':uniform_img['images'],
        'uniformEncoderSHA256':sha(f/'encoder-candidate.json'),'uniformImagesSHA256':sha(f/'images-candidate.json'),'rankReader':rank_reader,
        'rankReaderSHA256':sha(rank_parent/'result/reader-seal.json'), 'scope':'Observed full/rank/zero/uniform values and original rank diagnostic operands, opened only after complete selected-only candidates seal.'})
    save(p/'resource-basis.json',{'intentFirstPreview':preview,'measuredParentJobID':job['job_id'],
        'measuredParentResult':{k:c['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},
        'exception':'GenericB64/fallback40.2GBpreview inapplicable. Compatible measured14336MBRAM/13000MBVRAM envelope. SelectedreaderCPU1,encoderCPU1,imageCPU4,separatechildren/B1/no model co-residency.240sexpected/900sfinite/no retries; postseal cached-operand diagnostic additional.'})
    r={'schema':'tce-selected-reader-image-request-v1','prompts':PROMPTS,'manifestSHA256':old['manifestSHA256'],
        'scope':'Delete unselected-tail moments/envelopes/centroid/basis work; retain exact selected arithmetic, original fullKV/cache/scan and learned encoder/FLUX/VAE. No arbitrary-prompt zero-neural claim.',
        'protocol':'Rawstrings->sameatoms->frozen currentquery orderedtop4/routes and cachedanchor+delta selectedweights->ownfull512row encoder/E->2images. No tailstate/basis/fallback/refit/newreference/noisedraw/refresh. Seal before full/rank/zero/uniform targets and parentdiagnostic operands.',
        'readerPolicy':'current-top4-anchor-delta-selected-only-fp64-fp32-bf16-v1','conversion':'Original selected FP64 expression -> FP32 -> BF16 RNE via encoder_atoms.words','ownedSources':OWNED,
        'planSHA256':sha(p/'frozen-plan.md'),'parentRequestSHA256':old['requestSHA256'],'sourceHashes':{x.name:sha(x) for x in p.iterdir()},
        'model':'flux2-klein-4b-distilled','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'steps':4,'timeoutSeconds':900}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from run import command
    from mrun.client.submit import launch
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root / 'payload', r)
    if (root / 'job.json').exists(): raise ValueError('Already submitted')
    cmd = command()[:-5] + ['python', '-u', 'selected_reader_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = launch(cmd, experiment='tce-selected-reader-image-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs={'cuda': True},
        config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width']},
        ram_mb=14336, vram_mb=13000, cpu_threads=4, disk_gb=2, est_wall_s=240, timeout_s=900, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='2rawstrings->matched selected-only top4->own512row encoder/E->2images. Delete tailstate/basis/envelopes/fallback; retain fullKV/cache/scan and exact anchor+delta weights.52encoderSDPA/372linears,8denoisers/824imagelinears/2VAE paid.CPU1/1/4 CUDA B1.Inheritednoise/clocks. Seal before4old targets/parentcached diagnostics.No newrefs/fit.14336RAM/13000VRAM measured b2412411d329 exception vsbadB64plan.240sexpected900scap,noretries.')
    result = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root / 'job.json', result); return result


def seal(root):
    from paired_fields import load
    reader = read(root / 'reader-seal.json'); enc = read(root / 'encoder-candidate.json'); images = read(root / 'images-candidate.json')
    if reader['policy'] != read(root/'request.json')['readerPolicy']: raise ValueError('Selected-only law differs')
    for case in ['A', 'B']:
        item = reader['fields'][case]
        if item['shape'] != [1, 512, 4096] or item['logicalDtype'] != 'torch.bfloat16': raise ValueError('Selected-only bank geometry differs')
        load(item)
        if set(enc['fields'][case]) != {'H' + str(i) for i in range(28)} | {'E'}: raise ValueError('Encoder coverage differs')
        if set(images['fields'][case]) != {'C'+str(i) for i in range(4)} | {'velocity'+str(i) for i in range(4)} | {'latent'+str(i) for i in range(1,5)} | {'vaeInput','decodedRaw','rgbFloat','pngRGB'}: raise ValueError('Image coverage differs')
    if enc['readerSealSHA256'] != sha(root / 'reader-seal.json') or enc['lexicalSealSHA256'] != reader['lexicalSealSHA256']: raise ValueError('Selected-only/lexical encoder join differs')
    if images['encoderSHA256'] != sha(root / 'encoder-candidate.json') or images['tableSHA256'] != sha(root / 'clock-table.json'): raise ValueError('Image source joins differ')
    for group in [*enc['fields'].values(), *images['fields'].values()]:
        for item in group.values(): load(item)
    for item in [*reader['lexicalRows'], *reader['states'], *reader['packets'], *images['images'].values()]:
        if sha(item['path']) != item['sha256']: raise ValueError('Candidate lexical/image changed')
    save(root / 'candidate-seal.json', {'requestSHA256': read(root / 'request.json')['requestSHA256'], 'readerSHA256': sha(root / 'reader-seal.json'),
        'encoderSHA256': sha(root / 'encoder-candidate.json'), 'imagesSHA256': sha(root / 'images-candidate.json'), 'clockTableSHA256': sha(root / 'clock-table.json'),
        'readerFields': reader['fields'], 'encoderFields': enc['fields'], 'imageFields': images['fields'], 'images': images['images'],
        'scope': 'Complete own-state selected-only candidates authenticated before full/rank/zero/uniform targets or parent diagnostic operands are released.'})


def execute(root):
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root, r); started = time.perf_counter(); phases = []
    out = root / 'response-result'; out.mkdir()
    for action in ['reader', 'encoder', 'images']:
        tick = time.perf_counter(); subprocess.run([sys.executable, '-u', str(root / 'selected_reader_study.py'), action, str(root)], check=True, timeout=780)
        phases.append({'action': action, 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
    seal(root); print('SELECTED_READER_CANDIDATES_SEALED', sha(root / 'candidate-seal.json'), flush=True)
    score(root, out, phases, started)


def contribution(base, candidate, full):
    base,candidate,full=[np.asarray(x,dtype=np.float64) for x in [base,candidate,full]]
    if base.shape!=candidate.shape or base.shape!=full.shape:raise ValueError('Contribution geometry differs')
    eb=base-full;ec=candidate-full;q=candidate-base;energy=lambda x:float(np.sum(x*x))
    a,b,c=energy(eb),energy(ec),energy(q);cross=float(2*np.sum(eb*q))
    return {'baseErrorEnergy':a,'candidateErrorEnergy':b,'candidateMinusBaseEnergy':c,'twiceBaseErrorDotCandidateMinusBase':cross,
        'energyIdentityRemainder':b-a-c-cross,'convention':'q=candidate-base; e_candidate=e_base+q; E_candidate=E_base+||q||^2+2<e_base,q>','scalars':int(q.size)}


def parent_diagnostic(root, target, reader):
    from paired_fields import retain
    from selected_reader import logsum
    from encoder_atoms import floats, words
    parent=target['rankReader']; source=read(root/'inputs.json'); rows=[]; retained=[]
    def packet(item):
        if sha(item['path'])!=item['sha256']:raise ValueError('Diagnostic retained source identity differs')
        with np.load(item['path'],allow_pickle=False) as f:return {k:f[k] for k in f.files}
    if sha(source['readerAsset']['path'])!=source['readerAsset']['sha256']:raise ValueError('Diagnostic source asset differs')
    with np.load(source['readerAsset']['path'],allow_pickle=False) as f:landmarks=f['landmarks']
    for case in ['A','B']:
        own=packet(next(x for x in reader['packets'] if x['id']==case)); old=packet(next(x for x in parent['reader']['rows'] if x['id']==case))
        lex=packet(next(x for x in parent['lexicalRows'] if x['id']==case)); current=packet(next(x for x in reader['lexicalRows'] if x['id']==case))
        lexical_mismatches={k:int(np.count_nonzero(lex[k]!=current[k])) for k in ['ids','mask','positions','e','qpos','kpos','v']}
        reconstructed=np.zeros_like(own['raw']); logits=np.full_like(own['selectedLogits'],np.nan); weights=np.zeros_like(own['selectedWeights']); group_rows=[]; early=[]; d=128; tick=time.perf_counter()
        for g in range(8):
            state=packet(next(x for x in parent['reader']['states'] if x['id']==case and x['kvGroup']==g))
            own_state=packet(next(x for x in reader['states'] if x['id']==case and x['kvGroup']==g)); count=int(state['count'])
            cache={k:int(np.count_nonzero(state[k]!=own_state[k])) for k in ['keys','values','scores','positions']}
            group_rows.append({'kvGroup':g,'cachedOperandMismatches':cache,'parentCount':count,'candidateCount':int(own_state['count'])})
            positions=state['positions'][:count]
            for i in range(512):
                allowed=np.flatnonzero(positions<=i)
                for h in range(4):
                    head=g*4+h; selected_positions=old['supportPositions'][i,head]; selected_positions=selected_positions[selected_positions>=0]
                    selected=np.array([int(np.flatnonzero(positions==p)[0]) for p in selected_positions],dtype=np.int64)
                    if not set(selected).issubset(set(allowed)) or len(selected)!=min(4,len(allowed)):raise ValueError('Parent causal selected support invalid')
                    if not len(selected):continue
                    route=int(old['nearestIndex'][i,head]); q=floats(lex['qpos'][i,head*d:(head+1)*d]).astype(float)/d**.25
                    delta=q-landmarks[head,route]; U=state['keys'][:len(allowed)]; V=state['values'][:len(allowed)]; s=state['scores'][h,route,:len(allowed)]
                    a=s[selected]+U[selected]@delta; w=np.exp(a-logsum(a)); reconstructed[i,head*d:(head+1)*d]=w@V[selected]
                    logits[i,head,:len(selected)]=a;weights[i,head,:len(selected)]=w
                    if len(allowed)<=4:early.append((i,head))
        early_diff=[float(np.max(np.abs(own['raw'][i,h*d:(h+1)*d]-old['raw'][i,h*d:(h+1)*d]))) for i,h in early]
        item=retain(root/'arrays/diagnostics',case+'-parent-selected-raw',reconstructed);retained.append(item)
        rows.append({'id':case,'lexicalMismatches':lexical_mismatches,'groups':group_rows,
            'supportPositionMismatches':int(np.count_nonzero(own['supportPositions']!=old['supportPositions'])),
            'routeMismatches':int(np.count_nonzero(own['nearestIndex']!=old['nearestIndex'])),
            'selectedLogitMaxAbsDifference':float(np.max(np.abs(logits-own['selectedLogits']),where=np.isfinite(logits),initial=0.)),
            'selectedWeightMaxAbsDifference':float(np.max(np.abs(weights-own['selectedWeights']))),
            'parentSelectedRawMaxAbsDifference':float(np.max(np.abs(reconstructed-own['raw']))),
            'parentSelectedBF16WordMismatches':int(np.count_nonzero(words(reconstructed)!=own['rounded'])),
            'earlyParentSelectedOnlyQueries':len(early),'earlyParentRawMaxAbsDifference':max(early_diff,default=0.),
            'diagnosticSeconds':time.perf_counter()-tick,'reconstructedField':item})
    result={'schema':'tce-selected-reader-parent-diagnostic-v1','rows':rows,'retained':retained,
        'scope':'Postseal frozen-parent cache/support/route reconstruction; no native attention evaluation, candidate repair or accuracy gate. Nonzero discrepancies remain instrument observations.'}
    save(root/'parent-selected-diagnostic.json',result);shutil.copyfile(root/'parent-selected-diagnostic.json',root/'response-result/parent-selected-diagnostic.json')
    return result


def score(root,out,phases,started):
    import json
    from paired_fields import load,retain,metrics,paired
    r=read(root/'request.json');reader=read(root/'reader-seal.json');enc=read(root/'encoder-candidate.json');images=read(root/'images-candidate.json');target=read(root/'targets.json')
    seal_sha=sha(root/'candidate-seal.json'); diagnostic=parent_diagnostic(root,target,reader); accounts=[];pairs=[];contributions=[];differences={};observations=0
    if images['schedules']!=target['schedules']:raise ValueError('Original schedule differs')
    ef={'selected':enc['fields'],'rank':target['rankEncoderFields'],'zero':target['zeroEncoderFields'],'uniform':target['uniformEncoderFields']};im={'selected':images['fields'],'rank':target['rankFields'],'zero':target['zeroFields'],'uniform':target['uniformFields']}
    for case in ['A','B']:
        if len({f[case]['H0']['payloadSHA256'] for f in [*ef.values(),target['encoderFields']]})!=1:raise ValueError('Common raw H0 differs')
    with (out/'observations.jsonl').open('x') as stream:
        for case in ['A','B']:
            row=next(x for x in reader['lexicalRows'] if x['id']==case)
            with np.load(row['path'],allow_pickle=False) as lexical:mask=lexical['mask'].astype(bool)
            for port in ['H'+str(i) for i in range(1,28)]+['E',*images['fields'][case]]:
                encoder=port.startswith('H') or port=='E';branches=ef if encoder else im
                b=load((target['encoderFields'] if encoder else target['fields'])[case][port])
                for branch,fields in branches.items():
                    a=load(fields[case][port]);accounts.append({'case':case,'branch':branch,'port':port,'region':'all',**metrics(a,b)})
                    if encoder:
                        ar=a.reshape(512,-1);br=b.reshape(512,-1)
                        for region,use in [('active',mask),('padding',~mask)]:accounts.append({'case':case,'branch':branch,'port':port,'region':region,**metrics(ar[use],br[use])})
                    if encoder or port in ['rgbFloat','pngRGB']:
                        ar=a.reshape(512,-1);br=b.reshape(512,-1)
                        for i in range(512):
                            stream.write(json.dumps({'case':case,'branch':branch,'port':port,'position':i,'attended':bool(mask[i]) if encoder else None,**metrics(ar[i],br[i])},allow_nan=False)+'\n');observations+=1
                    if branch=='selected' and port in ['E','latent4','vaeInput','decodedRaw','rgbFloat','pngRGB']:
                        differences[case+'.'+port]=retain(root/'arrays/differences',case+'.'+port,a-b)
        for port in ['H1','H2','H9','H18','H27','E',*images['fields']['A']]:
            encoder=port.startswith('H') or port=='E';branches=ef if encoder else im;full=target['encoderFields'] if encoder else target['fields']
            fa,fb=[load(full[c][port]) for c in ['A','B']];deltas={}
            for branch,fields in branches.items():
                a,b=[load(fields[c][port]) for c in ['A','B']];deltas[branch]=b-a;pairs.append({'branch':branch,'port':port,**paired(a,b,fa,fb)})
            for base in ['zero','rank','uniform']:contributions.append({'port':port,'base':base,'candidate':'selected',**contribution(deltas[base],deltas['selected'],fb-fa)})
            if port in ['E','latent4','vaeInput','decodedRaw','rgbFloat','pngRGB']:
                for name,value in [('selected',deltas['selected']),('full',fb-fa),('error',deltas['selected']-(fb-fa)),('selectedMinusZero',deltas['selected']-deltas['zero']),('selectedMinusRank',deltas['selected']-deltas['rank']),('selectedMinusUniform',deltas['selected']-deltas['uniform'])]:
                    differences['paired.'+name+'.'+port]=retain(root/'arrays/differences','paired.'+name+'.'+port,value)
    if sha(root/'candidate-seal.json')!=seal_sha:raise ValueError('Candidate seal changed')
    for case in ['A','B']:
        for group,prefix in [(images['images'],'selected'),(target['rankImages'],'rank'),(target['zeroImages'],'zero'),(target['uniformImages'],'uniform'),(target['images'],'full')]:
            item=group[case]
            if sha(item['path'])!=item['sha256']:raise ValueError('Original PNG differs')
            shutil.copyfile(item['path'],out/(prefix+'-'+case+'.png'))
    for name in ['request.json','inputs.json','targets.json','resource-basis.json','frozen-plan.md','clock-table.json','reader-seal.json','encoder-candidate.json','images-candidate.json','candidate-seal.json']:shutil.copyfile(root/name,out/name)
    save(out/'differences.json',differences);calls=enc['primitives'];totals={k:sum(x[k] for x in images['counts'].values()) for k in ['denoiser','context','joint','single','VAE']}
    save(out/'report.json',{'schema':'tce-selected-reader-image-result-v1','requestSHA256':r['requestSHA256'],'scope':r['scope'],'readerWork':reader['work'],
        'parentSelectedDiagnostic':diagnostic,'accounts':accounts,'pairedPromptResponse':pairs,'selectedContribution':contributions,'observations':observations,'phases':phases,
        'workerSecondsAfterNumpyImports':time.perf_counter()-started,'candidateSealSHA256':seal_sha,
        'work':{**totals,'newImages':2,'newReferenceImages':0,'newRankImages':0,'newZeroImages':0,'newUniformImages':0,'newNoiseDraws':0,'firstReaderCalls':2,
            'queryKeyScoreProducts':sum(g['selectorScoreProducts'] for c in reader['work'] for g in c['groups']),
            'admissionAnchorMACs':sum(g['admissionAnchorMACs'] for c in reader['work'] for g in c['groups']),
            'selectedCoordinateProducts':sum(g['selectedCoordinateProducts'] for c in reader['work'] for g in c['groups']),
            'allocatedRetrievalAndCacheBytesEach':sum(g['allocatedIndexPayloadBytes']+g['countAndClockBytes'] for g in reader['work'][0]['groups']),'momentEnvelopeBytes':0,
            'nativeEncoderBlocks':enc['nativeComponentBlocks'],'encoderSDPA':sum(x['kind']=='SDPA' for x in calls),'encoderLinears':sum(x['kind']=='linear' for x in calls),
            'encoderNominalLinearMACs':sum(x['nominalMACs'] for x in calls if x['kind']=='linear'),'nativeTransformerLinears':sum(x['parameter'].startswith('transformer.') for x in images['linears']),
            'nativeTransformerNominalLinearMACs':sum(x['nominalMACs'] for x in images['linears'] if x['parameter'].startswith('transformer.')),'onlineClockAffines':0,'onlineClockGathers':len(images['clockCalls'])},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limitations':['One knownpair/onenoise; bundled lexicalE/QKV/position path and original learned encoder/FLUX/VAE paid. No arbitrarypromptquality/zero-neural or matchedspeedup claim.',
            'Selected-onlyFP64causalmeans are one compact state law, not all compact readers. Same-law directdiagnostic differs from native attention accuracy. Core8192B sums+count/control excludes input/output/serialization/diagnostics and compiled tables.']})
    print('SELECTED_READER_IMAGE_DONE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','reader','encoder','images','collect']);parser.add_argument('run');a=parser.parse_args()
    if a.action in ['reader','encoder','images']:result=__import__('selected_reader' if a.action=='reader' else 'selected_reader_'+a.action).execute(a.run)
    elif a.action=='collect':
        from response_study import collect
        result=collect(a.run)
    else:result=globals()[a.action](a.run)
    if result is not None:print(__import__('json').dumps(result))
