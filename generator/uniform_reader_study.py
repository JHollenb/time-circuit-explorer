"""Uniform causal value state through its own original encoder/image consumers."""
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
OWNED = ['uniform_reader_study.py', 'uniform_reader.py', 'uniform_reader_encoder.py', 'uniform_reader_images.py',
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
    parent=HERE.parent/'artifacts/zero-reader-image-v1';c,job,old=checked_collection(parent,current_workers=False)
    api=Api();live=api.json('GET','/api/jobs/'+job['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody'] or old['prompts']!=PROMPTS:raise ValueError('Qualified zero ancestor differs')
    f=parent/'result';source=read(f/'inputs.json');zero_enc=read(f/'encoder-candidate.json');zero_img=read(f/'images-candidate.json');targets=read(f/'targets.json');seal=read(f/'candidate-seal.json')
    for key,name in [('encoderSHA256','encoder-candidate.json'),('imagesSHA256','images-candidate.json')]:
        if seal[key]!=sha(f/name):raise ValueError('Zero parent seal differs')
    preview=_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for name in OWNED:
        if name in old['sourceHashes'] and sha(HERE/name)!=old['sourceHashes'][name]:raise ValueError('Qualified shared source changed: '+name)
        shutil.copyfile(HERE/name,p/name)
    for name in ['flux_functional_qwen_prefix.py','qwen_h18_state_history.py']:
        path=parent/'payload'/name
        if sha(path)!=old['sourceHashes'][name]:raise ValueError('Qualified graph/history differs')
        shutil.copyfile(path,p/name)
    shutil.copyfile(f/'clock-table.json',p/'clock-table.json');shutil.copyfile(HERE.parent/'docs/plans/uniform-causal-reader.md',p/'frozen-plan.md')
    source['forbiddenTargets'] += [x['path'] for group in [zero_enc['fields'],zero_img['fields']] for fields in group.values() for x in fields.values()]
    source['forbiddenTargets'] += [x['path'] for x in zero_img['images'].values()]
    source.update(zeroParentJobID=job['job_id'],zeroParentRoot=job['result']['diagnostics']['cwd'],zeroParentRequestSHA256=old['requestSHA256'],
        zeroParentSealSHA256=sha(f/'candidate-seal.json'),zeroParentByteVerificationSHA256=sha(parent/'retained-array-verification.json'))
    save(p/'inputs.json',source)
    save(p/'targets.json',{**targets,'zeroEncoderFields':zero_enc['fields'],'zeroFields':zero_img['fields'],'zeroImages':zero_img['images'],
        'zeroEncoderSHA256':sha(f/'encoder-candidate.json'),'zeroImagesSHA256':sha(f/'images-candidate.json'),
        'scope':'Previously observed full/rank/zero values; opened only after complete uniform candidates seal.'})
    save(p/'resource-basis.json',{'intentFirstPreview':preview,'measuredParentJobID':job['job_id'],
        'measuredParentResult':{k:c['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},
        'exception':'GenericB64/fallback40.2GBpreview inapplicable. Compatible measured14336MBRAM/13000MBVRAM envelope. UniformreaderCPU1,encoderCPU1,imageCPU4,separatechildren/B1/no model co-residency.180sexpected/900sfinite/no retries.'})
    r={'schema':'tce-uniform-reader-image-request-v1','prompts':PROMPTS,'manifestSHA256':old['manifestSHA256'],
        'scope':'Uniform causal value summary replaces first read;8192B sums+count/control, lexical/input/outputbuffers and remaining learned encoder/FLUX/VAE paid. No arbitrary-prompt zero-neural claim.',
        'protocol':'Freshrawstrings->samebundledatoms->chronologicalFP64causalVmeans->BF16all32headbank->originalblock0tail->ownfull512rowblocks1..26/E->2images. No QK/anchors/top4/index/quadraticmoments/envelopes/nativefallback/refit/referencecampaign/noisedraw/state refresh. Seal before saved full/rank/zero values.',
        'readerPolicy':'uniform-causal-values-fp64-fp32-bf16-v1','conversion':'Chronological FP64 sum/division -> FP32 -> BF16 RNE via encoder_atoms.words','ownedSources':OWNED,'planSHA256':sha(p/'frozen-plan.md'),'parentRequestSHA256':old['requestSHA256'],
        'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'model':'flux2-klein-4b-distilled','backend':'diffusers','device':'cuda','dtype':'bfloat16',
        'batch_size':1,'seq_len':1536,'height':512,'width':512,'steps':4,'timeoutSeconds':900}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from run import command
    from mrun.client.submit import launch
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root / 'payload', r)
    if (root / 'job.json').exists(): raise ValueError('Already submitted')
    cmd = command()[:-5] + ['python', '-u', 'uniform_reader_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = launch(cmd, experiment='tce-uniform-reader-image-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs={'cuda': True},
        config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width']},
        ram_mb=14336, vram_mb=13000, cpu_threads=4, disk_gb=2, est_wall_s=180, timeout_s=900, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='2rawstrings->uniform causalVmeans FP64/8192Bsum+count/clock->own full512row encoder/E->2images.NoQK/anchors/top4/index/moments/envelopes.52encoderSDPA/372linears,8denoisers/824imagelinears/2VAE paid.CPU1/1/4 CUDA B1.Inheritedclock/noise/profiles.Seal before full/rank/zero targets;no newrefs/noisedraw.14336RAM/13000VRAM measured0b613bbe336d exception vsbadB64plan.180sexpected900scap,noretries.')
    result = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root / 'job.json', result); return result


def seal(root):
    from paired_fields import load
    reader = read(root / 'reader-seal.json'); enc = read(root / 'encoder-candidate.json'); images = read(root / 'images-candidate.json')
    if reader['policy'] != read(root/'request.json')['readerPolicy']: raise ValueError('Uniform law differs')
    for case in ['A', 'B']:
        item = reader['fields'][case]
        if item['shape'] != [1, 512, 4096] or item['logicalDtype'] != 'torch.bfloat16': raise ValueError('Uniform bank geometry differs')
        load(item)
        if set(enc['fields'][case]) != {'H' + str(i) for i in range(28)} | {'E'}: raise ValueError('Encoder coverage differs')
        if set(images['fields'][case]) != {'C'+str(i) for i in range(4)} | {'velocity'+str(i) for i in range(4)} | {'latent'+str(i) for i in range(1,5)} | {'vaeInput','decodedRaw','rgbFloat','pngRGB'}: raise ValueError('Image coverage differs')
    if enc['readerSealSHA256'] != sha(root / 'reader-seal.json') or enc['lexicalSealSHA256'] != reader['lexicalSealSHA256']: raise ValueError('Uniform/lexical encoder join differs')
    if images['encoderSHA256'] != sha(root / 'encoder-candidate.json') or images['tableSHA256'] != sha(root / 'clock-table.json'): raise ValueError('Image source joins differ')
    for group in [*enc['fields'].values(), *images['fields'].values()]:
        for item in group.values(): load(item)
    for item in [*reader['lexicalRows'], *reader['states'], *reader['packets'], *images['images'].values()]:
        if sha(item['path']) != item['sha256']: raise ValueError('Candidate lexical/image changed')
    save(root / 'candidate-seal.json', {'requestSHA256': read(root / 'request.json')['requestSHA256'], 'readerSHA256': sha(root / 'reader-seal.json'),
        'encoderSHA256': sha(root / 'encoder-candidate.json'), 'imagesSHA256': sha(root / 'images-candidate.json'), 'clockTableSHA256': sha(root / 'clock-table.json'),
        'readerFields': reader['fields'], 'encoderFields': enc['fields'], 'imageFields': images['fields'], 'images': images['images'],
        'scope': 'Complete own-state uniform candidates authenticated before full/rank/zero target values are released.'})


def execute(root):
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root, r); started = time.perf_counter(); phases = []
    out = root / 'response-result'; out.mkdir()
    for action in ['reader', 'encoder', 'images']:
        tick = time.perf_counter(); subprocess.run([sys.executable, '-u', str(root / 'uniform_reader_study.py'), action, str(root)], check=True, timeout=780)
        phases.append({'action': action, 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
    seal(root); print('UNIFORM_READER_CANDIDATES_SEALED', sha(root / 'candidate-seal.json'), flush=True)
    score(root, out, phases, started)


def contribution(base, candidate, full):
    base,candidate,full=[np.asarray(x,dtype=np.float64) for x in [base,candidate,full]]
    if base.shape!=candidate.shape or base.shape!=full.shape:raise ValueError('Contribution geometry differs')
    eb=base-full;ec=candidate-full;q=candidate-base;energy=lambda x:float(np.sum(x*x))
    a,b,c=energy(eb),energy(ec),energy(q);cross=float(2*np.sum(eb*q))
    return {'baseErrorEnergy':a,'candidateErrorEnergy':b,'candidateMinusBaseEnergy':c,'twiceBaseErrorDotCandidateMinusBase':cross,
        'energyIdentityRemainder':b-a-c-cross,'convention':'q=candidate-base; e_candidate=e_base+q; E_candidate=E_base+||q||^2+2<e_base,q>','scalars':int(q.size)}


def score(root,out,phases,started):
    import json
    from paired_fields import load,retain,metrics,paired
    r=read(root/'request.json');reader=read(root/'reader-seal.json');enc=read(root/'encoder-candidate.json');images=read(root/'images-candidate.json');target=read(root/'targets.json')
    seal_sha=sha(root/'candidate-seal.json');accounts=[];pairs=[];contributions=[];differences={};observations=0
    if images['schedules']!=target['schedules']:raise ValueError('Original schedule differs')
    ef={'uniform':enc['fields'],'rank':target['rankEncoderFields'],'zero':target['zeroEncoderFields']};im={'uniform':images['fields'],'rank':target['rankFields'],'zero':target['zeroFields']}
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
                    if branch=='uniform' and port in ['E','latent4','vaeInput','decodedRaw','rgbFloat','pngRGB']:
                        differences[case+'.'+port]=retain(root/'arrays/differences',case+'.'+port,a-b)
        for port in ['H1','H2','H9','H18','H27','E',*images['fields']['A']]:
            encoder=port.startswith('H') or port=='E';branches=ef if encoder else im;full=target['encoderFields'] if encoder else target['fields']
            fa,fb=[load(full[c][port]) for c in ['A','B']];deltas={}
            for branch,fields in branches.items():
                a,b=[load(fields[c][port]) for c in ['A','B']];deltas[branch]=b-a;pairs.append({'branch':branch,'port':port,**paired(a,b,fa,fb)})
            for base in ['zero','rank']:contributions.append({'port':port,'base':base,'candidate':'uniform',**contribution(deltas[base],deltas['uniform'],fb-fa)})
            if port in ['E','latent4','vaeInput','decodedRaw','rgbFloat','pngRGB']:
                for name,value in [('uniform',deltas['uniform']),('full',fb-fa),('error',deltas['uniform']-(fb-fa)),('uniformMinusZero',deltas['uniform']-deltas['zero']),('uniformMinusRank',deltas['uniform']-deltas['rank'])]:
                    differences['paired.'+name+'.'+port]=retain(root/'arrays/differences','paired.'+name+'.'+port,value)
    if sha(root/'candidate-seal.json')!=seal_sha:raise ValueError('Candidate seal changed')
    for case in ['A','B']:
        for group,prefix in [(images['images'],'uniform'),(target['rankImages'],'rank'),(target['zeroImages'],'zero'),(target['images'],'full')]:
            item=group[case]
            if sha(item['path'])!=item['sha256']:raise ValueError('Original PNG differs')
            shutil.copyfile(item['path'],out/(prefix+'-'+case+'.png'))
    for name in ['request.json','inputs.json','targets.json','resource-basis.json','frozen-plan.md','clock-table.json','reader-seal.json','encoder-candidate.json','images-candidate.json','candidate-seal.json']:shutil.copyfile(root/name,out/name)
    save(out/'differences.json',differences);calls=enc['primitives'];totals={k:sum(x[k] for x in images['counts'].values()) for k in ['denoiser','context','joint','single','VAE']}
    save(out/'report.json',{'schema':'tce-uniform-reader-image-result-v1','requestSHA256':r['requestSHA256'],'scope':r['scope'],'readerWork':reader['work'],
        'accounts':accounts,'pairedPromptResponse':pairs,'uniformContribution':contributions,'observations':observations,'phases':phases,
        'workerSecondsAfterNumpyImports':time.perf_counter()-started,'candidateSealSHA256':seal_sha,
        'work':{**totals,'newImages':2,'newReferenceImages':0,'newRankImages':0,'newZeroImages':0,'newNoiseDraws':0,'firstReaderCalls':2,'queryKeyScoreProducts':0,'retrievalIndexBytes':0,'momentEnvelopeBytes':0,
            'nativeEncoderBlocks':enc['nativeComponentBlocks'],'encoderSDPA':sum(x['kind']=='SDPA' for x in calls),'encoderLinears':sum(x['kind']=='linear' for x in calls),
            'encoderNominalLinearMACs':sum(x['nominalMACs'] for x in calls if x['kind']=='linear'),'nativeTransformerLinears':sum(x['parameter'].startswith('transformer.') for x in images['linears']),
            'nativeTransformerNominalLinearMACs':sum(x['nominalMACs'] for x in images['linears'] if x['parameter'].startswith('transformer.')),'onlineClockAffines':0,'onlineClockGathers':len(images['clockCalls'])},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limitations':['One knownpair/onenoise; bundled lexicalE/QKV/position path and original learned encoder/FLUX/VAE paid. No arbitrarypromptquality/zero-neural or matchedspeedup claim.',
            'UniformFP64causalmeans are one compact state law, not all compact readers. Same-law directdiagnostic differs from native attention accuracy. Core8192B sums+count/control excludes input/output/serialization/diagnostics and compiled tables.']})
    print('UNIFORM_READER_IMAGE_DONE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','reader','encoder','images','collect']);parser.add_argument('run');a=parser.parse_args()
    if a.action in ['reader','encoder','images']:result=__import__('uniform_reader' if a.action=='reader' else 'uniform_reader_'+a.action).execute(a.run)
    elif a.action=='collect':
        from response_study import collect
        result=collect(a.run)
    else:result=globals()[a.action](a.run)
    if result is not None:print(__import__('json').dumps(result))
