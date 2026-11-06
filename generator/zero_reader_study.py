"""Ablate the first contextual reader; preserve the actual lexical residual/suffix."""
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
OWNED = ['zero_reader_study.py', 'zero_reader.py', 'zero_reader_encoder.py', 'zero_reader_images.py',
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
    parent = HERE.parent / 'artifacts/raw-rank-one-image-v2'; c, job, old = checked_collection(parent, current_workers=False)
    api = Api(); live = api.json('GET', '/api/jobs/' + job['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != c['custody']: raise ValueError('Rank-one ancestor differs')
    if old['prompts'] != PROMPTS: raise ValueError('Frozen prompt pair differs')
    f = parent / 'result'; source = read(f / 'inputs.json'); rank_enc = read(f / 'encoder-candidate.json'); rank_img = read(f / 'images-candidate.json')
    full = read(f / 'targets.json'); parent_seal = read(f / 'candidate-seal.json')
    for key, name in [('encoderSHA256', 'encoder-candidate.json'), ('imagesSHA256', 'images-candidate.json')]:
        if parent_seal[key] != sha(f / name): raise ValueError('Rank-one parent seal differs')
    preview = _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda')
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); p = root / 'payload'; p.mkdir()
    for name in OWNED:
        if name in old['sourceHashes'] and sha(HERE / name) != old['sourceHashes'][name]: raise ValueError('Shared qualified source changed: ' + name)
        shutil.copyfile(HERE / name, p / name)
    for name in ['flux_functional_qwen_prefix.py', 'qwen_h18_state_history.py']:
        path = parent / 'payload' / name
        if sha(path) != old['sourceHashes'][name]: raise ValueError('Qualified graph/history differs')
        shutil.copyfile(path, p / name)
    shutil.copyfile(f / 'clock-table.json', p / 'clock-table.json'); shutil.copyfile(HERE.parent / 'docs/plans/zero-reader-image.md', p / 'frozen-plan.md')
    for name in ['readerAsset', 'readerBasis', 'readerGeometry', 'readerPolicy', 'rankParentJobID', 'rankParentRoot', 'rankParentRequestSHA256', 'geometryParentRoot']:
        source.pop(name, None)
    source['forbiddenTargets'] += [x['path'] for group in [rank_enc['fields'], rank_img['fields']] for fields in group.values() for x in fields.values()]
    source['forbiddenTargets'] += [x['path'] for x in rank_img['images'].values()]
    source.update(rankImageParentJobID=job['job_id'], rankImageParentRoot=job['result']['diagnostics']['cwd'],
        rankImageParentRequestSHA256=old['requestSHA256'], rankImageParentSealSHA256=sha(f / 'candidate-seal.json'),
        rankImageParentByteVerificationSHA256=sha(parent / 'retained-array-verification.json'))
    save(p / 'inputs.json', source)
    save(p / 'targets.json', {**full, 'rankEncoderFields': rank_enc['fields'], 'rankFields': rank_img['fields'], 'rankImages': rank_img['images'],
        'rankEncoderSHA256': sha(f / 'encoder-candidate.json'), 'rankImagesSHA256': sha(f / 'images-candidate.json'),
        'scope': 'Previously observed full and rank-one candidates; value reads only after complete zero-reader candidates seal.'})
    save(p / 'resource-basis.json', {'intentFirstPreview': preview, 'measuredParentJobID': job['job_id'],
        'measuredParentResult': {k: c['result'][k] for k in ['elapsed_s', 'peak_rss_mb', 'peak_vram_mb']},
        'exception': 'GenericB64/fallback40.2GBpreview is inapplicable. Compatible measured14336MBRAM/13000MBVRAM envelope. Zero lexicalCPU1, originalencoderCPU1 then imageCPU4, separatechildren/B1/no co-residency.180sexpected/900sfinite/no retries.'})
    r = {'schema': 'tce-zero-reader-image-request-v1', 'prompts': PROMPTS, 'manifestSHA256': old['manifestSHA256'],
        'scope': 'Delete first contextual read only; lexical residual and remaining original learned encoder/FLUX/VAE stay paid. No arbitrary-prompt zero-neural claim.',
        'protocol': 'Rawstrings->same bundledlexicalatoms->positivezeroBF16[1,512,4096]bank->originalblock0tail->ownfull512rowblocks1..26/E->2images. No first reader/selection/moments/fallback. No state refresh/newrefs/noise draws/retune/compactKV. Seal before savedfull/rankone targets.',
        'ownedSources': OWNED, 'planSHA256': sha(p / 'frozen-plan.md'), 'parentRequestSHA256': old['requestSHA256'],
        'sourceHashes': {x.name: sha(x) for x in p.iterdir()}, 'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda',
        'dtype': 'bfloat16', 'batch_size': 1, 'seq_len': 1536, 'height': 512, 'width': 512, 'steps': 4, 'timeoutSeconds': 900}
    r['requestSHA256'] = identity(r); save(root / 'request.json', r); save(p / 'request.json', r)
    save(p / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [x.name for x in p.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': r['requestSHA256']}


def submit(root):
    from run import command
    from mrun.client.submit import launch
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root / 'payload', r)
    if (root / 'job.json').exists(): raise ValueError('Already submitted')
    cmd = command()[:-5] + ['python', '-u', 'zero_reader_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = launch(cmd, experiment='tce-zero-reader-image-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs={'cuda': True},
        config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width']},
        ram_mb=14336, vram_mb=13000, cpu_threads=4, disk_gb=2, est_wall_s=180, timeout_s=900, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='2 rawstrings->zero first32head bank->own full512row encoder/E->2images. No first reader/score/moments.52encoderSDPA/372linears,8denoisers/2VAE/824imagelinears remain.3children CPU1/1/4 CUDA B1. Inheritednoise/clocks/profiles, no newrefs. Seal before existingfull/rankonetargets.14336RAM/13000VRAM measured7b206109b072 exception vsbadB64plan.180sexpected/900scap/no retries.')
    result = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root / 'job.json', result); return result


def seal(root):
    from paired_fields import load
    reader = read(root / 'reader-seal.json'); enc = read(root / 'encoder-candidate.json'); images = read(root / 'images-candidate.json')
    if reader['readerCalls'] or reader['readerScoreProducts'] or reader['readerMomentCells']: raise ValueError('Deleted first reader executed')
    for case in ['A', 'B']:
        item = reader['fields'][case]
        if item['shape'] != [1, 512, 4096] or item['logicalDtype'] != 'torch.bfloat16': raise ValueError('Zero bank geometry differs')
        load(item)
        with Path(item['path']).open('rb') as f:
            if np.count_nonzero(np.load(f, allow_pickle=False)): raise ValueError('Positive-zero bits changed')
        if set(enc['fields'][case]) != {'H' + str(i) for i in range(28)} | {'E'}: raise ValueError('Encoder coverage differs')
        if set(images['fields'][case]) != {'C'+str(i) for i in range(4)} | {'velocity'+str(i) for i in range(4)} | {'latent'+str(i) for i in range(1,5)} | {'vaeInput','decodedRaw','rgbFloat','pngRGB'}: raise ValueError('Image coverage differs')
    if enc['readerSealSHA256'] != sha(root / 'reader-seal.json') or enc['lexicalSealSHA256'] != reader['lexicalSealSHA256']: raise ValueError('Zero/lexical encoder join differs')
    if images['encoderSHA256'] != sha(root / 'encoder-candidate.json') or images['tableSHA256'] != sha(root / 'clock-table.json'): raise ValueError('Image source joins differ')
    for group in [*enc['fields'].values(), *images['fields'].values()]:
        for item in group.values(): load(item)
    for item in [*reader['lexicalRows'], *images['images'].values()]:
        if sha(item['path']) != item['sha256']: raise ValueError('Candidate lexical/image changed')
    save(root / 'candidate-seal.json', {'requestSHA256': read(root / 'request.json')['requestSHA256'], 'readerSHA256': sha(root / 'reader-seal.json'),
        'encoderSHA256': sha(root / 'encoder-candidate.json'), 'imagesSHA256': sha(root / 'images-candidate.json'), 'clockTableSHA256': sha(root / 'clock-table.json'),
        'readerFields': reader['fields'], 'encoderFields': enc['fields'], 'imageFields': images['fields'], 'images': images['images'],
        'scope': 'Complete own-state zero-reader candidates authenticated before full/rank-one target values are released.'})


def execute(root):
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root, r); started = time.perf_counter(); phases = []
    out = root / 'response-result'; out.mkdir()
    for action in ['reader', 'encoder', 'images']:
        tick = time.perf_counter(); subprocess.run([sys.executable, '-u', str(root / 'zero_reader_study.py'), action, str(root)], check=True, timeout=780)
        phases.append({'action': action, 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
    seal(root); print('ZERO_READER_CANDIDATES_SEALED', sha(root / 'candidate-seal.json'), flush=True)
    score(root, out, phases, started)


def rank_contribution(zero, rank, full):
    zero, rank, full = [np.asarray(x, dtype=np.float64) for x in [zero, rank, full]]
    if zero.shape != rank.shape or zero.shape != full.shape: raise ValueError('Contribution geometry differs')
    e0 = zero - full; er = rank - full; q = rank - zero
    energy = lambda x: float(np.sum(x*x))
    a, b, c = energy(e0), energy(er), energy(q); cross = float(2*np.sum(e0*q))
    return {'zeroErrorEnergy': a, 'rankErrorEnergy': b, 'rankMinusZeroEnergy': c,
        'twiceZeroErrorDotRankMinusZero': cross, 'energyIdentityRemainder': b-a-c-cross,
        'convention': 'q=rank-zero; e_rank=e_zero+q; E_rank=E_zero+||q||^2+2<e_zero,q>', 'scalars': int(q.size)}


def score(root, out, phases, started):
    import json
    from paired_fields import load, retain, metrics, paired
    r = read(root / 'request.json'); reader = read(root / 'reader-seal.json'); enc = read(root / 'encoder-candidate.json'); images = read(root / 'images-candidate.json')
    target = read(root / 'targets.json'); seal_sha = sha(root / 'candidate-seal.json'); accounts = []; pair_accounts = []; contribution = []; differences = {}; observations = 0
    if images['schedules'] != target['schedules']: raise ValueError('Original schedule differs')
    for case in ['A', 'B']:
        if len({f[case]['H0']['payloadSHA256'] for f in [enc['fields'], target['encoderFields'], target['rankEncoderFields']]}) != 1: raise ValueError('Common raw input differs')
    with (out / 'observations.jsonl').open('x') as stream:
        for case in ['A', 'B']:
            row = next(x for x in reader['lexicalRows'] if x['id'] == case)
            with np.load(row['path'], allow_pickle=False) as lexical: mask = lexical['mask'].astype(bool)
            ports = [('H'+str(i), enc['fields'], target['rankEncoderFields'], target['encoderFields']) for i in range(1,28)] + [('E', enc['fields'], target['rankEncoderFields'], target['encoderFields'])]
            ports += [(port, images['fields'], target['rankFields'], target['fields']) for port in images['fields'][case]]
            for port, zero, rank, full in ports:
                b = load(full[case][port])
                for branch, fields in [('zero', zero), ('rank', rank)]:
                    a = load(fields[case][port]); accounts.append({'case': case, 'branch': branch, 'port': port, 'region': 'all', **metrics(a,b)})
                    encoder = port.startswith('H') or port == 'E'
                    if encoder:
                        ar = a.reshape(512,-1); br = b.reshape(512,-1)
                        for region, use in [('active',mask),('padding',~mask)]:
                            accounts.append({'case':case,'branch':branch,'port':port,'region':region,**metrics(ar[use],br[use])})
                    if encoder or port in ['rgbFloat','pngRGB']:
                        ar = a.reshape(512,-1); br = b.reshape(512,-1)
                        for i in range(512):
                            stream.write(json.dumps({'case':case,'branch':branch,'port':port,'position':i,'attended':bool(mask[i]) if encoder else None,**metrics(ar[i],br[i])},allow_nan=False)+'\n');observations+=1
                    if branch=='zero' and port in ['E','latent4','vaeInput','decodedRaw','rgbFloat','pngRGB']:
                        differences[case+'.'+port]=retain(root/'arrays/differences',case+'.'+port,a-b)
        for port in ['H1','H2','H9','H18','H27','E',*images['fields']['A']]:
            zero,rank,full=(enc['fields'],target['rankEncoderFields'],target['encoderFields']) if port.startswith('H') or port=='E' else (images['fields'],target['rankFields'],target['fields'])
            za,zb,ka,kb,fa,fb=[load(fields[case][port]) for fields,case in [(zero,'A'),(zero,'B'),(rank,'A'),(rank,'B'),(full,'A'),(full,'B')]]
            for branch,a,b in [('zero',za,zb),('rank',ka,kb)]: pair_accounts.append({'branch':branch,'port':port,**paired(a,b,fa,fb)})
            contribution.append({'port':port,**rank_contribution(zb-za,kb-ka,fb-fa)})
            if port in ['E','latent4','vaeInput','decodedRaw','rgbFloat','pngRGB']:
                for name,value in [('zero',zb-za),('full',fb-fa),('error',(zb-za)-(fb-fa)),('rankMinusZero',(kb-ka)-(zb-za))]:
                    differences['paired.'+name+'.'+port]=retain(root/'arrays/differences','paired.'+name+'.'+port,value)
    if sha(root/'candidate-seal.json')!=seal_sha: raise ValueError('Candidate seal changed')
    for case in ['A','B']:
        for group,prefix in [(images['images'],'zero'),(target['rankImages'],'rank'),(target['images'],'full')]:
            item=group[case]
            if sha(item['path'])!=item['sha256']:raise ValueError('Original PNG differs')
            shutil.copyfile(item['path'],out/(prefix+'-'+case+'.png'))
    for name in ['request.json','inputs.json','targets.json','resource-basis.json','frozen-plan.md','clock-table.json','reader-seal.json','encoder-candidate.json','images-candidate.json','candidate-seal.json']:
        shutil.copyfile(root/name,out/name)
    save(out/'differences.json',differences)
    calls=enc['primitives'];totals={k:sum(x[k] for x in images['counts'].values()) for k in ['denoiser','context','joint','single','VAE']}
    save(out/'report.json',{'schema':'tce-zero-reader-image-result-v1','requestSHA256':r['requestSHA256'],'scope':r['scope'],
        'accounts':accounts,'pairedPromptResponse':pair_accounts,'rankOneContributionAgainstZero':contribution,'observations':observations,'phases':phases,
        'workerSecondsAfterNumpyImports':time.perf_counter()-started,'candidateSealSHA256':seal_sha,
        'work':{**totals,'newImages':2,'newReferenceImages':0,'newRankImages':0,'newNoiseDraws':0,'firstReaderCalls':0,'firstReaderScoreProducts':0,'firstReaderMoments':0,
            'nativeEncoderBlocks':enc['nativeComponentBlocks'],'encoderSDPA':sum(x['kind']=='SDPA' for x in calls),'encoderLinears':sum(x['kind']=='linear' for x in calls),
            'encoderNominalLinearMACs':sum(x['nominalMACs'] for x in calls if x['kind']=='linear'),'nativeTransformerLinears':sum(x['parameter'].startswith('transformer.') for x in images['linears']),
            'nativeTransformerNominalLinearMACs':sum(x['nominalMACs'] for x in images['linears'] if x['parameter'].startswith('transformer.')),'onlineClockAffines':0,'onlineClockGathers':len(images['clockCalls'])},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limitations':['Known pair/one inherited noise, original lexical residual and remaining learned encoder/FLUX/VAE paid. No arbitrary-prompt quality or zero-neural claim.',
            'Zero-first-reader contribution control only; downstream comparisons are observations, not scientific abort gates. Original compiled lexical E/QKV lookup remains bundled and paid. No matched speedup.']})
    print('ZERO_READER_IMAGE_DONE',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','reader','encoder','images','collect']);parser.add_argument('run');a=parser.parse_args()
    if a.action in ['reader','encoder','images']:
        result=__import__('zero_reader' if a.action=='reader' else 'zero_reader_'+a.action).execute(a.run)
    elif a.action=='collect':
        from response_study import collect
        result=collect(a.run)
    else:result=globals()[a.action](a.run)
    if result is not None:print(__import__('json').dumps(result))
