"""One guarded source slice -> unchanged projected reader -> archived scoring."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read, save, sha, identity
from flux_clock_study import verify
from projected_complement_study import OWNED as PARENT_OWNED, WHEELS

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent.parent / 'artifacts/mdb-computation-family-20261107/fresh-family-02/compact'
OWNED = list(dict.fromkeys(['selected_source.py', 'selected_source_worker.py', 'selected_source_study.py',
                          'selected_source_analysis.py'] + PARENT_OWNED))
CASE = 'sphere-cube-plain-2027'


def source_metadata(plan, origins):
    code = '''import pathlib,json,hashlib
r=pathlib.Path(ROOT);spec=SPEC;out={};files={}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for role,origin in spec.items():
 cp=r/'state-store/cuts'/(origin['cut']+'.json');c=json.load(open(cp));files[str(cp)]=sha(cp);pages={}
 for slot in ['image','text','context']:
  pid=c['slots'][slot];p=r/'state-store/pages'/(pid+'.json');v=json.load(open(p))
  if sha(p)!=pid:raise ValueError('Source page digest differs')
  blob=r/'state-store/blobs'/(v['tensor_blob']+'.safetensors')
  if sha(blob)!=v['tensor_blob']:raise ValueError('Source blob digest differs')
  own={str(p):pid,str(blob):v['tensor_blob']};files.update(own)
  pages[slot]={'pageID':pid,'metadata':v,'store':str(r/'state-store'),'files':own}
 out[role]={'cutID':origin['cut'],'cut':c,'pages':pages,'cutFile':{'path':str(cp),'sha256':sha(cp)}}
print(json.dumps({'sources':out,'files':files}))
'''.replace('ROOT', repr(plan['output'])).replace('SPEC', repr(origins))
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', 'beast', 'python3', '-'],
                            input=code, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from projected_complement import fields, ACTOR_PARAMETERS
    from selected_source import PARAMETERS
    from sparse_text_kv import resolve
    from sparse_text_kv_study import metadata
    api = Api()
    parent, collection, job, previous = checked(HERE.parent / 'artifacts/projected-complement-v1')
    live = api.json('GET', '/api/jobs/' + job['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != collection['custody']:
        raise ValueError('Projected reader custody differs')
    p = parent / 'result'
    expected = {'compiler.json': 'cff9eb3adcc8c77555bed5c34c5a454affdf9e65798dac1c99365df21f96d5da',
                'compiler-seal.json': '3f90ec44bb280aa2a3eff61704410eecc14b31c7404efb202abc6c96abc4edd4',
                'primary.json': '9da36adb89ab25cf82a56baee608186d53bbe7aa40038a2e693a68c953742182',
                'primary-seal.json': 'c25d7afd1c8f67621cc466d711b051609c0eacc323e9831d4adb19e1942534d1',
                'report.json': '682afdcf5f9643995a21eb0bcc299a93222c2b3d740a86857f13ef7f0e771281'}
    if any(sha(p / n) != h for n, h in expected.items()):
        raise ValueError('Planned projected reader differs')
    c, a, old, target = [read(p / n) for n in ['compiler.json', 'primary.json', 'inputs.json', 'targets.json']]
    for n in ['compiler', 'primary']:
        if read(p / (n + '-seal.json'))[n + 'SHA256'] != sha(p / (n + '.json')):
            raise ValueError('Parent complete seal join differs')
    if sha(FAMILY / 'source-roots.json') != '3b07b8a59185736d31a1d2005c66296899a3335f6a0d83078ba1bf008fe727ac':
        raise ValueError('Original source root manifest differs')
    plan, roots, writers = [read(FAMILY / n) for n in ['plan.json', 'source-roots.json', 'native-source-writers.json']]
    origins = roots[CASE]['inputs']
    wanted = {'P': '8ccb66606eb879c23f819b49060f23624ae7bd523ad1aface825af3119d88bfa',
              'Q': '504bcfb16f8a8063162dca0cd318b760cbfc7a1dd3afecdda5f7e66c925d9f2d'}
    if {n: x['cut'] for n, x in origins.items()} != wanted:
        raise ValueError('Planned source cuts differ')
    family_job = read(FAMILY.parent / 'scheduler-job.json')
    family_live = api.json('GET', '/api/jobs/' + family_job['job_id'])
    if family_live['state'] != 'succeeded' or family_live['payload_custody'] != family_job['payload_custody']:
        raise ValueError('Original source scheduler custody differs')
    meta = source_metadata(plan, origins)
    for role, record in meta['sources'].items():
        cut, original = record['cut'], origins[role]
        if identity(cut['manifest']) != record['cutID'] or cut['fingerprint'] != record['cutID'] or cut['execution_point'] != original['execution_point'] or cut['manifest']['model_identity'] != original['model_identity']:
            raise ValueError('Actual source cut identity/point/model differs')
    # Re-run the original lexical support resolver, without new array/model work.
    old_meta = metadata(plan, writers[CASE]['sources']['Q']['state']['cut'])
    context = next(x for x in plan['family_rule']['fresh_contexts'] if x['id'] == CASE)
    resolved = resolve(context, old_meta['tokens'])
    if resolved['support'] != previous['support'] or resolved['support'] != roots[CASE]['support']:
        raise ValueError('Original support resolution differs')
    artifacts = read(FAMILY / 'parameter-artifacts.json')
    selected = set(PARAMETERS) | set(ACTOR_PARAMETERS)
    artifacts['artifacts'] = [x for x in artifacts['artifacts'] if x['source'].get('tensor') in selected]
    if {x['source']['tensor'] for x in artifacts['artifacts']} != selected:
        raise ValueError('Selected parameter recipes incomplete')
    remote = job['result']['diagnostics']['cwd'] + '/response-result/'
    def field(x):
        return {'path': plan['output'] + '/' + x['path'], 'sha256': x['sha256'], 'payloadSHA256': x['content_sha256'],
                'shape': x['shape'], 'logicalDtype': x['logical_dtype'], 'nativeStride': x['stride'], 'bytes': x['bytes']}
    source_targets = {role: {n: field(rec['ports'][n]) for n in ['text_in', 'text_attention_return_used', 'text_after_attention',
                       'text_mlp_input', 'text_mlp_return_used', 'text_out']} for role, rec in writers[CASE]['sources'].items()}
    new_target = {'family': target['family'], 'parent': {arm: a['actors'][arm]['fields'] for arm in ['QQQ', 'PQP']},
                  'imageIDs': target['imageIDs'], 'sourcePorts': source_targets,
                  'sourceTextTargets': read(HERE.parent / 'artifacts/sparse-text-kv-v1/result/targets.json')['text'],
                  'parentPrimary': {'path': remote + 'primary.json', 'sha256': expected['primary.json']},
                  'parentPrimarySeal': {'path': remote + 'primary-seal.json', 'sha256': expected['primary-seal.json']}}
    s = {k: old[k] for k in ['modelRoot', 'versions', 'nativeSources', 'numericalProfile', 'imageAssets', 'familyPoint', 'N']}
    s.update(sourceCuts=meta['sources'], sourceMetadataFiles={**meta['files'],
             **{plan['output'] + '/' + n: sha(FAMILY / n) for n in ['plan.json', 'source-roots.json', 'native-source-writers.json']}},
             supportResolution=resolved, originalParameterArtifacts=artifacts['artifacts'],
             parentCompiler={'path': remote + 'compiler.json', 'sha256': expected['compiler.json']},
             parentCompilerSeal={'path': remote + 'compiler-seal.json', 'sha256': expected['compiler-seal.json']},
             parentPrimaryJSON=remote + 'primary.json', parentPrimarySealJSON=remote + 'primary-seal.json',
             deniedArrays=sorted(set(old['deniedArrays'] + [x['path'] for x in fields(a)] + [x['path'] for x in fields(new_target)] +
                                     [x['path'] for n, x in c['actorInputs'].items() if n.endswith('.rows')])),
             parentJobID=job['job_id'], parentCustody=live['payload_custody'],
             sourceJobID=family_job['job_id'], sourceCustody=family_live['payload_custody'])
    root = Path(root).resolve()
    root.mkdir(exist_ok=False, parents=True)
    payload = root / 'payload'
    payload.mkdir()
    for name in OWNED:
        if name in previous['ownedSources'] and sha(HERE / name) != previous['sourceHashes'][name]:
            raise ValueError('Frozen reader/helper changed')
        shutil.copyfile(HERE / name, payload / name)
    for name in WHEELS:
        shutil.copyfile(parent / 'payload' / name, payload / name)
    shutil.copyfile(p / 'retrieve_weights.py', payload / 'retrieve_weights.py')
    shutil.copyfile(HERE.parent / 'docs/plans/selected-source-text-producer.md', payload / 'frozen-plan.md')
    save(payload / 'parameter-artifacts.json', artifacts)
    save(payload / 'inputs.json', s)
    save(payload / 'targets.json', new_target)
    preview = _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda')
    save(payload / 'resource-basis.json', {'intentFirstPreview': preview, 'measuredParent': live['result'],
         'geometry': {'sourceBF16ParameterBytes': 283116288, 'reusedProjectedCacheBytes': 604372992,
                      'readerBF16ParameterBytes': 56623360, 'fullSourceKVPayloadBytesPerRole': 18874368},
         'exception': 'Full-model alias sizing does not describe selected tensors. Separate source and reader processes. Source283MBweights/fullKV/small4rowMLP; reader unchanged measured3429.1MBRSS424MBdevice. Reuse6144RAM1536VRAMCPU4disk3GB,180sexpected900cap; no full transformer or new compiler.'})
    contract = one_lease_scheduling_contract(logical_branches=1, logical_operations=70, logical_checkpoint_replays=0,
         conditioner_policy='Authenticated P/Q joint0 inputs; original raw-prompt/native prefixes and receiving image queries/cache remain paid.',
         extra={'modelLoadScope': 'Source11selectedBF16tensors; separate unchanged4tensor projected reader.',
                'targetIsolation': 'Source denies old writer returns and finished rows; readers use own emitted rows. Both seal/exit before archived targets.'})
    r = {'schema': 'tce-selected-source-request-v1', 'ownedSources': OWNED,
         'sourceHashes': {x.name: sha(x) for x in payload.iterdir()}, 'scheduling_contract': contract,
         'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda', 'dtype': 'bfloat16',
         'batch_size': 1, 'seq_len': 1536, 'height': 512, 'width': 512, 'needs': {'cuda': True}, 'timeoutSeconds': 900,
         'case': CASE, 'support': resolved['support'], 'familyPoint': previous['familyPoint'], 'producerPoint': origins['Q']['execution_point'],
         'arithmeticSubplans': {'source': 'CUDA-BF16-fullKV-selectedQ-native-rectangularSDPA-selectedMLP',
                               'reader': 'unchanged-CUDA-fourrowKV-CPUFP64-projectedcache'},
         'protocol': 'Own2sources16affines58.813BMAC+2rectangularSDPA75.497MMAC; ownsequentialMLP thenQQQ/PQPsplice. Sealproducer, unchanged2readers855.638MMAC,sealexit thenarchivedscoring. No newfullreference/suffix/image/fit/precisionarm.'}
    r['requestSHA256'] = identity(r)
    save(root / 'request.json', r)
    save(payload / 'request.json', r)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [x.name for x in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': r['requestSHA256'], 'sourceCuts': wanted}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root = Path(root).resolve()
    r = read(root / 'request.json')
    verify(root / 'payload', r)
    if (root / 'job.json').exists():
        raise ValueError('Already submitted; inspect same handle')
    cmd = command()[:-5]
    for name in WHEELS:
        cmd += ['--with', './' + name]
    cmd += ['python', '-u', 'selected_source_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = guarded_mrun_launch(cmd, request=r, experiment='tce-selected-source-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs=r['needs'],
        mrun_config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width', 'scheduling_contract', 'arithmeticSubplans']},
        ram_mb=6144, vram_mb=1536, cpu_threads=4, disk_gb=3, est_wall_s=180, timeout_s=900, detach=True, retry_on_kill=False, preflight=True,
        note='One spherecube2027 step2 joint0 demanded-source cut:2sources16affines58.813BMAC+2native4x1536SDPA75.497MMAC;ownP/QMLP ancestry;thenunchanged2projectedreaders855.638MMAC,reuse604MBcache. Source11tensors283MB;reader4tensors56.6MB. No image queries/imageMLP/fullreference/suffix/VAE/image/fit. Sealexit beforetargetports. Measuredparent3429RSS424device;6144RAM1536VRAMCPU4disk3GB,180sexpected900cap,noretry.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}
    save(root / 'job.json', value)
    return value


def execute(root):
    root = Path(root).resolve()
    r, s = read(root / 'request.json'), read(root / 'inputs.json')
    verify(root, r)
    started, phases = time.perf_counter(), []
    tick = time.perf_counter()
    subprocess.run([sys.executable, '-u', str(root / 'selected_source_worker.py'), str(root)], check=True, cwd=root)
    phases.append({'phase': 'producer', 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
    out = root / 'response-result'
    producer, seal = read(out / 'producer.json'), read(out / 'producer-seal.json')
    if seal['producerSHA256'] != sha(out / 'producer.json') or seal['requestSHA256'] != r['requestSHA256']:
        raise ValueError('Complete own producer release differs')
    for name in ['parentCompiler', 'parentCompilerSeal']:
        if sha(s[name]['path']) != s[name]['sha256']:
            raise ValueError('Original projected cache metadata differs')
    c = read(s['parentCompiler']['path'])
    if read(s['parentCompilerSeal']['path'])['compilerSHA256'] != s['parentCompiler']['sha256']:
        raise ValueError('Original projected compiler seal differs')
    c = {**c, 'actorInputs': {**c['actorInputs'], **{arm + '.rows': producer['splice']['fields'][arm] for arm in ['QQQ', 'PQP']}},
         'sourceProducerSealSHA256': sha(out / 'producer-seal.json'), 'parentProjectedCompilerSHA256': s['parentCompiler']['sha256'],
         'scope': 'Metadata input binding only; reuses exact paid projected cache, with new own produced rows. No compiler matrix evaluation.'}
    save(out / 'compiler.json', c)
    save(out / 'compiler-seal.json', {'requestSHA256': r['requestSHA256'], 'compilerSHA256': sha(out / 'compiler.json')})
    save(out / 'compiler-execution.json', {'schema': 'tce-selected-source-reader-input-link-v1', 'nominalMatrixMACs': 0,
         'linears': [], 'contractions': [], 'loadedParameters': [], 'scope': 'Metadata binding of reused cache and own source rows; no new cache compiler.'})
    tick = time.perf_counter()
    subprocess.run([sys.executable, '-u', str(root / 'projected_complement_worker.py'), 'actors', str(root)], check=True, cwd=root)
    phases.append({'phase': 'readers', 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
    from selected_source_analysis import execute as score
    report = score(root)
    for name in ['request.json', 'inputs.json', 'targets.json', 'resource-basis.json', 'parameter-artifacts.json', 'retrieve_weights.py', 'frozen-plan.md']:
        shutil.copyfile(root / name, out / name)
    save(out / 'report.json', {**report, 'schema': 'tce-selected-source-result-v1', 'requestSHA256': r['requestSHA256'],
         'phases': phases, 'workerWallSeconds': time.perf_counter() - started, 'mechanics_status': 'complete',
         'instrument_status': 'executed', 'terminal_status': 'unassessed',
         'limits': ['One retrospective source family. Full current source KV and original learned maps remain paid.',
                    'Only four source text rows produced; original prefixes and receiving image queries/complement remain supplied.',
                    'Shortened GEMMs/rectangular SDPA change finite arithmetic; source and downstream discrepancies retained.',
                    'No full successor state, new reference, later diffusion, VAE, image or arbitrary-prompt zero-neural generator.']})
    print('SELECTED_SOURCE_DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'submit', 'execute', 'collect'])
    parser.add_argument('run')
    args = parser.parse_args()
    if args.action == 'collect':
        from response_study import collect
        value = collect(args.run)
    else:
        value = globals()[args.action](args.run)
    if value is not None:
        print(json.dumps(value))
