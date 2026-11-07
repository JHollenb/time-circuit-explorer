"""One source-bound compiler → fresh sparse actor → archived-target assay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read, save, sha, identity
from flux_clock_study import verify

HERE = Path(__file__).resolve().parent
CASE = 'sphere-cube-plain-2027'
FAMILY = HERE.parent.parent / 'artifacts/mdb-computation-family-20261107'
OWNED = ['sparse_text_kv.py', 'sparse_text_kv_study.py', 'sparse_text_kv_worker.py', 'sparse_text_kv_analysis.py',
         'encoder_atoms.py', 'paired_fields.py', 'flux_clock.py', 'flux_clock_study.py', 'complete_masked_study.py', 'masked_kv_study.py']
WHEELS = ['mdb_runtime-0.1.0-py3-none-any.whl', 'saturn_pub-0.4.1-py3-none-any.whl', 'mrun_pub-0.1.2-py3-none-any.whl']


def metadata(plan, cut):
    """Authenticated JSON/byte metadata I/O only; no array computation over SSH."""
    code = '''import json,pathlib,hashlib
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
r=pathlib.Path(ROOT); fresh=pathlib.Path(FRESH); cutpath=r/'state-store/cuts'/ (CUT+'.json')
c=json.load(open(cutpath));files={str(cutpath):sha(cutpath)};pages={}
for slot in ['context','img_ids']:
 pid=c['slots'][slot];p=r/'state-store/pages'/ (pid+'.json');v=json.load(open(p))
 if sha(p)!=pid:raise ValueError('State page identity differs')
 blob=r/'state-store/blobs'/ (v['tensor_blob']+'.safetensors')
 files[str(p)]=pid;files[str(blob)]=v['tensor_blob'];pages[slot]={'pageID':pid,'metadata':v,'store':str(r/'state-store')}
t=fresh/'token-addresses.json';f=fresh/'files.json';manifest=json.load(open(f))
token=json.load(open(t));files[str(t)]=sha(t);files[str(f)]=sha(f)
print(json.dumps({'cut':c,'pages':pages,'tokens':token['tokens'][CASE], 'addresses':token['addresses'][CASE],
 'files':files,'freshManifestSHA256':sha(f),'tokenManifestEntry':manifest.get('files',manifest).get('token-addresses.json')}))
'''.replace('ROOT', repr(plan['output'])).replace('FRESH', repr(plan['fresh_source']['output_root'])).replace('CUT', repr(cut)).replace('CASE', repr(CASE))
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', 'beast', 'python3', '-'], input=code, text=True, capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from sparse_text_kv import PARAMETERS, resolve
    api = Api()
    compact = FAMILY / 'fresh-family-02/compact'
    parent = read(FAMILY / 'fresh-family-02/scheduler-job.json')
    live = api.json('GET', '/api/jobs/' + parent['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != parent['payload_custody']:
        raise ValueError('Family scheduler custody differs')
    expected = {'native-source-writers.json': 'b282318bcd075315d823acdbc0f9c8418e93108f48e84db8ebc8ac1a568a39e2',
                'consumer-ports.json': '7add56418aa47672146b605269727c29dbf692d5fd8b47f61a8f5d36f7a9b466'}
    if any(sha(compact / n) != h for n, h in expected.items()) or sha(FAMILY / 'family-rule-v1.json') != '0609082da7815afb15a3780cb32300014ee6154501d0991e24095cddad9c226d':
        raise ValueError('Frozen family records differ')
    plan, writers, ports = [read(compact / n) for n in ['plan.json', 'native-source-writers.json', 'consumer-ports.json']]
    w = writers[CASE]
    point = w['sources']['Q']['state']['execution_point']
    if point['logical_step'] != 2 or point['index'] != 1 or point['next_operation'] != 'joint_block.1':
        raise ValueError('Wrong family invocation')
    meta = metadata(plan, w['sources']['Q']['state']['cut'])
    if meta['freshManifestSHA256'] != plan['fresh_source']['files_manifest_sha256']:
        raise ValueError('Fresh source custody differs')
    context = next(x for x in plan['family_rule']['fresh_contexts'] if x['id'] == CASE)
    selected = resolve(context, meta['tokens'])
    if selected['support'] != meta['addresses']['S'] or selected['support'] != [4, 11, 12, 13]:
        raise ValueError('Original address resolver differs')
    token_path = plan['fresh_source']['output_root'] + '/token-addresses.json'
    entry = meta['tokenManifestEntry']
    if (entry['sha256'] if isinstance(entry, dict) else entry) != meta['files'][token_path]:
        raise ValueError('Token source manifest differs')
    remote = plan['output']
    def field(item):
        return {'path': remote + '/' + item['path'], 'sha256': item['sha256'], 'payloadSHA256': item['content_sha256'],
                'shape': item['shape'], 'nativeStride': item['stride'], 'logicalDtype': item['logical_dtype'], 'bytes': item['bytes']}
    source_ports = {role: {n: field(v[n]) for n in ['text_in', 'text_attention_return_used', 'text_mlp_return_used', 'text_attention_gate', 'text_mlp_gate', 'text_out']}
                    for role, rec in w['sources'].items() for v in [rec['ports']]}
    image = field(w['sources']['Q']['ports']['image_out'])
    targets = {'R_attention': {arm: field(ports['native'][CASE + '/' + arm]['R_attention']) for arm in ['QQQ', 'PQP']},
               'text': {arm: field(w['endpoints'][arm]) for arm in ['QQQ', 'PQP']}}
    if targets['R_attention']['QQQ']['sha256'] != 'c19e777cefe5989406d10bd4c10e0679e8659dbdf1f40f0d49ba8e0873c0f2aa' or targets['R_attention']['PQP']['sha256'] != 'b870318d7b3a16febe6b2949a0a8a87e17ad785d7e05d81a4d4779dff3f1d16a':
        raise ValueError('Prespecified targets differ')
    artifacts = read(compact / 'parameter-artifacts.json')
    original = [x for x in artifacts['artifacts'] if x['source'].get('tensor') in PARAMETERS]
    if len(original) != 9:
        raise ValueError('Nine selected original tensors required')
    original = {**artifacts, 'artifacts': original}
    old = read(HERE.parent / 'artifacts/value-innovation-v3/result/inputs.json')
    model = plan['flux_parameter_root']
    root = Path(root).resolve()
    root.mkdir(exist_ok=False, parents=True)
    payload = root / 'payload'
    payload.mkdir()
    for name in OWNED:
        shutil.copyfile(HERE / name, payload / name)
    for name in WHEELS:
        shutil.copyfile(HERE.parent / 'artifacts/value-innovation-v3/payload' / name, payload / name)
    shutil.copyfile(compact / 'retrieve_weights.py', payload / 'retrieve_weights.py')
    shutil.copyfile(HERE.parent / 'docs/plans/sparse-text-kv-innovation.md', payload / 'frozen-plan.md')
    save(payload / 'parameter-artifacts.json', original)
    save(payload / 'family-metadata.json', meta)
    save(payload / 'targets.json', targets)
    source = {'modelRoot': model, 'versions': old['versions'], 'nativeSources': old['nativeSources'], 'numericalProfile': old['numericalProfile'],
              'imageAssets': {n: h for n, h in old['imageAssets'].items() if n.startswith('transformer/')},
              'writerPorts': source_ports, 'image': image, 'context': context, 'supportResolution': selected,
              'originalParameterArtifacts': original['artifacts'], 'familyPoint': point,
              'deniedTargets': [v['path'] for ports in targets.values() for v in ports.values()],
              'parentCustody': live['payload_custody'], 'parentJobID': parent['job_id'],
              'sourceMetadataFiles': {remote + '/' + n: sha(compact / n) for n in ['plan.json', 'native-source-writers.json', 'consumer-ports.json']},
              'sourceRootCorrection': {'planned': '/mnt/hdd8tb/mrun-work/job-a2048028d7bf/flux-binding-family-result', 'actual': remote,
                                       'authority': 'Original frozen family plan.output/output_scratch; original array hashes unchanged.'}}
    save(payload / 'inputs.json', source)
    preview = _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda')
    save(payload / 'resource-basis.json', {'intentFirstPreview': preview, 'measuredBasisJob': 'job-b187c2bccbdd',
         'measuredBasis': {'RSS_MB': 3086, 'device_MB': 748, 'seconds': 125.07},
         'exception': 'Full-model alias geometry is inapplicable to nine selected tensors. Separate compiler/actor; full FP64 QK and AV plus cache, no fullmodel. Reuse6144MBRAM1536MBVRAM CPU4; expected180s cap900s,3GBdisk. Full score302MB,weights302MB,numerator25MB; original parameter/source checks included.'})
    contract = one_lease_scheduling_contract(logical_branches=1, logical_operations=11, logical_checkpoint_replays=0,
        conditioner_policy='Historical source P/Q writer ports and original context; no fresh raw prompt interpretation.',
        extra={'modelLoadScope': 'Compiler five original QKV plus3norms; fresh actors K/V/Wo plus1norm only.',
               'targetIsolation': 'Original native targets denied in compiler and actor; actor seals and child exits before scoring.'})
    r = {'schema': 'tce-sparse-text-kv-request-v1', 'ownedSources': OWNED, 'sourceHashes': {x.name: sha(x) for x in payload.iterdir()},
         'scheduling_contract': contract, 'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda', 'dtype': 'bfloat16',
         'batch_size': 1, 'seq_len': 1536, 'height': 512, 'width': 512, 'needs': {'cuda': True}, 'timeoutSeconds': 900,
         'case': CASE, 'support': selected['support'], 'familyPoint': point,
         'arithmeticSubplans': {'projectionsNormsRoPEWo': 'source-native-cuda-bfloat16', 'complementAndUpdate': 'torch-cpu-float64'},
         'protocol': 'One paid compiler5QKV/allkeyQKAV; one fresh actor process2fourrowKV/Wo. Positive complement, no oldKV or targetfields in actors. Seal/exit before savedtargetscoring. No fullMLP/textcontinuation/suffix/VAE/image, no fit or precisionsearch.'}
    r['requestSHA256'] = identity(r)
    save(root / 'request.json', r)
    save(payload / 'request.json', r)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [x.name for x in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': r['requestSHA256'], 'support': r['support']}


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
    cmd += ['python', '-u', 'sparse_text_kv_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = guarded_mrun_launch(cmd, request=r, experiment='tce-sparse-text-kv-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs=r['needs'],
        mrun_config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width', 'scheduling_contract', 'arithmeticSubplans']},
        ram_mb=6144, vram_mb=1536, cpu_threads=4, disk_gb=3, est_wall_s=180, timeout_s=900, detach=True, retry_on_kill=False, preflight=True,
        note='Family step2joint1 spherecube2027: compile5QKV/allkeyFP64QKAV48.318BMAC. Fresh process2actors4rowKV/Wo19.529BMAC; fixedqueries/complement. Targetsdenied/sealexit then archivedscoring. No fullMLP/suffix/VAE/images/fit. CPU4CUDA BF16/CPUFP64;6144RAM1536VRAM3GBdisk180sexpected900cap. Basis51RSS3086MBdevice748MB; noautoretry.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}
    save(root / 'job.json', value)
    return value


def execute(root):
    root = Path(root).resolve()
    request = read(root / 'request.json')
    verify(root, request)
    started, phases = time.perf_counter(), []
    for phase in ['compile', 'actors']:
        tick = time.perf_counter()
        subprocess.run([sys.executable, '-u', str(root / 'sparse_text_kv_worker.py'), phase, str(root)], check=True, cwd=root)
        phases.append({'phase': phase, 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
        print('SPARSE_TEXT_KV_' + phase.upper() + '_EXITED', flush=True)
    from sparse_text_kv_analysis import execute as score
    report = score(root)
    out = root / 'response-result'
    for name in ['request.json', 'inputs.json', 'targets.json', 'family-metadata.json', 'resource-basis.json', 'parameter-artifacts.json', 'retrieve_weights.py', 'frozen-plan.md']:
        shutil.copyfile(root / name, out / name)
    save(out / 'report.json', {'schema': 'tce-sparse-text-kv-result-v1', 'requestSHA256': request['requestSHA256'], **report,
         'phases': phases, 'workerWallSeconds': time.perf_counter() - started,
         'mechanics_status': 'complete', 'instrument_status': 'executed', 'terminal_status': 'unassessed',
         'limits': ['Retrospective single family invocation; historical P/Q source formation and original learned matrices remain paid.',
                    'Cache assumes fixed queries, complement, support, positions, profile and revision.',
                    'Compact GEMM and FP64 softmax alter original numerical boundaries; sham errors remain separate.',
                    'No changed text continuation, joint1 MLP, later own-state diffusion, new image or arbitrary-prompt zero-neural generator.']})
    print('SPARSE_TEXT_KV_DONE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['stage', 'submit', 'execute', 'collect'])
    p.add_argument('run')
    a = p.parse_args()
    if a.action == 'collect':
        from response_study import collect
        value = collect(a.run)
    else:
        value = globals()[a.action](a.run)
    if value is not None:
        print(json.dumps(value))
