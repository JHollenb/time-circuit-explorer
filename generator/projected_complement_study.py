"""One paid Wo compiler, fresh projected readers and sealed-target scoring."""
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
OWNED = ['projected_complement.py', 'projected_complement_study.py', 'projected_complement_worker.py', 'projected_complement_analysis.py',
         'sparse_text_kv.py', 'encoder_atoms.py', 'paired_fields.py', 'flux_clock.py', 'flux_clock_study.py', 'complete_masked_study.py', 'masked_kv_study.py']
WHEELS = ['mdb_runtime-0.1.0-py3-none-any.whl', 'saturn_pub-0.4.1-py3-none-any.whl', 'mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from projected_complement import ACTOR_PARAMETERS, fields
    api = Api()
    parent, collection, job, previous = checked(HERE.parent / 'artifacts/sparse-text-kv-v1')
    live = api.json('GET', '/api/jobs/' + job['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != collection['custody']:
        raise ValueError('Original sparse-reader custody differs')
    p = parent / 'result'
    expected = {'compiler.json': 'b98d648d30df7f21948fc3854057c591eb79a025b5d29c0a3bc350e97688e7f2',
                'compiler-seal.json': '81677772d9dce36a3ca4da1c7629c3107f41ed59e3c5ef7bd2572fd9cfefcf7d',
                'primary.json': '1dd07f7ab9f3d226ae5cfc779dff958c696ecf568c80a3cd17585f9808c98d58',
                'primary-seal.json': '9241fec922084585312377815ff140c400fbffd9ee218b32076d28b48313df21',
                'report.json': '2eff58e258c605f9ec0c632a271427cdf2a7a87ebe77ec8c03ad46a3efda2558'}
    if any(sha(p / n) != h for n, h in expected.items()):
        raise ValueError('Planned parent differs')
    c, a, old, target, receipt = [read(p / n) for n in ['compiler.json', 'primary.json', 'inputs.json', 'targets.json'] ] + [read(parent / 'retained-array-verification.json')]
    if receipt['compilerSealSHA256'] != expected['compiler-seal.json'] or receipt['primarySealSHA256'] != expected['primary-seal.json']:
        raise ValueError('Parent byte release differs')
    for n in ['compiler', 'primary']:
        if read(p / (n + '-seal.json'))[n + 'SHA256'] != sha(p / (n + '.json')):
            raise ValueError('Parent complete seal join differs')
    for x in fields(c['actorInputs']):
        if receipt['files'][x['path']]['sha256'] != x['sha256']:
            raise ValueError('Original admitted field lacks byte custody')
    root = Path(root).resolve()
    root.mkdir(exist_ok=False, parents=True)
    payload = root / 'payload'
    payload.mkdir()
    for name in OWNED:
        if name in previous['ownedSources'] and sha(HERE / name) != previous['sourceHashes'][name]:
            raise ValueError('Frozen helper changed')
        shutil.copyfile(HERE / name, payload / name)
    for name in WHEELS:
        shutil.copyfile(parent / 'payload' / name, payload / name)
    shutil.copyfile(p / 'retrieve_weights.py', payload / 'retrieve_weights.py')
    shutil.copyfile(HERE.parent / 'docs/plans/projected-complement-reader.md', payload / 'frozen-plan.md')
    artifacts = read(p / 'parameter-artifacts.json')
    artifacts['artifacts'] = [x for x in artifacts['artifacts'] if x['source']['tensor'] in ACTOR_PARAMETERS]
    if len(artifacts['artifacts']) != 4:
        raise ValueError('Four selected original parameters required')
    save(payload / 'parameter-artifacts.json', artifacts)
    remote = job['result']['diagnostics']['cwd'] + '/response-result/'
    s = {k: old[k] for k in ['modelRoot', 'versions', 'nativeSources', 'numericalProfile', 'imageAssets', 'familyPoint']}
    s.update(N=c['actorInputs']['n'], actorInputs={n: x for n, x in c['actorInputs'].items() if n != 'n'},
             originalParameterArtifacts=artifacts['artifacts'], parentJobID=job['job_id'], parentCustody=live['payload_custody'],
             parentCompiler={'path': remote + 'compiler.json', 'sha256': expected['compiler.json']},
             parentCompilerSeal={'path': remote + 'compiler-seal.json', 'sha256': expected['compiler-seal.json']},
             deniedArrays=[x['path'] for x in fields(a)] + [x['path'] for x in fields(target)],
             parentPrimaryJSON=remote + 'primary.json', parentPrimarySealJSON=remote + 'primary-seal.json')
    save(payload / 'inputs.json', s)
    save(payload / 'targets.json', {'family': target['R_attention'], 'parent': {arm: a['actors'][arm]['fields'] for arm in ['QQQ', 'PQP']},
         'imageIDs': c['diagnostics']['imageIDs'], 'parentPrimary': {'path': remote + 'primary.json', 'sha256': expected['primary.json']},
         'parentPrimarySeal': {'path': remote + 'primary-seal.json', 'sha256': expected['primary-seal.json']},
         'scope': 'Both references already observed; admitted only after complete new actor seal and child exit.'})
    preview = _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda')
    save(payload / 'resource-basis.json', {'intentFirstPreview': preview, 'measuredParentJobID': job['job_id'], 'measuredParentResult': live['result'],
         'geometry': {'projectedCacheBytes': 604372992, 'WoFP64BytesPerProcess': 75497472, 'largestHeadWorkingFieldBytes': 25165824,
                      'actorSelectedBF16ParameterBytes': 56623360, 'compilerSelectedBF16ParameterBytes': 18874368},
         'exception': 'Full-model alias sizing is inapplicable to four selected tensors. Parent2194.3MBRSS/598MBdevice; new604MBcache plus retained-array copies and streamed25MB/head scratch. Reuse6144RAM1536VRAM CPU4disk3GB; expected180s cap900s. CPUFP64 compiler, CUDA4rowKV/CPUFP64 actors, no fullmodel.'})
    contract = one_lease_scheduling_contract(logical_branches=1, logical_operations=76, logical_checkpoint_replays=0,
        conditioner_policy='Parent sealed fixed-query complement and source-resolved raw rows; original source/cache preparation paid.',
        extra={'modelLoadScope': 'CPUcompiler selectedWo only; fresh actorK/V/normCUDA plusWoCPUFP64, no fullmodel.',
               'targetIsolation': 'Compiler oldN/Wo only; actor projectedcache/rows, no oldN/H/outputs. Seal/exit before both saved references.'})
    request = {'schema': 'tce-projected-complement-request-v1', 'ownedSources': OWNED, 'sourceHashes': {x.name: sha(x) for x in payload.iterdir()},
         'scheduling_contract': contract, 'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda', 'dtype': 'bfloat16',
         'batch_size': 1, 'seq_len': 1536, 'height': 512, 'width': 512, 'needs': {'cuda': True}, 'timeoutSeconds': 900,
         'case': previous['case'], 'support': previous['support'], 'familyPoint': previous['familyPoint'],
         'arithmeticSubplans': {'compiler': 'torch-cpu-float64-24-physical-Wo-headblocks', 'actors': 'native-cuda-bf16-KV-norm-rope;CPUFP64-projectedpositivecache;nativeBF16-gate-residual'},
         'finiteLaw': 'Perhead A=w@projectedV; T=(cacheScale*projectedN+A)/D; fixedascendingheadsum; FP64→FP32→BF16 then originalgate/residual.',
         'protocol': 'Compile24physicalWoNheadblocks9.664BMAC. Freshprocess2actors each2KV+24fourvalueWoblocks+QK/projectedAV427.819MMAC, plusseparatescalarwork. Seal/exit before parent+familytargets. No fit/precisionarm/fullmodel/MLP/suffix/VAE/image.'}
    request['requestSHA256'] = identity(request)
    save(root / 'request.json', request)
    save(payload / 'request.json', request)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [x.name for x in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': request['requestSHA256']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root = Path(root).resolve()
    request = read(root / 'request.json')
    verify(root / 'payload', request)
    if (root / 'job.json').exists():
        raise ValueError('Already submitted; inspect same handle')
    cmd = command()[:-5]
    for name in WHEELS:
        cmd += ['--with', './' + name]
    cmd += ['python', '-u', 'projected_complement_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = guarded_mrun_launch(cmd, request=request, experiment='tce-projected-complement-v1', model=request['model'], task='forward', payload=root / 'payload', pin='beast', needs=request['needs'],
        mrun_config={k: request[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width', 'scheduling_contract', 'arithmeticSubplans']},
        ram_mb=6144, vram_mb=1536, cpu_threads=4, disk_gb=3, est_wall_s=180, timeout_s=900, detach=True, retry_on_kill=False, preflight=True,
        note='Fixedfamily step2joint1: CPUFP64 Wo-complement24headblocks9.664BMAC,604MBcache. Fresh2actors each427.819MmatrixMAC+75.497Mscalepairs/75.497Mdiv/72.352Mheadadds. NativeCUDA4rowKV;CPUFP64ownvalueprojection. Seal/exit before savedparent/familyR. No fullMLP/suffix/image/fit/precisionarm. CPU4,6144RAM1536VRAM3GBdisk,180sexpected900cap,noretry. Parent2194RSS598device; streamed25MBheadscratch.')
    value = {'jobID': str(job), 'requestSHA256': request['requestSHA256']}
    save(root / 'job.json', value)
    return value


def execute(root):
    root = Path(root).resolve()
    request = read(root / 'request.json')
    verify(root, request)
    started, phases = time.perf_counter(), []
    for phase in ['compile', 'actors']:
        tick = time.perf_counter()
        subprocess.run([sys.executable, '-u', str(root / 'projected_complement_worker.py'), phase, str(root)], check=True, cwd=root)
        phases.append({'phase': phase, 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
        print('PROJECTED_COMPLEMENT_' + phase.upper() + '_EXITED', flush=True)
    from projected_complement_analysis import execute as score
    report = score(root)
    out = root / 'response-result'
    for name in ['request.json', 'inputs.json', 'targets.json', 'resource-basis.json', 'parameter-artifacts.json', 'retrieve_weights.py', 'frozen-plan.md']:
        shutil.copyfile(root / name, out / name)
    save(out / 'report.json', {'schema': 'tce-projected-complement-result-v1', 'requestSHA256': request['requestSHA256'], **report,
         'phases': phases, 'workerWallSeconds': time.perf_counter() - started, 'mechanics_status': 'complete', 'instrument_status': 'executed', 'terminal_status': 'unassessed',
         'limits': ['One retrospective fixed-query/source-resolved four-text-row family update; old source/cache production paid.',
                    'Original K/V maps, selected learned Wo value projections and new604MBprojectedcache remain required.',
                    'Real-arithmetic commutation removes intermediate BF16 head rounding; this is a changed finite program.',
                    'No new MLP, text continuation, later own-state diffusion, image or arbitrary-prompt zero-neural generator.',
                    'Matrix MACs omit separately counted scalar cache operations and I/O; no measured serving-speed comparison.']})
    print('PROJECTED_COMPLEMENT_DONE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['stage', 'submit', 'execute', 'collect'])
    p.add_argument('run')
    args = p.parse_args()
    if args.action == 'collect':
        from response_study import collect
        value = collect(args.run)
    else:
        value = globals()[args.action](args.run)
    if value is not None:
        print(json.dumps(value))
