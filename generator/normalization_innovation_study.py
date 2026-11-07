"""Stage, execute and collect one two-recipient shared-innovation experiment."""
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
OWNED = ['normalization_innovation.py', 'normalization_innovation_primary.py', 'normalization_innovation_analysis.py', 'normalization_innovation_study.py', 'product_transport.py', 'flux_clock_study.py', 'flux_clock.py', 'complete_masked_study.py', 'encoder_atoms.py', 'paired_fields.py', 'masked_kv_study.py', 'source_response.py']
WHEELS = ['mdb_runtime-0.1.0-py3-none-any.whl', 'saturn_pub-0.4.1-py3-none-any.whl', 'mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from normalization_innovation import IN, DOWN, STAGES
    parent, collection, job, previous = checked(HERE.parent / 'artifacts/first-consumer-v1')
    control, cc, cj, cr = checked(HERE.parent / 'artifacts/product-transport-v1')
    api = Api()
    for j, c in [(job, collection), (cj, cc)]:
        live = api.json('GET', '/api/jobs/' + j['job_id'])
        if live['state'] != 'succeeded' or live['payload_custody'] != c['custody']:
            raise ValueError('Native/control live custody differs')
    p = parent / 'result'
    s0, native = read(p / 'inputs.json'), read(p / 'primary.json')
    if read(parent / 'retained-array-verification.json')['primarySealSHA256'] != sha(p / 'primary-seal.json') or read(control / 'retained-array-verification.json')['primarySealSHA256'] != sha(control / 'result/primary-seal.json'):
        raise ValueError('Parent native/control release differs')
    selected = [x for x in read(p / 'execution.json')['loadedParameters'] if x['name'] in [IN, DOWN]]
    if len(selected) != 2 or sum(x['bytes'] for x in selected) != 169869312:
        raise ValueError('Two selected matrix admission differs')
    root = Path(root).resolve()
    root.mkdir(exist_ok=False, parents=True)
    payload = root / 'payload'
    payload.mkdir()
    for name in OWNED:
        for old in [previous, cr]:
            if name in old['ownedSources'] and sha(HERE / name) != old['sourceHashes'][name]:
                raise ValueError('Qualified helper changed: ' + name)
        shutil.copyfile(HERE / name, payload / name)
    for name in WHEELS:
        shutil.copyfile(control / 'payload' / name, payload / name)
    shutil.copyfile(HERE.parent / 'docs/plans/shared-normalization-innovation.md', payload / 'frozen-plan.md')
    s = {k: s0[k] for k in ['modelRoot', 'versions', 'nativeSources', 'imageAssets', 'numericalProfile', 'modulation']}
    s.update(fusedA={r: native['cells'][r + 'A']['fused'] for r in ['R0', 'R1']}, embeddings=native['embeddings'], carriers={h: native['shared'][h]['gatedCarrier'] for h in ['A', 'B']}, onePlusScale=native['constants']['onePlusScale'], selectedParameters=selected, parentJobID=job['job_id'], controlJobID=cj['job_id'],
        protectedRoots=[job['result']['diagnostics']['cwd'], cj['result']['diagnostics']['cwd'], *s0['protectedRoots']],
        deniedCanaries=[native['cells'][r + 'B']['fused']['path'] for r in ['R0', 'R1']] + [native['cells'][r + 'A']['product']['path'] for r in ['R0', 'R1']] + [read(control / 'result/primary.json')['fields']['imageExit']['path']],
        parentRecords={label + '.' + name: {'path': j['result']['diagnostics']['cwd'] + '/response-result/' + name, 'sha256': sha(local / 'result' / name)} for label, j, local in [('native', job, parent), ('control', cj, control)] for name in ['primary.json', 'primary-seal.json', 'inputs.json', 'execution.json', 'report.json']})
    save(payload / 'inputs.json', s)
    save(payload / 'targets.json', {'cells': {cell: {k: fs[k] for k in STAGES} for cell, fs in native['cells'].items()}, 'residuals': {cell: fs['residualCommit'] for cell, fs in native['cells'].items()}, 'priorConditioned': {k: read(control / 'result/primary.json')['fields'][k] for k in STAGES}, 'scope': 'Retrospective native endpoints/control. Available only to separate scorer after both candidates seal and primary exits.'})
    save(payload / 'resource-basis.json', {'intentFirstPreview': _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1024], backend='diffusers', dtype='bfloat16', device='cuda'), 'measuredParents': [{'jobID': j['job_id'], 'result': {k: j['result'][k] for k in ['elapsed_s', 'peak_rss_mb', 'peak_vram_mb']}} for j in [job, cj]],
        'workingSet': {'BF16ParameterBytes': 169869312, 'FP64InputMatrixBytes': 452984832, 'FP64SharedJBytes': 150994944, 'FP64Contractions': 'CPU4', 'nativeProductDownResidual': 'CUDA BF16 B1', 'separatePrimaryAndScorer': True, 'retainedBytesApprox': 3300000000},
        'exception': 'Generic full-model/B64 planning cannot represent2CPUFP64contractions+2CUDA BF16downs. Cycle49 measured1664MBRSS/494MBVRAM; add453MBFP64CPUweight and shared151MBJ with streamed recipient captures. Cycle48 four-matrix job measured2907MBRSS/700MBVRAM. Select4096MBRAM/1024MBVRAM CPU4/disk6GB;5325MBguardceil fits66513MB host less3326margin. Weights/operands are Beast-local; hard pin. Actual measurement replaces this basis for matching future runs.'})
    contract = one_lease_scheduling_contract(logical_branches=2, logical_operations=4, logical_checkpoint_replays=0, conditioner_policy='Native A fused baselines, embeddings/carriers and clock constants only; no supplied B fused endpoint.', extra={'modelLoadScope': 'Two original matrices. Shared FP64 CPU J/beta once, two native CUDA BF16 product/down paths. Primary child seals both B candidates/exits before targets.'})
    r = {'schema': 'tce-normalization-innovation-request-v1', 'parentJobID': job['job_id'], 'controlJobID': cj['job_id'], 'parentPrimarySealSHA256': sha(p / 'primary-seal.json'), 'ownedSources': OWNED, 'sourceHashes': {x.name: sha(x) for x in payload.iterdir()}, 'scheduling_contract': contract,
        'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda', 'dtype': 'bfloat16', 'batch_size': 1, 'seq_len': 1024, 'height': 512, 'width': 512, 'needs': {'cuda': True}, 'timeoutSeconds': 600,
        'arithmeticSubplans': {'sharedContractions': 'torch-cpu-float64', 'statisticsSecant': 'torch-cpu-float64', 'nonlinearDownResidual': 'torch-cuda-bfloat16'},
        'protocol': 'One shared centered carrier innovation and clock row, bothCPUFP64 W_in. Predict2Bfused fromnativeA/statistics;explicitFP64→FP32→BF16;2nativeCUDA SiLU/down/gatedexits. No Bfused/outputs/controltargets until both seal/primaryexit. No fit/nativeLayerNorm/attention/fullTransformer/encoder/clocknet/scheduler/VAE/images/extra-rho-projections.'}
    r['requestSHA256'] = identity(r)
    save(root / 'request.json', r)
    save(payload / 'request.json', r)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [x.name for x in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': r['requestSHA256']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root = Path(root).resolve()
    r = read(root / 'request.json')
    verify(root / 'payload', r)
    if (root / 'job.json').exists():
        raise ValueError('Already submitted; inspect same job')
    cmd = command()[:-5]
    for n in WHEELS:
        cmd += ['--with', './' + n]
    cmd += ['python', '-u', 'normalization_innovation_study.py', 'execute', '.']
    i = cmd.index('uv')
    cmd[i:i] = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = guarded_mrun_launch(cmd, request=r, experiment='tce-normalization-innovation-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs=r['needs'], mrun_config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width', 'scheduling_contract', 'arithmeticSubplans']}, ram_mb=4096, vram_mb=1024, cpu_threads=4, disk_gb=6, est_wall_s=120, timeout_s=600, detach=True, retry_on_kill=False, preflight=True,
        note='SharedLNinnovation:2CPUFP64J/beta+2CUDA BF16downs=4affines116.021BMAC/169.9MBparams. No Bfused supplied;bothcandidateexitsseal/childexit then savedtarget/control scoring. No normmodule/attention/fullmodel/fit/encoder/VAE/images/rhoprojections. CPU4,4096RAM1024VRAM,disk6GB,120s expected600s cap,noretry;parent1664MBRSS494MBVRAM plusCPU453MBweight151MBJ.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}
    save(root / 'job.json', value)
    return value


def execute(root):
    root = Path(root).resolve()
    r = read(root / 'request.json')
    verify(root, r)
    start = time.perf_counter()
    subprocess.run([sys.executable, str(root / 'normalization_innovation_primary.py'), str(root)], check=True, cwd=root)
    primary_wall = time.perf_counter() - start
    print('NORMALIZATION_INNOVATION_PRIMARY_EXITED', flush=True)
    from normalization_innovation_analysis import execute as analyze
    tick = time.perf_counter()
    result = analyze(root)
    analysis_wall = time.perf_counter() - tick
    out = root / 'response-result'
    for n in ['request.json', 'inputs.json', 'targets.json', 'resource-basis.json', 'frozen-plan.md']:
        shutil.copyfile(root / n, out / n)
    save(out / 'report.json', {'schema': 'tce-normalization-innovation-result-v1', 'requestSHA256': r['requestSHA256'], **result, 'primarySubprocessWallSeconds': primary_wall, 'savedAnalysisWallSeconds': analysis_wall, 'workerWallSeconds': time.perf_counter() - start, 'nominalAffineMACs': 116020740096,
        'work': {'selectedOriginalMatrices': 2, 'selectedBF16ParameterBytes': 169869312, 'affineCalls': 4, 'FP64CPUSharedProjections': 2, 'nativeDownProjections': 2, 'nativeSiLUScalars': 18874368, 'nativeProductScalars': 18874368, 'FP64NormalizationInputScalars': 12582912, 'nativeFusedBProjections': 0, 'extraRhoProjections': 0, 'nativeLayerNormCalls': 0, 'fullTransformerCalls': 0, 'attentionCalls': 0, 'newEncoders': 0, 'newClockNetworks': 0, 'newFits': 0, 'newSchedulerCommits': 0, 'newVAECalls': 0, 'newImages': 0}, 'mechanics_status': 'complete', 'instrument_status': 'executed', 'terminal_status': 'unassessed',
        'limits': ['Conditional local prediction on supplied native A baselines/embeddings/carriers. Both B fused operands omitted; no economical first-prompt or own-state closure.', 'Real-arithmetic shared-innovation law is not native-exact. Residual commit rho, native normalization/modulation/GEMM defects and final quantization remain unclosed.', 'Original learned W_in/W_down, FP64 contractions and historical preparation remain paid. No strictzero-neural claim.', 'No extra projections to assign causal shares. All signed/per-query/nonlocal trends and inherited-control costs retained; no effect-size gate.', 'Cycle48 source-order targets have no independent full-native-block-exit witness; no later whole-future mediation, image or matched serving-speed claim.']})
    print('NORMALIZATION_INNOVATION_DONE', flush=True)


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
