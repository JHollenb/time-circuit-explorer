"""One bounded single-down prediction, then separate saved-endpoint scoring."""
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
OWNED = ['product_transport.py', 'product_transport_primary.py', 'product_transport_analysis.py', 'product_transport_study.py', 'flux_clock_study.py', 'flux_clock.py', 'complete_masked_study.py', 'encoder_atoms.py', 'paired_fields.py', 'masked_kv_study.py', 'source_response.py']
WHEELS = ['mdb_runtime-0.1.0-py3-none-any.whl', 'saturn_pub-0.4.1-py3-none-any.whl', 'mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from product_transport import DOWN, STAGES
    parent, collection, job, previous = checked(HERE.parent / 'artifacts/first-consumer-v1')
    api = Api()
    live = api.json('GET', '/api/jobs/' + job['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != collection['custody']:
        raise ValueError('Source live custody differs')
    p = parent / 'result'
    inputs, primary = read(p / 'inputs.json'), read(p / 'primary.json')
    seal = read(p / 'primary-seal.json')
    verification = read(parent / 'retained-array-verification.json')
    if seal['primarySHA256'] != sha(p / 'primary.json') or verification['primarySealSHA256'] != sha(p / 'primary-seal.json'):
        raise ValueError('Source release differs')
    for cell in ['R0A', 'R0B', 'R1A', 'R1B']:
        for name, view in primary['views'][cell].items():
            if verification['views'][cell + '.' + name]['payloadSHA256'] != view['payloadSHA256']:
                raise ValueError('Source semantic gate/up view not byte verified')
    selected = [x for x in read(p / 'execution.json')['loadedParameters'] if x['name'] == DOWN]
    if len(selected) != 1 or selected[0]['bytes'] != 56623104:
        raise ValueError('Single-down source closure differs')
    root = Path(root).resolve()
    root.mkdir(exist_ok=False, parents=True)
    payload = root / 'payload'
    payload.mkdir()
    for name in OWNED:
        if name in previous['ownedSources'] and sha(HERE / name) != previous['sourceHashes'][name]:
            raise ValueError('Qualified helper changed')
        shutil.copyfile(HERE / name, payload / name)
    for name in WHEELS:
        shutil.copyfile(parent / 'payload' / name, payload / name)
    shutil.copyfile(HERE.parent / 'docs/plans/receiver-conditioned-product.md', payload / 'frozen-plan.md')
    s = {key: inputs[key] for key in ['modelRoot', 'versions', 'nativeSources', 'imageAssets', 'numericalProfile', 'modulation']}
    s.update(fused={cell: primary['cells'][cell]['fused'] for cell in ['R0A', 'R0B', 'R1A']}, embedding=primary['embeddings']['R1'], carrier=primary['shared']['B']['gatedCarrier'], selectedParameters=selected,
        parentJobID=job['job_id'], protectedRoots=[job['result']['diagnostics']['cwd'], *inputs['protectedRoots']],
        deniedCanaries=[primary['cells']['R1B']['fused']['path'], primary['cells']['R1A']['product']['path'], primary['cells']['R1A']['mlpWrite']['path'], primary['cells']['R1A']['imageExit']['path']],
        parentRecords={name: {'path': job['result']['diagnostics']['cwd'] + '/response-result/' + name, 'sha256': sha(p / name)} for name in ['primary.json', 'primary-seal.json', 'inputs.json', 'execution.json', 'report.json']},
        sourceByteVerificationSHA256=sha(parent / 'retained-array-verification.json'))
    save(payload / 'inputs.json', s)
    save(payload / 'targets.json', {'cells': {cell: {name: fields[name] for name in STAGES} for cell, fields in primary['cells'].items()}, 'fusedR1A': primary['cells']['R1A']['fused'], 'fusedR1B': primary['cells']['R1B']['fused'], 'residualR1B': primary['cells']['R1B']['residualCommit'], 'scope': 'Retrospective native targets. Separate scoring only after complete primary seal and child exit.'})
    save(payload / 'resource-basis.json', {'intentFirstPreview': _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1024], backend='diffusers', dtype='bfloat16', device='cuda'), 'measuredParentJobID': job['job_id'], 'measuredParentResult': {k: live['result'][k] for k in ['elapsed_s', 'peak_rss_mb', 'peak_vram_mb']}, 'schedulerParentRecommendation': job.get('rightsize_recommendation'),
        'workingSet': {'BF16DownBytes': 56623104, 'threeFusedPayloadBytes': 113246208, 'oneFP64FusedBytes': 150994944, 'transportDevice': 'cpu', 'nativeProductDownDevice': 'cuda', 'separatePrimaryAndScoringProcesses': True, 'retainedBytesApprox': 1100000000},
        'exception': 'Generic complete-alias/B64 planning is inapplicable to one selected down/one product. Parent12affine assay measured2907.1MBRSS/700MBVRAM, scheduler1024MBVRAM recommendation. New56.6MBweight/nativeGPUworkingfields under160MB, CPUFP64transport and streamed scoring in separate process. Reserve4096MBRAM/1024MBVRAM CPU4/disk4GB; guard-ceiling5325MB fits66513MB host less3326MBmargin. All immutable inputs Beast-local, pin required.'})
    contract = one_lease_scheduling_contract(logical_branches=1, logical_operations=1, logical_checkpoint_replays=0, conditioner_policy='Three supplied fused native operands and recipient embedding/carrier/clock gate; no encoder or gate/up producer.', extra={'modelLoadScope': 'One original down matrix, one native SiLU/product path. Primary child exits before saved-target scoring.'})
    r = {'schema': 'tce-product-transport-request-v1', 'parentJobID': job['job_id'], 'parentRequestSHA256': previous['requestSHA256'], 'parentPrimarySealSHA256': sha(p / 'primary-seal.json'), 'ownedSources': OWNED, 'sourceHashes': {x.name: sha(x) for x in payload.iterdir()}, 'scheduling_contract': contract,
        'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda', 'dtype': 'bfloat16', 'batch_size': 1, 'seq_len': 1024, 'height': 512, 'width': 512, 'needs': {'cuda': True}, 'timeoutSeconds': 600,
        'protocol': 'Fixed source increments +actual recipientA baseline,CPUFP64→FP32→BF16. NativeCUDA SiLU/product+one original down/gate/residual. Primary seal/exit then savednative/control scoring. No fit,norm,gateup,attention,fullTransformer,encoder,clocknetwork,scheduler,VAE,image.'}
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
    for name in WHEELS:
        cmd += ['--with', './' + name]
    cmd += ['python', '-u', 'product_transport_study.py', 'execute', '.']
    i = cmd.index('uv')
    cmd[i:i] = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = guarded_mrun_launch(cmd, request=r, experiment='tce-product-transport-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs=r['needs'], mrun_config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width', 'scheduling_contract']}, ram_mb=4096, vram_mb=1024, cpu_threads=4, disk_gb=4, est_wall_s=60, timeout_s=600, detach=True, retry_on_kill=False, preflight=True,
        note='One recipient-baseline transport:CPU FP64→FP32→BF16,1CUDA SiLU/product/down=28.991BMAC/56.6MBweights. Seal/childexit before savedtarget scoring. No norm/gateup/attention/fullTransformer/encoder/fit/VAE/image. Parent2907MBRSS700MBVRAM basis4096RAM1024VRAM CPU4disk4GB,60s expected600s cap,no retry.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}
    save(root / 'job.json', value)
    return value


def execute(root):
    root = Path(root).resolve()
    r = read(root / 'request.json')
    verify(root, r)
    start = time.perf_counter()
    subprocess.run([sys.executable, str(root / 'product_transport_primary.py'), str(root)], check=True, cwd=root)
    primary_wall = time.perf_counter() - start
    print('PRODUCT_TRANSPORT_PRIMARY_EXITED', flush=True)
    from product_transport_analysis import execute as analyze
    tick = time.perf_counter()
    result = analyze(root)
    analysis_wall = time.perf_counter() - tick
    out = root / 'response-result'
    for name in ['request.json', 'inputs.json', 'targets.json', 'resource-basis.json', 'frozen-plan.md']:
        shutil.copyfile(root / name, out / name)
    save(out / 'report.json', {'schema': 'tce-product-transport-result-v1', 'requestSHA256': r['requestSHA256'], **result, 'primarySubprocessWallSeconds': primary_wall, 'savedAnalysisWallSeconds': analysis_wall, 'workerWallSeconds': time.perf_counter() - start, 'nominalAffineMACs': 28991029248,
        'work': {'selectedOriginalMatrices': 1, 'selectedParameterBytes': 56623104, 'affineCalls': 1, 'nativeSiLUScalars': 9437184, 'nativeProductScalars': 9437184, 'fullTransformerCalls': 0, 'attentionCalls': 0, 'newNorms': 0, 'newGateUpProjections': 0, 'newEncoders': 0, 'newClockNetworks': 0, 'newFits': 0, 'newSchedulerCommits': 0, 'newVAECalls': 0, 'newImages': 0}, 'mechanics_status': 'complete', 'instrument_status': 'executed', 'terminal_status': 'unassessed',
        'limits': ['Retrospective conditional local prediction from supplied original operands; not raw-prompt serving or a cheap baseline/increment producer.', 'Original down/nativeSiLU and historical donor/receiver/clock/MLP preparation remain paid. No strictzero-neural claim.', 'Local native outputs are the cycle48 source-order implementation; no independent complete-block parity witness or later whole-future mediation.', 'All-row and dependent per-query observations, signed controls, quantization and nonlocal countertrends retained; no effect-size gate.', 'Wall times include imports/hash/transfers/captures/diagnostics and lack a matched speedup baseline.']})
    print('PRODUCT_TRANSPORT_DONE', flush=True)


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
