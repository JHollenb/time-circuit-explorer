"""One guarded value-address producer, post-seal reference and saved scoring."""
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
OWNED = ['value_innovation.py', 'value_innovation_primary.py', 'value_innovation_reference.py', 'value_innovation_analysis.py', 'value_innovation_study.py',
         'source_native_reader.py', 'native_reader_corpus.py', 'first_consumer.py', 'normalization_innovation.py', 'product_transport.py', 'flux_clock_study.py',
         'flux_clock.py', 'complete_masked_study.py', 'encoder_atoms.py', 'paired_fields.py', 'masked_kv_study.py', 'source_response.py']
WHEELS = ['mdb_runtime-0.1.0-py3-none-any.whl', 'saturn_pub-0.4.1-py3-none-any.whl', 'mrun_pub-0.1.2-py3-none-any.whl']


def host_rows(response):
    rows = response.get('hosts') if isinstance(response, dict) else response
    if not isinstance(rows, list) or not all(isinstance(x, dict) and 'name' in x for x in rows):
        raise ValueError('Invalid scheduler host inventory')
    return rows


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from value_innovation import IN, DOWN, WO, STAGES
    api = Api()
    parents = [checked(HERE.parent / 'artifacts' / name) for name in ['first-consumer-v1', 'normalization-innovation-v1', 'source-native-reader-v1']]
    for local, collection, job, previous in parents:
        live = api.json('GET', '/api/jobs/' + job['job_id'])
        if live['state'] != 'succeeded' or live['payload_custody'] != collection['custody']:
            raise ValueError('Live parent custody differs')
    native_root, _, native_job, _ = parents[0]
    normalization_root, _, normalization_job, _ = parents[1]
    reader_root, _, reader_job, _ = parents[2]
    p = native_root / 'result'
    s0, old, execution = [read(p / n) for n in ['inputs.json', 'primary.json', 'execution.json']]
    reader, corpus, read_execution = [read(reader_root / 'result' / n) for n in ['inputs.json', 'corpus.json', 'images.json']]
    prefix = next(x for x in corpus['records'] if x['id'] == 's60672.base.canonical')
    if prefix['z'] != s0['receivers']['R0']['z'] or reader['clockTable']['rows'][3]['fields']['double_img'] != s0['modulation']:
        raise ValueError('Natural A source/clock differs')
    for local, _, _, _ in parents[:2]:
        if read(local / 'retained-array-verification.json')['primarySealSHA256'] != sha(local / 'result/primary-seal.json'):
            raise ValueError('Parent byte release differs')
    root = Path(root).resolve()
    root.mkdir(exist_ok=False, parents=True)
    payload = root / 'payload'
    payload.mkdir()
    for name in OWNED:
        for _, _, _, previous in parents:
            if name in previous['ownedSources'] and sha(HERE / name) != previous['sourceHashes'][name]:
                raise ValueError('Qualified helper changed: ' + name)
        shutil.copyfile(HERE / name, payload / name)
    for name in WHEELS:
        shutil.copyfile(normalization_root / 'payload' / name, payload / name)
    shutil.copyfile(HERE.parent / 'docs/plans/addressed-value-innovation.md', payload / 'frozen-plan.md')
    selected = read_execution['loadedParameters'] + [x for x in execution['loadedParameters'] if x['name'] in [WO, IN, DOWN]]
    if len(selected) != 13 or sum(x['bytes'] for x in selected) != 331088640:
        raise ValueError('Selected thirteen-tensor geometry differs')
    s = {k: s0[k] for k in ['modelRoot', 'versions', 'nativeSources', 'imageAssets', 'numericalProfile', 'carrierGeometry']}
    s.update(z=prefix['z'], conditioning=corpus['conditioningPages']['base.conditioning'], txtIDs=corpus['conditioningPages']['base.txt_ids'], imgIDs=reader['ids'],
             modImage=s0['modulation'], modText=reader['clockTable']['rows'][3]['fields']['double_txt'], carrierA=old['shared']['A']['gatedCarrier'], fusedA=old['cells']['R0A']['fused'],
             onePlusScale=old['constants']['onePlusScale'], beta=read(normalization_root / 'result/primary.json')['shared']['beta'],
             expectedEmbeddingPayloadSHA256=old['embeddings']['R0']['payloadSHA256'], expectedH0APayloadSHA256=s0['donors']['A']['payloadSHA256'], selectedParameters=selected,
             protectedRoots=sorted(set([*s0['protectedRoots'], *corpus['sourceRoots'], *[j['result']['diagnostics']['cwd'] for _, _, j, _ in parents]])),
             deniedCanaries=[old['cells']['R0A'][n]['path'] for n in ['product', 'imageExit']] + [old['cells']['R0B']['fused']['path'], s0['donors']['A']['path']],
             parentRecords={j['job_id'] + '.' + name: {'path': j['result']['diagnostics']['cwd'] + '/response-result/' + name, 'sha256': sha(local / 'result' / name)}
                            for local, _, j, _ in parents for name in ['request.json', 'inputs.json', 'primary-seal.json'] if (local / 'result' / name).exists()})
    save(payload / 'inputs.json', s)
    baseline = {n: old['cells']['R0A'][n] for n in STAGES if n != 'gatedCarrier'}
    baseline['gatedCarrier'] = old['shared']['A']['gatedCarrier']
    save(payload / 'targets.json', {'A': baseline, 'nativeH0A': s0['donors']['A'], 'scope': 'Historical A comparison/inheritance only, admitted after primary seal and child exit. New native B does not exist before reference.'})
    hosts = api.json('GET', '/api/hosts')
    save(payload / 'resource-basis.json', {'intentFirstPreview': _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda'),
          'measuredParent': normalization_job['result'], 'workingSet': {'selectedBF16ParameterBytes': 331088640, 'FP64WinBytes': 452984832, 'FP64WoBytes': 75497472, 'FP64ScoresAndProbabilitiesBytes': 603979776,
          'FP64JBytes': 150994944, 'candidateMACs': 86856695808, 'normalReferenceMACs': 125627793408, 'cpuThreads': 4, 'separateCandidateReferenceProcesses': True},
          'exception': 'Full-model alias sizing is inapplicable to selected thirteen-tensor QKV/value-event path. Parent measured2719.9MBRSS/534MBdevice. Add prefix/FP64Wo/scores;6144RAM1536VRAM CPU4disk6GB expected240s cap900s. 7987.2MB RAMguard fits66512MB host less3326margin. Beast-local authenticated inputs; hardpin. Actual peaks replace this basis.', 'hostCapacitySnapshot': [{k: x.get(k) for k in ['name', 'ram_total_mb', 'vram_total_mb', 'admission', 'telemetry']} for x in host_rows(hosts) if x['name'] == 'beast']})
    contract = one_lease_scheduling_contract(logical_branches=1, logical_operations=36, logical_checkpoint_replays=0, conditioner_policy='Supplied natural A conditioning/latent/baseline and clock; fresh seven-affine QKV, no new B before seal.',
                                           extra={'modelLoadScope': 'Primary thirteen tensors; reference reloads three. Fixed QK all-key scores, two-address V event, own BF16 B local state.', 'targetIsolation': 'Primary child seals/exits before native reference child; no automatic retries.'})
    r = {'schema': 'tce-value-innovation-request-v1', 'ownedSources': OWNED, 'sourceHashes': {x.name: sha(x) for x in payload.iterdir()}, 'scheduling_contract': contract,
         'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda', 'dtype': 'bfloat16', 'batch_size': 1, 'seq_len': 1536, 'height': 512, 'width': 512,
         'needs': {'cuda': True}, 'timeoutSeconds': 900, 'event': {'receiver': 's60672.base.canonical', 'clock': 3, 'imageRows': [264, 792], 'keyIndices': [776, 1304], 'heads': 24, 'fullKeyCount': 1536, 'nearGridChebyshevRadius': 2},
         'arithmeticSubplans': {'acquisition': 'source-native-cuda-bf16-seven-affine-before-SDPA', 'scoresFactorsSecant': 'torch-cpu-float64', 'candidateProductDownResidual': 'native-cuda-bfloat16', 'reference': 'post-seal-native-cuda-SDPA-Wo-MLP'},
         'parentJobIDs': [j['job_id'] for _, _, j, _ in parents], 'protocol': 'One frozen Vswap; no fittedrank/extraevents. Candidate33affines+QK/2assemblies86.857BMAC seals/exits; postseal2SDPA/2Wo/1BMLP125.628BMAC, extraAMLPonlyifshamfailsinheritance. Savedcompletequeries, nofullTransformer/encoder/clocknet/fit/scheduler/VAE/image. Alloriginalmaps/baselines/historicalbeta paid.'}
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
        raise ValueError('Already submitted; inspect the same job')
    cmd = command()[:-5]
    for name in WHEELS:
        cmd += ['--with', './' + name]
    cmd += ['python', '-u', 'value_innovation_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = guarded_mrun_launch(cmd, request=r, experiment='tce-value-innovation-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs=r['needs'],
          mrun_config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width', 'scheduling_contract', 'arithmeticSubplans']},
          ram_mb=6144, vram_mb=1536, cpu_threads=4, disk_gb=6, est_wall_s=240, timeout_s=900, detach=True, retry_on_kill=False, preflight=True,
          note='FixedQK Vswap rows264/792, full1536keys. Fresh7affineQKV;24headblocks+24rowWin+2assemblies+down=86.857BMACcandidate;seal/childexit then2SDPA2Wo1BMLP125.628BMAC. ExtraAMLP onlyifshamnotexact. No fit/fullmodel/encoder/VAE/images. CPU4/CUDA BF16+CPUFP64,6144RAM1536VRAM,disk6GB,240s expected900cap,noretry. Parent2719.9RSS534device; newscore604MB+Wo75.5MB accounted.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}
    save(root / 'job.json', value)
    return value


def execute(root):
    root = Path(root).resolve()
    r = read(root / 'request.json')
    verify(root, r)
    start, phases = time.perf_counter(), []
    for name in ['primary', 'reference']:
        tick = time.perf_counter()
        subprocess.run([sys.executable, '-u', str(root / ('value_innovation_' + name + '.py')), str(root)], check=True, cwd=root)
        phases.append({'phase': name, 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
        print('VALUE_INNOVATION_' + name.upper() + '_EXITED', flush=True)
    from value_innovation_analysis import execute as analyze
    tick = time.perf_counter()
    result = analyze(root)
    phases.append({'phase': 'saved-scoring', 'wallSeconds': time.perf_counter() - tick})
    out = root / 'response-result'
    for name in ['request.json', 'inputs.json', 'targets.json', 'resource-basis.json', 'frozen-plan.md']:
        shutil.copyfile(root / name, out / name)
    pe, re = read(out / 'execution.json'), read(out / 'reference-execution.json')
    save(out / 'report.json', {'schema': 'tce-value-innovation-result-v1', 'requestSHA256': r['requestSHA256'], **result, 'event': r['event'], 'phases': phases,
         'workerWallSeconds': time.perf_counter() - start, 'candidateMACs': pe['nominalCandidateMACs'], 'referenceMACs': re['nominalReferenceMACs'],
         'totalNewMACs': pe['nominalCandidateMACs'] + re['nominalReferenceMACs'], 'historicalBetaMACs': 56623104,
         'work': {'candidateAffineCalls': len(pe['linears']), 'candidateOtherContractions': len(pe['contractions']), 'referenceAffineCalls': len(re['linears']), 'referenceSDPACalls': len(re['attention']),
         'selectedDistinctBF16ParameterBytes': 331088640, 'processLevelSelectedParameterBytes': 331088640 + 188743680, 'newFullTransformerCalls': 0, 'newEncoders': 0, 'newClockNetworks': 0, 'newFits': 0, 'newSchedulerCommits': 0, 'newVAECalls': 0, 'newImages': 0},
         'mechanics_status': 'complete', 'instrument_status': 'executed', 'terminal_status': 'unassessed',
         'limits': ['One synthetic value-address intervention on natural R0A with full Q/K fixed, not a natural raw prompt edit.',
         'Supplied native A baseline/carrier, old conditioning/latent/clock/beta and original learned maps remain paid; fresh QKV and all-key scores are additional computation.',
         'Real finite factorization changes native numerical boundaries. Complete endpoint/response/row discrepancies are retained without effect-size gates.',
         'No changed text branch, later own-state diffusion, full velocity, VAE/image or arbitrary-prompt zero-neural generator.',
         'Instrumented costs include selected-weight reloads, source hashing, transfers/captures and saved diagnostics. No matched serving-speed claim.']})
    print('VALUE_INNOVATION_DONE', flush=True)


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
