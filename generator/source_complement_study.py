"""One selected-operand producer and two isolated complementary-context arms."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read, save, sha, identity
from flux_clock_study import verify

HERE = Path(__file__).resolve().parent
OWNED = ['source_complement.py', 'source_complement_worker.py', 'source_complement_study.py', 'source_complement_analysis.py',
         'selected_source_v2.py', 'sparse_text_kv.py', 'projected_complement.py', 'encoder_atoms.py', 'paired_fields.py',
         'flux_clock.py', 'flux_clock_study.py', 'complete_masked_study.py', 'masked_kv_study.py']
WHEELS = ['mdb_runtime-0.1.0-py3-none-any.whl', 'saturn_pub-0.4.1-py3-none-any.whl', 'mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from projected_complement import fields, ACTOR_PARAMETERS
    from source_complement import SHARED_PARAMETERS, RETURN_PARAMETERS
    parent, collection, job, previous = checked(HERE.parent / 'artifacts/selected-source-v2')
    api = Api()
    live = api.json('GET', '/api/jobs/' + job['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != collection['custody']:
        raise ValueError('Own source/reader scheduler custody differs')
    p = parent / 'result'
    expected = {'report.json': 'dbcab2d0e9439de4b1c63540c30bdc4e4f29b326bf6bb791367d9e50bbaf818b',
                'producer-seal.json': 'e8d40460096a08aaf4127a6734e6f6b077d17e1cfe7d3f55f96a8632779e8e61',
                'primary-seal.json': 'fa1a8d03467b99181dd0e21588b829c489727e8b5dfd71654c68dd60f38d47b4'}
    if any(sha(p / n) != h for n, h in expected.items()):
        raise ValueError('Planned historical source differs')
    old, prod, compiler, primary, targets = [read(p / n) for n in ['inputs.json', 'producer.json', 'compiler.json', 'primary.json', 'targets.json']]
    for name in ['producer', 'compiler', 'primary']:
        if read(p / (name + '-seal.json'))[name + 'SHA256'] != sha(p / (name + '.json')):
            raise ValueError('Complete historical source seal differs')
    if prod['commonGatePayloadsExact'] is not True or previous['support'] != [4, 11, 12, 13]:
        raise ValueError('Frozen gate/address premise differs')
    byte_receipt = read(parent / 'retained-array-verification.json')
    remote = job['result']['diagnostics']['cwd'] + '/response-result/'
    old_records = {n: {'path': remote + n, 'sha256': sha(p / n)} for n in
                   ['producer.json', 'producer-seal.json', 'compiler.json', 'compiler-seal.json', 'primary.json', 'primary-seal.json']}
    pcut = copy.deepcopy(old['sourceCuts']['P'])
    pcut['pages'] = {n: item for n, item in pcut['pages'].items() if n in ['text', 'context']}
    declared = {x['key']['value']: x['value'] for x in pcut['cut']['manifest']['payload']['items']}
    if identity(pcut['cut']['manifest']) != pcut['cutID']:
        raise ValueError('Original P cut manifest differs')
    for name, page in pcut['pages'].items():
        if pcut['cut']['slots'][name] != page['pageID'] or page['metadata']['value'] != declared[name] or page['declaredDescriptor'] != declared[name]:
            raise ValueError('P cut/page membership differs')
    banks = {role: {n: prod['sources'][role]['fields'][n] for n in ['K', 'V']} for role in ['P', 'Q']}
    q_attention = prod['sources']['Q']['fields']['A']
    reader_inputs = {n: x for n, x in compiler['actorInputs'].items() if not n.endswith('.rows')}
    admitted = [q_attention, *fields(banks), *reader_inputs.values()]
    if any(byte_receipt['files'][x['path']]['sha256'] != x['sha256'] for x in admitted):
        raise ValueError('Retained input lacks complete parent byte custody')
    target = {'nativeReader': targets['family'], 'parentReader': {arm: primary['actors'][arm]['fields']['R_attention'] for arm in ['QQQ', 'PQP']},
              'parentSource': {n: prod['sources']['P']['fields'][n] for n in ['R', 'A', 'U', 'X', 'M', 'T']},
              'nativeSource': targets['sourcePorts']['P'], 'parentRows': prod['splice']['fields']['PQP'],
              'nativeRows': targets['sourceTextTargets']['PQP'], 'imageIDs': targets['imageIDs'],
              'scope': 'All historical comparisons remain unavailable until both new arms seal and exit. Retained cycle54 QQQ is a paid post-seal comparison baseline.'}
    artifacts = read(p / 'parameter-artifacts.json')
    required = set(SHARED_PARAMETERS + RETURN_PARAMETERS + ACTOR_PARAMETERS)
    artifacts['artifacts'] = [x for x in artifacts['artifacts'] if x['source']['tensor'] in required]
    if {x['source']['tensor'] for x in artifacts['artifacts']} != required:
        raise ValueError('Original parameter recipes incomplete')
    source = {k: old[k] for k in ['modelRoot', 'versions', 'nativeSources', 'numericalProfile', 'imageAssets', 'familyPoint']}
    source.update(pCut=pcut, banks=banks, qAttention=q_attention, readerInputs=reader_inputs,
                  qGatePayloads={n: prod['sources']['Q']['fields'][n]['payloadSHA256'] for n in ['gateA', 'gateM']},
                  parentRecords=old_records, parentJobID=job['job_id'], parentCustody=live['payload_custody'],
                  sourceJobID=old['sourceJobID'], sourceCustody=old['sourceCustody'],
                  originalParameterArtifacts=artifacts['artifacts'], deniedTargets=sorted({x['path'] for x in fields(target)}))
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
    shutil.copyfile(HERE.parent / 'docs/plans/source-complement-transfer.md', payload / 'frozen-plan.md')
    save(payload / 'inputs.json', source)
    save(payload / 'targets.json', target)
    save(payload / 'parameter-artifacts.json', artifacts)
    preview = _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda')
    save(payload / 'resource-basis.json', {'intentFirstPreview': preview, 'measuredParent': live['result'],
         'geometry': {'sharedBF16ParameterBytes': 56623616, 'perArmBF16ParameterBytes': 245367040,
                      'reusedProjectedCacheBytes': 604372992, 'perArmComplementKVBytes': 18874368},
         'exception': 'Full-model alias sizing is inapplicable. Three sequential child processes; selected source57MB, each arm245MBweights/19MBbank/604MBCPUcache/75MBFP64Wo. Prior source-reader3195.1MBRSS756MBdevice; reuse6144RAM1536VRAMCPU4disk3GB,180sexpected900cap. No full model or fresh bank/compiler.'})
    contract = one_lease_scheduling_contract(logical_branches=2, logical_operations=63, logical_checkpoint_replays=0,
         conditioner_policy='Supplied own-qualified P/Q complement banks, Q direct return and receiving cache; raw-prompt prefixes and QQQ baseline remain paid.',
         extra={'modelLoadScope': 'Shared5selectedtensors; each isolated arm7selectedtensors. No full model or bank formation.',
                'targetIsolation': 'Shared and primary cannot open P complement. Control admitted only after primary seal/process exit; cannot read primary arrays. All archived comparisons after both seals/exits.'})
    r = {'schema': 'tce-source-complement-request-v1', 'ownedSources': OWNED, 'sourceHashes': {x.name: sha(x) for x in payload.iterdir()},
         'scheduling_contract': contract, 'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda', 'dtype': 'bfloat16',
         'batch_size': 1, 'seq_len': 1536, 'height': 512, 'width': 512, 'needs': {'cuda': True}, 'timeoutSeconds': 900,
         'case': previous['case'], 'support': previous['support'], 'familyPoint': previous['familyPoint'], 'producerPoint': previous['producerPoint'],
         'arithmeticSubplans': {'shared': 'CUDA-BF16-P-selected-QKV', 'arms': 'CUDA-BF16-rectangular-source-SDPA-MLP;unchanged-CPUFP64-projected-recipient'},
         'protocol': 'Shared3selectedQKV113.246MMAC;primaryQcomplement thenpostsealPcontrol,identicalselectedoperands. Eachsource3affines/SDPA plus26recipientaffines. Total1,799,356,416matrixMAC plus scalarwork. No bankformation,fullreference,suffix,image,fit,precisionarm.'}
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
        raise ValueError('Already submitted; inspect same handle')
    cmd = command()[:-5]
    for name in WHEELS:
        cmd += ['--with', './' + name]
    cmd += ['python', '-u', 'source_complement_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = guarded_mrun_launch(cmd, request=r, experiment='tce-source-complement-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs=r['needs'],
        mrun_config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width', 'scheduling_contract', 'arithmeticSubplans']},
        ram_mb=6144, vram_mb=1536, cpu_threads=4, disk_gb=3, est_wall_s=180, timeout_s=900, detach=True, retry_on_kill=False, preflight=True,
        note='Source information cut spherecube2027 step2:shared selectedP QKV113.246MMAC;Qcomplementprimary thenPcomplementcontrol,eachownsourceMLP+projectedreader843.055MMAC. Total1.799BmatrixMAC plus scalarcachework. PrimaryPbankdenied;controlpostsealexit;alltargetsdenied. SuppliedKVbanks,Qreturn,QQQbaseline,604MBrecipientcachepaid. No freshbank/fullreference/image/fit. ThreeprocessesCPU4,6144RAM1536VRAMdisk3GB,180sexpected900cap,noretry. Parent3195RSS756device.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}
    save(root / 'job.json', value)
    return value


def execute(root):
    root = Path(root).resolve()
    r, s = read(root / 'request.json'), read(root / 'inputs.json')
    verify(root, r)
    for record in s['parentRecords'].values():
        if sha(record['path']) != record['sha256']:
            raise ValueError('Original source/cache metadata changed')
    started, phases = time.perf_counter(), []
    for phase in ['shared', 'primary', 'control']:
        tick = time.perf_counter()
        subprocess.run([sys.executable, '-u', str(root / 'source_complement_worker.py'), phase, str(root)], check=True, cwd=root)
        phases.append({'phase': phase, 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
        print('SOURCE_COMPLEMENT_' + phase.upper() + '_EXITED', flush=True)
    from source_complement_analysis import execute as score
    report = score(root)
    out = root / 'response-result'
    for name in ['request.json', 'inputs.json', 'targets.json', 'parameter-artifacts.json', 'retrieve_weights.py', 'resource-basis.json', 'frozen-plan.md']:
        shutil.copyfile(root / name, out / name)
    save(out / 'report.json', {'schema': 'tce-source-complement-result-v1', 'requestSHA256': r['requestSHA256'], **report,
         'phases': phases, 'workerWallSeconds': time.perf_counter() - started, 'mechanics_status': 'complete',
         'instrument_status': 'executed', 'terminal_status': 'unassessed', 'limits': [
             'One retrospective fixed-address context substitution. Retained full banks and Q direct return remain supplied and paid.',
             'Historical prefixes, original learned maps, fixed receiving query/cache and post-seal QQQ comparison baseline remain paid.',
             'Matched control shares fresh selected operands and measures finite drift; no general context sufficiency claim.',
             'No new full reference, raw-prompt completion, own complete diffusion, VAE or image; arbitrary-prompt zero-neural generation remains open.']})
    print('SOURCE_COMPLEMENT_DONE', flush=True)


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
