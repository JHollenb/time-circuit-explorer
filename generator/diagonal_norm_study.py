"""One finite lease for source-only norm compilation and own-state continuation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read, save, sha, identity
from flux_clock_study import verify
from image_dual_study import OWNED as PREVIOUS_OWNED, WHEELS

HERE = Path(__file__).resolve().parent
OWNED = ['diagonal_key_norm.py', 'diagonal_norm_worker.py', 'diagonal_norm_analysis.py', 'diagonal_norm_study.py'] + PREVIOUS_OWNED


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from projected_complement import fields
    from propagated_context import RETURN_PARAMETERS, NEXT_PARAMETERS
    from diagonal_norm_worker import WEIGHT
    from image_dual_analysis import NAMES
    parent, collection, job, previous = checked(HERE.parent / 'artifacts/image-dual-v1')
    api = Api();live = api.json('GET', '/api/jobs/' + job['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != collection['custody']:
        raise ValueError('Parent scheduler/custody changed')
    p = parent / 'result'
    if sha(p / 'report.json') != '84dc610d01f1fa98adfbf209fe248687a5abebbeb1eb2ccaa86db742f43b78a8':
        raise ValueError('Frozen parent result differs')
    compiler, actor, reference, s = [read(p / n) for n in ['compiler.json', 'actor.json', 'reference.json', 'inputs.json']]
    for n in ['compiler', 'actor', 'reference']:
        if read(p / (n + '-seal.json'))[n + 'SHA256'] != sha(p / (n + '.json')):
            raise ValueError('Parent phase seal differs')
    if reference['reference']['sourceFrame']['slots']['x']['fingerprint'] != actor['modln']['fields']['x']['payloadSHA256']:
        raise ValueError('Saved Q norm is not bound to the retained actor input')
    trajectories = read(p / 'targets.json')
    trajectories['retained'] = {n: {**actor['source']['fields'], **actor['next_query']['fields']}[n] for n in NAMES}
    targets = {'trajectories': trajectories, 'norms': {'P': compiler['compile']['fields']['pDenominator'], 'Q': reference['reference']['fields']['qDenominator']},
        'smoothImageLogits': reference['reference']['fields']['smoothImageLogits'], 'smoothLogPartition': reference['reference']['fields']['smoothLogPartition']}
    reuse = {n: compiler['compile']['fields'][n] for n in ['duals', 'groupIDs']}
    receipt = read(parent / 'retained-array-verification.json')
    for x in fields({'selected': s['selected'], 'reuse': reuse, 'textKeySlice': s['textKeySlice'], 'bankV': s['banks']['Q']['V'], 'targets': targets}):
        if receipt['files'][x['path']]['sha256'] != x['sha256']:
            raise ValueError('Full parent operand custody absent')
    denied = {x['path'] for x in fields(targets)} | {x['path'] for x in fields(actor)}
    denied |= {compiler['compile']['fields'][n]['path'] for n in ['rawP', 'pDenominator']}
    denied |= {reference['reference']['fields'][n]['path'] for n in ['rawQ', 'qDenominator']}
    denied |= {s['banks'][n]['K']['path'] for n in ['P', 'Q']} | {s['banks']['P']['V']['path']}
    forbidden = set()
    for cut in s['sourceCuts'].values():
        forbidden.add(cut['cutFile']['path']);forbidden.update(p for page in cut['pages'].values() for p in page['files'])
    forbidden |= {x['path'] for x in s['selected'].values()} | {s['textKeySlice']['path'], s['banks']['Q']['V']['path']}
    forbidden |= {x['path'] for x in reuse.values()}
    remote = job['result']['diagnostics']['cwd'] + '/response-result/'
    s.update(reuse=reuse, controlModLN=actor['modln']['fields']['x'],
        matchedControlFields={n: actor['read']['fields'][n] for n in ['imageNumerator', 'textLogits', 'textK_hybrid', 'V_hybrid']},
        dualParentJobID=job['job_id'], dualParentCustody=collection['custody'],
        dualParentRecords={n: {'path': remote + n, 'sha256': sha(p / n)} for n in ['compiler.json', 'compiler-seal.json', 'actor.json', 'actor-seal.json', 'reference.json', 'reference-seal.json', 'report.json', 'request.json', 'inputs.json', 'targets.json']},
        deniedNumericalInputs=sorted(denied), compilerForbiddenSources=sorted(forbidden))
    artifacts = read(p / 'parameter-artifacts.json');required = set((WEIGHT,) + RETURN_PARAMETERS + NEXT_PARAMETERS)
    artifacts['artifacts'] = [x for x in artifacts['artifacts'] if x['source']['tensor'] in required]
    if {x['source']['tensor'] for x in artifacts['artifacts']} != required:
        raise ValueError('Required original map recipes incomplete')
    s['originalParameterArtifacts'] = artifacts['artifacts']
    root = Path(root).resolve();root.mkdir(parents=True, exist_ok=False);payload = root / 'payload';payload.mkdir()
    for n in OWNED:
        if n in previous['ownedSources'] and sha(HERE / n) != previous['sourceHashes'][n]:
            raise ValueError('Frozen helper changed')
        shutil.copyfile(HERE / n, payload / n)
    for n in WHEELS:
        shutil.copyfile(parent / 'payload' / n, payload / n)
    shutil.copyfile(p / 'retrieve_weights.py', payload / 'retrieve_weights.py')
    shutil.copyfile(HERE.parent / 'docs/plans/source-defined-diagonal-key-norm.md', payload / 'frozen-plan.md')
    save(payload / 'inputs.json', s);save(payload / 'targets.json', targets);save(payload / 'parameter-artifacts.json', artifacts)
    preview = _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda')
    save(payload / 'resource-basis.json', {'intentFirstPreview': preview, 'measuredParent': live['result'],
        'geometry': {'newCoefficientFP64Bytes': 589824, 'reusedDualFP64Bytes': 153354240,
                     'distinctBF16ParameterBytes': 226492672, 'newFullKeyProjections': 0},
        'exception': 'Two sequential selected-tensor children, not full model. Parent1712.7MBRSS568MBdevice includes same153MBdual actor. Weight-only compiler18.9MBBF16+75.5MBFP64 temporary; no source input. RAM4096VRAM1024CPU4disk3GB,120sexpected600cap. No fullreference/fit/suffix/image.'})
    contract = one_lease_scheduling_contract(logical_branches=1, logical_operations=5, logical_checkpoint_replays=0,
        conditioner_policy='Paid own native Q source cut, sealed query-specific duals, QtextK/fullV, selectedP and original receiving maps.',
        extra={'modelLoadScope': 'Compiler1originalW; actor5consumer tensors only.',
               'targetIsolation': 'Compiler source-free, actor denies P/Qnorms/rawkeys/archived outputs. Same own norm→H→MLP→T→nextQ. Post-seal saved-only diagnosis.'})
    r = {'schema': 'tce-diagonal-norm-request-v1', 'ownedSources': OWNED, 'sourceHashes': {x.name: sha(x) for x in payload.iterdir()},
        'scheduling_contract': contract, 'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda', 'dtype': 'bfloat16',
        'batch_size': 1, 'seq_len': 1536, 'height': 512, 'width': 512, 'needs': {'cuda': True}, 'timeoutSeconds': 600,
        'case': previous['case'], 'support': previous['support'], 'logicalStep': 2, 'sourceJoint': 1, 'nextJoint': 2,
        'arithmeticSubplans': {'compiler': 'W-onlyFP64channel-diagonal;squares/reductions/no matrixMAC',
            'actor': 'ownBF16ModLN→FP64positivequadraticnorm→samesealeddualread→ownfiniteMLP/nextQ'},
        'plannedMatrixMACs': {'compiler': 0, 'actor': 1421869056, 'total': 1421869056},
        'protocol': 'One fixed positive diagonal law from originalW only; compilersealexit→freshactorsealexit→saved-onlynorm/crossterm/response diagnosis. No suppliedP/Qnorm or imageW in actor, no newfullkey reference,fit,sweep,extraconsumer,suffix,image.'}
    r['requestSHA256'] = identity(r);save(root / 'request.json', r);save(payload / 'request.json', r)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [x.name for x in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': r['requestSHA256']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root = Path(root).resolve();r = read(root / 'request.json');verify(root / 'payload', r)
    if (root / 'job.json').exists():
        raise ValueError('Already submitted; inspect same handle')
    cmd = command()[:-5]
    for n in WHEELS:
        cmd += ['--with', './' + n]
    cmd += ['python', '-u', 'diagonal_norm_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = guarded_mrun_launch(cmd, request=r, experiment='tce-diagonal-norm-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs=r['needs'],
        mrun_config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width', 'scheduling_contract', 'arithmeticSubplans']},
        ram_mb=4096, vram_mb=1024, cpu_threads=4, disk_gb=3, est_wall_s=120, timeout_s=600, detach=True, retry_on_kill=False, preflight=True,
        note='Diagonal key norm spherecube2027step2joint1→2. W-onlycompiler9.437Msquares/reductions,0matrixMAC; actor ownQBF16ModLN→24x3072positivequadratic75.497MMAC→sealeddualimage/textQK/AV+4nativeconsumer affines,total1.422BMAC. No P/Qnorm/rawkey/fulltarget beforeseal, no newfullkey/reference/fit/suffix/image. TwochildrenCPU4RAM4096VRAM1024disk3GB,120sexpected600cap,noretry;parent1713RSS568device. Native sources/dualcache/maps/diagonals paid.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']};save(root / 'job.json', value);return value


def execute(root):
    root = Path(root).resolve();r = read(root / 'request.json');verify(root, r);started = time.perf_counter();phases = []
    for phase in ['compiler', 'actor']:
        tick = time.perf_counter();subprocess.run([sys.executable, '-u', str(root / 'diagonal_norm_worker.py'), phase, str(root)], check=True, cwd=root)
        phases.append({'phase': phase, 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
        print('DIAGONAL_NORM_' + phase.upper() + '_EXITED', flush=True)
    from diagonal_norm_analysis import execute as analyze
    report = analyze(root);out = root / 'response-result'
    for n in ['request.json', 'inputs.json', 'targets.json', 'parameter-artifacts.json', 'retrieve_weights.py', 'resource-basis.json', 'frozen-plan.md']:
        shutil.copyfile(root / n, out / n)
    save(out / 'report.json', {'schema': 'tce-diagonal-norm-result-v1', 'requestSHA256': r['requestSHA256'], **report,
        'phases': phases, 'workerWallSeconds': time.perf_counter() - started,
        'mechanics_status': 'complete', 'instrument_status': 'executed', 'terminal_status': 'unassessed'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']);parser.add_argument('run');args = parser.parse_args()
    if args.action == 'collect':
        from response_study import collect
        value = collect(args.run)
    else:
        value = globals()[args.action](args.run)
    if value is not None:
        print(json.dumps(value))
