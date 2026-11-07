"""One finite lease for a query-side score compiler and its own-state continuation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read, save, sha, identity
from flux_clock_study import verify
from propagated_context_study import OWNED as PREVIOUS_OWNED, WHEELS

HERE = Path(__file__).resolve().parent
OWNED = ['image_dual.py', 'image_dual_intake.py', 'image_dual_worker.py', 'image_dual_analysis.py', 'image_dual_study.py'] + PREVIOUS_OWNED


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from projected_complement import fields
    from propagated_context import RETURN_PARAMETERS, NEXT_PARAMETERS
    from image_dual_worker import KEY_PARAMETERS
    from image_dual_analysis import NAMES
    from image_dual_intake import prepare, inventory_positions
    parent, collection, job, previous = checked(HERE.parent / 'artifacts/stale-key-context-v1')
    older, old_collection, old_job, old_request = checked(HERE.parent / 'artifacts/propagated-context-v1')
    api = Api()
    for c, j in [(collection, job), (old_collection, old_job)]:
        live = api.json('GET', '/api/jobs/' + j['job_id'])
        if live['state'] != 'succeeded' or live['payload_custody'] != c['custody']:
            raise ValueError('Parent scheduler/custody changed')
    p = parent / 'result';old = older / 'result';s = read(p / 'inputs.json')
    if sha(p / 'report.json') != 'b0bc1e8a4185baebea34e089237c5bad2ce04e824233e9aa5db61c3705b635bb':
        raise ValueError('Frozen latest result differs')
    stale = read(p / 'primary.json')
    if read(p / 'primary-seal.json')['primarySHA256'] != sha(p / 'primary.json'):
        raise ValueError('Stale-key baseline seal differs')
    targets = read(p / 'targets.json')['diagonals']
    targets['stale'] = {n: {**stale['source']['fields'], **stale['nextQuery']['fields']}[n] for n in NAMES}
    byte_receipt = read(parent / 'retained-array-verification.json')
    for x in fields({'selected': s['selected'], 'banks': s['banks'], 'targets': targets}):
        if byte_receipt['files'][x['path']]['sha256'] != x['sha256']:
            raise ValueError('Full historical operand custody absent')
    source = read(old / 'inputs.json')
    positions = inventory_positions(source['sourceCuts']['P']['pages']['context'])
    if [x['imageDistinctByteGroups'] for x in positions['axes']] != [1, 32, 32, 1] or [x['textDistinctByteGroups'] for x in positions['axes']] != [1, 1, 1, 512] or not positions['axes'][1]['imageHeightMembershipExact'] or not positions['axes'][2]['imageWidthMembershipExact']:
        raise ValueError('Actual stored THWL byte groups differ from pre-stage correction')
    # No image-key tensor is decoded by this metadata/byte-only preparation.
    text_slice = prepare(s['banks']['Q']['K'], '/mnt/big/mdb/research/tce-image-dual-intake-v1/textK_Q.npy')
    artifacts = read(old / 'parameter-artifacts.json');required = set(KEY_PARAMETERS + RETURN_PARAMETERS + NEXT_PARAMETERS)
    artifacts['artifacts'] = [x for x in artifacts['artifacts'] if x['source']['tensor'] in required]
    if {x['source']['tensor'] for x in artifacts['artifacts']} != required:
        raise ValueError('Original map recipes incomplete')
    remote = job['result']['diagnostics']['cwd'] + '/response-result/'
    s.update(sourceCuts=source['sourceCuts'], sourceMetadataFiles=source['sourceMetadataFiles'],
             sourceJobID=source['sourceJobID'], sourceCustody=source['sourceCustody'],
             originalParameterArtifacts=artifacts['artifacts'], textKeySlice=text_slice, positionByteInventory=positions,
             latestParentJobID=job['job_id'], latestParentCustody=collection['custody'],
             latestParentRecords={n: {'path': remote + n, 'sha256': sha(p / n)} for n in ['report.json', 'primary.json', 'primary-seal.json', 'request.json', 'inputs.json', 'targets.json']},
             deniedTargets=sorted({x['path'] for x in fields(targets)}))
    root = Path(root).resolve();root.mkdir(parents=True, exist_ok=False);payload = root / 'payload';payload.mkdir()
    for n in OWNED:
        if n in old_request['ownedSources'] and sha(HERE / n) != old_request['sourceHashes'][n]:
            raise ValueError('Frozen runtime helper changed')
        shutil.copyfile(HERE / n, payload / n)
    for n in WHEELS:
        shutil.copyfile(older / 'payload' / n, payload / n)
    shutil.copyfile(old / 'retrieve_weights.py', payload / 'retrieve_weights.py')
    for name in ['image-dual-score-producer', 'image-dual-execution-addendum']:
        shutil.copyfile(HERE.parent / ('docs/plans/' + name + '.md'), payload / (name + '.md'))
    save(payload / 'inputs.json', s);save(payload / 'targets.json', targets);save(payload / 'parameter-artifacts.json', artifacts)
    preview = _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda')
    save(payload / 'resource-basis.json', {'intentFirstPreview': preview, 'measuredParent': job['result'],
        'geometry': {'dualFP64PayloadBytes': 153354240, 'distinctBF16ParameterBytes': 226492928,
                     'processBF16ParameterBytes': 245367552, 'fullQImageKeyFormationInActor': 0},
        'exception': 'Selected-tensor three sequential children, not full-model load. Prior receiving actor1776.5MBRSS586MBdevice; compiler dual table153MB plus FP64 maps/inputs, reference<200MB extra arrays. RAM4096VRAM1024CPU4disk3GB,180sexpected600cap. No overlapping model/fullsuffix/image.'})
    contract = one_lease_scheduling_contract(logical_branches=1, logical_operations=4, logical_checkpoint_replays=0,
        conditioner_policy='Paid authenticated P/Q source cuts, selectedP, native text-only Q keys/full Q values, original matrices and archived diagonals.',
        extra={'modelLoadScope': 'Compiler2originaltensors; actor5consumer tensors; reference2key tensors.',
               'targetIsolation': 'Compiler/actor deny both full key arrays and archived outputs. Actor has no image-key matrix. Reference admitted after actor seal/process exit; no feedback.'})
    request = {'schema': 'tce-image-dual-request-v1', 'ownedSources': OWNED,
        'sourceHashes': {x.name: sha(x) for x in payload.iterdir()}, 'scheduling_contract': contract,
        'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda', 'dtype': 'bfloat16',
        'batch_size': 1, 'seq_len': 1536, 'height': 512, 'width': 512, 'needs': {'cuda': True}, 'timeoutSeconds': 600,
        'case': previous['case'], 'support': previous['support'], 'logicalStep': 2, 'sourceJoint': 1, 'nextJoint': 2,
        'arithmeticSubplans': {'compiler': 'nativePModLN→FP64PrawRMS+65querysidecoefficientgroups',
            'actor': 'nativeQModLN→groupedFP64logits/stalePnorm→ownnativeMLP→ownnextQ',
            'reference': 'postsealactualactorQinput→oneFP64Qprojection/twoQK/twoAVdiagnostics'},
        'plannedMatrixMACs': {'compiler': 10286530560, 'actor': 1346371584, 'reference': 9726590976, 'total': 21359493120},
        'protocol': 'One fixed image-dual candidate; compilersealexit→actorsealexit→reference. No image-key matrix/fullkey numeric input to actor; exact storedRoPE transpose then gamma; smoothP denominator explicit. FP64head→FP32→BF16 admitted to original consumer. No extraMLP/refit/precisionsearch/suffix/image.'}
    request['requestSHA256'] = identity(request);save(root / 'request.json', request);save(payload / 'request.json', request)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [x.name for x in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': request['requestSHA256']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root = Path(root).resolve();r = read(root / 'request.json');verify(root / 'payload', r)
    if (root / 'job.json').exists():
        raise ValueError('Already submitted; inspect same handle')
    cmd = command()[:-5]
    for n in WHEELS:
        cmd += ['--with', './' + n]
    cmd += ['python', '-u', 'image_dual_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = guarded_mrun_launch(cmd, request=r, experiment='tce-image-dual-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs=r['needs'],
        mrun_config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width', 'scheduling_contract', 'arithmeticSubplans']},
        ram_mb=4096, vram_mb=1024, cpu_threads=4, disk_gb=3, est_wall_s=180, timeout_s=600, detach=True, retry_on_kill=False, preflight=True,
        note='Image dual spherecube2027 step2joint1→joint2: compiler PrawRMS+65dual groups10.287BMAC, actor groupedimage/textQK/fullAV+4ownconsumer affines1.346B; postsealQprojection2QK2AV9.727B,total21.359B. CPU4FP64 contractions/nativeCUDABF16ModLN/consumer. Actornoimagekeymatrix/fullkeytargets. Three sequential children, RAM4096VRAM1024disk3GB,180sexpected600cap,noretry. Paid sourcecuts/values/textkeys/maps/diagonals; no fit/extraconsumer/suffix/image.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']};save(root / 'job.json', value);return value


def execute(root):
    root = Path(root).resolve();r = read(root / 'request.json');verify(root, r)
    started = time.perf_counter();phases = []
    for phase in ['compiler', 'actor', 'reference']:
        tick = time.perf_counter();subprocess.run([sys.executable, '-u', str(root / 'image_dual_worker.py'), phase, str(root)], cwd=root, check=True)
        phases.append({'phase': phase, 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
        print('IMAGE_DUAL_' + phase.upper() + '_EXITED', flush=True)
    from image_dual_analysis import execute as analyze
    report = analyze(root);out = root / 'response-result'
    for n in ['request.json', 'inputs.json', 'targets.json', 'parameter-artifacts.json', 'retrieve_weights.py', 'resource-basis.json', 'image-dual-score-producer.md', 'image-dual-execution-addendum.md']:
        shutil.copyfile(root / n, out / n)
    save(out / 'report.json', {'schema': 'tce-image-dual-result-v1', 'requestSHA256': r['requestSHA256'], **report,
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
