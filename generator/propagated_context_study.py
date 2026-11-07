"""One finite CUDA lease for propagated-context reads and their own next query."""
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
OWNED = ['propagated_context.py', 'propagated_context_worker.py', 'propagated_context_analysis.py', 'propagated_context_study.py',
         'propagated_context_intake.py', 'selected_source_v2.py', 'projected_complement.py', 'paired_fields.py', 'encoder_atoms.py',
         'flux_clock.py', 'flux_clock_study.py', 'complete_masked_study.py', 'masked_kv_study.py', 'sparse_text_kv.py']
WHEELS = ['mdb_runtime-0.1.0-py3-none-any.whl', 'saturn_pub-0.4.1-py3-none-any.whl', 'mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from propagated_context import BANK_PARAMETERS, SELECTED_PARAMETERS, RETURN_PARAMETERS, NEXT_PARAMETERS
    from projected_complement import fields
    api = Api()
    intake = HERE.parent / 'artifacts/propagated-context-intake-v1'
    meta, inv = read(intake / 'source-metadata.json'), read(intake / 'byte-inventory.json')
    if sha(intake / 'source-metadata.json') != inv['sourceMetadataSHA256'] or sha(HERE / 'propagated_context_intake.py') != inv['verifierSourceSHA256']:
        raise ValueError('Authenticated support inventory changed')
    if not inv['contextContrastApplicable'] or not inv['contextDescriptorsExact']:
        raise ValueError('Declared changed-context instrument inapplicable')
    source = api.json('GET', '/api/jobs/' + inv['sourceJobID'])
    if source['state'] != 'succeeded' or source['payload_custody'] != inv['sourceCustody']:
        raise ValueError('Native source scheduler custody differs')
    wanted = {'P': '8551691e8094d927d8521d0b5e6a0416520b2d90abea800d65d14c8d986062e3',
              'Q': 'b7f0fe0b6ef01e52bb5b14dafcf9202188c3b354e84f53d9979b1706c242e500'}
    if {n: x['cutID'] for n, x in meta['sources'].items()} != wanted:
        raise ValueError('Frozen source boundary differs')
    for role, record in meta['sources'].items():
        cut = record['cut']
        if identity(cut['manifest']) != record['cutID'] or cut['execution_point'] != meta['origins'][role]['execution_point']:
            raise ValueError('Source cut identity/point differs')
        declared = {x['key']['value']: x['value'] for x in cut['manifest']['payload']['items']}
        for name, page in record['pages'].items():
            if cut['slots'][name] != page['pageID'] or page['declaredDescriptor'] != declared[name] or page['metadata']['value'] != declared[name]:
                raise ValueError('Actual source page membership differs')
    family = HERE.parent.parent / 'artifacts/mdb-computation-family-20261107/fresh-family-02/compact'
    for n, digest in meta['familyMetadataSHA256'].items():
        if sha(family / n) != digest:
            raise ValueError('Original source record changed')
    old = read(HERE.parent / 'artifacts/source-complement-v1/result/inputs.json')
    s = {k: old[k] for k in ['modelRoot', 'versions', 'nativeSources', 'numericalProfile', 'imageAssets']}
    artifacts = read(family / 'parameter-artifacts.json')
    required = set(BANK_PARAMETERS + SELECTED_PARAMETERS + RETURN_PARAMETERS + NEXT_PARAMETERS)
    artifacts['artifacts'] = [x for x in artifacts['artifacts'] if x['source'].get('tensor') in required]
    if {x['source']['tensor'] for x in artifacts['artifacts']} != required:
        raise ValueError('Original selected parameter recipes incomplete')
    writers = read(family / 'native-source-writers.json')['sphere-cube-plain-2027']['sources']
    denied = [str(Path(meta['familyRoot']) / x['path']) for writer in writers.values() for x in writer['ports'].values()]
    previous = read(HERE.parent / 'artifacts/source-complement-v1/result/primary.json')
    denied += [x['path'] for x in fields(previous)]
    s.update(sourceCuts=meta['sources'], sourceMetadataFiles=meta['files'], sourceJobID=inv['sourceJobID'], sourceCustody=inv['sourceCustody'],
             originalParameterArtifacts=artifacts['artifacts'], deniedHistoricalArrays=sorted(set(denied)),
             contextContract='Original FLUX2 forward forms joint img_mod/txt_mod once and passes these same values to every joint block. Joint1 and joint2 consume the authenticated context; no clock network rerun.')
    root = Path(root).resolve();root.mkdir(parents=True, exist_ok=False)
    payload = root / 'payload';payload.mkdir()
    for n in OWNED:
        shutil.copyfile(HERE / n, payload / n)
    for n in WHEELS:
        shutil.copyfile(HERE.parent / 'artifacts/source-complement-v1/payload' / n, payload / n)
    shutil.copyfile(family / 'retrieve_weights.py', payload / 'retrieve_weights.py')
    shutil.copyfile(HERE.parent / 'docs/plans/propagated-context-support.md', payload / 'frozen-plan.md')
    for n in ['source-metadata.json', 'byte-inventory.json']:
        shutil.copyfile(intake / n, payload / n)
    save(payload / 'inputs.json', s);save(payload / 'parameter-artifacts.json', artifacts)
    preview = _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda')
    basis = api.json('GET', '/api/jobs/job-3505d6effb40')
    save(payload / 'resource-basis.json', {'intentFirstPreview': preview, 'measuredFullKVSelectedMLPParent': basis['result'],
         'geometry': {'bankPhaseBF16ParameterBytes': 94372608, 'perArmBF16ParameterBytes': 207618304,
                      'distinctBF16ParameterBytes': 301990912, 'fullKVPayloadBytesPerRole': 18874368, 'reusedReceivingCacheBytes': 0},
         'exception': 'Selected-tensor assay, not full-model load. Prior fullKV/source3195.1MBRSS756MBdevice; use6144RAM1536VRAMCPU4disk3GB,180sexpected900cap. Three sequential children; no fulltransformer/suffix/image/fit.'})
    contract = one_lease_scheduling_contract(logical_branches=2, logical_operations=19, logical_checkpoint_replays=0,
        conditioner_policy='Authenticated actual post-joint0 native states. Global modulation/positions and original learned maps remain supplied/paid.',
        extra={'modelLoadScope': 'Bankphase8selectedtensors; each arm5selectedtensors. No full model.',
               'targetIsolation': 'Primary denies full P bank. Control admitted after primary seal/process exit and denies all primary arrays. No archived target/refresh or receiving-cache actor.'})
    r = {'schema': 'tce-propagated-context-request-v1', 'ownedSources': OWNED, 'sourceHashes': {p.name: sha(p) for p in payload.iterdir()},
         'scheduling_contract': contract, 'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda', 'dtype': 'bfloat16',
         'batch_size': 1, 'seq_len': 1536, 'height': 512, 'width': 512, 'needs': {'cuda': True}, 'timeoutSeconds': 900,
         'case': 'sphere-cube-plain-2027', 'support': [4, 11, 12, 13], 'logicalStep': 2, 'sourceJoint': 1, 'nextJoint': 2,
         'arithmeticSubplans': {'banks': 'two-full-original-KV-and-one-selectedP-query', 'arms': 'native4x1536attention-ownMLP-ownnextquery'},
         'protocol': 'Inventory source byte support then actual fresh KV support. Shared selectedP operands; Qcomplement primary,postsealexitPcontrol. Own joint1 residual/MLP feeds own joint2 query.17affines+2SDPA58,925,776,896MAC; no splice,oldtarget,suffix,image,fit.'}
    r['requestSHA256'] = identity(r)
    save(root / 'request.json', r);save(payload / 'request.json', r)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [p.name for p in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': r['requestSHA256'], 'sourceCuts': wanted}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root = Path(root).resolve();r = read(root / 'request.json');verify(root / 'payload', r)
    if (root / 'job.json').exists():
        raise ValueError('Already submitted; inspect same handle')
    cmd = command()[:-5]
    for n in WHEELS:
        cmd += ['--with', './' + n]
    cmd += ['python', '-u', 'propagated_context_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = guarded_mrun_launch(cmd, request=r, experiment='tce-propagated-context-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs=r['needs'],
        mrun_config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width', 'scheduling_contract', 'arithmeticSubplans']},
        ram_mb=6144, vram_mb=1536, cpu_threads=4, disk_gb=3, est_wall_s=180, timeout_s=900, detach=True, retry_on_kill=False, preflight=True,
        note='Propagated context spherecube2027 step2 joint1:sourcebyteinventory512text1024imagechanged,contextsame. TwofullKV57.982BMAC+sharedPquery37.749MMAC;twoown4rowSDPA/MLP/nextjoint2query905.970MMAC,total58.926B17affines2SDPA. PrimaryQcomplement deniesPbank;postsealPcontrol deniesprimaryarrays. No archivedtarget/refit/splice/cacheactor/suffix/VAE/image. Three sequential processes,CPU4RAM6144VRAM1536disk3GB,180sexpected900cap,noretry; measuredfullKVparent3195RSS756device.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']};save(root / 'job.json', value);return value


def execute(root):
    root = Path(root).resolve();r = read(root / 'request.json');verify(root, r)
    started = time.perf_counter();phases = []
    for phase in ['banks', 'primary', 'control']:
        tick = time.perf_counter()
        subprocess.run([sys.executable, '-u', str(root / 'propagated_context_worker.py'), phase, str(root)], cwd=root, check=True)
        phases.append({'phase': phase, 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
        print('PROPAGATED_CONTEXT_' + phase.upper() + '_EXITED', flush=True)
        if phase == 'banks' and not read(root / 'response-result/banks.json')['outsideSupportKVChanged']:
            out = root / 'response-result'
            save(out / 'report.json', {'schema': 'tce-propagated-context-result-v1', 'mechanics_status': 'complete',
                 'instrument_status': 'inapplicable-identical-projected-complements', 'terminal_status': 'unassessed',
                 'requestSHA256': r['requestSHA256'], 'phases': phases, 'totalNewMatrixMACs': 58019807232})
            break
    else:
        from propagated_context_analysis import execute as analyze
        report = analyze(root)
        out = root / 'response-result'
        save(out / 'report.json', {'schema': 'tce-propagated-context-result-v1', 'requestSHA256': r['requestSHA256'], **report,
             'phases': phases, 'workerWallSeconds': time.perf_counter() - started, 'mechanics_status': 'complete',
             'instrument_status': 'executed-changed-complement', 'terminal_status': 'unassessed',
             'limits': ['One known supplied native-state pair with original learned maps. Full bank formation charged.',
                        'Selected P operands fixed; each own MLP output advances into its own query. No full-width native joint1 reference is run.',
                        'No raw-prompt completion, full own-state diffusion, VAE/image or arbitrary-prompt zero-neural capability.']})
    for n in ['request.json', 'inputs.json', 'parameter-artifacts.json', 'retrieve_weights.py', 'resource-basis.json', 'frozen-plan.md', 'source-metadata.json', 'byte-inventory.json']:
        shutil.copyfile(root / n, out / n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']);parser.add_argument('run');args = parser.parse_args()
    if args.action == 'collect':
        from response_study import collect
        value = collect(args.run)
    else:
        value = globals()[args.action](args.run)
    if value is not None:
        print(json.dumps(value))
