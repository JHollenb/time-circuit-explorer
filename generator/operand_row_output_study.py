"""One guarded current-read/product basis cut, followed by post-exit comparisons."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read, save, sha, identity
from flux_clock_study import verify
from operand_row_output import SOURCE_NAMES, abis, work
from operand_row_output_analysis import analysis_abis

HERE = Path(__file__).resolve().parent
OWNED = ['operand_row_output.py', 'operand_row_output_worker.py', 'operand_row_output_ledger.py', 'operand_row_output_analysis.py', 'operand_row_output_study.py', 'test_operand_row_output.py',
    'row_output.py', 'gate_up_basis_ledger.py', 'gate_up_basis_analysis.py', 'encoder_atoms.py', 'paired_fields.py', 'flux_clock.py', 'flux_clock_study.py', 'complete_masked_study.py', 'masked_kv_study.py', 'terminal_capture_guard_v2.py', 'projected_complement.py', 'sparse_text_kv.py']
WHEELS = ['mdb_runtime-0.1.0-py3-none-any.whl', 'saturn_pub-0.4.1-py3-none-any.whl', 'mrun_pub-0.1.2-py3-none-any.whl']


def logical_bytes(item):
    return math.prod(item['shape'])*(2 if item['dtype'] == 'torch.bfloat16' else 8)


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from projected_complement import fields
    from publish_source_native import checked
    from publish_row_output import audit
    api = Api(); audit(HERE.parent/'docs/results/current-row-output')
    parent, col, job, pr = checked(HERE.parent/'artifacts/current-state-gate-up-v1'); p = parent/'result'; s = read(p/'inputs.json'); c = read(p/'candidate.json')
    prior, pc, pj, pp = checked(HERE.parent/'artifacts/current-row-output-v1'); q = prior/'result'; c83 = read(q/'candidate.json')
    for completed, custody in [(job, col['custody']), (pj, pc['custody'])]:
        actual = api.json('GET', '/api/jobs/'+completed['job_id'])
        if actual['state'] != 'succeeded' or actual['payload_custody'] != custody:
            raise ValueError('Source/comparison parent custody differs')
    source = {n: (c['fields'][n] if n == 'product' else s['sourceFields'][n]) for n in SOURCE_NAMES}
    comparison = {'82': {n: c['fields'][n] for n in ['preResidualUpdate', 'exit']}, '83': {n: c83['fields'][n] for n in ['preResidualUpdate', 'exit']}, '73': {n: s['comparisonFields'][n] for n in ['preResidualUpdate', 'exit']}}
    witnesses = [read(x/'retained-array-verification.json') for x in [parent, prior]]
    for item in [*source.values(), *list(fields(comparison))]:
        if not any(v['files'].get(item['path'], {}).get('sha256') == item['sha256'] and v['npyPayloadWitnesses'].get(item['path'], {}).get('payloadSHA256') == item['payloadSHA256'] for v in witnesses):
            raise ValueError('Complete source/comparison byte witness absent')
    for n in OWNED:
        for old in [pr, pp]:
            if n in old['ownedSources'] and sha(HERE/n) != old['sourceHashes'][n]:
                raise ValueError('Reused frozen helper differs: '+n)
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); payload = root/'payload'; payload.mkdir()
    for n in OWNED:
        shutil.copyfile(HERE/n, payload/n)
    for n in WHEELS:
        shutil.copyfile(parent/'payload'/n, payload/n)
    for src, dest in [('docs/plans/post-nonlinearity-row-basis.md', 'frozen-plan.md'), ('docs/plans/post-nonlinearity-row-basis-implementation.md', 'implementation-notes.md')]:
        shutil.copyfile(HERE.parent/src, payload/dest)
    denied = set(s['deniedNumericalInputs']) | {a['path'] for x in [p, q] for a in fields(read(x/'candidate.json'))} | {a['path'] for x in [p, q] for a in fields(read(x/'analysis-fields.json'))} | {a['path'] for a in fields(s['sourceFields'])}
    config = {k: s[k] for k in ['modelRoot', 'nativeSources', 'versions', 'imageAssets', 'numericalProfile']}
    config.update(sourceFields=source, comparisonFields=comparison, originalParameters=[s['originalParameters'][1]], parentJobID=job['job_id'], parentCustody=col['custody'], parentRoot=job['result']['diagnostics']['cwd'],
        parentCandidateSealSHA256=sha(p/'candidate-seal.json'), parentByteVerificationSHA256=sha(parent/'retained-array-verification.json'),
        parentRecords={n: sha(p/n) for n in ['request.json', 'inputs.json', 'candidate.json', 'execution.json', 'candidate-seal.json', 'producer-exit.json', 'report.json']},
        comparisonParent83={'jobID': pj['job_id'], 'custody': pc['custody'], 'root': pj['result']['diagnostics']['cwd'], 'records': {n: sha(q/n) for n in ['request.json', 'inputs.json', 'candidate.json', 'execution.json', 'candidate-seal.json', 'producer-exit.json', 'report.json']}},
        sourceRoots=sorted({job['result']['diagnostics']['cwd'], pj['result']['diagnostics']['cwd'], *s['sourceRoots']}), deniedNumericalInputs=sorted(denied-{a['path'] for a in source.values()}),
        preStageReviewSnapshots={n: sha(HERE.parent/n) for n in ['docs/results/fixed-row-projection-evidence/advisor.json', 'docs/results/research-review-cycle84/advisor-survey.json', 'docs/results/storage-audit-cycle84/publication.json']})
    save(payload/'inputs.json', config); host = next(a for a in api.json('GET', '/api/hosts') if a['name'] == 'beast')
    save(payload/'resource-basis.json', {'intentFirstFullModelPreview': _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda'),
        'measuredParentJobID': pj['job_id'], 'measuredParent': {k: pj['result'][k] for k in ['elapsed_s', 'peak_rss_mb', 'peak_vram_mb']},
        'liveHost': {k: host[k] for k in ['name', 'active_jobs', 'admission', 'telemetry', 'ram_total_mb', 'vram_total_mb']},
        'sourceFileBytes': sum(a['bytes'] for a in source.values()), 'comparisonFileBytes': sum(a['bytes'] for a in fields(comparison)),
        'originalLoadedParameterBytes': 75497472, 'promotedOutFP64Bytes': 3072*12288*8, 'joinedBF16Bytes': 1536*12288*2, 'joinedFP64Bytes': 1536*12288*8,
        'twoRawPCG64WordBytes': 2*12288*128*8, 'twoFP64SketchBytes': 2*12288*128*8,
        'producerOutputLogicalBytes': sum(map(logical_bytes, abis().values())), 'analysisOutputLogicalBytes': sum(map(logical_bytes, analysis_abis().values())),
        'producerBytesByField': {n: logical_bytes(a) for n, a in abis().items()}, 'analysisBytesByField': {n: logical_bytes(a) for n, a in analysis_abis().items()},
        'extraProducerTemporaryBudgetBytes': 512 << 20, 'analysisExtraLogicalBudgetBytes': 1 << 30,
        'scope': 'Closest83measured1795.6MBRSS/1564device used same originalout/FP64weight/join and six receiving products. New two source sketches add25MBretainedsigns and1.6MBF, rawPCG/temp budget explicit. Retain source-derived4096RAM3072VRAM4CPU3GBdisk90expected300cap; fullmodel sizing inapplicable. EightFP64products/twoQR; postexit targetaccounts sequential under1GiBextra. No source/attention/GU/encoder/VAE execution.'})
    contract = one_lease_scheduling_contract(logical_branches=1, logical_operations=2, logical_checkpoint_replays=0, conditioner_policy='Own82product, paid73read/gate/residual. Originaloutputmap and historical82GU/source/attention remain paid.',
        extra={'purpose': 'Post-product/read source rowbasis, fixedrank128/sketch rule, seal/exit then82primary83and73secondary fullcentered signed everyrow.', 'modelLoadScope': 'One originaloutput75497472B, eightFP64products15.099BMAC;twoQR95.071MFLOPs separate;postexit4diagnosticproducts50.332MMAC. No output-derived fit/oldbases/coordinates/newreference/continuation/image.'})
    costs = work(); r = {'schema': 'tce-current-operand-row-output-request-v1', 'ownedSources': OWNED, 'sourceHashes': {a.name: sha(a) for a in payload.iterdir()}, 'scheduling_contract': contract,
        'model': 'flux2-klein-4b-distilled', 'device': 'cuda', 'backend': 'torch', 'dtype': 'float64', 'batch_size': 1, 'seq_len': 1536, 'needs': {'cuda': True}, 'timeoutSeconds': 300,
        'prospectiveCandidateMACs': costs['candidateMACs'], 'prospectiveDiagnosticMACs': costs['diagnosticMACs'], 'prospectiveAssayMACs': costs['assayMACs'], 'fixedAddress': {'clock': 0, 'singleBlock': 0},
        'protocol': 'Fivepaidfields+originalout. ActualJBF16joinhash->FP64; PCG64SeedSequence[20261107,85,s]random_raw lowbitCorder unscaledOmega;F=J Omega,canonicalthinQR,Fbasis->U.TJ->originalWout->Ucompleteupdate. OneFP64->FP32->BF16, separateBF16gate/add. Oldbases/coordinates/82/83/73outputs denied before seal+exit;4postexitUR/Gram products. No rank/seed/fit/GU/source/attention/reference/continuation/image/zero-neuralclaim.'}
    r['requestSHA256'] = identity(r); save(root/'request.json', r); save(payload/'request.json', r)
    save(payload/'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [a.name for a in payload.iterdir()]+['preflight.v1.json']})
    return {'root': str(root), 'requestSHA256': r['requestSHA256'], 'candidateMACs': costs['candidateMACs'], 'diagnosticMACs': costs['diagnosticMACs']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root = Path(root).resolve(); r = read(root/'request.json'); verify(root/'payload', r)
    if (root/'job.json').exists():
        raise ValueError('Already submitted; inspect same handle')
    cmd = command()[:-5]
    for name in WHEELS:
        cmd += ['--with', './'+name]
    cmd += ['python', '-u', 'operand_row_output_study.py', 'execute', '.']; cmd[cmd.index('uv'):cmd.index('uv')] = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = guarded_mrun_launch(cmd, request=r, experiment='tce-current-operand-row-output-v1', model=r['model'], task='analysis', payload=root/'payload', pin='beast', needs=r['needs'],
        mrun_config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'device', 'backend', 'dtype', 'batch_size', 'seq_len', 'scheduling_contract']}, ram_mb=4096, vram_mb=3072, cpu_threads=4, disk_gb=3, est_wall_s=90, timeout_s=300, detach=True, retry_on_kill=False, preflight=True,
        note='Own82product+paidread/gate/incoming/originalout75MB. FixedPCG64[20261107,85,s] lowbitCorder Jsketch->rank128QR->U.TJ/originalout/Uwrite,8FP64products15.099BMAC+2QR95.071MFLOPs;onefinaldoubleadmission/nativeBF16gate/add.4postexitUR/Gramproducts50.332MMAC and82primary83/73secondary fullcentered/all1536rows/signedaccounts. Denyoldbases/coordinates/outputs untilseal+exit.4096RAM3072VRAM4CPU3GBdisk90expected300cap,noretry,onelease; no source/GU/attention/reference/image,allhistory/maps paid.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root/'job.json', value); return value


def execute(root):
    root = Path(root).resolve(); verify(root, read(root/'request.json')); (root/'response-result').mkdir(); started = time.monotonic()
    with subprocess.Popen([sys.executable, '-u', str(root/'operand_row_output_worker.py'), str(root)], cwd=root) as producer:
        code = producer.wait()
        if code:
            raise subprocess.CalledProcessError(code, producer.args)
        save(root/'response-result/producer-exit.json', {'schema': 'tce-current-operand-row-output-producer-exit-v1', 'producerPID': producer.pid, 'returncode': code, 'requestSHA256': read(root/'request.json')['requestSHA256'],
            'candidateSealSHA256': sha(root/'response-result/candidate-seal.json'), 'producerProcessSeconds': time.monotonic()-started, 'scope': 'Parent observed successful producer exit before82/83/73 target analyzer starts.'})
    subprocess.run([sys.executable, '-u', str(root/'operand_row_output_analysis.py'), str(root)], check=True, cwd=root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']); parser.add_argument('root'); args = parser.parse_args()
    if args.action == 'collect':
        from response_study import collect
        value = collect(args.root)
    else:
        value = globals()[args.action](args.root)
    print(json.dumps(value))
