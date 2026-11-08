"""Stage one bounded raw-input branch/writer and native-QKV consumer experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from contextual_writer_program import BUDGET, FINITE, read, save, sha

HERE = Path(__file__).resolve().parent
OWNED = ['contextual_writer_program.py', 'contextual_writer_worker.py', 'contextual_writer_study.py',
         'test_contextual_writer.py', 'template_cofactor_program.py', 'token_history_process.py', 'token_history_address.py', 'contextual_writer_saturn_init.py', 'publish_contextual_writer.py']
SELECTED = ['model.layers.1.'+n for n in ('input_layernorm.weight', 'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight')]


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify(root, request):
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']: raise ValueError('Request differs')
    for name, digest in request['sourceHashes'].items():
        if sha(Path(root)/name) != digest: raise ValueError('Executed source differs: '+name)


def stage(root):
    from mrun.client.api import Api
    from mdb.job_support import one_lease_scheduling_contract
    api = Api(); results = HERE.parent/'docs/results'; root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); payload = root/'payload'; payload.mkdir()
    for name in OWNED: shutil.copyfile(HERE/name, payload/name)
    # Export the unchanged canonical Frame/Act dependency closure, not a new runtime.
    saturn = HERE.parents[1]/'saturn/src/saturn'; exported = payload/'saturn'; exported.mkdir()
    for name in ('titan.py', 'tensor_hash.py'): shutil.copyfile(saturn/name, exported/name)
    shutil.copyfile(HERE/'contextual_writer_saturn_init.py', exported/'__init__.py')
    shutil.copyfile(HERE.parent/'docs/plans/contextual-writer-assay.md', payload/'frozen-plan.md')
    roots = read(results/'source-order-program/inputs.json'); atom_manifest = results/'raw-prompt-encoder-atoms/manifest.json'
    if sha(atom_manifest) != roots['manifestSHA256']: raise ValueError('Qualified atom manifest differs')
    atom_job = roots['atomOriginalJob']; atom_live = api.json('GET', '/api/jobs/'+atom_job['jobID'])
    if atom_live['state'] != 'succeeded' or atom_live['payload_custody'] != atom_job['payloadCustody']: raise ValueError('Original atom custody differs')
    shutil.copyfile(atom_manifest, payload/'atom-manifest.json')
    cases = {}; parents = []; parameters = None; assets = None; torch_version = None
    for group in ('complete-masked-image', 'fresh-prompt-square'):
        base = results/group; provenance = read(base/'provenance.json'); meta = read(base/'encoder-full.json'); request = read(base/'request.json')
        for name in ('encoder-full.json', 'request.json'):
            if sha(base/name) != provenance['completeCollectedInventory']['files'][name]['sha256']: raise ValueError('Historical native metadata differs')
        live = api.json('GET', '/api/jobs/'+provenance['jobID'])
        if live['state'] != 'succeeded' or live['payload_custody'] != provenance['payloadCustody'] or live['config']['requestSHA256'] != provenance['requestSHA256'] or meta['requestSHA256'] != provenance['requestSHA256']:
            raise ValueError('Native input/custody differs')
        selected = [x for x in meta['parameterReads'] if x['name'] in SELECTED]
        if sorted(x['name'] for x in selected) != sorted(SELECTED): raise ValueError('Native consumer source missing')
        if parameters is not None and selected != parameters: raise ValueError('Native consumer parameter identities differ')
        if torch_version is not None and (torch_version != meta['torchVersion'] or numerical_profile != meta['numericalProfile']): raise ValueError('Native numerical profiles differ')
        if meta['profile']['rms_norm_eps'] != 1e-6: raise ValueError('Qualified RMS epsilon differs')
        parameters = selected; assets = meta['assetChecks']; torch_version = meta['torchVersion']; numerical_profile = meta['numericalProfile']
        parents.append({'group': group, 'jobID': provenance['jobID'], 'custody': live['payload_custody'], 'requestSHA256': provenance['requestSHA256'],
                        'metadataSHA256': sha(base/'encoder-full.json'), 'historicalNominalLinearMACs': sum(x.get('nominalMACs', 0) for x in meta['primitives'])})
        for item in meta['lexicalRows']:
            key = item['id']; cases[key] = {'prompt': item['prompt'], 'H0': meta['fields'][key]['H0'], 'H1': meta['fields'][key]['H1'],
                'tokenIDsSHA256': item['tokenIDsSHA256'], 'maskSHA256': item['maskSHA256'], 'activeRows': item['activeRows']}
        for name in ('provenance.json', 'encoder-full.json', 'request.json'): shutil.copyfile(base/name, payload/(group+'-'+name))
    if set(cases) != {'A', 'B', '00', '10', '01', '11'}: raise ValueError('Frozen corpus differs')
    forbidden = [x[n]['path'] for x in cases.values() for n in ('H0', 'H1')]+sorted({x['file'] for x in parameters})
    source = {'atomStore': roots['store'], 'atomManifestSHA256': roots['manifestSHA256'], 'atomOriginalCustody': atom_job,
        'cases': cases, 'parents': parents, 'parameters': parameters, 'parameterAssets': assets, 'torchVersion': torch_version, 'numericalProfile': numerical_profile, 'rmsNormEpsilon': 1e-6,
        'forbiddenServingPaths': forbidden, 'scope': 'Historical full neural preparation paid; no retrospective fresh-heldout claim.'}
    save(payload/'inputs.json', source)
    scheduling = one_lease_scheduling_contract(model_loads=1, logical_branches=28, logical_operations=28, logical_checkpoint_replays=0,
        extra={'modelLoadScope': 'One selected layer1 inputnorm/QKV parameter set;no full encoder/transformer.',
               'branchIsolation': 'Fresh non-neural serving children share one outer lease; no per-child scheduler submissions.'})
    prompt_cases = [{'id': key, 'prompt': cases[key]['prompt']} for key in ('A', 'B', '00', '10', '01', '11')]
    prompt_cases.append({'id': 'novel', 'prompt': 'A small glass observatory on a cliff above a violet ocean at sunrise.'})
    request = {'schema': 'tce-contextual-writer-assay-v1', 'cases': prompt_cases, 'budget': BUDGET, 'finitePolicy': FINITE,
        'discovery': ['A', 'B', '00', '10'], 'transfer': ['01', '11'], 'novelNoTarget': ['novel'], 'sourceHashes': {str(p.relative_to(payload)): sha(p) for p in payload.rglob('*') if p.is_file()},
        'device': 'cuda', 'backend': 'torch-selected-qkv-bf16;offline-numpy-CPU1;stdlib-serving', 'dtype': 'bfloat16', 'batch_size': 1, 'seq_len': 512,
        'needs': {'cuda': True}, 'scheduling_contract': scheduling, 'timeoutSeconds': 600,
        'physicalWork': {'consumerUnits': 28, 'originalLinears': 84, 'nominalMACs': 225485783040, 'fullBlocks': 0, 'images': 0},
        'protocol': 'Fullvocab ownH0 ->positionwriter+finite-residual contentwriter;equalcapacity positioncontrol. One half-slot1 proposal selected by balanced discovery native-QKV-increment error;immutable exactrestore. Fresh final own-input predictions seal before reused transfer targets. No expansion/sweep/newteacher/image.'}
    request['requestSHA256'] = identity(request); save(root/'request.json', request); save(payload/'request.json', request)
    save(payload/'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': sorted([p.name for p in payload.iterdir()]+['preflight.v1.json'])})
    return {'root': str(root), 'requestSHA256': request['requestSHA256']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root = Path(root).resolve(); r = read(root/'request.json'); verify(root/'payload', r)
    if (root/'job.json').exists(): raise ValueError('Already submitted; inspect existing job')
    cmd = command()[:-5]+['python', '-u', 'contextual_writer_study.py', 'execute', '.']; i = cmd.index('uv')
    cmd[i:i] = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = guarded_mrun_launch(cmd, request=r, experiment='tce-contextual-writer-v1', task='analysis', model=None, payload=root/'payload', pin='beast', needs=r['needs'],
        mrun_config={k: r[k] for k in ('schema', 'requestSHA256', 'device', 'backend', 'dtype', 'batch_size', 'seq_len', 'scheduling_contract')},
        timeout_s=600, detach=True, retry_on_kill=False, preflight=True,
        note='Ownraw->fullH1 two depth4 constant writers;4discovery/2reusedtransfer+1no-targetprompt. CPU1fit,1half-amplitude native-consumer future/immutable restore;fresh stdlib predictionsseal before transfer.28serialB1 inputnorm/QKV units84affines225.486BMAC;no attention/fullblock/image. FullvocabH0copy0.78GBstaysBeast;native selectedweights31.46MB. Intentfirst,hostlocal;expected<5min/600scap,noretries.')
    result = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root/'job.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']); parser.add_argument('root'); args = parser.parse_args()
    if args.action == 'execute':
        from contextual_writer_worker import execute
        value = execute(args.root)
    elif args.action == 'collect':
        from response_study import collect
        value = collect(args.root)
    else: value = globals()[args.action](args.root)
    if value is not None: print(json.dumps(value))
