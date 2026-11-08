"""Stage one same-budget contrast-aware writer transaction."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from contextual_writer_program import BUDGET, FINITE, read, save, sha

HERE = Path(__file__).resolve().parent
OWNED = ['contrast_writer_fit.py', 'contrast_writer_worker.py', 'contrast_writer_study.py', 'publish_contrast_writer.py', 'test_contrast_writer.py',
         'contextual_writer_program.py', 'contextual_writer_worker.py', 'template_cofactor_program.py', 'token_history_process.py', 'token_history_address.py', 'contextual_writer_saturn_init.py']
OWNED.append('publish_contextual_writer.py')
PARENT_NAMES = ['inputs.json', 'request.json', 'programs.json', 'consumer-fields.json', 'typed-consumers.json', 'lexical-manifest.json', 'publication.json']


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify(root, request):
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']: raise ValueError('Request differs')
    for name, digest in request['sourceHashes'].items():
        if sha(Path(root)/name) != digest: raise ValueError('Executed source differs: '+name)


def stage(root):
    from mrun.client.api import Api
    from mdb.job_support import one_lease_scheduling_contract
    api = Api(); base = HERE.parent/'docs/results/contextual-writer'; publication = read(base/'publication.json'); old = read(base/'scheduler-job.json'); live = api.json('GET', '/api/jobs/'+old['job_id'])
    if old['job_id'] != publication['jobID'] or live['state'] != 'succeeded' or live['payload_custody'] != old['payload_custody'] or not old['payload_custody']['required']:
        raise ValueError('Qualified parent custody differs')
    for name, item in publication['files'].items():
        if sha(base/name) != item['sha256'] or (base/name).stat().st_size != item['bytes']: raise ValueError('Parent publication changed')
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); payload = root/'payload'; payload.mkdir()
    for name in OWNED: shutil.copyfile(HERE/name, payload/name)
    saturn = HERE.parents[1]/'saturn/src/saturn'; exported = payload/'saturn'; exported.mkdir()
    for name in ('titan.py', 'tensor_hash.py'): shutil.copyfile(saturn/name, exported/name)
    shutil.copyfile(HERE/'contextual_writer_saturn_init.py', exported/'__init__.py')
    shutil.copyfile(HERE.parent/'docs/plans/contrast-writer-assay.md', payload/'frozen-plan.md')
    for name in PARENT_NAMES: shutil.copyfile(base/name, payload/('parent-'+name))
    catalog = read(base/'programs.json'); old_request = read(base/'request.json'); old_report = read(base/'report.json')
    if old_report['selected'] != 'initial' or catalog['accepted']['programSHA256'] != catalog['initial']['programSHA256']: raise ValueError('Qualified endpoint branch differs')
    cwd = old['result']['diagnostics']['cwd']; lexical = read(base/'lexical-manifest.json'); assets = {cwd+'/lexical/manifest.json': sha(base/'lexical-manifest.json')}
    for name, info in lexical['files'].items(): assets[cwd+'/lexical/'+name] = info['sha256']
    for key in ('tokenizer', 'template'): assets[cwd+'/lexical/'+lexical[key]['path']] = lexical[key]['sha256']
    for method in ('accepted', 'position'):
        item = catalog[method]; folder = Path(item['directory']); assets[str(folder/'program.json')] = item['programSHA256']; assets[str(folder/'prefix.bf16')] = item['program']['prefixSHA256']
        for dictionary in item['program']['dictionaries']: assets[str(folder/dictionary['path'])] = dictionary['sha256']
    remote_inventory = {x['path']: x['sha256'] for x in publication['freshRemoteFiles']}
    if any(remote_inventory.get(name) != digest for name, digest in assets.items()): raise ValueError('Reused serving asset lacks original publication custody')
    reuse = {'jobID': old['job_id'], 'custody': old['payload_custody'], 'requestSHA256': old_request['requestSHA256'], 'publicationSHA256': sha(base/'publication.json'),
        'selected': old_report['selected'], 'lexicalDirectory': cwd+'/lexical', 'servingAssets': assets,
        'scope': 'Historical neural/fit/compile cost paid;fresh current file auth required on worker. Original public histories are reused transfer,not fresh holdout.'}
    save(payload/'reuse.json', reuse)
    retained = ['request.json', 'reuse.json', 'frozen-plan.md']+['parent-'+n for n in PARENT_NAMES]
    scheduling = one_lease_scheduling_contract(model_loads=1, logical_branches=6, logical_operations=6, logical_checkpoint_replays=0,
        extra={'modelLoadScope': 'One selected layer1 inputnorm/QKV set;historical native/H0/endpoint/position outputs reused with exact input binding.',
               'branchIsolation': 'Fresh non-neural serving children within one outer lease;no per-child scheduler submissions.'})
    request = {'schema': 'tce-contrast-writer-assay-v1', 'cases': old_request['cases'], 'budget': BUDGET, 'finitePolicy': FINITE,
        'discovery': ['A', 'B', '00', '10'], 'transfer': ['01', '11'], 'novelNoTarget': ['novel'], 'retainedMetadata': retained,
        'sourceHashes': {str(p.relative_to(payload)): sha(p) for p in payload.rglob('*') if p.is_file()},
        'device': 'cuda', 'backend': 'torch-selected-qkv-bf16;paired-fit-numpy-CPU1;stdlib-serving', 'dtype': 'bfloat16', 'batch_size': 1, 'seq_len': 512,
        'needs': {'cuda': True}, 'scheduling_contract': scheduling, 'timeoutSeconds': 900,
        'physicalWork': {'consumerUnits': 6, 'originalLinears': 18, 'nominalMACs': 48318382080, 'fullBlocks': 0, 'images': 0},
        'protocol': 'Fixed prior slot0/features/budget;one lambda1 paired residual tree. Relaxed global split Schur scoring,FP32 atoms,BF16 serving. Discovery actualQKV endpoint+response selects candidate/endpoint,exactimmutable restore. Ownfinalseal before reusedtransfer targetaccess. No expansion/newteacher/image.'}
    request['requestSHA256'] = identity(request); save(root/'request.json', request); save(payload/'request.json', request)
    save(payload/'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': sorted([p.name for p in payload.iterdir()]+['preflight.v1.json'])})
    return {'root': str(root), 'requestSHA256': request['requestSHA256']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root = Path(root).resolve(); r = read(root/'request.json'); verify(root/'payload', r)
    if (root/'job.json').exists(): raise ValueError('Already submitted;inspect existing job')
    cmd = command()[:-5]+['python', '-u', 'contrast_writer_study.py', 'execute', '.']; i = cmd.index('uv')
    cmd[i:i] = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = guarded_mrun_launch(cmd, request=r, experiment='tce-contrast-writer-v1', task='analysis', model=None, payload=root/'payload', pin='beast', needs=r['needs'],
        mrun_config={k: r[k] for k in ('schema', 'requestSHA256', 'device', 'backend', 'dtype', 'batch_size', 'seq_len', 'scheduling_contract')},
        timeout_s=900, detach=True, retry_on_kill=False, preflight=True,
        note='Samebudget2slot/depth4 rawinputH1:fixedslot0,155features,one paired-loss slot1 fit lambda1. CPU1Schur split scoring,cutstatscache<1GB+temps;expected<10min/900scap. Reuse authenticated lexical/native/control fields;6serialB1newQKVunits18affines48.318BMAC. Discovery endpoint+response choice,immutable restore,seal final before transfer. No attention/fullencoder/image,newteacher,sweep,retries. OneCUDAhostlocal lease,intentfirst sizing.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root/'job.json', value); return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']); parser.add_argument('root'); args = parser.parse_args()
    if args.action == 'execute':
        from contrast_writer_worker import execute
        value = execute(args.root)
    elif args.action == 'collect':
        from response_study import collect
        value = collect(args.root)
    else: value = globals()[args.action](args.root)
    if value is not None: print(json.dumps(value))
