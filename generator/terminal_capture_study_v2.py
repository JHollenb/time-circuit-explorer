"""One explicitly paid native suffix capture after metadata proves no terminal cut."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['terminal_capture_study_v2.py','terminal_capture_worker_v2.py','terminal_capture_cut_v2.py','terminal_capture_guard_v2.py','terminal_capture_mechanics.py','terminal_liveness_intake.py','encoder_atoms.py','paired_fields.py',
       'flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py','projected_complement.py','sparse_text_kv.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']
METADATA=r'''
import hashlib,json,sys
from pathlib import Path
s=json.load(sys.stdin);source=s['source'];p=Path(source['cutFile']['path']);raw=p.read_bytes()
if hashlib.sha256(raw).hexdigest()!=source['cutFile']['sha256']:raise ValueError('Source cut bytes differ')
cut=json.loads(raw);root=p.parent.parent;declared={x['key']['value']:x['value'] for x in cut['manifest']['payload']['items']};pages={}
for name,page_id in cut['slots'].items():
 p=root/'pages'/(page_id+'.json');raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=page_id:raise ValueError('Source page differs')
 page=json.loads(raw)
 if page['value']!=declared[name]:raise ValueError('Source membership descriptor differs')
 files={str(p):page_id}
 if page['tensor_blob'] is not None:
  blob=root/'blobs'/(page['tensor_blob']+'.safetensors');digest=hashlib.sha256(blob.read_bytes()).hexdigest()
  if digest!=page['tensor_blob']:raise ValueError('Source blob differs')
  files[str(blob)]=digest
 pages[name]={'pageID':page_id,'declaredDescriptor':declared[name],'metadata':page,'files':files}
config=Path(s['modelRoot'])/'transformer/config.json'
if hashlib.sha256(config.read_bytes()).hexdigest()!=s['configSHA256']:raise ValueError('Source config differs')
print(json.dumps({'cutID':source['cutID'],'store':str(root),'cutFile':source['cutFile'],'cut':cut,'pages':pages,'modelConfig':json.loads(config.read_bytes())}))
'''


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    parent,c,j,old=checked(HERE.parent/'artifacts/source-reader-v1');api=Api();s0=read(parent/'result/inputs.json')
    census=read(HERE.parent/'artifacts/terminal-liveness-intake-v1/state-cut-census.json')
    if census['terminalSingle19Cuts'] or census['intakeSourceSHA256']!=sha(HERE/'terminal_liveness_intake.py'):raise ValueError('Capture is not justified by this census')
    for job_id,custody in [(j['job_id'],c['custody']),(s0['sourceJobID'],s0['sourceCustody'])]:
        live=api.json('GET','/api/jobs/'+job_id)
        if live['state']!='succeeded' or live['payload_custody']!=custody:raise ValueError('Source/parent custody differs')
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(METADATA)],
        input=json.dumps({'source':s0['sourceCuts']['P'],'modelRoot':s0['modelRoot'],'configSHA256':s0['imageAssets']['transformer/config.json']}),text=True,capture_output=True,timeout=90)
    if proc.returncode:raise RuntimeError(proc.stderr)
    source=json.loads(proc.stdout);config=source.pop('modelConfig')
    if identity(source['cut']['manifest'])!=source['cutID'] or source['cut']!=s0['sourceCuts']['P']['cut']:raise ValueError('Qualified source identity differs')
    if {n:config[n] for n in ['num_layers','num_single_layers','num_attention_heads','attention_head_dim','mlp_ratio','in_channels','patch_size','eps']}!={
        'num_layers':5,'num_single_layers':20,'num_attention_heads':24,'attention_head_dim':128,'mlp_ratio':3.,'in_channels':128,'patch_size':1,'eps':1e-6} or config['patch_size']**2*(config.get('out_channels') or config['in_channels'])!=128:
        raise ValueError('Declared native geometry differs')
    family=HERE.parent.parent/'artifacts/mdb-computation-family-20261107/fresh-family-02/compact';artifacts=read(family/'parameter-artifacts.json')
    prefixes=tuple('transformer_blocks.'+str(i)+'.' for i in range(1,5))+tuple('single_transformer_blocks.'+str(i)+'.' for i in range(20))
    artifacts['artifacts']=[x for x in artifacts['artifacts'] if x['source']['tensor'].startswith(prefixes) or x['source']['tensor'] in ['norm_out.linear.weight','proj_out.weight']]
    if len(artifacts['artifacts'])!=146:raise ValueError('Native suffix parameter inventory differs')
    failed=api.json('GET','/api/jobs/job-c266fd593526')
    if failed['state']!='failed':raise ValueError('Previous attempt is not authoritatively terminal')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for n in OWNED:
        if n in old['ownedSources'] and sha(HERE/n)!=old['sourceHashes'][n]:raise ValueError('Qualified frozen helper changed')
        shutil.copyfile(HERE/n,payload/n)
    for n in WHEELS:shutil.copyfile(parent/'payload'/n,payload/n)
    shutil.copyfile(HERE.parent/'docs/plans/terminal-native-capture-v2.md',payload/'frozen-plan.md')
    shutil.copyfile(HERE.parent/'docs/results/terminal-native-capture-failed-v1/failure.json',payload/'previous-failure.json')
    shutil.copyfile(HERE.parent/'artifacts/terminal-liveness-intake-v1/state-cut-census.json',payload/'state-cut-census.json')
    s={k:s0[k] for k in ['modelRoot','nativeSources','imageAssets','versions','numericalProfile','sourceJobID','sourceCustody']}
    s.update(sourceCut=source,modelConfig=config,originalParameterArtifacts=artifacts['artifacts'],deniedNumericalInputs=s0['deniedNumericalInputs'],
        parentJobID=j['job_id'],parentCustody=c['custody'],sourceReaderReportSHA256=sha(parent/'result/report.json'))
    save(payload/'inputs.json',s);save(payload/'parameter-artifacts.json',artifacts)
    save(payload/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'failedNativeAttempt':{'jobID':failed['job_id'],'state':failed['state'],'observed':{k:failed['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'scope':'Reached first persistence after native prefix; counters not durable, no sealed capture.'},
        'measuredSelectedMapParent':{k:j['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},
        'geometry':{'largestPerBlockOriginalParameterBytes':max(sum(x['source']['data_offsets'][1]-x['source']['data_offsets'][0] for x in artifacts['artifacts'] if x['source']['tensor'].startswith(prefix)) for prefix in prefixes),
                    'residentBlocks':1,'fullTransformerConstructed':False,'nativePrefixJointBlocks':4,'nativePrefixSingleBlocks':19,'targetSingleBlocks':1,'targetVelocityHeads':1},
        'exception':'Stream one originalblock: largest joint491MB,single245MB vs302MBselected-map parent1824RSS774device; full1536row native activation/transient workspace additional.6144RAM3072VRAMCPU4disk2GB,120expected600cap,no retries. No fullmodel/VAE/encoder/clocknetwork/scheduler.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=26,logical_checkpoint_replays=0,
        conditioner_policy='Complete authenticated native post-joint0 P StateCut; previous native prefix and clock context remain paid.',
        extra={'modelLoadScope':'One original joint/single block resident at a time;146distinct tensors across24blocks and finalhead.',
               'purpose':'Capture missing complete single19 input, matched native output and velocity for later target-isolated liveness producer. This entire native suffix is new paid work.'})
    r={'schema':'tce-terminal-native-capture-request-v2','ownedSources':OWNED,'sourceHashes':{p.name:sha(p) for p in payload.iterdir()},
        'scheduling_contract':contract,'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'needs':{'cuda':True},'timeoutSeconds':600,
        'arithmeticSubplans':{'mechanics':'Tiny supplied CPU StateCut with actual worker wheel, no model/checkpoint arrays','prefix':'4nativejoint+19nativesingle','target':'single19+finalvelocityhead'},
        'previousFailureJobID':'job-c266fd593526',
        'plannedNativeAffineMACs':4523022090240,'plannedNativeAttentionMACs':347892350976,
        'protocol':'One paid native suffix from postjoint0P. Persist complete13slot single19input and matched postblock StateCuts plus finalvelocity; verify durable roundtrips.25typedActs90affines24SDPA. No reducedproducer/encoder/clocknet/scheduler/VAE/image/noise/fit.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(payload/'request.json',r)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256'],'sourceCutID':source['cutID']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect the same handle')
    cmd=command()[:-5]
    for n in WHEELS:cmd+=['--with','./'+n]
    cmd+=['python','-u','terminal_capture_study_v2.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='Wheel-mechanics first; paid terminal-cut capture from native postjoint0P:4joint+20single+velocityhead,90affines4.523T+24attention.348TMAC. Persist complete single19source/matchedtarget/velocity; no reducedproducer/encoder/clocknet/scheduler/VAE/image/fit. Stream oneblock:maxjoint491MB/single245MB; measured302MBmaps parent1824RSS774device,6144RAM3072VRAM4CPU2GBdisk,120sexpect600cap,noretries. Missingcut confirmed342filecensus. Originalprefix/context paid.'
    if len(note)>500:raise ValueError('Scheduler note exceeds limit')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-terminal-native-capture-v2',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width','scheduling_contract','arithmeticSubplans']},
        ram_mb=6144,vram_mb=3072,cpu_threads=4,disk_gb=2,est_wall_s=120,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r)
    subprocess.run([sys.executable,'-u',str(root/'terminal_capture_mechanics.py'),str(root)],cwd=root,check=True)
    subprocess.run([sys.executable,'-u',str(root/'terminal_capture_worker_v2.py'),str(root)],cwd=root,check=True)
    for n in ['capture-mechanics.json','previous-failure.json']:shutil.copyfile(root/n,root/'response-result'/n)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
