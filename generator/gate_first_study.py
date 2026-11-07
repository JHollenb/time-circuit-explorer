"""One source-bound gate-first terminal/head assay, no new full image/reference."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify
from terminal_liveness import PARAMETERS

HERE=Path(__file__).resolve().parent
OWNED=['gate_first_mlp.py','gate_first_worker.py','gate_first_ledger.py','gate_first_analysis.py','gate_first_study.py','test_gate_first_mlp.py',
    'terminal_liveness.py','terminal_attention_ledger.py','terminal_capture_guard_v2.py','encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py',
    'complete_masked_study.py','masked_kv_study.py','projected_complement.py','sparse_text_kv.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    parent,collection,job,pr=checked(HERE.parent/'artifacts/image-partition-moments-v1');p=parent/'result';pub=HERE.parent/'docs/results/image-partition-moments';api=Api()
    for name,digest in read(pub/'files.json').items():
        if sha(pub/name)!=digest:raise ValueError('Published own-state parent differs')
    candidate=read(p/'candidate.json');execution=read(p/'execution.json');s0=read(p/'inputs.json');seal=read(p/'candidate-seal.json');bytecheck=read(parent/'retained-array-verification.json')
    if sha(p/'candidate-seal.json')!='71122d424dd57e0d3f306a8ff8c4ec7c9e816bf941358eb60a7c7949cb538457' or bytecheck['candidateSealSHA256']!=sha(p/'candidate-seal.json') or seal['candidateSHA256']!=sha(p/'candidate.json') or seal['executionSHA256']!=sha(p/'execution.json'):raise ValueError('Qualified source parent release differs')
    source={n:candidate['fields'][field] for n,field in {'rows':'terminalRows3','mod':'singleMod3','temb':'temb3','cosine':'cosine3','sine':'sine3'}.items()}
    targets={'image':candidate['fields']['terminalImage3'],'velocity':candidate['fields']['velocity3']}
    for item in [*source.values(),*targets.values()]:
        if bytecheck['files'][item['path']]['sha256']!=item['sha256'] or bytecheck['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Qualified source/target bytes absent')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for name in OWNED:
        if name in pr['ownedSources'] and sha(HERE/name)!=pr['sourceHashes'][name]:raise ValueError('Inherited frozen helper changed:'+name)
        shutil.copyfile(HERE/name,payload/name)
    for name in WHEELS:shutil.copyfile(parent/'payload'/name,payload/name)
    shutil.copyfile(HERE.parent/'docs/plans/gate-first-terminal-mlp.md',payload/'frozen-plan.md')
    s={n:s0[n] for n in ['modelRoot','nativeSources','imageAssets','versions','numericalProfile']}
    originals=[next(x for x in execution['loadedTransformerParameters'] if x['name']==n) for n in PARAMETERS]
    allowed={x['path'] for x in source.values()}
    denied=set(s0['deniedNumericalInputs'])|{x['path'] for x in candidate['fields'].values()}|{candidate['image']['path']}|{x['path'] for x in read(p/'differences.json').values()};denied-=allowed
    s.update(sourceFields=source,originalParameters=originals,parentJobID=job['job_id'],parentCustody=collection['custody'],parentCandidateSealSHA256=sha(p/'candidate-seal.json'),
        parentRecords={n:sha(p/n) for n in ['candidate.json','candidate-seal.json','execution.json','report.json','request.json']},parentByteVerifierSHA256=sha(parent/'retained-array-verification.json'),deniedNumericalInputs=sorted(denied),step=3,phase='single',cursor=19)
    save(payload/'inputs.json',s);save(payload/'targets.json',{'fields':targets,'sourceCombined':source['rows'],'scope':'Existing own70 terminal/head outputs, denied until candidate seal and actor exit; no new reference.'})
    old=read(HERE.parent/'docs/results/terminal-liveness/scheduler.json')
    save(payload/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredTerminalHistory':{k:old['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'originalParameterBytes':283902464,'physicalMapsLogicalBytes':160432128,
        'preparedBackingLogicalBytes':72351744,'sourceNormMaxTwoTemporaryFP32BanksBytes':226492416,'nominalCandidateArrayBytesAbout':240000000,
        'exception':'Actual singleterminal65:1682.7MBRSS1058MBdevice/38.07s, sixoriginaltensors284MB. New160MBphysicalmaps,72.4MBpreparedbacking inclviews,226.5MBtemporaryFP32normbanks;~240MBcandidate+26MBdifferences ondisk, sixActs. Same4096MBRAM2048MBVRAM4CPU1GBdisk,90sexpected240cap,noretries. No fullmodel/prefix/reference/image/encoder/VAE.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=7,logical_checkpoint_replays=0,conditioner_policy='Authenticated70 own terminalRows3/mod/temb/rope, paid historical prefix, no supplied gate/up/attention activations.',
        extra={'purpose':'Six typed stages plus fresh saved comparison: sourceMaps/currentGates/selection/gather/image/head. No fullU to select, fixed2304/9216sharedsupport.',
            'modelLoadScope':'Sixoriginalterminal/head tensors only; original K/V,Q/fullG,selectedU/output physicalmaps. Full current SiLU/norm/ranking/copy costs paid separately.'})
    r={'schema':'tce-gate-first-terminal-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in payload.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'needs':{'cuda':True},'timeoutSeconds':240,
        'selectedChannels':2304,'totalMLPChannels':9216,'prospectiveMatrixMACs':101890129920,'matchedTerminalHeadMatrixMACs':145376673792,'prospectiveAvoidedMatrixMACs':43486543872,
        'protocol':'CurrentBF16fullgate;FP32Urow/Dcolnorms+currentinputnorm weightedgateenergy, stabledescendingranking,ascending source support. Physical selectedU/out only, fullKV/imageQ/attention/nativehead; target70outputs denied untilseal+exit. No fullUselection/ref/image/fit/sweep. Roundedbounddiagnostic, notgate; learnedmaps/sourceprefixpaid.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(payload/'request.json',r)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in payload.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':r['requestSHA256'],'sourceJobID':job['job_id']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    cmd=command()[:-5]
    for name in WHEELS:cmd+=['--with','./'+name]
    cmd+=['python','-u','gate_first_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='Gatefirst terminal19/clock3:fullKV+imageQ/fullG; currentSiLU/sourceweightnormranking selects2304/9216 U/downchannels.6affines+1attention101.890BMAC,43.487Bbelowmatched.6originaltensors284MB+160MBmaps+72MBbacking+226MBnormtemps,~266MBarrays. Paid70prefix,targetsdenieduntilseal+exit. No fullUselection/fullmodel/ref/image/fit/sweep. Prior1683RSS1058device;4096RAM2048VRAM4CPU1GBdisk,90sexpected240cap,noretries.'
    if len(note)>500:raise ValueError('Queue note too long')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-gate-first-terminal-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width','scheduling_contract']},
        ram_mb=4096,vram_mb=2048,cpu_threads=4,disk_gb=1,est_wall_s=90,timeout_s=240,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r)
    for name in ['gate_first_worker.py','gate_first_analysis.py']:subprocess.run([sys.executable,'-u',str(root/name),str(root)],cwd=root,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
