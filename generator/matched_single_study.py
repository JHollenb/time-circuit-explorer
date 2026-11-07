"""Stage two isolated common-input local replays and their signed-write diagnostic."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify
from matched_single_worker import PARAMETERS

HERE=Path(__file__).resolve().parent
OWNED=['matched_single.py','matched_single_worker.py','matched_single_ledger.py','matched_single_analysis.py','matched_single_study.py','test_matched_single.py',
    'gate_first_trajectory.py','gate_first_mlp.py','image_partition_moments.py','fixed_text_memory.py','terminal_liveness.py','terminal_attention_ledger.py',
    'terminal_capture_guard_v2.py','paired_fields.py','encoder_atoms.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py','projected_complement.py','sparse_text_kv.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    api=Api();parents={};source=None;targets={};denied=set();parent_seals={};records={};sources={}
    for mode,directory in [('sparse','gate-first-trajectory-v1'),('dense','image-partition-moments-v1')]:
        parent,col,job,pr=checked(HERE.parent/'artifacts'/directory);p=parent/'result';c=read(p/'candidate.json');e=read(p/'execution.json');s=read(p/'inputs.json');b=read(parent/'retained-array-verification.json')
        seal=read(p/'candidate-seal.json')
        if b['candidateSealSHA256']!=sha(p/'candidate-seal.json') or seal['candidateSHA256']!=sha(p/'candidate.json') or seal['executionSHA256']!=sha(p/'execution.json'):raise ValueError('Parent release/complete bytes differ')
        mapping={'text':'b_joint4_textOut0','image':'b_joint4_imageOut0','mod':'singleMod0','cosine':'cosine0','sine':'sine0'}
        current={n:c['fields'][field] for n,field in mapping.items()}
        if source is not None and any(any(source[n][k]!=current[n][k] for k in ['shape','logicalDtype','payloadSHA256']) for n in mapping):raise ValueError('Matched common input differs')
        if mode=='sparse':
            source={**current,**{n:c['fields'][n] for n in ['sourceUpNorm','sourceDownNorm']}};originals=[next(x for x in e['loadedTransformerParameters'] if x['name']==name) for name in PARAMETERS];source_config=s
        targets[mode]={name:c['fields']['b_single0_'+name+'0'] for name in ['textOut','imageOut','alpha','textRead']}
        targets[mode].update({'moment_'+name:c['fields']['b_single0_moment_'+name+'0'] for name in ['mean','covariance','logZI','logZT','quadratic']})
        if mode=='sparse':targets[mode].update({'selection_'+name:c['fields']['s_single0_'+name+'0'] for name in ['rowNormSquared','weightedGateEnergy','score','rankedIndices','support','omittedScoreSum','selectedScoreSum']})
        for item in [*current.values(),*targets[mode].values(),*([source['sourceUpNorm'],source['sourceDownNorm']] if mode=='sparse' else [])]:
            if b['files'][item['path']]['sha256']!=item['sha256'] or b['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Required retained byte witness absent')
        denied|={x['path'] for x in c['fields'].values()}|{x['path'] for x in read(p/'differences.json').values()}|set(s['deniedNumericalInputs'])|{c['image']['path']}
        parents[mode]={'jobID':job['job_id'],'custody':col['custody'],'result':{k:job['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'root':job['result']['diagnostics']['cwd']}
        parent_seals[mode]=sha(p/'candidate-seal.json');records[mode]={n:sha(p/n) for n in ['request.json','candidate.json','execution.json','candidate-seal.json','report.json']};sources[mode]=pr['sourceHashes']
    root=Path(root).resolve();root.mkdir(exist_ok=False,parents=True);p=root/'payload';p.mkdir()
    for name in OWNED:
        for hashes in sources.values():
            if name in hashes and sha(HERE/name)!=hashes[name]:raise ValueError('Frozen source changed:'+name)
        shutil.copyfile(HERE/name,p/name)
    for name in WHEELS:shutil.copyfile(HERE.parent/'artifacts/gate-first-trajectory-v1/payload'/name,p/name)
    shutil.copyfile(HERE.parent/'docs/plans/matched-first-single-write.md',p/'frozen-plan.md')
    config={n:source_config[n] for n in ['modelRoot','nativeSources','versions','numericalProfile']}
    config.update(imageAssets={n:h for n,h in source_config['imageAssets'].items() if n.startswith('transformer/')},sourceFields=source,originalParameters=originals,
        parents=parents,parentSeals=parent_seals,parentRecords=records,deniedNumericalInputs=sorted(denied-{x['path'] for x in source.values()}))
    save(p/'inputs.json',config);save(p/'targets.json',{'fields':targets,'scope':'Archived70dense/72sparse clock0single0 outputs and72selector only after both fresh actor seals/exits; no new reference.'})
    hosts=api.json('GET','/api/hosts');host=next(x for x in hosts if x['name']=='beast')
    if 6144*1.1+host['admission']['ram_margin_mb']>=host['ram_total_mb']:raise ValueError('Declared RAM guard cannot fit physically')
    history=read(HERE.parent/'docs/results/gate-first-terminal/scheduler.json')
    save(p/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredLocalTerminalHistory':{k:history['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'liveHost':{k:host[k] for k in ['name','active_jobs','admission','telemetry','ram_total_mb','vram_total_mb']},
        'originalFourParameterBytes':245367296,'maxSparsePhysicalMapBytes':113246208,'FP64FullProductBytes':113246208,'FP64DownBytes':226492416,'oneFP64WriteBytes':37748736,
        'scope':'Three serial fresh children, oneblock only. Replay capturesslices actualprojection/productbackings, no fullmodel/head/VAE. Analysis roughly1.5GBlogicalworkspace inclFP64gatednormtemps/writes, mixedBF16/FP64. Priorlocal1855.9MBRSS956MBdevice with284MBparams; allow6144RAM3072VRAM4CPU2GBdisk,180expected360cap,noretries. Candidate/capture/rawarraysremainnearcompute. Logical byte model not measuredresidency.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=4,logical_checkpoint_replays=0,
        conditioner_policy='Authenticated70/72byte-identical clock0single0input plus72sourceNormcache, historicalprefix/conditioning/normworkpaid; no newprefix orencoder.',
        extra={'purpose':'Sparse72localreplay seal+exit;dense70localreplay seal+exit;commonP FP64signedwriteanatomy;thenarchivedcomparison. Fixedquarter,no downstreamconsumer/image/fit/sweep.',
            'modelLoadScope':'4single0parameters245MB peractor;analysisloads75MBoriginalout only.5affines2imageSDPA10momentproducts2FP64downcontractions total389771427840MAC;gating/norm/energy/rank/hash/copy/I/Oadditional.'})
    r={'schema':'tce-matched-single-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','diagnosticDtype':'float64','batch_size':1,'seq_len':1536,'height':512,'width':512,'needs':{'cuda':True},'timeoutSeconds':360,
        'prospectiveAffineMACs':{'sparse':123211874304,'dense':188441690112},'prospectiveMatrixMACs':389771427840,'selectedChannels':2304,
        'protocol':'Exactfrozen72sparse/70densemoment localoperators on common1536rows, original4parameters, passiveactualoperands. Sparse sealbeforedensefullU, bothsealsbeforearchivedexits. Ownfixedquarter unchanged. DenseadmittedP FP64full/selecteddown, signedomission/diag/crosschannel/textimage/gatedenergies, delta=-gate*O+Rfinite; no oracleselector/downstream/image/fit/sweep/zero-neuralclaim.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r);save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':r['requestSHA256'],'parents':{n:v['jobID'] for n,v in parents.items()}}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    cmd=command()[:-5]
    for name in WHEELS:cmd+=['--with','./'+name]
    cmd+=['python','-u','matched_single_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='Matched clock0single0:sparse72seal+exit,dense70seal+exit,FP64commonP signedomission thenarchivedcompare.5affines2imageSDPA10momentproducts2FP64down=389.771BMAC,4params245MB,nomodelprefix/head/VAE/image/oracleselector. Actualproducts capturedfromoutinputs, finite residualnotcalledrounding. Serialchildren,6144RAM3072VRAM4CPU2GBdisk180expected360cap,noretries. Norm/energy/rank/copies/hashpaid;strictzero-neuralopen.'
    if len(note)>500:raise ValueError('Queue note too long')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-matched-single-write-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','diagnosticDtype','batch_size','seq_len','height','width','scheduling_contract']},
        ram_mb=6144,vram_mb=3072,cpu_threads=4,disk_gb=2,est_wall_s=180,timeout_s=360,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);out=root/'response-result';out.mkdir()
    for mode in ['sparse','dense']:subprocess.run([sys.executable,'-u',str(root/'matched_single_worker.py'),str(root),mode],check=True,cwd=root)
    cases={mode:read(out/('replay-'+mode+'.json')) for mode in ['sparse','dense']};executions={mode:read(out/('execution-'+mode+'.json')) for mode in cases}
    if executions['dense']['precedingSparseSealSHA256']!=sha(out/'seal-sparse.json'):raise ValueError('Dense did not follow the sealed sparse replay')
    save(out/'candidate.json',{'fields':{mode+'_'+name:item for mode,case in cases.items() for name,item in case['fields'].items()},'replays':cases,'requestSHA256':r['requestSHA256']})
    save(out/'execution.json',{'replays':executions,'requestSHA256':r['requestSHA256']})
    save(out/'candidate-seal.json',{'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json'),'sparseSealSHA256':sha(out/'seal-sparse.json'),'denseSealSHA256':sha(out/'seal-dense.json')})
    print('MATCHED_BOTH_ACTORS_SEALED',sha(out/'candidate-seal.json'),flush=True)
    subprocess.run([sys.executable,'-u',str(root/'matched_single_analysis.py'),str(root)],check=True,cwd=root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
