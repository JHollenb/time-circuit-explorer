"""Finite-feedback custody and public scalar audit; tensors stay on Beast."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
from template_cofactor_program import read,save,sha
from cauchy_writer_study import OWNED,PARENTS,PUB_SHA,FAMILY_SHA,WORK,identity,verify
from cauchy_writer_math import POLICY,TERMS,ETA
from coupled_normalization import SOURCE_POLICY
from publish_writer_direction import fresh_bytes


def frame_check(f,request_sha):
    body={k:f[k] for k in ('schema','slots','parent_frame_id','act','implementation','program_fingerprint','metadata')}
    if identity(body)!=f['frame_id'] or f['program_fingerprint']!=request_sha:raise ValueError('Canonical Frame identity differs')


def transaction(row,request_sha):
    a,b,s,e=[row[k] for k in ('input','output','spec','evidence')]
    frame_check(a,request_sha);frame_check(b,request_sha)
    if b['parent_frame_id']!=a['frame_id'] or b['act']!=s['identity'] or e['input_frame_id']!=a['frame_id'] or e['output_frame_id']!=b['frame_id'] or e['act']!=s['identity'] or row['inputPayloadBefore']!=row['inputPayloadAfter']:raise ValueError('Transaction lineage/mutation differs')
    for name,digest in row['inputPayloadBefore'].items():
        if a['slots'][name]!=b['slots'][name]:raise ValueError('Immutable input slot differs')
    return a,b,s


def publish(root,mechanics,destination):
    from mrun.client.api import Api
    root=Path(root).resolve();dest=Path(destination).resolve();req=read(root/'request.json');verify(root/'payload',req);tests=read(mechanics)
    if (tests['testsRun'],tests['failures'],tests['errors'])!=(8,0,0) or set(tests['sourceHashes'])!=set(OWNED):raise ValueError('Eight source-bound mechanics absent')
    for name in OWNED:
        if sha(Path(__file__).parent/name)!=req['sourceHashes'][name] or tests['sourceHashes'][name]!=req['sourceHashes'][name]:raise ValueError('Tested/current/staged source differs')
    job=read(root/'scheduler-job.json');live=Api().json('GET','/api/jobs/'+job['job_id']);collection=read(root/'collection.json');custody=job['payload_custody'];phases=[custody[k] for k in ('declared','sealed','executed')]
    if job['state']!='succeeded' or job['result']['returncode']!=0 or live['state']!='succeeded' or live['result']!=job['result'] or live['payload_custody']!=custody or collection['custody']!=custody or collection['result']!=job['result'] or collection['jobID']!=job['job_id'] or not custody['required'] or not live.get('admission') or len({(x['sha256'],x['size_bytes']) for x in phases})!=1 or phases[-1]['host']!='beast':raise ValueError('Successful guarded113 custody differs')
    if any(x['config']['requestSHA256']!=req['requestSHA256'] for x in (live,job)):raise ValueError('Actual request binding differs')
    result=root/'result';actual={str(p.relative_to(result)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in result.rglob('*') if p.is_file()}
    if actual!=collection['inventory']['files']:raise ValueError('Compact collection differs')
    parent=read(result/'parent-publication.json')
    if sha(result/'parent-publication.json')!=PUB_SHA or sha(result/'family-plan.md')!=FAMILY_SHA or req['physicalWork']!=WORK or req['policy']!=POLICY or req['eta']!=ETA:raise ValueError('Frozen authority differs')
    for name in PARENTS:
        if name!='publication.json' and sha(result/('parent-'+name))!=parent['files'][name]['sha256']:raise ValueError('Published parent member differs')
    for name in req['retainedMetadata']:
        if name=='request.json':
            if read(result/name)!=req:raise ValueError('Collected request differs')
        elif sha(result/name)!=req['sourceHashes'][name]:raise ValueError('Staged metadata differs')
    report=read(result/'report.json');selection=read(result/'selection.json');programs=read(result/'programs.json');arrays=read(result/'array-manifest.json');typed=read(result/'typed-proposal.json');norm=read(result/'typed-normalization.json');admission=read(result/'admission.json');seal=read(result/'analysis-seal.json');release=read(result/'release-seal.json');serving=read(result/'final-serving.json')
    old_request=read(result/'parent-request.json');old_catalog=read(result/'parent-programs.json');old_report=read(result/'parent-report.json')
    for key in ('cases','discovery','transfer','novelNoTarget','numericalProfile','torchVersion','rmsNormEpsilon','lexicalDirectory'):
        if req[key]!=old_request[key]:raise ValueError('Frozen parent case/profile authority differs')
    if req['parent']['program']!=old_catalog['accepted'] or req['fields']['normGamma']['payloadSHA256']!=old_request['gammaParameter']['payloadSHA256']:raise ValueError('Parent/gamma authority differs')
    members={'selection.json','release-seal.json','observations.jsonl.gz','admission.json','typed-proposal.json','typed-normalization.json','programs.json','array-manifest.json','final-serving.json','final-seal.json'}
    if set(actual)!=set(req['retainedMetadata'])|members|{'analysis-seal.json','report.json'} or set(seal)!=members or any(sha(result/n)!=h for n,h in seal.items()) or report['analysisSealSHA256']!=sha(result/'analysis-seal.json') or report['releaseSealSHA256']!=sha(result/'release-seal.json'):raise ValueError('Complete result/seals differ')
    if release['selectionSHA256']!=sha(result/'selection.json') or release['servingSealSHA256']!=sha(result/'final-seal.json') or report['physicalWork']!=WORK or report['policy']!=POLICY or report['requestSHA256']!=req['requestSHA256'] or report['numericalProfile']!=req['numericalProfile'] or report['torchVersion']!=req['torchVersion'] or report['sourcePolicy']!=SOURCE_POLICY:raise ValueError('Finite source policy differs')
    if report['selection']!=selection or selection['selected']!=('candidate' if selection['candidate']['score']<selection['parent']['score'] else 'parent') or report['normalizationUnits']!=len(norm['rows']) or len(norm['rows'])>6 or report['originalLinears']!=0 or report['nominalMatrixMACs']!=0 or report['images']!=0:raise ValueError('Continuation/work differs')
    model=report['proposal']
    if selection['parent']!=model['parent'] or selection['parent']!=old_report['initialObjective'] or selection['candidate']['nativeEnergies']!=selection['parent']['nativeEnergies']:raise ValueError('Original parent/four fixed denominators differ')
    for key in ('gDotQ','negativePreconditionedGradientEnergy','signIdentityRegrouping','curvature','rowRelativeTau','eta','alpha','modelPolynomialScore','modelRegroupingDifference','predictedReduction','directionRMS'):
        if not math.isfinite(model[key]):raise ValueError('Nonfinite proposal scalar')
    if model['eta']!=ETA or model['curvature']!=math.fsum(model['curvatureTerms'].values()) or set(model['curvatureTerms'])!=set(TERMS) or any(not math.isfinite(x) or x<0 for x in model['curvatureTerms'].values()):raise ValueError('Curvature scalar law differs')
    if model['alpha']==0:
        if model['curvature']!=0 or model['rowRelativeTau']!=0 or model['alphaModel'] is not None or model['alphaTrust'] is not None or programs['candidate']['files']!=programs['parent']['files']:raise ValueError('Exact zero-ray alias differs')
    elif model['alphaModel']!=-model['gDotQ']/model['curvature'] or model['alphaTrust']!=ETA/model['rowRelativeTau'] or model['alpha']!=min(model['alphaModel'],model['alphaTrust']):raise ValueError('One analytic step scalar law differs')
    polynomial=selection['parent']['score']+model['alpha']*model['gDotQ']/2+model['alpha']*model['alpha']*model['curvature']/4
    if model['modelPolynomialScore']!=polynomial or model['predictedReduction']!=selection['parent']['score']-polynomial or model['modelRegroupingDifference']!=polynomial-model['linearResidualSquareObjective']['score'] or selection['actualReduction']!=selection['parent']['score']-selection['candidate']['score']:raise ValueError('Model/actual scalar laws differ')
    for score in (selection['parent'],selection['candidate']):
        if set(score['terms'])!=set(TERMS) or score['score']!=math.fsum(score['terms'].values())/4 or any(not math.isfinite(v) or v<0 for v in score['terms'].values()):raise ValueError('Finite loss terms differ')
    old_norm=read(result/'parent-typed-normalization.json');old_fields=read(result/'parent-array-manifest.json');old_rows={r['state']:r for r in old_norm['rows']}
    for case in req['discovery']+req['transfer']:
        for method in ('parent','target'):
            row=old_rows[method+'-'+case]
            if row['spec']['numerical_contract']!=SOURCE_POLICY or row['inputPayloadBefore']['normWeight']!=req['fields']['normGamma']['payloadSHA256']:raise ValueError('Reused actual normalized source differs')
            for port,key in [('H','inputField'),('Z','outputField')]:
                item=req['fields'][method+'-'+case+'-'+port]
                if item!=old_fields[method+'-'+case+'-'+port] or item!=row[key]:raise ValueError('Reused typed H/Z field join differs')
    old_fit=read(result/'parent-typed-proposals.json')[0]
    for name in ('h','z','target','zt','features','weights','edges','pairWeights','gamma','scaleVector','gradient'):
        item=(old_fit['outputFields'] if name=='gradient' else old_fit['inputFields'])[name]
        if req['fields'][name]!=item:raise ValueError('Source-qualified proposal field join differs')
    a,b,s=transaction(typed,req['requestSHA256']);expected_abis={'r0':{'shape':[14,2560],'dtype':'float64'},'r1':{'shape':[9,2560],'dtype':'float64'},'q':{'shape':[2040,2560],'dtype':'float64'},'Jq':{'shape':[2040,2560],'dtype':'float64'}}
    if s['identity']!='cauchy_writer.joint_constants:v1' or s['numerical_contract']!=POLICY or s['write_abis']!=expected_abis or set(s['reads'])!=set(typed['inputABIs']) or set(s['writes'])!=set(expected_abis):raise ValueError('Proposal ABI differs')
    input_shapes={k:[2040,2560] for k in ('h','z','target','zt','gradient')};input_shapes.update({'features':[2040,155],'weights':[2040],'edges':[1017,2],'pairWeights':[1017],'gamma':[2560],'scaleVector':[4],'labels0':[2040],'labels1':[2040]})
    if set(typed['inputABIs'])!=set(input_shapes) or set(a['slots'])!=set(input_shapes) or set(b['slots'])!=set(input_shapes)|set(expected_abis):raise ValueError('Exact13-input/output domain differs')
    for name,item in typed['inputABIs'].items():
        if item['shape']!=input_shapes[name] or item['dtype']!=('torch.int64' if name in ('edges','labels0','labels1') else 'torch.float64'):raise ValueError('Exact input shape/dtype differs')
        if a['slots'][name]['fingerprint']!=identity({'dtype':item['dtype'],'shape':item['shape'],'bytes':item['payloadSHA256']}) or item['payloadSHA256']!=typed['inputPayloadBefore'][name]:raise ValueError('Input fingerprint differs')
        if name in req['fields'] and item!={k:req['fields'][name][k] for k in ('shape','dtype','payloadSHA256')}:raise ValueError('Original proposal input differs')
    for name,abi in expected_abis.items():
        item=arrays[name]
        if item['shape']!=abi['shape'] or item['dtype']!='torch.float64' or item['storedDtype']!='<f8' or b['slots'][name]['fingerprint']!=identity({'dtype':item['dtype'],'shape':item['shape'],'bytes':item['payloadSHA256']}):raise ValueError('Proposal output fingerprint/ABI differs')
    if norm['parameterPayloadBefore']!=norm['parameterPayloadAfter'] or norm['parameterPayloadBefore']!=req['fields']['normGamma']['payloadSHA256']:raise ValueError('Gamma closure differs')
    cases=req['discovery']+req['transfer']
    aliases=report['normalizationAliases'];physical=[r['case'] for r in norm['rows']];weight_parent=norm['weightParent'];frame_check(weight_parent,req['requestSHA256'])
    expected_arrays=set(expected_abis)|{'admission-diagnostics'}|{'candidate-'+case+'-H' for case in cases}|{'candidate-'+case+'-Z' for case in physical}
    if set(arrays)!=expected_arrays:raise ValueError('Exact retained array domain differs')
    for name,item in arrays.items():
        bf16=name.startswith('candidate-');shape=[1,512,2560] if bf16 else [2040,10] if name=='admission-diagnostics' else expected_abis[name]['shape']
        if item['shape']!=shape or item['dtype']!=('torch.bfloat16' if bf16 else 'torch.float64') or item['storedDtype']!=('<u2' if bf16 else '<f8') or item['format']!=('raw' if bf16 else 'npy') or item['bytes']!=math.prod(shape)*(2 if bf16 else 8)+(0 if bf16 else 128):raise ValueError('Exact retained array ABI differs')
    if set(weight_parent['slots'])!={'normWeight'} or weight_parent['slots']['normWeight']['fingerprint']!=identity({'dtype':'torch.bfloat16','shape':[2560],'bytes':norm['parameterPayloadBefore']}):raise ValueError('Weight parent ABI differs')
    if set(norm['fields'])!=set(cases) or set(physical)!=set(cases)-set(aliases) or len(physical)!=len(set(physical)):raise ValueError('Normalization physical/alias domain differs')
    for case,fields in norm['fields'].items():
        if fields['H']!=arrays['candidate-'+case+'-H']:raise ValueError('Candidate H field differs')
        if case in aliases:
            if aliases[case]!='exact-parent-H' or fields.get('alias')!=aliases[case] or fields['H']['payloadSHA256']!=req['fields']['parent-'+case+'-H']['payloadSHA256'] or fields['Z']!=req['fields']['parent-'+case+'-Z']:raise ValueError('Byte-identical normalization alias differs')
        elif fields['Z']!=arrays['candidate-'+case+'-Z'] or 'alias' in fields:raise ValueError('Physical normalization output descriptor differs')
    for row in norm['rows']:
        ni,no,ns=transaction(row,req['requestSHA256'])
        if ns['identity']!='cauchy_writer.source_operand:v1' or ns['numerical_contract']!=SOURCE_POLICY or ns['write_abis']!={'Z':{'shape':[1,512,2560],'dtype':'bfloat16'}} or set(ns['reads'])!={'H1','normWeight'} or ns['writes']!=['Z'] or set(ni['slots'])!={'H1','normWeight'} or set(no['slots'])!={'H1','normWeight','Z'} or row['inputPayloadBefore']['H1']!=row['inputField']['payloadSHA256'] or row['inputPayloadBefore']['normWeight']!=norm['parameterPayloadBefore'] or row['outputField']!=arrays['candidate-'+row['case']+'-Z']:raise ValueError('Finite candidate input/output binding differs')
        if ni['parent_frame_id']!=weight_parent['frame_id'] or ni['metadata']!={'case':row['case'],'programSHA256':programs['candidate']['programSHA256']} or ni['act']!='cauchy_writer.install:v1' or ni['slots']['normWeight']!=weight_parent['slots']['normWeight'] or ni['slots']['H1']['fingerprint']!=identity({'dtype':'torch.bfloat16','shape':[1,512,2560],'bytes':row['inputField']['payloadSHA256']}) or row['inputField']!=norm['fields'][row['case']]['H'] or row['outputField']!=norm['fields'][row['case']]['Z']:raise ValueError('Normalization install ancestry/input ABI differs')
        if no['slots']['Z']['fingerprint']!=identity({'dtype':'torch.bfloat16','shape':[1,512,2560],'bytes':row['outputField']['payloadSHA256']}):raise ValueError('Actual normalized output fingerprint differs')
    for name,program in programs.items():
        old=req['parent']['program']['program']
        if any(program['program'][k]!=old[k] for k in old if k!='dictionaries'):raise ValueError('Fixed routes/feature/prefix/budget changed')
        if set(program['files'])!={'program.json','prefix.bf16','slot0.f32','slot1.f32'} or program['files']['program.json']['sha256']!=program['programSHA256'] or program['files']['prefix.bf16']['sha256']!=old['prefixSHA256']:raise ValueError('Program file/prefix identity differs')
        for item in program['program']['dictionaries']:
            if program['files'][item['path']]!={k:item[k] for k in ('sha256','bytes')}:raise ValueError('Dictionary catalog/file identity differs')
    if programs['accepted']['programSHA256']!=programs[selection['selected']]['programSHA256'] or selection['acceptedProgramSHA256']!=programs['accepted']['programSHA256'] or release['candidateProgramSHA256']!=programs['candidate']['programSHA256']:raise ValueError('Program selection/restore differs')
    if selection['parentProgramSHA256']!=req['parent']['program']['programSHA256'] or selection['parentProgramSHA256']!=programs['parent']['programSHA256'] or selection['candidateProgramSHA256']!=programs['candidate']['programSHA256'] or release['acceptedProgramSHA256']!=programs['accepted']['programSHA256'] or selection['restoredFiles']!={k:v['sha256'] for k,v in programs['accepted']['files'].items()} or programs['accepted']['files']!=programs[selection['selected']]['files']:raise ValueError('Exact parent/candidate/restored branch identity differs')
    final_seal=read(result/'final-seal.json');old_packets={r['id']:r['packet'] for r in read(result/'parent-final-serving.json')['rows']}
    if final_seal['receiptSHA256']!=sha(result/'final-serving.json') or [r['id'] for r in serving['rows']]!=[r['id'] for r in req['cases']] or final_seal['programs']!=[{'name':k,'path':programs[k]['directory'],'sha256':programs[k]['programSHA256']} for k in ('parent','candidate','accepted')]:raise ValueError('Final serving receipt/seven-case/program seal differs')
    if serving['loadedNumericalRuntimeRoots'] or serving['newNeuralEvaluations']!=0 or len(serving['rows'])!=7:raise ValueError('Fresh owned serving differs')
    label_hashes=[hashlib.sha256(),hashlib.sha256()]
    remote=dict(req['remoteInputs']);npys={x['path']:x for x in req['fields'].values() if x['format']=='npy'}
    for item in arrays.values():remote[item['path']]={k:item[k] for k in ('sha256','bytes')};npys.update({item['path']:item} if item['format']=='npy' else {})
    for program in programs.values():
        for name,item in program['files'].items():remote[str(Path(program['directory'])/name)]=item
    for row in serving['rows']:
        predictions={x['method']:x for x in row['predictions']}
        if row['packet']!=old_packets[row['id']] or any(x['programSHA256']!=programs[x['method']]['programSHA256'] for x in row['predictions']):raise ValueError('Fresh raw packet/program binding differs')
        if set(predictions)!={'parent','candidate','accepted'} or predictions['accepted']['sha256']!=predictions[selection['selected']]['sha256']:raise ValueError('Serving accepted alias differs')
        for item in [row['H0'],*row['predictions']]:remote[item['path']]={k:item[k] for k in ('sha256','bytes')}
        if row['id'] in cases:
            if predictions['candidate']['sha256']!=norm['fields'][row['id']]['H']['sha256'] or predictions['parent']['sha256']!=req['fields']['parent-'+row['id']+'-H']['sha256']:raise ValueError('Fresh candidate/parent and finite consumer H byte join differs')
        if row['id'] in req['discovery']:
            for leaf in predictions['parent']['leaves']:
                if len(leaf)!=2:raise ValueError('Two-slot serving assignment differs')
                for slot,value in enumerate(leaf):label_hashes[slot].update(struct.pack('<q',value))
            if any(predictions[name]['leaves']!=predictions['parent']['leaves'] for name in ('candidate','accepted')):raise ValueError('Fresh fixed-route assignments differ')
    if any(h.hexdigest()!=typed['inputPayloadBefore']['labels'+str(slot)] for slot,h in enumerate(label_hashes)):raise ValueError('Fresh serving leaves/typed discovery labels byte join differs')
    if len(admission['rows'])!=2040 or len(admission['columns'])!=10:raise ValueError('Admission scalar domain differs')
    packed=hashlib.sha256();sums=[[] for _ in range(10)]
    for i,row in enumerate(admission['rows']):
        if row['case']!=req['discovery'][i//510] or row['position']!=i%510+2 or len(row['values'])!=10 or any(not math.isfinite(x) for x in row['values']):raise ValueError('Admission row ordering/nonfinite differs')
        packed.update(struct.pack('<10d',*row['values']))
        for at,x in enumerate(row['values']):sums[at].append(x)
    if packed.hexdigest()!=arrays['admission-diagnostics']['payloadSHA256']:raise ValueError('Typed admission field and public scalar packing differ')
    reaggregated={name:{'producer':admission['globalSums'][name],'fsum':math.fsum(values),'difference':admission['globalSums'][name]-math.fsum(values)} for name,values in zip(admission['columns'],sums)}
    with gzip.open(result/'observations.jsonl.gz','rt') as stream:rows=[json.loads(line) for line in stream]
    if len(rows)!=18432 or report['observationRows']!=len(rows) or len(report['summaries'])!=96 or len(report['pairs'])!=48:raise ValueError('Complete endpoint/pair scalar coverage differs')
    packets={r['id']:r['packet'] for r in serving['rows']};expected=[];metrics={'errorRMS','normalizerRMS','relativeRMS','predictionRMS','targetRMS','cosine'}
    for case in cases:
        for method in ('parent','candidate'):
            for port in ('H','Z'):
                for position in range(512):expected.append({'case':case,'method':method,'port':port,'position':position,'active':bool(packets[case]['mask'][position])})
    for left,right in [('A','B'),('00','10'),('01','11')]:
        for method in ('parent','candidate'):
            for port in ('H','Z'):
                for position in range(512):expected.append({'pair':[left,right],'method':method,'port':port,'position':position,'active':bool(packets[left]['mask'][position]),'sameToken':packets[left]['ids'][position]==packets[right]['ids'][position]})
    for row,tags in zip(rows,expected):
        if {k:v for k,v in row.items() if k not in metrics}!=tags or set(row)!=set(tags)|metrics:raise ValueError('Exact observation row domain/order differs')
        if any(v is not None and not math.isfinite(v) for k,v in row.items() if k in ('errorRMS','normalizerRMS','relativeRMS','predictionRMS','targetRMS','cosine')):raise ValueError('Nonfinite observed metric')
    endpoint={(r['case'],r['method'],r['port'],r['position']):r for r in rows if 'case' in r};paired={(tuple(r['pair']),r['method'],r['port'],r['position']):r for r in rows if 'pair' in r}
    panel_checks=[];panel_expectations=[]
    for case in cases:
        for method in ('parent','candidate'):
            for port in ('H','Z'):
                for label in ('all','learned','active','padding'):
                    positions=[i for i in range(512) if label=='all' or i>=2 and (label=='learned' or bool(packets[case]['mask'][i])==(label=='active'))]
                    panel_expectations.append(({'case':case,'method':method,'port':port,'stratum':label,'rows':len(positions)},[endpoint[case,method,port,i] for i in positions]))
    for left,right in [('A','B'),('00','10'),('01','11')]:
        for method in ('parent','candidate'):
            for port in ('H','Z'):
                for label in ('all','unchangedLearned','unchangedActive','unchangedPadding'):
                    positions=[i for i in range(512) if label=='all' or i>=2 and packets[left]['ids'][i]==packets[right]['ids'][i] and (label=='unchangedLearned' or bool(packets[left]['mask'][i])==(label=='unchangedActive'))]
                    panel_expectations.append(({'pair':[left,right],'method':method,'port':port,'stratum':label,'rows':len(positions)},[paired[(left,right),method,port,i] for i in positions]))
    for producer,(tags,values) in zip(report['summaries']+report['pairs'],panel_expectations):
        if {k:v for k,v in producer.items() if k not in metrics}!=tags or set(producer)!=set(tags)|metrics:raise ValueError('Exact report panel domain differs')
        rebuilt={k:math.sqrt(math.fsum(r[k]**2 for r in values)/len(values)) for k in ('errorRMS','normalizerRMS','predictionRMS','targetRMS')}
        rebuilt['relativeRMS']=rebuilt['errorRMS']/rebuilt['normalizerRMS'] if rebuilt['normalizerRMS'] else None
        dot=math.fsum((r['cosine'] or 0.)*r['predictionRMS']*r['targetRMS'] for r in values)/len(values)
        rebuilt['cosine']=dot/(rebuilt['predictionRMS']*rebuilt['targetRMS']) if rebuilt['predictionRMS']*rebuilt['targetRMS'] else None
        panel_checks.append({'tags':tags,'reconstructed':rebuilt,'differences':{k:None if producer[k] is None or rebuilt[k] is None else producer[k]-rebuilt[k] for k in metrics},'nullPairs':{k:[producer[k] is None,rebuilt[k] is None] for k in metrics}})
    global_terms={};regressions=[]
    for method in ('parent','candidate'):
        parts={k:[] for k in TERMS};strata=[]
        for case in req['discovery']:
            for active in (True,False):
                positions=[i for i in range(2,512) if bool(packets[case]['mask'][i])==active];weight=1/(2*len(req['discovery'])*len(positions))
                for port in ('H','Z'):
                    term=port+'endpoint';value=math.fsum(endpoint[case,method,port,i]['errorRMS']**2*2560*weight for i in positions)/(selection['parent']['nativeEnergies'][term] or 1.)
                    parts[term].append(value);strata.append({'case':case,'stratum':'active' if active else 'padding','term':term,'originalGlobalContribution':value})
        for left,right in [('A','B'),('00','10')]:
            for active in (True,False):
                positions=[i for i in range(2,512) if bool(packets[left]['mask'][i])==active and packets[left]['ids'][i]==packets[right]['ids'][i]];weight=1/(4*len(positions))
                for port in ('H','Z'):
                    term=port+'response';value=math.fsum(paired[(left,right),method,port,i]['errorRMS']**2*2560*weight for i in positions)/(selection['parent']['nativeEnergies'][term] or 1.)
                    parts[term].append(value);strata.append({'pair':[left,right],'stratum':'unchangedActive' if active else 'unchangedPadding','term':term,'originalGlobalContribution':value})
        global_terms[method]={'terms':{k:math.fsum(v) for k,v in parts.items()},'strata':strata,'producerDifferences':{k:selection[method]['terms'][k]-math.fsum(v) for k,v in parts.items()}}
    for case in cases:
        for port in ('H','Z'):
            for active in (True,False):
                values=[endpoint[case,'candidate',port,i]['errorRMS']-endpoint[case,'parent',port,i]['errorRMS'] for i in range(2,512) if bool(packets[case]['mask'][i])==active]
                regressions.append({'case':case,'port':port,'stratum':'active' if active else 'padding','rows':len(values),'improved':sum(v<0 for v in values),'worsened':sum(v>0 for v in values),'tied':sum(v==0 for v in values)})
    def finite_scalars(value):
        if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite published/reconstructed scalar')
        if isinstance(value,dict):
            for v in value.values():finite_scalars(v)
        elif isinstance(value,list):
            for v in value:finite_scalars(v)
    for value in (report,selection,admission,reaggregated,global_terms,regressions,panel_checks):finite_scalars(value)
    fresh=fresh_bytes(remote,npys,{})
    if dest.exists():raise ValueError('Publication already exists')
    dest.mkdir(parents=True)
    for name in actual:shutil.copyfile(result/name,dest/name)
    for path,name in [(root/'scheduler-job.json','scheduler-job.json'),(root/'collection.json','collection.json'),(Path(mechanics),'mechanics.json')]:shutil.copyfile(path,dest/name)
    save(dest/'scalar-reaggregation.json',{'admission':reaggregated,'reportPanels':panel_checks,'originalGlobalWeightObjectives':global_terms,'endpointRegressions':regressions,'scope':'Independent public scalar fsum/typed-payload packing and original-global-weight loss reconstruction from rounded per-row RMS;not vector or finite-normalization replay. RMS resquaring/reduction-order differences remain diagnostics without thresholds.'})
    files={str(p.relative_to(dest)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in dest.rglob('*') if p.is_file()}
    publication={'schema':'tce-cauchy-writer-publication-v1','jobID':job['job_id'],'requestSHA256':req['requestSHA256'],'publisherSHA256':sha(__file__),'files':files,'freshRemoteFiles':fresh,'observationRows':len(rows),'admissionRows':2040,'typedAdmissionPayloadSHA256':packed.hexdigest(),'scope':'Complete compact custody/typed/byte audit and public scalar checks. Vector construction,losses and source-policy evaluations remain source-qualified producer measurements. Arrays remain Beast;no arbitrary-prompt zero-neural image claim.'}
    save(dest/'publication.json',publication);return publication


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('mechanics');parser.add_argument('destination');args=parser.parse_args();p=publish(args.root,args.mechanics,args.destination);print(json.dumps({k:v for k,v in p.items() if k not in ('files','freshRemoteFiles')}))
