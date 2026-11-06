"""Finite saved-array diagnostic; never fits a representation or runs a consumer."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

from encoder_atoms import np,read,save,sha,identity,floats,words,guard
from attention_bank import geometry,packet
from attention_bank_study import loaded,verify_request
from selector_objective_study import strata,describe
from tail_factorial import ordered_support
from tail_factorial_study import energy
from rank_one_omission import compare,TERMS,SCALARS

HERE=Path(__file__).resolve().parent


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,j,r=checked_collection(parent,current_workers=False);api=Api();live=api.json('GET','/api/jobs/'+c['jobID']);ancestor=api.json('GET','/api/jobs/'+r['parentJobID'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody'] or live['config']['requestSHA256']!=r['requestSHA256']:raise ValueError('Rank-one parent differs')
    if ancestor['state']!='succeeded' or ancestor['config']['requestSHA256']!=r['parentRequestSHA256']:raise ValueError('Full-quadratic ancestor differs')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir();old=read(parent/'result/source.json');seal=read(parent/'result/prediction-seal.json');base=j['result']['diagnostics']['cwd']
    source={k:old[k] for k in ['asset','ports','geometry','pairs','parentBanks','parentControls','completeBankRoot']};source.update(basis=seal['basis'],rankBanks=seal['rows'],parentSeal={'path':base+'/predictions/prediction-seal.json','sha256':sha(parent/'result/prediction-seal.json')},roots=[base,ancestor['result']['diagnostics']['cwd'],old['completeBankRoot']])
    save(p/'source.json',source)
    for name in sorted({name for name in r['sourceHashes'] if name.endswith('.py')}|{'rank_one_omission.py','rank_one_omission_study.py'}):shutil.copyfile(HERE/name,p/name)
    request={'schema':'tce-rank-one-omission-request-v1','parentJobID':c['jobID'],'parentRequestSHA256':r['requestSHA256'],'fullQuadraticJobID':r['parentJobID'],'fullQuadraticRequestSHA256':r['parentRequestSHA256'],
        'scope':'Retrospective samepair/savedoperands, frozen rank1/source/routes/currenttop4. Factor omittedscore into signedlinear/nonnegativequadratic mass and centeredvalue transport. No fitting/newrank/rawprompt/image claim.',
        'law':'epsilon=t-ttilde; f-g=epsilon*(1+ttilde)+epsilon^2/2. Wholehead correction mass+(fulltailweight/Kf)*Ep[(L+Q)*(V-Hg)]. Separate contractionarithmetic, oldfiniteprofile and BF16defects; actualcausaltailenergy ratios neednot be <=1.',
        'observables':'All32heads/512queries percase and paired effects, all/active/padding/before8/at8/after8. Signedterm/crossenergies and alignment with initial rank1headerror, finite/undefined distributions. Old receivingstate unchanged/inherited, no repairedconsumer or downstreamattribution claim.',
        'guardPolicy':'Neural-framework imports/originalcheckpoint access/network prohibited by worker guard. Saved arrays only; zero original projections, exacthead references, encoder/denoiser/VAE/consumer evaluations. Analysis seal is retrospective, not a heldout prediction seal.',
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':600}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request);save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']});return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    verify_request(root/'payload',r);cmd=command()[:-5]+['python','-u','rank_one_omission_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-rank-one-omission-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512,'heads':32},timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='Savedonly CPU1:2retainedportsx32headsx512queries/27causalkeys; fixed rank1/fullquadratic commonprofile factor L=epsilon*(1+ttilde),Q=epsilon^2/2 into mass/centeredvalue corrections. Signedrawvectors/canonicalbanks staynearcompute, compactallrow reports. No originalweights/learnedprojections/exacthead/consumer/model calls, no rankfit/sweep. Hostlocal frozen parents,defaultadmission,expectedunder2min,finite600s,no retry; independent work while neighboringrowsign campaign qualifies.')
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def analyze(port,m,asset,basis,frozen,oldrank,oldfull):
    d,heads,groups=geometry(m);hpg=heads//groups;n=len(port['mask']);full=np.zeros((n,heads*d));rank=full.copy();fields={k:np.full((n,heads),np.nan) for k in SCALARS};terms={k:np.zeros_like(full) for k in TERMS}
    for g in range(groups):
        first=g*hpg;p=packet(port,m,first);U=floats(p['k']).astype(float)/d**.25-asset['keyCenters'][g];V=floats(p['v']).astype(float);ell=asset['landmarks'][first:first+hpg];nr=ell.shape[1];scores=np.full((hpg,nr,n),np.nan);z=np.full(n,np.nan)
        for j in np.flatnonzero(port['mask']):scores[:,:,j]=(ell.reshape(-1,d)@U[j]).reshape(hpg,nr);z[j]=float(basis['b'][g]@U[j])
        for h in range(first,first+hpg):
            x=floats(packet(port,m,h)['q']).astype(float)/d**.25;span=slice(h*d,(h+1)*d)
            for i in range(n):
                allowed=np.flatnonzero(port['mask'][:i+1]);fields['hasTail'][i,h]=0
                if not len(allowed):continue
                route=int(frozen['nearestIndex'][i,h]);selected=ordered_support(frozen['supportPositions'][i,h],allowed,4);delta=x[i]-ell[h-first,route]
                f,q,t,s=compare(scores[h-first,route,allowed],U[allowed],V[allowed],z[allowed],selected,delta,float(basis['a'][g]@delta));full[i,span]=f;rank[i,span]=q
                for k,v in t.items():terms[k][i,span]=v
                for k,v in s.items():fields[k][i,h]=v
    terms['legacyFiniteProfile']=oldfull['raw']-oldrank['raw']-(full-rank);terms['rounding']=floats(oldfull['rounded']).astype(float)-floats(oldrank['rounded']).astype(float)-(oldfull['raw']-oldrank['raw'])
    return {'canonicalFull':full,'canonicalRank':rank,**terms,**fields}


def score(label,correction,baseline,arrays,masks,d,heads):
    rows=[];correction=correction.reshape(-1,heads,d);baseline=baseline.reshape(-1,heads,d);terms={k:arrays[k].reshape(-1,heads,d) for k in TERMS}
    for region,use in masks:
        if not use.any():continue
        for h in [None,*range(heads)]:
            part=lambda a:a[use] if h is None else a[use,h];err=part(baseline);change=part(correction);e=float(np.mean(err*err));new=float(np.mean((err+change)**2));cor=float(np.mean(change*change));alignment={k:float(2*np.mean(err*part(v))) for k,v in terms.items()}
            rows.append({**label,'region':region,'queryHead':h,**energy(change,{k:part(v) for k,v in terms.items()}),'initialRankErrorEnergy':e,'fullQuadraticErrorEnergy':new,'correctionEnergyOverInitialError':cor/e if e else None,
                'signedInitialErrorAlignment':alignment,'signedAlignmentOverInitialError':{k:v/e if e else None for k,v in alignment.items()},'errorEnergyChange':new-e,'errorChangeIdentityDefect':new-e-sum(alignment.values())-cor})
    return rows


def execute(root):
    guard();root=Path(root).resolve();r=read(root/'request.json');verify_request(root,r);start=time.perf_counter();cpu=time.process_time();source=read(root/'source.json');out=root/'response-result';out.mkdir(exist_ok=False);work=root/'analysis-arrays';work.mkdir()
    if sha(source['parentSeal']['path'])!=source['parentSeal']['sha256']:raise ValueError('Parent seal changed')
    m=source['geometry'];d,heads,_=geometry(m);asset=loaded(source['asset']);basis=loaded(source['basis']);rows=[];checks=[];descriptions=[];scores=[];cases={};masks={};timings=[];observations=0
    with (out/'observations.jsonl').open('x') as stream:
        for item in source['ports']:
            name=item['id'];tick=time.perf_counter();port=loaded(item);oldrank=loaded(next(x for x in source['rankBanks'] if x['id']==name));oldfull=loaded(next(x for x in source['parentBanks'] if x['id']==name));exact=loaded(next(x for x in source['parentControls'] if x['id']==name))
            if not np.array_equal(oldrank['supportPositions'],oldfull['supportPositions']) or not np.array_equal(oldrank['nearestIndex'],oldfull['nearestIndex']):raise ValueError('Frozen supports/routes differ')
            a=analyze(port,m,asset,basis,oldrank,oldrank,oldfull);path=work/(name+'-omission.npz');np.savez(path,**a);rows.append({'id':name,'path':str(path),'sha256':sha(path)})
            checks.append({'id':name,'canonicalRankVsParentRawMaxAbs':float(np.max(np.abs(a['canonicalRank']-oldrank['raw']))),'canonicalFullVsParentRawMaxAbs':float(np.max(np.abs(a['canonicalFull']-oldfull['raw']))),
                'canonicalRankVsParentBF16Mismatches':int(np.count_nonzero(words(a['canonicalRank'])!=oldrank['rounded'])),'canonicalFullVsParentBF16Mismatches':int(np.count_nonzero(words(a['canonicalFull'])!=oldfull['rounded']))})
            correction=floats(oldfull['rounded']).astype(float)-floats(oldrank['rounded']).astype(float);baseline=floats(oldrank['rounded']).astype(float)-floats(exact['11__rounded']).astype(float);mask=port['mask'].astype(bool);masks[name]=mask;cases[name]=(a,correction,baseline);sm=strata(mask);label={'id':name,'kind':'endpoint'};scores+=score(label,correction,baseline,a,sm,d,heads)
            for key in SCALARS:
                for region,use in sm:
                    if use.any():
                        for h in [None,*range(heads)]:descriptions.append({**label,'field':key,'region':region,'queryHead':h,**describe(a[key][use] if h is None else a[key][use,h])})
            for i in range(len(mask)):
                for h in range(heads):
                    values={k:float(a[k][i,h]) if np.isfinite(a[k][i,h]) else None for k in SCALARS};stream.write(json.dumps({**label,'position':i,'queryHead':h,**values},allow_nan=False)+'\n');observations+=1
            timings.append({'id':name,'analysisSerializationAndSummariesSeconds':time.perf_counter()-tick});print('RANK_OMISSION_CASE_DONE',name,flush=True)
        left,right=source['pairs'][0];aa,ac,ae=cases[left];ba,bc,be=cases[right];parts={k:ba[k]-aa[k] for k in TERMS};correction=bc-ac;baseline=be-ae;label={'a':left,'b':right,'kind':'paired'};scores+=score(label,correction,baseline,parts,strata(masks[left],masks[right]),d,heads)
        for i in range(len(baseline)):
            for h in range(heads):
                span=slice(h*d,(h+1)*d);terms={k:v[i,span] for k,v in parts.items()};stream.write(json.dumps({**label,'position':i,'queryHead':h,**energy(correction[i,span],terms)},allow_nan=False)+'\n');observations+=1
    seal={'schema':'tce-rank-one-omission-analysis-seal-v1','requestSHA256':r['requestSHA256'],'rows':rows,'sourceSHA256':sha(root/'source.json'),'scope':'Retrospective saved-array analysis, not a target-free prediction seal.'};save(out/'analysis-seal.json',seal)
    for name in ['source.json','request.json']:shutil.copyfile(root/name,out/name)
    save(out/'scalar-diagnostics.json',{'rows':descriptions,'scope':'Dependent causal-tail score/mass/closure descriptions. No rank fitting, consumer attribution or scientific accuracy gate; zero denominators undefined.'})
    save(out/'report.json',{'schema':'tce-rank-one-omission-result-v1','requestSHA256':r['requestSHA256'],'analysisSealSHA256':sha(out/'analysis-seal.json'),'scope':r['scope'],'checks':checks,'scores':scores,'timings':timings,'workerSeconds':time.perf_counter()-start,'cpuSeconds':time.process_time()-cpu,'observations':observations,
        'work':{'newNeuralFrameworkCalls':0,'originalParameterReads':0,'originalDenseProjections':0,'newExactHeadReferences':0,'newOriginalConsumers':0,'rankFits':0},'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed','goalStatus':'Active. Explains one frozen component representation; no new arbitraryprompt/encoder/image serving.'});print('RANK_OMISSION_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':v=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(json.dumps(v))
