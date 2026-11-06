"""Offline attribution of an already sealed transport error, not a new predictor."""
import argparse
import json
from pathlib import Path
import shutil
import time

from encoder_atoms import np,read,save,sha,identity,guard,floats,words
from attention_bank import packet,geometry,head_asset
from attention_bank_study import loaded,regions
from context_statistics_study import exact_head,summary,rowrms


def decompose(packet,asset,indices):
    q,k,v=[floats(packet[key]).astype(float) for key in ['q','k','v']];n,d=q.shape
    x=q/d**.25;u=k/d**.25-asset['keyCenter'];out={key:np.zeros((n,d)) for key in ['H','A','P','V','exact']}
    scalar={key:np.zeros(n) for key in ['beta','logTiltPartition','weightedTMean','tVariance','negativeMass','minimumT','maximumT','smallTRemainderClosure','remainderMass','maximumRemainderMassShare','minimumRemainderCoefficient']}
    key_positions=np.flatnonzero(packet['mask']);coefficients=np.zeros((n,len(key_positions)))
    for i in range(n):
        allowed=np.flatnonzero(packet['mask'][:i+1])
        if not len(allowed):scalar['beta'][i]=1.;continue
        ell=asset['landmarks'][int(indices[i])];U=u[allowed];values=v[allowed];delta=x[i]-ell
        logits=U@ell;top=logits.max();logp=logits-top-np.log(np.exp(logits-top).sum());p=np.exp(logp)
        H=p@values;mu=p@U;t=(U-mu)@delta;centered=values-H
        A=(p*t)@centered;tilted=logp+t;top=tilted.max();logZ=float(top+np.log(np.exp(tilted-top).sum()))
        w=np.exp(tilted-logZ);beta=float(np.exp(-logZ));coefficient=w-beta*p*(1+t)
        # Near zero expm1 avoids subtracting two almost identical weights.
        small=np.abs(t)<=1.;stable=beta*p[small]*(np.expm1(t[small])-t[small]);small_diff=float(np.max(np.abs(coefficient[small]-stable))) if small.any() else 0.
        coefficient[small]=stable;P=(1-beta)*A;V=-(coefficient@centered)
        coefficients[i,np.searchsorted(key_positions,allowed)]=coefficient;mass=float(coefficient.sum())
        scalar['remainderMass'][i]=mass;scalar['maximumRemainderMassShare'][i]=float(coefficient.max()/mass) if mass else 0.;scalar['minimumRemainderCoefficient'][i]=coefficient.min()
        for name,value in [('H',H),('A',A),('P',P),('V',V),('exact',w@values)]:out[name][i]=value
        scalar['beta'][i]=beta;scalar['logTiltPartition'][i]=logZ;scalar['weightedTMean'][i]=p@t;scalar['tVariance'][i]=p@(t*t)
        effective=p*(1+t);scalar['negativeMass'][i]=-effective[effective<0].sum();scalar['minimumT'][i]=t.min();scalar['maximumT'][i]=t.max();scalar['smallTRemainderClosure'][i]=small_diff
    return {**out,**scalar,'normalizedRemainderCoefficients':coefficients}


def one_key_change(base,other,position):
    """Exact baseline-assisted key/payload update; changed-query rows excluded."""
    q,ka,va=[floats(base[k]).astype(float) for k in ['q','k','v']];kb,vb=[floats(other[k]).astype(float) for k in ['k','v']];n,d=q.shape
    key=np.zeros_like(q);value=np.zeros_like(q);valid=np.zeros(n,dtype=bool)
    common=np.array_equal(base['mask'],other['mask']);outside=np.arange(n)!=position
    unchanged=common and np.array_equal(base['k'][outside],other['k'][outside]) and np.array_equal(base['v'][outside],other['v'][outside])
    if not unchanged:return {'key':key,'value':value,'valid':valid,'applicable':False}
    for i in range(position+1,n):
        if not np.array_equal(base['q'][i],other['q'][i]) or not base['mask'][position]:continue
        allowed=np.flatnonzero(base['mask'][:i+1]);rest=allowed[allowed!=position];x=q[i]/np.sqrt(d)
        old_score=x@ka[position];new_score=x@kb[position]
        if len(rest):
            logits=ka[rest]@x;top=logits.max();log_rest=float(top+np.log(np.exp(logits-top).sum()))
            h_rest=np.exp(logits-log_rest)@va[rest]
            pa=float(np.exp(old_score-np.logaddexp(log_rest,old_score)));pb=float(np.exp(new_score-np.logaddexp(log_rest,new_score)))
            key[i]=(pb-pa)*(va[position]-h_rest)
        else:pb=1.
        value[i]=pb*(vb[position]-va[position]);valid[i]=True
    return {'key':key,'value':value,'valid':valid,'applicable':True}


def attribution(error,partition,value,rounding,target,selected):
    use=selected&np.isfinite(error).all(1)&np.isfinite(partition).all(1)&np.isfinite(value).all(1)&np.isfinite(rounding).all(1)
    if not use.any():return {'queries':int(selected.sum()),'validQueries':0}
    E,P,V,Q,T=[z[use] for z in [error,partition,value,rounding,target]];energy=float(np.sum(T*T))
    def normalized(z):return float(np.sum(z*z))/energy if energy else None
    return {'queries':int(selected.sum()),'validQueries':int(use.sum()),'targetRms':float(np.sqrt(np.mean(T*T))),
        'roundedError':summary(T+E,T),'partitionTermRms':float(np.sqrt(np.mean(P*P))),
        'valueRemainderTermRms':float(np.sqrt(np.mean(V*V))),'roundingTermRms':float(np.sqrt(np.mean(Q*Q))),
        'errorEnergyOverTarget':normalized(E),'partitionEnergyOverTarget':normalized(P),'valueRemainderEnergyOverTarget':normalized(V),
        'roundingEnergyOverTarget':normalized(Q),'partitionValueCrossEnergyOverTarget':float(2*np.sum(P*V))/energy if energy else None,
        'smoothRoundingCrossEnergyOverTarget':float(2*np.sum((P+V)*Q))/energy if energy else None,
        'partitionCorrectedOracleError':summary(T+E-P,T),'valueRemainderCorrectedOracleError':summary(T+E-V,T),
        'telescopeMaxAbsClosure':float(np.max(np.abs(E-(P+V+Q)))),
        'oracleScope':'Diagnostic term deletion uses actual query-prefix exponential remainder, not an available compact-state serving rule. Rounding term is retained unchanged in these diagnostic vector corrections.'}


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,j,r=checked_collection(parent);live=Api().json('GET','/api/jobs/'+c['jobID'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody'] or live['config']['requestSHA256']!=r['requestSHA256']:raise ValueError('Complete-bank parent changed')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir();base=j['result']['diagnostics']['cwd'];seal=read(parent/'result/prediction-seal.json');ref=read(parent/'result/reference-receipt.json')
    save(p/'source.json',{'predictionSeal':{'path':base+'/predictions/prediction-seal.json','sha256':sha(parent/'result/prediction-seal.json')},
        'asset':seal['asset'],'predictions':seal['rows'],'references':ref['rows'],'pairs':r['pairs'],
        'geometry':{'headDim':128,'qWidth':4096,'kvWidth':1024}})
    here=Path(__file__).resolve().parent
    for name in ['attention_remainder.py','attention_bank.py','attention_bank_study.py','encoder_atoms.py','context_source_geometry.py',
                 'context_landmarks.py','context_transport.py','context_statistics.py','context_statistics_study.py','context_transport_study.py','transport_consumer.py']:shutil.copyfile(here/name,p/name)
    request={'schema':'tce-attention-remainder-request-v1','parentJobID':c['jobID'],'parentRequestSHA256':r['requestSHA256'],
        'scope':'Saved-only allquery/allhead exact exponential remainder attribution on already scored pair. No new forecast, target fit, original checkpoint or learned projection. Target-assisted diagnostic corrections are not serving laws.',
        'law':'Centered t=delta dot(y-E_l[y]), H=E_l[v], A=E_l[(v-H)t], beta=1/E_l[exp(t)]. Error H+A-h_exact=P+V, P=(1-beta)A, V=-beta E_l[(v-H)(expm1(t)-t)]. Separate BF16 forecast/reference defects telescope rounded error.',
        'numericalPolicy':'Stable logsumexp tilt, normalized remainder coefficients. expm1(t)-t for abs(t)<=1, otherwise w_tilt-beta*p*(1+t); no clipping. Original direct attention separately verifies exact identity. All signed/raw terms retained.',
        'observables':'Paired vectors/energies/cross terms and target-assisted single-term removals, active/padding/maskchanged plus before8/at8/after8 clocks; full endpoint/head traces and normalized per-key remainder coefficients. Verify sole changed token8/identical later queries/routes; exact baseline-assisted one-key key/payload split on applicable later rows. Retrospective control, not serving.',
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'backend':'numpy-fp64','device':'cpu','cpuThreads':1}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    from attention_bank_study import verify_request
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    verify_request(root/'payload',r);cmd=command()[:-5]+['python','-u','attention_remainder.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-attention-remainder-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512,'heads':32},
        timeout_s=600,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 saved-only complete-bank2promptsx32headsx512queries; exact centered exponential partition/value remainder and BF16 telescopes. No fit/newprediction/checkpoint/framework/GPU/MLP. Parent banks/source assets retainedhost-local. Expected under2min,finite600s,no retry.')
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def execute(root):
    guard();root=Path(root).resolve();r=read(root/'request.json');started=time.perf_counter()
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for name,h in r['sourceHashes'].items():
        if sha(root/name)!=h:raise ValueError('Source changed')
    source=read(root/'source.json');seal=source['predictionSeal']
    if sha(seal['path'])!=seal['sha256']:raise ValueError('Parent forecast seal changed')
    asset=loaded(source['asset']);m=source['geometry'];d,heads,groups=geometry(m);out=root/'response-result';out.mkdir(exist_ok=False);work=root/'remainder-arrays';work.mkdir()
    summaries=[];paired=[];references=[];raw={};masks={};ports={};predictions={};observations=0;checks=[]
    with (out/'observations.jsonl').open('x') as f:
        for record in source['predictions']:
            name=record['id'];candidate=loaded(record);port=loaded(record['port']);mask=port['mask'].astype(bool);masks[name]=mask
            ports[name]=port;predictions[name]=candidate
            reference=loaded(next(x for x in source['references'] if x['id']==name));fields={key:np.zeros((len(mask),heads*d)) for key in ['H','A','P','V','exact','rounding','roundedError']};scalars={};coefficients=np.zeros((len(mask),heads,int(mask.sum())))
            for head in range(heads):
                p=packet(port,m,head);parts=decompose(p,head_asset(asset,m,head),candidate['nearestIndex'][:,head]);direct=exact_head(p)
                span=slice(head*d,(head+1)*d);forecast=candidate['transport64_raw'][:,span];rounded=floats(candidate['transport64_rounded'][:,span]).astype(float)
                exact_bits=reference['exactBank'][:,span];exact_rounded=floats(exact_bits).astype(float)
                rounding=(rounded-forecast)-(exact_rounded-parts['exact']);error=rounded-exact_rounded
                for key in ['H','A','P','V','exact']:fields[key][:,span]=parts[key]
                coefficients[:,head,:]=parts['normalizedRemainderCoefficients']
                fields['rounding'][:,span]=rounding;fields['roundedError'][:,span]=error
                for key,value in parts.items():
                    if value.ndim==1:
                        if key not in scalars:scalars[key]=np.zeros((len(mask),heads))
                        scalars[key][:,head]=value
                checks.append({'id':name,'queryHead':head,
                    'nearestMaxAbsClosure':float(np.max(np.abs(parts['H']-candidate['nearest64_raw'][:,span]))),
                    'firstOrderMaxAbsClosure':float(np.max(np.abs(parts['H']+parts['A']-forecast))),
                    'exactTiltMaxAbsClosure':float(np.max(np.abs(parts['exact']-direct['raw']))),
                    'exactBF16WordMismatches':int(np.count_nonzero(words(parts['exact'])!=exact_bits)),
                    'telescopeMaxAbsClosure':float(np.max(np.abs(error-(parts['P']+parts['V']+rounding))))})
                for region,selected in regions(mask):
                    if selected.any():summaries.append({'id':name,'queryHead':head,'region':region,**attribution(error,parts['P'],parts['V'],rounding,exact_rounded,selected)})
                rms={key:rowrms(value) for key,value in [('roundedError',error),('partition',parts['P']),('valueRemainder',parts['V']),('rounding',rounding)]}
                for i in range(len(mask)):
                    f.write(json.dumps({'id':name,'queryHead':head,'position':i,'active':bool(mask[i]),
                        **{key:float(value[i]) for key,value in rms.items()},**{key:float(parts[key][i]) for key in scalars}},allow_nan=False)+'\n');observations+=1
            path=work/(name+'-remainder.npz');np.savez(path,**fields,**scalars,mask=mask,keyPositions=np.flatnonzero(mask),normalizedRemainderCoefficients=coefficients)
            references.append({'id':name,'path':str(path),'sha256':sha(path)});raw[name]=fields
        a,b=source['pairs'][0];diff={key:raw[b][key]-raw[a][key] for key in raw[a]};target=floats(loaded(next(x for x in source['references'] if x['id']==b))['exactBank']).astype(float)-floats(loaded(next(x for x in source['references'] if x['id']==a))['exactBank']).astype(float)
        clock=np.arange(len(masks[a]));pair_regions=regions(masks[a],masks[b])+[('before8',clock<8),('at8',clock==8),('after8',clock>8)]
        interface={'changedTokenPositions':np.flatnonzero(ports[a]['ids']!=ports[b]['ids']).tolist(),
            'identicalMasks':bool(np.array_equal(masks[a],masks[b])),'changedQueryRows':np.flatnonzero(np.any(ports[a]['qpos']!=ports[b]['qpos'],axis=1)).tolist(),
            'changedKeyRows':np.flatnonzero(np.any(ports[a]['kpos']!=ports[b]['kpos'],axis=1)).tolist(),
            'changedValueRows':np.flatnonzero(np.any(ports[a]['v']!=ports[b]['v'],axis=1)).tolist(),
            'laterRouteMismatches':int(np.count_nonzero(predictions[a]['nearestIndex'][9:]!=predictions[b]['nearestIndex'][9:]))}
        one_key=[];key_terms=np.zeros_like(target);value_terms=np.zeros_like(target);key_valid=np.zeros((len(clock),heads),dtype=bool)
        for head in range(heads):
            span=slice(head*d,(head+1)*d);control=one_key_change(packet(ports[a],m,head),packet(ports[b],m,head),8)
            key_terms[:,span]=control['key'];value_terms[:,span]=control['value'];key_valid[:,head]=control['valid'];use=control['valid']
            one_key.append({'queryHead':head,'applicable':control['applicable'],'validQueries':int(use.sum()),
                'smoothClosureMaxAbs':float(np.max(np.abs((control['key']+control['value'])[use]-diff['exact'][use,span]))) if use.any() else None,
                'keyTermRms':float(np.sqrt(np.mean(control['key'][use]**2))) if use.any() else None,
                'valueTermRms':float(np.sqrt(np.mean(control['value'][use]**2))) if use.any() else None})
        path=work/'one-key-control.npz';np.savez(path,keyTerm=key_terms,valueTerm=value_terms,valid=key_valid,exactSmoothDelta=diff['exact'])
        references.append({'id':'one-key-control','path':str(path),'sha256':sha(path)})
        for head in [None,*range(heads)]:
            span=slice(None) if head is None else slice(head*d,(head+1)*d)
            for region,selected in pair_regions:
                if selected.any():paired.append({'a':a,'b':b,'queryHead':head,'region':region,
                    **attribution(diff['roundedError'][:,span],diff['P'][:,span],diff['V'][:,span],diff['rounding'][:,span],target[:,span],selected)})
    if sha(seal['path'])!=seal['sha256']:raise ValueError('Parent forecast seal changed')
    shutil.copyfile(root/'request.json',out/'request.json');shutil.copyfile(root/'source.json',out/'source.json')
    save(out/'reference-receipt.json',{'rows':references,'closureChecks':checks,'interfaceChecks':interface,'oneKeyControls':one_key,
        'scope':'Saved-only exact exponential-remainder decomposition and baseline-assisted one-key controls. No reference learned projections or future consumer runs.'})
    save(out/'report.json',{'schema':'tce-attention-remainder-result-v1','requestSHA256':r['requestSHA256'],'predictionSealSHA256':seal['sha256'],
        'scope':r['scope'],'scores':summaries,'pairedScores':paired,'checks':checks,'interfaceChecks':interface,'oneKeyControls':one_key,'observations':observations,'workerSeconds':time.perf_counter()-started,
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed','newPredictions':0,'frameworkCalls':0,'originalCheckpointBytesRead':0,
        'goalStatus':'Active. Retrospective attribution and target-assisted term corrections do not supply a new compact serving law or image.'})
    print('ATTENTION_REMAINDER_ATTRIBUTED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
