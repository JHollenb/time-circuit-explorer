"""Saved-receiver CPU diagnostic separating smooth curvature and finite errors."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import np,read,save,sha,identity,floats
from paired_fields import load,retain,metrics
from padding_mlp import SourceWeights,smooth_numpy

HERE=Path(__file__).resolve().parent
OWNED=['padding_mlp_remainder.py','padding_mlp.py','encoder_atoms.py','paired_fields.py','masked_kv_study.py']


def decomposition(error,terms):
    error=np.asarray(error,dtype=np.float64);terms={k:np.asarray(v,dtype=np.float64) for k,v in terms.items()}
    if any(v.shape!=error.shape for v in terms.values()):raise ValueError('Error term geometry differs')
    total=sum(terms.values(),np.zeros_like(error));names=list(terms)
    return {'actualErrorEnergy':float(np.sum(error*error)),'termEnergy':{k:float(np.sum(v*v)) for k,v in terms.items()},
        'twiceCrossTerms':[{'a':a,'b':b,'value':float(2*np.sum(terms[a]*terms[b]))} for i,a in enumerate(names) for b in names[i+1:]],
        'closureRMS':float(np.sqrt(np.mean((error-total)**2))),'closureMaxAbs':float(np.max(np.abs(error-total))),'scalars':int(error.size)}


class CPUWeights(SourceWeights):
    def parameter(self,name,row=None):
        p=self.model/'text_encoder'/self.index[name];f=self.files[str(p)];entry=f['header'][name]
        if self.signature(p)!=f['signature'] or entry['dtype']!='BF16':raise ValueError('Source parameter differs')
        lo,hi=entry['data_offsets'];shape=entry['shape']
        if row is not None or hi-lo!=int(np.prod(shape))*2:raise ValueError('Full parameter geometry differs')
        with p.open('rb') as stream:stream.seek(f['offset']+lo);data=stream.read(hi-lo)
        if len(data)!=hi-lo:raise ValueError('Truncated source parameter')
        self.reads.append({'name':name,'bytes':len(data),'payloadSHA256':hashlib.sha256(data).hexdigest(),'file':str(p)})
        return floats(np.frombuffer(data,dtype='<u2').reshape(shape)).astype(np.float64)


def verify(root,r):
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(Path(root)/n)!=h:raise ValueError('Frozen source changed: '+n)


def stage(root):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    parent=HERE.parent/'artifacts/padding-mlp-tangent-v2';c,j,r=checked_collection(parent,current_workers=False);live=Api().json('GET','/api/jobs/'+j['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody']:raise ValueError('Primary scientific job differs')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for n in OWNED:
        if n in r['sourceHashes'] and sha(HERE/n)!=r['sourceHashes'][n]:raise ValueError('Primary helper changed')
        shutil.copyfile(HERE/n,p/n)
    f=parent/'result';source=read(f/'inputs.json');compiled=read(f/'compile-receipt.json');encoder=read(f/'encoder-candidate.json');diag=read(f/'native-write-diagnostic.json');base=j['result']['diagnostics']['cwd']
    if diag['releasedAfterCandidateSealSHA256']!=sha(f/'candidate-seal.json'):raise ValueError('Native diagnostic release differs')
    save(p/'inputs.json',{'source':source,'compiled':compiled,'encoder':encoder,'diagnostic':diag,'primaryRoot':base,'primaryJobID':j['job_id'],'primaryRequestSHA256':r['requestSHA256'],
        'primaryRecords':{n:{'path':base+'/response-result/'+n,'sha256':sha(f/n)} for n in ['candidate-seal.json','compile-receipt.json','encoder-candidate.json','native-write-diagnostic.json','report.json']}})
    request={'schema':'tce-padding-remainder-request-v1','parentJobID':j['job_id'],'parentRequestSHA256':r['requestSHA256'],'candidateSealSHA256':sha(f/'candidate-seal.json'),'sourceHashes':{x.name:sha(x) for x in p.iterdir()},
        'ownedSources':OWNED,'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':300,'scope':'Postseal source-weight smooth MLP on primary actual saved receivers; separate curvature/native-vs-smooth/anchor offset/runtime emission and rounding. No candidate change, new image or native BF16 model execution.'}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request);save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from response_study import command
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted')
    cmd=command();cmd[-3]='padding_mlp_remainder.py'
    job=launch(cmd,experiment='tce-padding-mlp-remainder-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},config={'requestSHA256':r['requestSHA256'],'device':'cpu','backend':'numpy-fp64','cpu_threads':1,'rows':512,'hidden':2560,'intermediate':9728},
        timeout_s=300,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='Postseal successfulv2 primary actual2normalizedreceivers. CPU1 sourceFP64SwiGLU/ownrawJdelta separates nonlinear remainder, nativefinite, sourceanchor offset, runtimeFP32/BF16. No newimage/state repair/nativeBF16 calls. Paid76.5Bsource smoothMAC+6.71BJdeltaMAC/sourcehashIO. Intentfirstsizing;60sexpected300scap,noretries.')
    v={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',v);return v


def execute(root):
    import os
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json');start=time.perf_counter();out=root/'response-result';out.mkdir()
    for x in s['primaryRecords'].values():
        if sha(x['path'])!=x['sha256']:raise ValueError('Sealed primary record differs')
    source=s['source'];compiled=s['compiled'];enc=s['encoder'];native=s['diagnostic'];weights=CPUWeights(source);G,U,D=[weights.parameter('model.layers.0.mlp.'+n+'.weight') for n in ['gate_proj','up_proj','down_proj']]
    x0=load(compiled['fields']['x0']);b=load(compiled['fields']['b']);anchor=load(compiled['fields']['smoothAnchor']);J=load(compiled['fields']['J'])-load(compiled['fields']['jacobianEmissionError'])
    anchor_cpu=smooth_numpy(x0,G,U,D);anchor_difference=metrics(anchor_cpu,anchor);fields={};rows=[];observation_count=0
    with (out/'observations.jsonl').open('x') as stream:
        for c in ['00','10']:
            f=enc['firstMLPFields']['constant-'+c];other=enc['firstMLPFields']['tangent-'+c]
            if f['normalized']['payloadSHA256']!=other['normalized']['payloadSHA256']:raise ValueError('Policies do not share actual receiver')
            x=load(f['normalized']).reshape(512,2560);mask=load(f['mask']).astype(bool);delta=x-x0;g=x@G.T;u=x@U.T;sig=np.exp(-np.logaddexp(0.,-g));smooth=(g*sig*u)@D.T
            linear=delta@J.T;curvature=smooth-anchor-linear;actual=load(native['fields'][c]['write']).reshape(512,2560);finite=actual-smooth;offset=np.broadcast_to(anchor-b,x.shape)
            fields[c]={n:retain(root/'arrays'/c,n,v) for n,v in [('smoothWrite',smooth),('linearDelta',linear),('curvature',curvature),('nativeMinusSmooth',finite)]}
            for policy in ['constant','tangent']:
                own=load(enc['firstMLPFields'][policy+'-'+c]['write']).reshape(512,2560);terms={'nativeMinusSmooth':finite,'smoothCurvature':curvature,'smoothAnchorMinusNativeBase':offset}
                if policy=='tangent':terms['runtimeVsRawTangent']=(b+linear)-own
                else:terms.update(omittedLinearResponse=linear,runtimeVsNativeBase=np.broadcast_to(b,x.shape)-own)
                use=~mask;account=decomposition((actual-own)[use],{k:v[use] for k,v in terms.items()});rows.append({'case':c,'policy':policy,'region':'padding',**account,
                    'actualWrite':metrics(own[use],actual[use]),'smoothCurvature':metrics(curvature[use],smooth[use]),'receiverDeltaRMS':float(np.sqrt(np.mean(delta[use]**2)))})
                for i in np.flatnonzero(use):
                    stream.write(json.dumps({'case':c,'policy':policy,'position':int(i),**decomposition((actual-own)[i],{k:v[i] for k,v in terms.items()})},allow_nan=False)+'\n');observation_count+=1
    weights.unchanged()
    for x in s['primaryRecords'].values():
        if sha(x['path'])!=x['sha256']:raise ValueError('Sealed primary changed')
    for n in ['request.json','inputs.json']:shutil.copyfile(root/n,out/n)
    save(out/'fields.json',fields);save(out/'report.json',{'schema':'tce-padding-remainder-result-v1','requestSHA256':r['requestSHA256'],'primaryJobID':s['primaryJobID'],'candidateSealSHA256':r['candidateSealSHA256'],
        'sourceAnchorCPUvsCompilerCUDA':anchor_difference,'rows':rows,'observations':observation_count,'sourceParameterReads':weights.reads,'sourceAssetChecks':weights.checks,'numpyVersion':np.__version__,
        'threadEnvironment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},'workerSecondsAfterImports':time.perf_counter()-start,
        'work':{'newImages':0,'newNativeBF16MLPs':0,'sourceSmoothDenseProducts':6,'sourceSmoothMACs':2*512*3*2560*9728,'rawJacobianDeltaProducts':2,'rawJacobianDeltaMACs':2*512*2560*2560},
        'scope':'Additive postseal diagnostic; original primary immutable. CPUFP64source maps on actual saved normalized receivers, reconstructed rawFP64J. Smooth curvature includes qualified CPU/CUDA/GEMM reassociation floor; native finite and runtime rounding terms stay separate. No input-rank truncation, fit or serving result update.',
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed'})
    print('PADDING_REMAINDER_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(json.dumps(v))
