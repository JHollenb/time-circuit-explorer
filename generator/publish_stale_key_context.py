"""Full-byte custody, independent hybrid splices and own-state typed joins."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read, save, sha, identity
from projected_complement import fields
from publish_source_native import checked
from publish_source_complement import BYTES, check_frame
from propagated_context import RETURN_PARAMETERS, NEXT_PARAMETERS
from stale_key_context_analysis import NAMES
from stale_key_context_worker import CORNERS

TEMPLATES = ['publish_source_complement.py', 'publish_sparse_text_kv.py']


def items(p):
    s = read(p/'inputs.json')
    return [*s['parentRecords'].values(), *fields({'selected':s['selected'],'banks':s['banks']}),
            *fields(read(p/'targets.json')), *fields(read(p/'report.json')),
            *fields(read(p/'primary.json')), *fields(read(p/'diagnostic.json')),
            *[{'path':str(Path(s['modelRoot'])/n),'sha256':h} for n,h in s['imageAssets'].items()]]


def verify(root):
    from mrun.client.api import Api
    root, collection, job, request = checked(root);p=root/'result';s=read(p/'inputs.json')
    parent=Api().json('GET','/api/jobs/'+s['parentJobID'])
    if parent['state']!='succeeded' or parent['payload_custody']!=s['parentCustody']:raise ValueError('Parent custody changed')
    base=job['result']['diagnostics']['cwd'];unique={}
    for x in items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()]:
        if x['path'] in unique and unique[x['path']]['sha256']!=x['sha256']:raise ValueError('Conflicting complete byte identity')
        unique[x['path']]=x
    views=[{'arm':phase,'field':n,'support':request['support'],'selected':s['selected'][n],
            'bank':s['banks'][role][n],'hybrid':read(p/(phase+'.json'))['source']['fields'][n+'_hybrid']}
           for phase,roles in CORNERS.items() for n,role in zip(['K','V'],roles)]
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],
        input=json.dumps({'allowedRoots':[base,parent['result']['diagnostics']['cwd'],s['modelRoot']],
                          'items':list(unique.values()),'spliceChecks':views}),text=True,capture_output=True,timeout=240)
    if result.returncode:raise RuntimeError(result.stderr)
    receipt=json.loads(result.stdout);receipt.update(schema='tce-stale-key-context-bytes-v1',jobID=job['job_id'],
        verifierSourceSHA256=sha(__file__),templateSourceHashes={n:sha(Path(__file__).with_name(n)) for n in TEMPLATES},
        phaseSeals={n:sha(p/(n+'-seal.json')) for n in CORNERS})
    save(root/'retained-array-verification.json',receipt)
    return {'files':len(receipt['files']),'bytes':receipt['totalBytesVerified'],'NPYPayloadWitnesses':len(receipt['npyPayloadWitnesses']),'bankSpliceWitnesses':len(receipt['bankSpliceWitnesses'])}


def publish(root,destination):
    root,collection,job,request=checked(root);p=root/'result';receipt=read(root/'retained-array-verification.json')
    if receipt['verifierSourceSHA256']!=sha(__file__) or receipt['templateSourceHashes']!={n:sha(Path(__file__).with_name(n)) for n in TEMPLATES} or receipt['phaseSeals']!={n:sha(p/(n+'-seal.json')) for n in CORNERS}:raise ValueError('Verified sources/seals changed')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for x in p.iterdir():
        if x.suffix=='.jsonl':
            raw=x.read_bytes();target=out/(x.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless observations differ')
            streams.append({'source':x.name,'sourceSHA256':sha(x),'sourceBytes':len(raw),'published':target.name,'publishedSHA256':sha(target)})
        else:shutil.copyfile(x,out/x.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-stale-key-context-publication-v1','jobID':job['job_id'],'requestSHA256':request['requestSHA256'],
        'payloadCustody':collection['custody'],'schedulerResult':collection['result'],'completeCollectedInventory':collection['inventory'],
        'losslessStreams':streams,'scope':'Two fixed cross-corners with paid banks/diagonals/originalmaps; actual-response and finite interaction measurements, no image or serving saving.'})
    save(out/'files.json',{x.name:sha(x) for x in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    p=Path(source);r,s,target,report,receipt=[read(p/n) for n in ['request.json','inputs.json','targets.json','report.json','retained-array-verification.json']]
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256'] or r['corners']!={n:list(x) for n,x in CORNERS.items()}:raise ValueError('Request/candidate prescription differs')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Published member changed')
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Executed source changed')
    if receipt['verifierSourceSHA256']!=sha(__file__) or receipt['templateSourceHashes']!={n:sha(Path(__file__).with_name(n)) for n in TEMPLATES}:raise ValueError('Byte verifier changed')
    for x in items(p):
        if receipt['files'][x['path']]['sha256']!=x['sha256'] or (x.get('payloadSHA256') and receipt['npyPayloadWitnesses'][x['path']]['payloadSHA256']!=x['payloadSHA256']):raise ValueError('Complete consumed/emitted bytes absent')
    executions={n:read(p/(n+'-execution.json')) for n in CORNERS};acts=[]
    for phase,(keys,values) in CORNERS.items():
        arm=read(p/(phase+'.json'));seal=read(p/(phase+'-seal.json'));ev=executions[phase]
        if seal[phase+'SHA256']!=sha(p/(phase+'.json')) or seal['requestSHA256']!=r['requestSHA256'] or receipt['phaseSeals'][phase]!=sha(p/(phase+'-seal.json')):raise ValueError('Phase seal differs')
        if arm['corner']!=keys+values or arm['parentBankSealSHA256']!=s['parentBankSealSHA256']:raise ValueError('Actual corner/parent release differs')
        weights={x['name']:x['payloadSHA256'] for x in ev['loadedParameters']}
        if set(weights)!=set(RETURN_PARAMETERS+NEXT_PARAMETERS):raise ValueError('Actual parameter closure differs')
        expected={n:x['payloadSHA256'] for n,x in s['selected'].items()};expected.update(bankK=s['banks'][keys]['K']['payloadSHA256'],bankV=s['banks'][values]['V']['payloadSHA256'],**{n:weights[n] for n in RETURN_PARAMETERS})
        check_frame(arm['source'],expected,r['requestSHA256']);acts.append(arm['source'])
        meta=arm['source']['sourceFrame']['metadata']
        if (meta['keysFrom'],meta['valuesFrom'])!=(keys,values):raise ValueError('Typed key/value origin differs')
        expected={'rows':arm['source']['fields']['T']['payloadSHA256'],**{n:s['selected'][n]['payloadSHA256'] for n in ['txtMod','ropeCos','ropeSin']},**{n:weights[n] for n in NEXT_PARAMETERS}}
        check_frame(arm['nextQuery'],expected,r['requestSHA256']);acts.append(arm['nextQuery'])
        if arm['nextQuery']['sourceFrame']['metadata']['sourceOutputFrame']!=arm['source']['outputFrame']['frame_id']:raise ValueError('Actual T→query join differs')
        if ev['nominalMatrixMACs']!=452984832 or sum(x['nominalMACs'] for x in ev['linears']+ev['attentions'])!=452984832 or len(ev['linears'])!=4 or len(ev['attentions'])!=1:raise ValueError('Physical work differs')
        guard=ev['fileQuarantine'];denied=set(guard['deniedCanaries']);allowed=set(guard['allowedFiles'])
        required={x['path'] for x in fields(target)}|{s['banks']['Q' if keys=='P' else 'P']['K']['path'],s['banks']['Q' if values=='P' else 'P']['V']['path']}
        if phase=='diagnostic':required|={x['path'] for x in fields(read(p/'primary.json'))}
        if not required.issubset(denied) or required&allowed or not set(guard['successfulFileOpenCounts']).issubset(allowed) or not ev['parameterInputRNGClosure']:raise ValueError('Actual numerical read/parameter/quarantine closure differs')
    if read(p/'diagnostic.json')['primarySealSHA256']!=sha(p/'primary-seal.json'):raise ValueError('Primary-before-diagnostic release differs')
    for act in acts:
        for n in act['evidence']['metrics']['capturedPayloads']:
            x=act['fields'][n]
            if act['outputFrame']['slots'][n]['abi']!={'shape':x['shape'],'dtype':x['logicalDtype']}:raise ValueError('Actual typed output ABI differs')
    views=receipt['bankSpliceWitnesses']
    if len(views)!=4 or any(not x['selectedRowsByteExact'] or not x['outsideSupportByteExact'] for x in views):raise ValueError('Independent hybrid splice differs')
    if set(report['interactions'])!=set(NAMES) or report['totalNewMatrixMACs']!=905969664 or report['affineCalls']!=8 or report['attentionCalls']!=2:raise ValueError('Complete field/work coverage differs')
    raw=gzip.decompress((p/'observations.jsonl.gz').read_bytes());stream=read(p/'provenance.json')['losslessStreams'][0]
    if hashlib.sha256(raw).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless observations differ')
    rows=[json.loads(x) for x in raw.splitlines()];actual=[(x['field'],x['panel'],x['row'],x.get('head')) for x in rows]
    expected={(n,'whole-row',row,None) for n in NAMES for row in r['support']}|{(n,'query-head',row,h) for n in ['nextQProjected','nextQNormalized','nextQ'] for row in r['support'] for h in range(24)}
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Complete unique observation addresses differ')
    return {'schema':'tce-stale-key-context-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'typedActs':len(acts),'ownStateQueryJoins':2,
        'explicitOutputABIs':10,'independentSpliceWitnesses':4,'observations':len(rows),'retainedFullInteractions':15,
        'fullByteFiles':len(receipt['files']),'NPYPayloadWitnesses':len(receipt['npyPayloadWitnesses']),
        'scope':'Engineering source/seal/bytes/typed-own-state/work/quarantine evidence. No response-effect gate, serving saving or general zero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
