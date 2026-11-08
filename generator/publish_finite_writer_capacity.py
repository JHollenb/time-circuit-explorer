"""Compact scalar/custody publisher; scientific tensors remain near compute."""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
from contextual_writer_program import read,save,sha
from publish_contextual_writer import check_frame,digest
from publish_contrast_writer import pooled,close


def validate_component(row,count):
    pair = row['kind']=='pair'
    if row['kind'] not in ('pair','single') or len(row['rows']) != (2 if pair else 1): raise ValueError('Component kind differs')
    norms = row['endpointNorms']; alpha = row['alpha']
    if len(norms)!=len(alpha) or len(norms)!=(2 if pair else 1) or any(len(v)!=count for v in norms): raise ValueError('Endpoint option geometry differs')
    if any(not math.isfinite(x) or x<0 for v in norms for x in v) or any(not math.isfinite(x) or x<=0 for x in alpha): raise ValueError('Invalid finite endpoint scores')
    if pair:
        response = row['responseNorms']; scores = row['scores']; beta = row['beta']
        if len(response)!=count or len(scores)!=count or any(len(v)!=count for v in response+scores) or not math.isfinite(beta) or beta<=0: raise ValueError('Pair option geometry differs')
        if any(not math.isfinite(x) or x<0 for v in response for x in v): raise ValueError('Invalid response norm')
        endpoint = [alpha[0]*norms[0][a]+alpha[1]*norms[1][b] for a in range(count) for b in range(count)]
        response = [beta*x for v in response for x in v]; flat = [x for v in scores for x in v]
    else:
        endpoint = [alpha[0]*x for x in norms[0]]; response = [0.]*count; flat = row['scores']
        if len(flat)!=count: raise ValueError('Single option geometry differs')
    if any(not math.isfinite(x) for x in flat) or any(a+b!=c for a,b,c in zip(endpoint,response,flat)): raise ValueError('Component scalar expression differs')
    selected = min(range(len(flat)),key=flat.__getitem__); current = row['currentIndex']; distinct = sorted(set(flat))
    if type(current) is not int or not 0<=current<len(flat) or row['selectedIndex']!=selected or row['ties']!=flat.count(flat[selected]): raise ValueError('Canonical exhaustive choice differs')
    gap = distinct[1]-distinct[0] if len(distinct)>1 else None
    if row['nextDistinctGap']!=gap: raise ValueError('Tie/gap evidence differs')
    return {m:{'joint':flat[i],'endpoint':endpoint[i],'response':response[i]} for m,i in [('current',current),('capacity',selected)]}


def publish(root,mechanics,destination):
    from mrun.client.api import Api
    from finite_writer_capacity_study import OWNED,PARENTS,verify
    root,destination = Path(root).resolve(),Path(destination).resolve(); req = read(root/'request.json'); verify(root/'payload',req)
    for name in OWNED:
        if sha(Path(__file__).parent/name)!=req['sourceHashes'][name]: raise ValueError('Current executed source differs')
    job = read(root/'scheduler-job.json'); live = Api().json('GET','/api/jobs/'+job['job_id']); custody = job['payload_custody']; phases = [custody[k] for k in ('declared','sealed','executed')]
    collection = read(root/'collection.json'); result = root/'result'
    if live['state']!='succeeded' or live['result']['returncode']!=0 or live['result']!=job['result'] or collection['result']!=job['result'] or live['payload_custody']!=custody or not custody['required'] or not live.get('admission'): raise ValueError('Successful guarded job association differs')
    if any((p['sha256'],p['size_bytes'])!=(phases[0]['sha256'],phases[0]['size_bytes']) for p in phases) or phases[2]['host']!='beast': raise ValueError('Executed payload differs')
    if collection['jobID']!=job['job_id'] or collection['custody']!=custody or any(x['config']['requestSHA256']!=req['requestSHA256'] for x in (job,live)): raise ValueError('Request/custody association differs')
    actual = {str(p.relative_to(result)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in result.rglob('*') if p.is_file()}
    if actual!=collection['inventory']['files']: raise ValueError('Complete collection inventory differs')
    for name in req['retainedMetadata']:
        if name=='request.json':
            if read(result/name)!=req: raise ValueError('Collected request differs')
        elif sha(result/name)!=req['sourceHashes'][name]: raise ValueError('Retained source metadata differs')
    tests = read(mechanics)
    if (tests['testsRun'],tests['failures'],tests['errors'])!=(8,0,0): raise ValueError('Tiny mechanics incomplete')
    for name,h in tests['sourceHashes'].items():
        if req['sourceHashes'].get(name)!=h or sha(Path(__file__).parent/name)!=h: raise ValueError('Tested source differs')
    parent_pub = read(result/'parent-publication.json'); parent_live = Api().json('GET','/api/jobs/'+req['parent']['jobID'])
    if sha(result/'parent-publication.json')!=req['parent']['publicationSHA256'] or parent_pub['jobID']!=req['parent']['jobID'] or parent_live['state']!='succeeded' or parent_live['payload_custody']!=req['parent']['custody']: raise ValueError('Parent lineage differs')
    for name in PARENTS:
        if name!='publication.json' and sha(result/('parent-'+name))!=parent_pub['files'][name]['sha256']: raise ValueError('Parent metadata authority differs')
    report = read(result/'report.json'); arrays = read(result/'array-manifest.json'); typed = read(result/'typed-capacity.json'); reconstruction = read(result/'input-reconstruction.json')
    source = read(result/'parent-parent-inputs.json'); catalog = read(result/'parent-programs.json'); child = read(result/'parent-final-serving.json'); fit = read(result/'parent-fit.json')
    packets = {x['id']:x['packet'] for x in child['rows']}; parent_rows = {x['id']:x for x in child['rows']}; current_labels = {}
    for key in req['cases']: current_labels[key] = next(x for x in parent_rows[key]['predictions'] if x['method']=='candidate')['leaves']
    expected_groups = {'discovery','reusedTransfer'}
    if req['cases']!=['A','B','00','10','01','11'] or req['discovery']!=['A','B','00','10'] or req['transfer']!=['01','11']: raise ValueError('Fixed history domain differs')
    if len(typed)!=2 or {x['group'] for x in typed}!=expected_groups or len(report['groups'])!=2 or {x['group'] for x in report['groups']}!=expected_groups: raise ValueError('Complete unique typed/report group domain differs')
    if report['requestSHA256']!=req['requestSHA256'] or report['physicalWork']!=req['physicalWork'] or report['device']!='cpu' or report['threads']!=1: raise ValueError('Executed geometry/profile differs')
    if report['finitePolicy']!=req['finitePolicy'] or report['analysisSealSHA256']!=sha(result/'analysis-seal.json'): raise ValueError('Finite policy/analysis seal differs')
    for name,h in read(result/'analysis-seal.json').items():
        if sha(result/name)!=h: raise ValueError('Sealed analysis member differs')
    group_info = {g['group']:g for g in report['groups']}; contexts = {}; expected_components = {}; weights = {}; betas = {}
    for group,cases,pairs in [('discovery',req['discovery'],[('A','B'),('00','10')]),('reusedTransfer',req['transfer'],[('01','11')])]:
        g = group_info[group]; keys = [(k,i) for k in cases for i in range(2,512)]; offsets = {k:j*510 for j,k in enumerate(cases)}
        if g['casePositionOrder']!=[list(x) for x in keys] or g['cases']!=cases: raise ValueError('Complete group row order differs')
        if group=='discovery' and g['referenceEnergies']!=[fit['endpointEnergy'],fit['responseEnergy']]: raise ValueError('Frozen discovery normalizers differ')
        e,t = g['referenceEnergies']; w = []; edges = []; v = []
        if any(not math.isfinite(x) or x<0 for x in (e,t)): raise ValueError('Invalid reference energy')
        for key in cases:
            active = sum(packets[key]['mask'][2:]); padding = 510-active
            if not active or not padding: raise ValueError('Empty endpoint stratum')
            w.extend(.5/(active if packets[key]['mask'][i] else padding)/len(cases) for i in range(2,512))
        for a,b in pairs:
            for active in (True,False):
                positions = [i for i in range(2,512) if packets[a]['ids'][i]==packets[b]['ids'][i] and bool(packets[a]['mask'][i]) is active]
                if not positions: raise ValueError('Empty paired stratum')
                for i in positions: edges.append((offsets[a]+i-2,offsets[b]+i-2));v.append(1/(2*len(pairs)*len(positions)))
        covered = {i for edge in edges for i in edge}; expected_components[group] = {tuple(edge) for edge in edges}|{(i,) for i in range(len(keys)) if i not in covered}
        contexts[group] = keys; weights[group] = w; betas[group] = {edge:weight/(t if t else 1) for edge,weight in zip(edges,v)}
    totals = {group:{m:{k:[] for k in ('joint','endpoint','response')} for m in ('current','capacity')} for group in group_info}; choices = {g:[None]*len(k) for g,k in contexts.items()}; components = collections.defaultdict(list); count = 0
    with gzip.open(result/'components.jsonl.gz','rt') as stream:
        for line in stream:
            row = json.loads(line); group = row['group']; key = tuple(row['rows'])
            if key not in expected_components[group]: raise ValueError('Duplicate/out-of-domain component')
            expected_components[group].remove(key); values = validate_component(row,9); g = group_info[group]; e,t = g['referenceEnergies']
            labels = [current_labels[c][i-2][1] for c,i in (contexts[group][j] for j in key)]
            if row['currentIndex']!=(labels[0]*9+labels[1] if len(key)==2 else labels[0]): raise ValueError('Current feasible choice differs')
            for j,a in zip(key,row['alpha']): close(a,weights[group][j]/(e if e else 1))
            if len(key)==2: close(row['beta'],betas[group][key]);selected = divmod(row['selectedIndex'],9)
            else: selected = [row['selectedIndex']]
            for j,a in zip(key,selected): choices[group][j] = a
            for method,terms in values.items():
                for term,value in terms.items(): totals[group][method][term].append(value)
            components[group].append({k:row[k] for k in ('rows','currentIndex','selectedIndex','ties','nextDistinctGap')});count += len(row['scores'])*(9 if len(key)==2 else 1)
    if any(expected_components.values()) or count!=req['physicalWork']['evaluatedChoices']: raise ValueError('Incomplete exhaustive score domain')
    for group,g in group_info.items():
        if components[group]!=g['components']: raise ValueError('Compact choice trajectory differs')
        for method,terms in totals[group].items():
            for term,values in terms.items(): close(g['scores'][method][term],math.fsum(values))
    groups = collections.defaultdict(list); raw = {}; expected = {('case',c,m,i) for c in req['cases'] for m in ('current','capacity') for i in range(512)}
    expected.update(('pair',a,b,m,i) for a,b in [('A','B'),('00','10'),('01','11')] for m in ('current','capacity') for i in range(512))
    with gzip.open(result/'observations.jsonl.gz','rt') as stream:
        for line in stream:
            r = json.loads(line);i = r['position']
            if 'case' in r:
                k = ('case',r['case'],r['method'],i);mask = bool(packets[r['case']]['mask'][i]);labels = ['all','active' if mask else 'padding']+(['learned'] if i>=2 else [])
            else:
                a,b = r['pair'];k = ('pair',a,b,r['method'],i);mask = bool(packets[a]['mask'][i]);same = packets[a]['ids'][i]==packets[b]['ids'][i]
                if r['sameToken'] is not same: raise ValueError('Paired token binding differs')
                labels = ['all']+(['unchangedToken'] if same else [])+(['unchangedLearned','unchangedActive' if mask else 'unchangedPadding'] if same and i>=2 else [])
            if k not in expected or k in raw or r['port']!='H1' or r['active'] is not mask: raise ValueError('Raw observation domain differs')
            raw[k] = r
            for label in labels: groups[(*k[:-1],label)].append(r)
    if set(raw)!=expected or len(raw)!=9216 or report['observationRows']!=9216: raise ValueError('Incomplete raw scalar domain')
    scalars = 0
    for item in report['summaries']+report['pairs']:
        k = ('case',item['case'],item['method'],item['stratum']) if 'case' in item else ('pair',item['A'],item['B'],item['method'],item['selection'])
        rows = groups.pop(k)
        if len(rows)!=item['rows']: raise ValueError('Pooled coverage differs')
        for name,value in pooled(rows).items(): close(item[name],value);scalars += 1
    if groups: raise ValueError('Missing pooled groups')
    # Bind the complete tensor ABI independently of self-reported manifest geometry.
    expected_arrays = {}
    for group,keys in contexts.items():
        n = len(keys); edge_count = len(betas[group])
        for name in ('H0','S','Y','current','capacityH1'): expected_arrays[group+'-'+name] = ([n,2560],'torch.bfloat16','<u2')
        for name in ('labels','capacityLabels'): expected_arrays[group+'-'+name] = ([n],'torch.int64','<i8')
        for name in ('weights',): expected_arrays[group+'-'+name] = ([n],'torch.float64','<f8')
        expected_arrays[group+'-atoms'] = ([9,2560],'torch.float32','<f4')
        expected_arrays[group+'-edges'] = ([edge_count,2],'torch.int64','<i8')
        expected_arrays[group+'-pairWeights'] = ([edge_count],'torch.float64','<f8')
        expected_arrays[group+'-frozenEnergies'] = ([2],'torch.float64','<f8')
    for key in req['cases']: expected_arrays['capacity-full-'+key] = ([512,2560],'torch.bfloat16','<u2')
    if set(arrays)!=set(expected_arrays) or len({x['path'] for x in arrays.values()})!=len(expected_arrays): raise ValueError('Complete unique array domain differs')
    for name,abi in expected_arrays.items():
        field = arrays[name]
        if (field['shape'],field['dtype'],field['storedDtype'])!=abi: raise ValueError('Expected tensor ABI differs: '+name)
    for name in ('rows','caseOutputs'):
        records = reconstruction[name]
        if len(records)!=6 or {x['case'] for x in records}!=set(req['cases']): raise ValueError('Complete unique reconstruction/output domain differs')
    for record in reconstruction['caseOutputs']:
        if record['field']!=arrays['capacity-full-'+record['case']]: raise ValueError('Complete case output field differs')
    authenticated = reconstruction['authenticatedRemoteInputs']
    if len(authenticated)!=len(req['remoteInputs']) or {x['path']:{k:x[k] for k in ('bytes','sha256')} for x in authenticated}!=req['remoteInputs']: raise ValueError('Complete authenticated input domain differs')
    # Full raw-file checks, small label/segment byte joins and typed fingerprints only.
    remote = {path:dict(x) for path,x in req['remoteInputs'].items()}; relations = []; npys = {}; labels_expected = {}
    for x in arrays.values():
        if x['sha256']!=x['payloadSHA256'] or x['bytes']!=math.prod(x['shape'])*{'<u2':2,'<f4':4,'<f8':8,'<i8':8}[x['storedDtype']]: raise ValueError('Tensor/file ABI differs')
        remote[x['path']] = {'sha256':x['sha256'],'bytes':x['bytes']}
    for record in reconstruction['rows']:
        key = record['case'];original = next(x for x in parent_rows[key]['predictions'] if x['method']=='candidate')
        if record['H0']!=parent_rows[key]['H0'] or record['target']!=source['cases'][key]['H1'] or record['current']!=original or record['currentReconstructedPayloadSHA256']!=original['sha256']: raise ValueError('Actual source reconstruction binding differs')
        item = source['cases'][key]['H1'];npys[item['path']] = item
    prefix_path = str(Path(catalog['accepted']['directory'])/'prefix.bf16')
    for item in typed:
        group = item['group'];g = group_info[group];before,after,ev,spec = item['input'],item['output'],item['evidence'],item['spec'];check_frame(before);check_frame(after)
        names = ['H0','S','Y','current','atoms','labels','weights','edges','pairWeights','frozenEnergies']
        if set(before['slots'])!=set(names) or set(after['slots'])!=set(names+['capacityH1','capacityLabels']) or spec['reads']!=names or spec['writes']!=['capacityH1','capacityLabels']: raise ValueError('Typed exact read/write boundary differs')
        if before['program_fingerprint']!=req['requestSHA256'] or after['program_fingerprint']!=req['requestSHA256'] or after['parent_frame_id']!=before['frame_id'] or ev['input_frame_id']!=before['frame_id'] or ev['output_frame_id']!=after['frame_id'] or ev['act']!=spec['identity']: raise ValueError('Typed ancestry differs')
        if item['inputPayloadBefore']!=item['inputPayloadAfter']: raise ValueError('Typed immutable inputs changed')
        for name in names+['capacityH1','capacityLabels']:
            field = arrays[group+'-'+name];fingerprint = digest({'dtype':field['dtype'],'shape':field['shape'],'bytes':field['sha256']})
            frame = before if name in names else after
            if frame['slots'][name]['fingerprint']!=fingerprint or name in names and (after['slots'][name]!=before['slots'][name] or item['inputPayloadBefore'][name]!=field['sha256']): raise ValueError('Actual typed array payload differs')
        if arrays[group+'-atoms']['sha256']!=catalog['accepted']['program']['dictionaries'][1]['sha256']: raise ValueError('Frozen atom bytes differ')
        labels_expected[arrays[group+'-labels']['path']] = [current_labels[c][i-2][1] for c,i in contexts[group]]
        labels_expected[arrays[group+'-capacityLabels']['path']] = choices[group]
        for name in ('H0','Y','current'):
            parts = []
            for key in g['cases']:
                source_item = parent_rows[key]['H0'] if name=='H0' else source['cases'][key]['H1'] if name=='Y' else next(x for x in parent_rows[key]['predictions'] if x['method']=='candidate')
                parts.append({'path':source_item['path'],'payload':name=='Y','offset':10240,'bytes':2611200})
            relations.append({'parts':parts,'sha256':arrays[group+'-'+name]['sha256']})
        for index,key in enumerate(g['cases']):
            reconstructed = next(x for x in reconstruction['rows'] if x['case']==key)
            relations.append({'parts':[{'path':arrays[group+'-S']['path'],'offset':index*2611200,'bytes':2611200}],'sha256':reconstructed['slot0PayloadSHA256']})
            relations.append({'parts':[{'path':prefix_path,'offset':0,'bytes':10240},{'path':arrays[group+'-capacityH1']['path'],'offset':index*2611200,'bytes':2611200}],'sha256':arrays['capacity-full-'+key]['sha256']})
    code = '''import ast,hashlib,json,stat,struct
from pathlib import Path
expected=json.loads(%r);npys=json.loads(%r);relations=json.loads(%r);labels=json.loads(%r);rows=[];payload_offsets={}
for name,item in expected.items():
 p=Path(name);s=p.lstat()
 if not name.startswith('/mnt/hdd8tb/mrun-work/') or not stat.S_ISREG(s.st_mode) or s.st_size!=item['bytes']:raise ValueError('Unsafe retained file')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 if h.hexdigest()!=item['sha256']:raise ValueError('Fresh retained bytes differ')
 if name in npys:
  with p.open('rb') as f:
   if f.read(8)!=b'\\x93NUMPY\\x01\\x00':raise ValueError('NPY format differs')
   n=struct.unpack('<H',f.read(2))[0];header=ast.literal_eval(f.read(n).decode('latin1'));payload_offsets[name]=f.tell()
   if header!={'descr':'<u2','fortran_order':False,'shape':(1,512,2560)}:raise ValueError('NPY ABI differs')
   body=hashlib.sha256(f.read()).hexdigest()
   if body!=npys[name]['payloadSHA256']:raise ValueError('Native payload differs')
 if name in labels:
  data=p.read_bytes()
  if list(struct.unpack('<'+str(len(labels[name]))+'q',data))!=labels[name]:raise ValueError('Actual label choices differ')
 rows.append({'path':name,'bytes':s.st_size,'sha256':item['sha256']})
for relation in relations:
 h=hashlib.sha256()
 for part in relation['parts']:
  with open(part['path'],'rb') as f:
   f.seek(part['offset']+(payload_offsets[part['path']] if part.get('payload') else 0));data=f.read(part['bytes'])
   if len(data)!=part['bytes']:raise ValueError('Short actual tensor segment')
   h.update(data)
 if h.hexdigest()!=relation['sha256']:raise ValueError('Actual byte segment join differs')
print(json.dumps(rows))
''' % tuple(json.dumps(x) for x in (remote,npys,relations,labels_expected))
    witnesses = json.loads(subprocess.run(['ssh','-T','beast','python3','-'],input=code,text=True,capture_output=True,check=True,timeout=90).stdout)
    destination.mkdir(exist_ok=False)
    for p in result.iterdir(): shutil.copyfile(p,destination/p.name)
    for p,name in [(root/'collection.json','collection.json'),(root/'scheduler-job.json','scheduler-job.json'),(Path(mechanics),'mechanics.json')]: shutil.copyfile(p,destination/name)
    publication = {'schema':'tce-fixed-writer-capacity-publication-v1','jobID':job['job_id'],'requestSHA256':req['requestSHA256'],'publisherSHA256':sha(__file__),
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in destination.iterdir()},'exhaustiveEvaluatedChoices':count,'rawObservationRows':len(raw),'independentlyRebuiltPooledScalars':scalars,
        'freshRemoteFiles':witnesses,'byteSegmentJoins':len(relations),'actualLabelVectorsChecked':len(labels_expected),
        'scope':'Source/custody/complete component scalar expressions and canonical minima, complete observation domain/pooled scalars, actual typed raw payload/label/segment joins. Vector norms and historical einsum reductions are producer measurements,not independent tensor replay. No local numerical arrays/model or new scientific scoring.'}
    save(destination/'publication.json',publication);return publication


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('mechanics');parser.add_argument('destination');a=parser.parse_args();print(json.dumps(publish(a.root,a.mechanics,a.destination)))
