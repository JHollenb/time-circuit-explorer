"""Metadata-only, source-bound discovery/feedback inventory for one native read."""
from pathlib import Path
from encoder_atoms import read,sha


def inventory(workspace, image_assets):
    workspace=Path(workspace);campaign=workspace/'artifacts/flux-global-reader-20261106'
    plan=read(campaign/'run-01/plan.json');request=read(campaign/'run-01/request.json')
    if request['config']!=plan:raise ValueError('Historical discovery contract differs')
    files={};roots=set();records=[]
    def member(local,remote,name,expected):
        p=local/name
        if sha(p)!=expected:raise ValueError('Historical source differs: '+str(p))
        path=remote+'/'+name;files[path]={'path':path,'sha256':expected,'bytes':p.stat().st_size};roots.add(remote)
        return files[path]
    def archive(local,remote,item,manifest):
        f=member(local,remote,item['path'],manifest[item['path']])
        if f['sha256']!=item['sha256'] or item.get('encoding')!='bf16-bits-v1':raise ValueError('Native field identity differs')
        return {**f,'format':'legacy-npy','shape':item['stored_shape'],'payloadSHA256':item['descriptor']['sha256'],
            'nativeStride':item['native_stride'],'logicalDtype':'torch.bfloat16'}
    def packet(local,remote,name,key,expected,expected_payload,stride):
        f=member(local,remote,name,expected)
        return {**f,'format':'legacy-npz','member':key,'shape':[1,1024,128],'payloadSHA256':expected_payload,
            'nativeStride':stride,'logicalDtype':'torch.bfloat16'}
    def add(label,context,kind,z,n,target,role=None,population=None):
        seed=int(context.split('.')[0][1:]);population=population or ('discovery' if seed in plan['discovery_seeds'] else 'feedback')
        role=role or ('base' if context.split('.')[1]=='base' else 'shape')
        row={'id':label,'context':context,'kind':kind,'clock':3,'population':population,'conditioning':role,'z':z,'previous':n,'target':target}
        records.append(row);return row
    canonical={}
    for spec in plan['corpus']:
        remote=spec['remote'];name=remote.rsplit('/',1)[-1];label=name.removeprefix('velocity-').removesuffix('-result')
        local=workspace/('artifacts/flux-velocity-'+label+'-20261106/run-01/extracted/'+name)
        if label=='closure-student':local=workspace/'artifacts/flux-velocity-closure-20261106/run-02-law/extracted/velocity-closure-student-result'
        manifest=spec['files']
        for fn in ['history.json','source.json']:member(local,remote,fn,manifest[fn])
        src=read(local/'source.json')
        if any(src['checkpoint_files'][n]!=h for n,h in image_assets.items()):raise ValueError('Discovery model bytes differ')
        for context,h in read(local/'history.json').items():
            row=add(context+'.canonical',context,'canonical',archive(local,remote,h[3]['z'],manifest),
                archive(local,remote,h[2]['n'],manifest),archive(local,remote,h[3]['n'],manifest))
            canonical[context]=row
    local=workspace/'artifacts/flux-velocity-closure-20261106/run-01/extracted/velocity-closure-result'
    remote=plan['diagnostic_source'];manifest=plan['diagnostic_files'];member(local,remote,'rows.json',manifest['rows.json'])
    for row in read(local/'rows.json'):
        if row['method'] not in ['constant','linear','stencil','conditioning']:continue
        role='base' if row['context'].endswith('.base') else 'shape'
        if row['method']=='conditioning':role='shape' if role=='base' else 'base'
        add(row['id'],row['context'],row['method'],archive(local,remote,row['input_z'],manifest),
            archive(local,remote,row['previous_n'],manifest),archive(local,remote,row['native_n'],manifest),role)
    local=workspace/'artifacts/flux-velocity-closure-20261106/run-05-confirm/extracted/velocity-nonlocal-result'
    remote=plan['nonlocal_source'];manifest=plan['nonlocal_files'];member(local,remote,'rows.json',manifest['rows.json'])
    pairs=[]
    for row in read(local/'rows.json'):
        if row['label']!='far_bottom_right':continue
        prior=canonical[row['context']]['previous']
        value=next(x['value'] for x in row['action']['act']['parameters']['items'] if x['key']['value']=='value')
        z=packet(local,remote,row['input'],'current',row['input_sha256'],value['sha256'],value['stride'])
        n=packet(local,remote,row['input'],'n_last',row['input_sha256'],prior['payloadSHA256'],prior['nativeStride'])
        added=add(row['id'],row['context'],'remote-edit',z,n,archive(local,remote,row['native_n'],manifest))
        if added['population']=='discovery':pairs.append([canonical[row['context']]['id'],added['id']])
    local=campaign/'run-01/extracted/global-reader-result';remote='/home/beast/.cache/mdb-research/job-3264b738b16d/global-reader-result';manifest=read(local/'files.json')
    for fn in ['history.json','remote-history.json','remote.binding.json']:member(local,remote,fn,manifest[fn])
    histories=read(local/'history.json');remote_history=read(local/'remote-history.json');eval_pairs=[]
    for context,h in histories.items():
        base=add(context+'.canonical',context,'canonical',archive(local,remote,h[3]['z'],manifest),
            archive(local,remote,h[2]['n'],manifest),archive(local,remote,h[3]['n'],manifest),population='evaluation')
        changed=remote_history[context]
        edited=add(context+'.remote',context,'remote-edit',archive(local,remote,changed['z'],manifest),
            base['previous'],archive(local,remote,changed['n'],manifest),population='evaluation')
        eval_pairs.append([base['id'],edited['id']])
    oldlocal=workspace/'artifacts/flux-eye-pose-transfer-20261106/run-02/extracted/eye-pose-result/store'
    oldremote=plan['source_root']+'/store';pages={}
    for role,cut_id in plan['origins'].items():
        cutname='cuts/'+cut_id+'.json';member(oldlocal,oldremote,cutname,plan['source_files']['store/'+cutname])
        cut=read(oldlocal/cutname)
        for slot in ['conditioning','img_ids','txt_ids','sigmas','timesteps']:
            page_id=cut['slots'][slot];name='pages/'+page_id+'.json'
            member(oldlocal,oldremote,name,page_id);page=read(oldlocal/name);blob='blobs/'+page['tensor_blob']+'.safetensors'
            member(oldlocal,oldremote,blob,page['tensor_blob'])
            pages[role+'.'+slot]={'format':'saturn-slot','store':oldremote,'pageID':page_id,'payloadSHA256':page['value']['sha256'],
                'shape':page['value']['shape'],'logicalDtype':page['value']['dtype'],'nativeStride':page['value']['stride'],
                'files':[oldremote+'/'+name,oldremote+'/'+blob]}
    counts={k:sum(x['population']==k for x in records) for k in ['discovery','feedback','evaluation']}
    if counts!={'discovery':18,'feedback':18,'evaluation':8} or len(pairs)!=2 or len(eval_pairs)!=4:raise ValueError('Declared historical population differs')
    return {'schema':'tce-native-reader-corpus-v1','records':records,'conditioningPages':pages,'discoveryPairs':pairs,'evaluationPairs':eval_pairs,
        'sourceFiles':files,'sourceRoots':sorted(roots),'counts':counts,'discoverySeeds':plan['discovery_seeds'],'feedbackSeeds':plan['feedback_seeds'],
        'oldGlobalReaderRequestSHA256':request['request_sha256'],'oldGlobalReaderPlanSHA256':sha(campaign/'run-01/plan.json'),
        'decoderScope':'Foreign native velocity labels share transformer/profile; their historical VAE normalization on CPU is not used for new pixel comparisons.',
        'scope':'Metadata/byte inventory only; no array calculations, fitting or model calls. Raw previews omitted. Discovery labels alone may enter fitting.'}


def load(item,device='cpu'):
    import numpy as np
    import torch
    from paired_fields import payload
    if item['format']=='saturn-slot':
        from saturn_pub.store import LocalStore
        value,_=LocalStore(item['store'])._read_page(item['pageID'])
    else:
        if sha(item['path'])!=item['sha256']:raise ValueError('Historical array bytes differ')
        raw=np.load(item['path'],allow_pickle=False)
        if item['format']=='legacy-npz':
            with raw as packet:value=torch.from_numpy(np.array(packet[item['member']],copy=True)).to(torch.bfloat16)
        else:value=torch.from_numpy(raw.copy()).view(torch.bfloat16)
    if list(value.shape)!=item['shape'] or str(value.dtype)!=item['logicalDtype'] or payload(value)!=item['payloadSHA256']:raise ValueError('Actual historical input differs')
    stride=tuple(item['nativeStride'])
    result=torch.empty_strided(tuple(value.shape),stride,dtype=value.dtype,device=device);result.copy_(value.to(device))
    if payload(result)!=item['payloadSHA256']:raise ValueError('Restored input layout changes values')
    return result
