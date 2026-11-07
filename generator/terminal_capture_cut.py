"""Durable complete terminal boundary with an explicit component capture contract."""
from pathlib import Path
from encoder_atoms import sha,identity

SLOTS={'conditioning','context','cursor','image','img_ids','latent','noise_prediction','phase','sigmas','step','text','timesteps','txt_ids'}


def persist(state,original,root,request_id,phase,cursor,parent):
    from saturn_pub.core import StateCut
    from saturn_pub.contracts import SurfaceManifest,SlotSpec,ExecutionPoint
    from saturn_pub.store import LocalStore
    from terminal_capture_guard import local_store_only
    if set(state)!=SLOTS or (state['phase'],state['cursor'])!=(phase,cursor):raise ValueError('Complete terminal state/clock differs')
    if (phase,cursor) not in [('single',19),('readout',20)]:raise ValueError('Unsupported terminal capture point')
    point=ExecutionPoint('flux2-klein',state['step'],phase,operator=phase,index=cursor if phase=='single' else None,edge='before',
        next_operation='single_block.19' if phase=='single' else 'final_velocity_head',local_clock=cursor)
    surface=SurfaceManifest(tuple(SlotSpec(n,role='runtime',producer='original-native-prefix',consumer='terminal-single19-and-velocity',
        schema={'shape':list(v.shape),'dtype':str(v.dtype)} if hasattr(v,'shape') else {}) for n,v in sorted(state.items())),
        'tce-complete-flux-terminal-capture-v1','terminal-single19-and-velocity','one-single-block-and-final-head')
    surface.validate(state)
    execution={'family':'flux2-klein','adapter':'tce-native-component-capture-v1','composition':'tce-terminal-native-capture-v1',
        'requestSHA256':request_id,'sourceExecutionSHA256':identity(dict(original.execution)),
        'device':{'type':state['image'].device.type,'index':state['image'].device.index},'dtype':str(state['image'].dtype),
        'environment_versions':dict(original.execution).get('environment_versions',{}),'numeric_scope':dict(original.execution).get('numeric_scope',{}),
        'program':'Stream original joint1…4/single0…19/final velocity modules from supplied post-joint0 state.',
        'nativeAdapterSessionCompatible':False,'parity':'unassessed against an independent full native forward'}
    cut=StateCut(original.model_identity,execution,
        'terminal-source' if phase=='single' else 'terminal-target',parent,state,point.to_dict(),surface.to_dict())
    root=Path(root);store=LocalStore(root)
    with local_store_only(root):store.save(cut)
    reopened=store.load(cut.fingerprint,device=str(state['image'].device))
    if reopened.manifest()!=cut.manifest():raise ValueError('Complete terminal StateCut roundtrip differs')
    path=root/'cuts'/(cut.fingerprint+'.json')
    return {'cutID':cut.fingerprint,'store':str(root),'cutFile':{'path':str(path),'sha256':sha(path)},
        'point':point.to_dict(),'roundtripExact':True,'manifest':cut.manifest()}
