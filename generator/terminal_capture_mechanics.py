"""Tiny complete-cut persistence proof in the actual worker environment, before prefix work."""
import importlib.util
from pathlib import Path
import sys
from encoder_atoms import read,save,sha


def execute(root):
    import os
    import torch
    import saturn_pub.store as store_module
    from saturn_pub.core import StateCut
    from terminal_capture_cut_v2 import persist
    from terminal_capture_guard_v2 import quarantine
    root=Path(root).resolve();request=read(root/'request.json');store=root/'mechanics-store'
    denied=root/'mechanics-denied.npy';denied.write_bytes(b'tiny-denied-control')
    unknown=root/'mechanics-unlisted.safetensors';unknown.write_bytes(b'tiny-unlisted-control')
    os.environ['MDB_SHARED_TENSOR_STORE']=str(root/'mechanics-external-pool-must-not-open')
    guard=quarantine(root,[],[denied],store)
    rows=torch.arange(20,dtype=torch.bfloat16).reshape(1,5,4)
    state={'conditioning':torch.ones(1,2,6,dtype=torch.bfloat16),'context':{'rope':(torch.ones(5,2),torch.zeros(5,2)),'single_mod':torch.ones(1,12,dtype=torch.bfloat16)},
        'cursor':19,'image':rows[:,2:],'img_ids':torch.zeros(1,3,4),'latent':torch.zeros(1,3,2,dtype=torch.bfloat16),'noise_prediction':None,
        'phase':'single','sigmas':torch.tensor([1.,.5,0.]),'step':1,'text':rows[:,:2],'timesteps':torch.tensor([1000.,500.]),'txt_ids':torch.zeros(1,2,4)}
    original=StateCut('mechanics-only-no-model',{},'source',None,state)
    result=persist(state,original,store,request['requestSHA256'],'single',19,original.fingerprint)
    blocked=[]
    for path in [denied,unknown]:
        try:path.read_bytes()
        except PermissionError:blocked.append(path.name)
        else:raise ValueError('Numerical denial fixture escaped guard')
    if (root/'mechanics-external-pool-must-not-open').exists() or not guard['ownStoreLinkedFiles']:raise ValueError('Output publication policy differs')
    source=Path(store_module.__file__)
    if source.is_file():store_source_sha=sha(source)
    else:
        import hashlib,inspect
        store_source_sha=hashlib.sha256(inspect.getsource(store_module).encode()).hexdigest()
    receipt={'schema':'tce-terminal-capture-worker-mechanics-v1','requestSHA256':request['requestSHA256'],'roundtripExact':result['roundtripExact'],
        'completeSlots':len(state),'blocked':blocked,'sharedStoreModuleAvailable':importlib.util.find_spec('saturn_pub.shared_store') is not None,
        'storeModule':str(source),'storeSourceSHA256':store_source_sha,'pythonVersion':sys.version,'torchVersion':torch.__version__,
        'guard':guard,'externalPoolOpened':False,'newModelEvaluations':0,'scope':'Tiny supplied CPU toy state only; actual package/guard/durable publication mechanics, no model/checkpoint or scientific arrays.'}
    save(root/'capture-mechanics.json',receipt);print('TERMINAL_CAPTURE_WHEEL_MECHANICS_OK',flush=True)
    return receipt


if __name__=='__main__':execute(sys.argv[1])
