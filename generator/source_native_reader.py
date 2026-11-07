"""Seven source affines and one live first-joint image attention read."""
from encoder_atoms import read
from paired_fields import payload

AFFINES=('x_embedder.weight','context_embedder.weight','transformer_blocks.0.attn.to_q.weight',
    'transformer_blocks.0.attn.to_k.weight','transformer_blocks.0.attn.to_v.weight',
    'transformer_blocks.0.attn.add_k_proj.weight','transformer_blocks.0.attn.add_v_proj.weight')
NORMS=('norm_q','norm_k','norm_added_k')
PARAMETERS=AFFINES+tuple('transformer_blocks.0.attn.'+n+'.weight' for n in NORMS)


def form(weights,z,e,img_ids,txt_ids,mod_image,mod_text,normalizers,pos_embed,native,heads,operations,attention,phase):
    import torch
    import torch.nn.functional as F
    if mod_image.ndim!=2 or mod_text.ndim!=2 or mod_image.shape[-1]!=6*weights['x_embedder.weight'].shape[0] or mod_text.shape!=mod_image.shape:
        raise ValueError('Expected source two-set compiled modulation geometry')
    def linear(name,x):
        w=weights[name];y=F.linear(x,w)
        operations.append({**phase,'parameter':name,'inputShape':list(x.shape),'weightShape':list(w.shape),
            'inputStride':list(x.stride()),'dtype':str(x.dtype),'nominalMACs':x.numel()//x.shape[-1]*w.numel()})
        return y
    image=linear('x_embedder.weight',z);text=linear('context_embedder.weight',e)
    mi=mod_image[:,None,:].chunk(6,dim=-1);mt=mod_text[:,None,:].chunk(6,dim=-1)
    xi=(1+mi[1])*F.layer_norm(image,(image.shape[-1],),eps=1e-6)+mi[0]
    xt=(1+mt[1])*F.layer_norm(text,(text.shape[-1],),eps=1e-6)+mt[0]
    prefix='transformer_blocks.0.attn.'
    q=normalizers['norm_q'](linear(prefix+'to_q.weight',xi).unflatten(-1,(heads,-1)))
    ki=normalizers['norm_k'](linear(prefix+'to_k.weight',xi).unflatten(-1,(heads,-1)))
    vi=linear(prefix+'to_v.weight',xi).unflatten(-1,(heads,-1))
    kt=normalizers['norm_added_k'](linear(prefix+'add_k_proj.weight',xt).unflatten(-1,(heads,-1)))
    vt=linear(prefix+'add_v_proj.weight',xt).unflatten(-1,(heads,-1))
    image_rope=pos_embed(img_ids[0] if img_ids.ndim==3 else img_ids)
    text_rope=pos_embed(txt_ids[0] if txt_ids.ndim==3 else txt_ids)
    full_rope=tuple(torch.cat([t,i],0) for t,i in zip(text_rope,image_rope))
    q=native.apply_rotary_emb(q,image_rope,sequence_dim=1)
    k=native.apply_rotary_emb(torch.cat([kt,ki],1),full_rope,sequence_dim=1);v=torch.cat([vt,vi],1)
    y=native.dispatch_attention_fn(q,k,v,attn_mask=None,backend=None,parallel_config=None).flatten(2,3).to(q.dtype)
    attention.append({**phase,'queryShape':list(q.shape),'keyShape':list(k.shape),'valueShape':list(v.shape),
        'queryDtype':str(q.dtype),'keyDtype':str(k.dtype),'valueDtype':str(v.dtype),
        'queryStride':list(q.stride()),'keyStride':list(k.stride()),'valueStride':list(v.stride()),
        'requestedBackend':'auto','mask':None,'outputPayloadSHA256':payload(y)})
    return y


def load(path,config,device='cuda'):
    import torch
    from safetensors import safe_open
    from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
    weights={};loaded=[]
    with safe_open(str(path),framework='pt',device='cpu') as stream:
        for n in PARAMETERS:
            x=stream.get_tensor(n)
            if x.dtype!=torch.bfloat16:raise ValueError('Original parameter dtype differs')
            weights[n]=x.to(device)
            loaded.append({'name':n,'shape':list(x.shape),'bytes':x.numel()*x.element_size(),'payloadSHA256':payload(x)})
    norms={}
    for n in NORMS:
        with torch.device('meta'):m=torch.nn.RMSNorm(config['attention_head_dim'],eps=config['eps'],elementwise_affine=True)
        m.weight=torch.nn.Parameter(weights['transformer_blocks.0.attn.'+n+'.weight'],requires_grad=False)
        m.eval();norms[n]=m
    if len(weights)!=10 or sum(x['bytes'] for x in loaded)!=142344960:raise ValueError('Selected parameter closure differs')
    pos=Flux2PosEmbed(theta=config['rope_theta'],axes_dim=config['axes_dims_rope'])
    return weights,norms,pos,loaded
