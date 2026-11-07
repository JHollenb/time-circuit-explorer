"""Fixed packed-cell geometry and innovation law; neural imports remain lazy."""
import math
import numpy as np
from paired_fields import payload


def pool(value):
    a=np.asarray(value)
    if a.shape!=(1,1024,128) or not np.isfinite(a).all():raise ValueError('Expected finite complete packed field')
    x=a.reshape(1,32,32,128)
    return (((x[:,0::2,0::2]+x[:,0::2,1::2])+(x[:,1::2,0::2]+x[:,1::2,1::2]))*.25).reshape(1,256,128)


def lift(value):
    a=np.asarray(value)
    if a.shape!=(1,256,128) or not np.isfinite(a).all():raise ValueError('Expected finite coarse packed field')
    return np.repeat(np.repeat(a.reshape(1,16,16,128),2,axis=1),2,axis=2).reshape(1,1024,128)


def geometry(ids):
    a=np.asarray(ids)
    h,w=np.meshgrid(np.arange(32),np.arange(32),indexing='ij')
    expected=np.stack([np.zeros_like(h),h,w,np.zeros_like(h)],axis=-1).reshape(1,1024,4)
    if a.shape!=expected.shape or not np.array_equal(a,expected):raise ValueError('Original physical image grid differs')
    h,w=np.meshgrid(np.arange(16),np.arange(16),indexing='ij')
    return np.stack([np.zeros_like(h),2*h+.5,2*w+.5,np.zeros_like(h)],axis=-1).reshape(1,256,4).astype(np.float32)


def mask():
    return np.concatenate([np.zeros(512,dtype=np.float32),np.full(256,math.log(4),dtype=np.float32)]).reshape(1,1,1,768)


def pool_torch(value):
    x=value.float().reshape(1,32,32,128)
    return (((x[:,0::2,0::2]+x[:,0::2,1::2])+(x[:,1::2,0::2]+x[:,1::2,1::2]))*.25).reshape(1,256,128)


def lift_torch(value):
    return value.reshape(1,16,16,128).repeat_interleave(2,dim=1).repeat_interleave(2,dim=2).reshape(1,1024,128)


def check(kwargs,numerical,contract,expected):
    if numerical!=contract['numericalProfile']:raise ValueError('Coarse profile differs')
    if any(kwargs.get(n) is not None for n in ['guidance','kv_cache','kv_cache_mode']) or kwargs.get('num_ref_tokens',0)!=0:
        raise ValueError('Unsupported coarse guidance/cache/reference')
    for n,shape,dtype in [('hidden_states',[1,256,128],contract['dtype']),('encoder_hidden_states',contract['EShape'],contract['dtype']),
                          ('img_ids',[1,256,4],'torch.float32'),('txt_ids',[1,512,4],'torch.int64')]:
        a=kwargs[n]
        if list(a.shape)!=shape or str(a.dtype)!=dtype or payload(a)!=expected[n]:raise ValueError('Coarse actual operand differs: '+n)
    overrides=kwargs.get('joint_attention_kwargs')
    if not isinstance(overrides,dict) or set(overrides)!={'attention_mask'}:raise ValueError('Only fixed coarse mask admitted')
    a=overrides['attention_mask']
    if list(a.shape)!=[1,1,1,768] or str(a.dtype)!='torch.float32' or payload(a)!=expected['attention_mask']:
        raise ValueError('Coarse key multiplicity differs')
    if payload(kwargs['timestep'])!=expected['timestep']:raise ValueError('Coarse clock differs')


def split_error(coarse,held,native,emitted):
    coarse,held,native,emitted=[np.asarray(a,dtype=np.float64) for a in [coarse,held,native,emitted]]
    projected=lift(pool(native-held))
    coarse_error=lift(coarse-pool(native));missing=-(native-held-projected)
    smooth=held+lift(coarse-pool(held));finite=emitted-smooth
    return {'coarseProducerError':coarse_error,'omittedFineInnovation':missing,'finiteEmissionError':finite,
        'actualVelocityError':emitted-native,'nativeInnovation':native-held,'nativeCoarseInnovation':projected,
        'smoothPrediction':smooth}
