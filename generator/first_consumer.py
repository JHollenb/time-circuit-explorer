"""Source-ordered image-local path and finite-change identities."""
from encoder_atoms import np
from paired_fields import metrics

SELECTED={'x_embedder.weight':[3072,128],'transformer_blocks.0.attn.to_out.0.weight':[3072,3072],
    'transformer_blocks.0.ff.linear_in.weight':[18432,3072],'transformer_blocks.0.ff.linear_out.weight':[3072,9216]}
STAGES=['projection','gatedCarrier','residualCommit','normalization','normScaled','mlpInput','gateOperand','upOperand','activation','product','mlpWrite','gatedMLP','imageExit']


def make_local(torch,native):
    # The generic selected loader admits exactly these four named parameters.
    model=torch.nn.Module();model.x_embedder=torch.nn.Linear(128,3072,bias=False)
    block=torch.nn.Module();block.attn=torch.nn.Module();block.attn.to_out=torch.nn.ModuleList([torch.nn.Linear(3072,3072,bias=False),torch.nn.Dropout(0.)]);block.norm2=torch.nn.LayerNorm(3072,elementwise_affine=False,eps=1e-6);block.ff=native.Flux2FeedForward(3072,mult=3.,bias=False)
    model.transformer_blocks=torch.nn.ModuleList([block])
    if {n:list(p.shape) for n,p in model.named_parameters()}!=SELECTED:raise ValueError('Selected local parameter closure differs')
    return model


def path(block,embedding,write,modulation,capture):
    (_,_,_), (shift,scale,gate)=modulation
    u=embedding+write;capture('residualCommit',u)
    n=block.norm2(u);capture('normalization',n)
    scaled=n*(1+scale);capture('normScaled',scaled)
    z=scaled+shift;capture('mlpInput',z)
    ff=block.ff(z);capture('mlpWrite',ff)
    gated=gate*ff;capture('gatedMLP',gated)
    y=u+gated;capture('imageExit',y);return y


def ln_terms(a,b,epsilon=1e-6):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    if a.shape!=b.shape or a.ndim<1 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Normalization input contract differs')
    c=a-a.mean(-1,keepdims=True);delta=b-a;h=delta-delta.mean(-1,keepdims=True);cb=b-b.mean(-1,keepdims=True)
    sa=np.sqrt(np.mean(c*c,axis=-1,keepdims=True)+epsilon);sb=np.sqrt(np.mean(cb*cb,axis=-1,keepdims=True)+epsilon)
    direct=h/sb;rescale=c*(1/sb-1/sa);reference=cb/sb-c/sa
    return {'direct':direct,'rescale':rescale,'reference':reference,'closure':reference-direct-rescale}


def product_terms(fa,fb,ua,ub):
    fa,fb,ua,ub=[np.asarray(x,np.float64) for x in [fa,fb,ua,ub]]
    if any(x.shape!=fa.shape for x in [fb,ua,ub]) or not all(np.isfinite(x).all() for x in [fa,fb,ua,ub]):raise ValueError('Product input contract differs')
    gate=(ua+ub)*.5*(fb-fa);up=(fa+fb)*.5*(ub-ua);reference=fb*ub-fa*ua
    return {'gate':gate,'up':up,'reference':reference,'closure':reference-gate-up}


def silu64(x):
    x=np.asarray(x,np.float64);e=np.exp(-np.abs(x));return x*np.where(x>=0,1/(1+e),e/(1+e))


def energy_terms(a,b,total):
    ea=float(np.sum(a*a));eb=float(np.sum(b*b));cross=float(2*np.sum(a*b));et=float(np.sum(total*total))
    return {'firstEnergy':ea,'secondEnergy':eb,'twiceCross':cross,'sumEnergy':et,'energyClosure':et-ea-eb-cross}


def response_account(d0,d1):
    return {**metrics(d1,d0),'responseR0Energy':float(np.sum(d0*d0)),'responseR1Energy':float(np.sum(d1*d1)),'interactionEnergy':float(np.sum((d1-d0)**2)),
        'interactionOverR0':float(np.linalg.norm(d1-d0)/np.linalg.norm(d0)) if np.any(d0) else None}
