"""Tiny independent numerical oracles and fresh-process target quarantine."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from encoder_atoms import read,save,sha,words
from paired_fields import retain,load
from source_native_writer import fit,predict


def test_ridge_matches_augmented_least_squares():
    rng=np.random.default_rng(17);raw=rng.normal(size=(4,5,4));raw[:,:,-1]=3;target=rng.normal(size=(4,5,2));mask=np.array([1,1,0,1,1],bool)
    model,info=fit(raw,target,[[0,1]],mask)
    x=((raw-model['mean'])/model['scale']).reshape(-1,4);y=target.reshape(-1,2)
    dx=((raw[1]-raw[0])/model['scale'])[mask];dy=(target[1]-target[0])[mask]
    design=np.vstack([np.column_stack([x,np.ones(len(x))])/np.sqrt(len(x)),np.column_stack([dx,np.zeros(len(dx))])/np.sqrt(len(dx)),
        np.column_stack([np.sqrt(.01)*np.eye(4),np.zeros(4)])])
    rhs=np.vstack([y/np.sqrt(len(y)),dy/np.sqrt(len(dy)),np.zeros((4,2))])
    expected=np.linalg.lstsq(design,rhs,rcond=None)[0]
    np.testing.assert_allclose(model['weight'],expected[:-1],atol=2e-13);np.testing.assert_allclose(model['bias'],expected[-1],atol=2e-13)
    assert model['scale'][-1]==1 and info['pairedRows']==4


def test_raw_zero_head_and_two_stage_rounding():
    model={'mean':np.array([0.,0.,2.]),'scale':np.array([1.,1.,2.]),'weight':np.array([[0.],[0.],[3.]]),'bias':np.array([.25])}
    z=np.zeros((1,2,1));previous=np.ones_like(z);head=np.full_like(z,4.)
    raw,bits=predict(model,z,previous,head,True)
    np.testing.assert_array_equal(raw,np.full_like(z,-1.75));np.testing.assert_array_equal(bits,words(raw.astype(np.float32)))
    assert not np.array_equal(raw,predict(model,z,previous,head,False)[0])


def test_seven_affines_against_independent_numpy_attention():
    torch=pytest.importorskip('torch')
    from source_native_reader import form,AFFINES,NORMS
    rng=np.random.default_rng(91);width=8;heads=2;head_dim=4;ni=4;nt=3
    shapes=[(width,3),(width,5),*[(width,width)]*5]
    weights={n:torch.tensor(rng.normal(size=shape)*.3,dtype=torch.float64) for n,shape in zip(AFFINES,shapes)}
    z=rng.normal(size=(1,ni,3));e=rng.normal(size=(1,nt,5));mi=rng.normal(size=(1,6*width))*.1;mt=rng.normal(size=(1,6*width))*.1
    norms={}
    for n in NORMS:
        m=torch.nn.RMSNorm(head_dim,eps=1e-6,dtype=torch.float64);m.weight.data.copy_(torch.tensor(rng.uniform(.5,1.5,size=head_dim)));norms[n]=m
    def pos(ids):
        angle=ids[:,0,None]*torch.tensor([.12,.27],dtype=torch.float64)
        return angle.cos(),angle.sin()
    def rotate(q,cs,sequence_dim=1):
        c,s=[a[None,:,None,:] for a in cs];a,b=q.chunk(2,-1)
        return torch.cat([a*c-b*s,b*c+a*s],-1)
    def attention(q,k,v,**kw):return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
    native=SimpleNamespace(apply_rotary_emb=rotate,dispatch_attention_fn=attention);ops=[];reads=[]
    ids=torch.arange(ni,dtype=torch.float64)[:,None];txt=torch.arange(nt,dtype=torch.float64)[:,None]+10
    args=[torch.tensor(a,dtype=torch.float64) for a in [z,e,mi,mt]]
    actual=form(weights,args[0],args[1],ids,txt,args[2],args[3],norms,pos,native,heads,ops,reads,{})
    w={n:v.numpy() for n,v in weights.items()}
    def mod(x,m):
        m=m[:,None,:].reshape(1,1,6,width)
        return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-6)*(1+m[:,:,1])+m[:,:,0]
    xi=mod(z@w[AFFINES[0]].T,mi);xt=mod(e@w[AFFINES[1]].T,mt)
    def rms(x,name):return x/np.sqrt(np.mean(x*x,-1,keepdims=True)+1e-6)*norms[name].weight.detach().numpy()
    q=rms((xi@w[AFFINES[2]].T).reshape(1,ni,heads,head_dim),'norm_q')
    ki=rms((xi@w[AFFINES[3]].T).reshape(1,ni,heads,head_dim),'norm_k');vi=(xi@w[AFFINES[4]].T).reshape(1,ni,heads,head_dim)
    kt=rms((xt@w[AFFINES[5]].T).reshape(1,nt,heads,head_dim),'norm_added_k');vt=(xt@w[AFFINES[6]].T).reshape(1,nt,heads,head_dim)
    def rotation(x,positions):
        a,b=np.split(x,2,-1);angle=np.asarray(positions)[None,:,None,None]*np.array([.12,.27])[None,None,None,:]
        return np.concatenate([a*np.cos(angle)-b*np.sin(angle),b*np.cos(angle)+a*np.sin(angle)],-1)
    q=rotation(q,np.arange(ni));k=rotation(np.concatenate([kt,ki],1),np.r_[np.arange(nt)+10,np.arange(ni)]);v=np.concatenate([vt,vi],1)
    scores=np.einsum('bqhd,bkhd->bhqk',q,k)/np.sqrt(head_dim);probs=np.exp(scores-scores.max(-1,keepdims=True));probs/=probs.sum(-1,keepdims=True)
    expected=np.einsum('bhqk,bkhd->bqhd',probs,v).reshape(1,ni,width)
    np.testing.assert_allclose(actual.detach().numpy(),expected,atol=2e-14)
    assert len(ops)==7 and len(reads)==1 and reads[0]['queryShape']==[1,ni,heads,head_dim]
    changed=args[0].clone();changed[:,-1]+=torch.tensor([.3,-.7,.5])
    second=form(weights,changed,args[1],ids,txt,args[2],args[3],norms,pos,native,heads,[],[],{})
    assert not torch.equal(actual[:,0],second[:,0])


def test_fresh_fit_and_serve_actors_quarantine_targets(tmp_path):
    rng=np.random.default_rng(32);raw=rng.normal(size=(4,5,3));target=rng.normal(size=(4,5,1));packet=tmp_path/'discovery.npz'
    np.savez(packet,features=raw,targets=target,pairs=np.array([[0,1]]),mask=np.ones(5,bool))
    denied=tmp_path/'future.npy';np.save(denied,np.ones(2));quarantine={'protectedRoots':[str(tmp_path)],'deniedCanaries':[str(denied)]}
    fit_spec=tmp_path/'fit-spec.json';save(fit_spec,{'packet':{'path':str(packet),'sha256':sha(packet)},'output':str(tmp_path/'fit'),**quarantine})
    worker=Path(__file__).parent/'source_native_writer.py'
    subprocess.run([sys.executable,str(worker),'fit',str(fit_spec)],check=True,capture_output=True,text=True)
    info=read(tmp_path/'fit/fit.json');seal=tmp_path/'seal.json';save(seal,{'fitReceiptSHA256':sha(tmp_path/'fit/fit.json'),'fields':info['fields']})
    fields={k:retain(tmp_path/'inputs',k,a) for k,a in zip(['z','previous','H0'],[raw[:1,:,:1],raw[:1,:,1:2],raw[:1,:,2:3]])}
    spec=tmp_path/'serve.json';save(spec,{'coefficientSeal':str(seal),'fitReceipt':str(tmp_path/'fit/fit.json'),'inputs':[{'id':'toy','population':'main',**fields,'output':str(tmp_path/'outputs')}],
        'receipt':str(tmp_path/'receipt.json'),**quarantine})
    subprocess.run([sys.executable,str(worker),'serve',str(spec)],check=True,capture_output=True,text=True)
    receipt=read(tmp_path/'receipt.json');assert len(receipt['rows'])==2 and receipt['frameworksLoaded']==[]
    assert receipt['fileQuarantine']['deniedCanaries']==[str(denied)]
    assert load(receipt['rows'][0]['velocity']).shape==(1,5,1)
