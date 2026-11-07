"""Small independent ridge oracle, local-operand mask and actor quarantine."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from paired_fields import retain
from source_response import ridge,mask,features


def test_dual_ridge_matches_augmented_least_squares():
    rng=np.random.default_rng(83);x=rng.normal(size=(6,9));y=rng.normal(size=(6,3));b,info=ridge(x,y)
    expected=np.linalg.lstsq(np.vstack([x,np.sqrt(6*.01)*np.eye(9)]),np.vstack([y,np.zeros((9,3))]),rcond=None)[0]
    np.testing.assert_allclose(b,expected,atol=2e-13)
    assert info['gramNominalMACs']==6*6*9 and info['intercept'] is False


def test_mask_and_actual_local_operand_admission(tmp_path):
    assert mask('bottom-right').sum()==944 and mask('top-left').sum()==944
    z=np.zeros((1,1024,128),np.float32);h=np.zeros((1,1024,3072),np.float32)
    a={k:retain(tmp_path/'A',k,v) for k,v in [('z',z),('previous',z),('H0',h)]}
    zz=z.copy();zz[:,1023]=1;zz[:,0,0]=2;hh=h.copy();hh[:,1,17]=4
    b={k:retain(tmp_path/'B',k,v) for k,v in [('z',zz),('previous',z),('H0',hh)]}
    x,selected,app=features({'A':a,'B':b,'location':'bottom-right'},np.full(3072,2.))
    assert x.shape==(943,3072) and app['inapplicableQueries']==[0]
    assert not selected[0] and x[0,17]==2 and np.count_nonzero(x)==1


def test_framework_and_target_denial_in_fresh_actor(tmp_path):
    allowed=tmp_path/'source.npy';future=tmp_path/'future.npy';np.save(allowed,np.ones(2));np.save(future,np.ones(2))
    code='''import sys,json
sys.path.insert(0,sys.argv[1])
from encoder_atoms import guard
from source_response import quarantine
guard()
q=quarantine(sys.argv[2],{'protectedRoots':[sys.argv[2]],'deniedCanaries':[sys.argv[4]]},[sys.argv[3]],[sys.argv[5]])
with open(sys.argv[3],'rb') as f: f.read()
try: import torch
except ImportError: pass
else: raise RuntimeError('Framework guard absent')
print(json.dumps(q))
'''
    p=subprocess.run([sys.executable,'-c',code,str(Path(__file__).parent),str(tmp_path/'spec.json'),str(allowed),str(future),str(tmp_path/'out')],check=True,capture_output=True,text=True)
    assert json.loads(p.stdout)['deniedCanaries']==[str(future)]
