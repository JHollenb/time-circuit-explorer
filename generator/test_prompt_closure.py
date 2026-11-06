"""Small synthetic checks of the diagnostic instrument, without model data."""
from pathlib import Path
import subprocess
import sys
import unittest

class PromptClosureMechanics(unittest.TestCase):
    def test_coordinates_transport_and_finite_identity(self):
        code='''import numpy as np
from prompt_closure import permutation,transform,candidate,stored
ids=np.array([[[0,y,x,0] for y in range(2) for x in range(3)]])
h,v=permutation(ids,2),permutation(ids,1)
b=np.arange(12,dtype=np.float64).reshape(1,6,2);c=b+np.arange(12).reshape(1,6,2);o=b+3
assert np.array_equal(transform(c,'tokens',h,v),c.reshape(1,2,3,2)[:,:,::-1].reshape(1,6,2))
assert np.array_equal(transform(c,'text',h,v),c)
t=candidate(b,c,o,'transported','tokens',h,v)
m=t-o-c+b;correction=transform(c-b,'tokens',h,v)-(c-b)
assert np.array_equal(m,correction)
assert np.array_equal(t-o-transform(c-b,'tokens',h,v),m-correction)
assert np.array_equal(stored(np.array([-.5,.5,1.5,255.5]),'rgb'),[0,0,2,255])
'''
        subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,check=True,capture_output=True,text=True)

    def test_span_separates_adjustable_coefficients_from_orthogonal_residual(self):
        code='''import numpy as np
from prompt_span_study import project_span
d1=np.array([1.,0.,0.]);d2=np.array([0.,1.,0.]);m=np.array([2.,-3.,4.])
r,stats=project_span(d1,d2,m)
assert np.allclose(stats['gamma'],[2,-3])
assert np.allclose(r,[0,0,4]) and stats['rank']==2
r,stats=project_span(d1,2*d1,m)
assert np.allclose(r,[0,-3,4]) and stats['rank']==1
r,stats=project_span(np.zeros(3),np.zeros(3),np.zeros(3))
assert stats['rank']==0 and stats['unexplainedOverMixed'] is None
'''
        subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,check=True,capture_output=True,text=True)

if __name__=='__main__':unittest.main()
