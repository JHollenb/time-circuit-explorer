import numpy as np
from deferred_values import form


def test_distinct_head_and_stream_weights_with_spliced_input():
    text=np.asarray([[1.,2.,-1.],[-3.,1.,2.],[2.,0.,4.]])
    selected=np.asarray([7.,-2.,3.]);text[1]=selected
    image=np.asarray([[2.,-1.,3.],[4.,2.,0.]])
    probability=np.asarray([[[.1,.2,.3,.15,.25],[.3,.1,.2,.05,.35]],
                            [[.2,.05,.1,.4,.25],[.05,.15,.3,.2,.3]]])
    tw=np.arange(12,dtype=np.float64).reshape(4,3)/5-1
    iw=np.arange(12,dtype=np.float64)[::-1].reshape(4,3)/7+.2
    operations=[]
    def mm(a,b,name):operations.append(name);return a@b
    result=form(probability,text,image,tw,iw,mm)
    expected=np.zeros((2,2,2))
    for h in range(2):
        for q in range(2):
            for c in range(2):
                for j in range(5):
                    x=text[j] if j<3 else image[j-3]
                    w=tw[2*h+c] if j<3 else iw[2*h+c]
                    expected[h,q,c]+=probability[h,q,j]*sum(float(a)*float(b) for a,b in zip(x,w))
    np.testing.assert_allclose(result['H_FP64'],expected,rtol=1e-14,atol=1e-14)
    assert operations==['weighted-text-input','weighted-image-input','deferred-text-V','deferred-image-V']


def test_stream_probability_mass_is_not_renormalized():
    p=np.asarray([[[.1,.2,.7]]]);xt=np.asarray([[1.,0.],[1.,0.]]);xi=np.asarray([[0.,1.]])
    w=np.eye(2)
    result=form(p,xt,xi,w,w,lambda a,b,n:a@b)
    np.testing.assert_allclose(result['textContribution'],[[[.3,0.]]])
    np.testing.assert_allclose(result['imageContribution'],[[[0.,.7]]])
