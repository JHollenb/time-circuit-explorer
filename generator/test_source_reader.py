import math
import torch
from source_reader import derive,project_norm


def test_selected_inputs_keep_original_source_and_position_order():
    text=torch.arange(42,dtype=torch.float64).reshape(1,7,6)
    mod=torch.tensor([[.25,-.5]])
    cos=torch.arange(60,dtype=torch.float32).reshape(10,6);sin=-cos
    got=derive(text,mod,cos,sin,[4,1,5])
    assert got['Ptext'] is text and got['txtMod'] is mod
    assert got['R'].tolist()==[[[24.,25.,26.,27.,28.,29.],[6.,7.,8.,9.,10.,11.],[30.,31.,32.,33.,34.,35.]]]
    assert got['ropeCos'].tolist()==[[24.,25.,26.,27.,28.,29.],[6.,7.,8.,9.,10.,11.],[30.,31.,32.,33.,34.,35.]]
    assert torch.equal(got['ropeSin'],-got['ropeCos']) and got['R'].shape==(1,3,6)
    assert got['fullRopeCos'] is cos and got['fullRopeSin'] is sin


def test_projection_then_per_head_native_rms_matches_scalar_formula():
    x=torch.tensor([[[1.,-2.,.5],[2.,1.,-1.]]],dtype=torch.float64)
    weight=torch.tensor([[1.,2.,3.],[-2.,1.,.5],[.5,-1.,2.],[2.,.25,-1.]],dtype=torch.float64)
    gamma=torch.tensor([.75,1.25],dtype=torch.float64)
    projected,norm=project_norm(x,weight,gamma,heads=2)
    for row in range(2):
        for head in range(2):
            z=[sum(x[0,row,a].item()*weight[head*2+c,a].item() for a in range(3)) for c in range(2)]
            expected=[z[c]*gamma[c].item()/math.sqrt(sum(t*t for t in z)/2+1e-6) for c in range(2)]
            assert projected[0,row,head].tolist()==z
            torch.testing.assert_close(norm[0,row,head],torch.tensor(expected,dtype=torch.float64),rtol=1e-14,atol=1e-14)
