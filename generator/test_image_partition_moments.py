import math
import torch
from image_partition_moments import text_read,mechanics


def evaluate(q,k,v):
    captured={};products=[]
    def product(name,a,b):
        result=torch.matmul(a,b);products.append((name,list(a.shape),list(b.shape),str(result.dtype)));return result
    output,alpha=text_read(q,k,v,product,lambda values:captured.update(values))
    return output,alpha,captured,products


def test_constant_image_keys_exact_partition_and_one_output_boundary():
    q=torch.tensor([[[[.2,.5]],[[.7,-.4]]]],dtype=torch.float32)
    k=torch.tensor([[[[0.,0.]],[[.1,-.2]],[[.3,.4]],[[.3,.4]],[[.3,.4]]]])
    v=torch.tensor([[[[.4,.8]],[[.7,-.1]]]])
    output,alpha,c,products=evaluate(q,k,v)
    scores=torch.matmul(q.transpose(1,2),k.transpose(1,2).transpose(-1,-2))/math.sqrt(2)
    probabilities=scores.softmax(-1)
    expected=torch.matmul(probabilities[...,:2],v.transpose(1,2)).transpose(1,2).flatten(2,3)
    torch.testing.assert_close(output,expected,rtol=2e-6,atol=2e-7)
    torch.testing.assert_close(alpha,probabilities[...,:2].sum(-1),rtol=2e-6,atol=2e-7)
    torch.testing.assert_close(c['covariance'],torch.zeros(1,1,2,2),atol=1e-12,rtol=0)
    assert [x[0] for x in products]==['imageCovariance','queryCovariance','queryMean','textScores','textAV']


def test_population_covariance_retains_off_diagonal_structure():
    q=torch.tensor([[[[1.,1.]],[[1.,-1.]]]])
    k=torch.tensor([[[[0.,0.]],[[0.,0.]],[[1.,1.]],[[-1.,-1.]]]])
    _,_,c,_=evaluate(q,k,torch.ones_like(q))
    torch.testing.assert_close(c['covariance'],torch.ones(1,1,2,2))
    torch.testing.assert_close(c['quadratic'],torch.tensor([[[2.,0.]]]))
    assert not torch.equal(c['covariance'],torch.diag_embed(c['covariance'].diagonal(dim1=-2,dim2=-1)))


def test_image_key_translation_changes_only_first_cumulant():
    q=torch.arange(12,dtype=torch.float32).reshape(1,3,2,2)/10;k=torch.arange(28,dtype=torch.float32).reshape(1,7,2,2)/30
    v=torch.ones_like(q);_,_,before,_=evaluate(q,k,v);delta=torch.tensor([.3,-.2]);changed=k.clone();changed[:,3:]+=delta
    _,_,after,_=evaluate(q,changed,v)
    torch.testing.assert_close(before['covariance'],after['covariance'],rtol=2e-6,atol=2e-7)
    torch.testing.assert_close(after['logZI']-before['logZI'],(q.transpose(1,2)*delta).sum(-1)/math.sqrt(2),rtol=3e-6,atol=3e-7)
    torch.testing.assert_close(before['logZT'],after['logZT'],rtol=0,atol=0)


def test_all_products_stay_fp32_under_autocast():
    q=torch.ones(1,2,1,2,dtype=torch.bfloat16);k=torch.zeros(1,6,1,2,dtype=torch.bfloat16)
    with torch.autocast('cpu',dtype=torch.bfloat16):output,alpha,c,products=evaluate(q,k,q)
    assert output.dtype==torch.bfloat16 and alpha.dtype==torch.float32
    assert all(x[-1]=='torch.float32' for x in products) and all(x.dtype==torch.float32 for x in c.values())


def test_native_block_direct_score_moment_reference_two_own_commits():
    result=mechanics()
    assert result['ownInputChanges'] and result['pretrainedModelEvaluations']==0
    assert len(result['blocks'])==10 and sum(x['textUpdated'] for x in result['blocks'])==8
    assert len(result['explicitProducts'])==40 and len(result['momentShapes'])==8 and result['terminalMapCalls']==[2,2]
    assert all(x['covariance']==[1,2,8,8] for x in result['momentShapes'])
