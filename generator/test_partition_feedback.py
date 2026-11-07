import torch
from partition_feedback import text_read,mechanics


def test_one_denominator_and_live_image_key_competition():
    q=torch.ones(1,2,1,2);key=torch.zeros(1,6,1,2);v=torch.ones_like(q);calls=[]
    def product(kind,a,b):calls.append((kind,tuple(a.shape),tuple(b.shape)));return torch.matmul(a,b)
    read,alpha=text_read(q,key,v,product)
    torch.testing.assert_close(alpha,torch.full((1,1,2),1/3));torch.testing.assert_close(read,torch.full((1,2,2),1/3))
    key[:,2:]=3;changed,changed_alpha=text_read(q,key,v,product)
    assert torch.all(changed_alpha<alpha) and torch.all(changed<read)
    assert calls[:2]==[('textScores',(1,1,2,2),(1,1,2,6)),('textAV',(1,1,2,2),(1,1,2,2))]


def test_native_reference_and_two_own_commits():
    result=mechanics()
    assert result['ownInputChanges'] and result['pretrainedModelEvaluations']==0
    assert len(result['blocks'])==10 and sum(x['textUpdated'] for x in result['blocks'])==8
    assert result['terminalMapCalls']==[2,2] and len(result['explicitProducts'])==16
    assert all(a[-2:]==('torch.float32','torch.float32') for a in result['explicitProducts'])


def test_fp32_products_even_inside_cpu_autocast():
    q=torch.ones(1,2,1,2,dtype=torch.bfloat16);k=torch.zeros(1,6,1,2,dtype=torch.bfloat16);v=q.clone();types=[]
    def product(kind,a,b):
        out=torch.matmul(a,b);types.append((a.dtype,b.dtype,out.dtype));return out
    with torch.autocast('cpu',dtype=torch.bfloat16):output,alpha=text_read(q,k,v,product)
    assert types==[(torch.float32,torch.float32,torch.float32)]*2 and output.dtype==torch.bfloat16 and alpha.dtype==torch.float32
