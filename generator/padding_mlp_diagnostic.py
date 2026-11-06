"""Paid original MLP on the sealed candidates' actual normalized receivers."""
from pathlib import Path
import time
from encoder_atoms import read,save,sha
from paired_fields import load,retain,payload
from padding_mlp import SourceWeights


def execute(root):
    import torch
    from torch.nn import functional as F
    from complete_masked_study import profile
    from padding_mlp_study import verify,install_guard
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json')
    seal=read(root/'candidate-seal.json');seal_sha=sha(root/'candidate-seal.json')
    if seal['requestSHA256']!=r['requestSHA256'] or seal['encoderSHA256']!=sha(root/'encoder-candidate.json') or seal['imagesSHA256']!=sha(root/'images-candidate.json'):raise ValueError('Candidates must seal before diagnostic')
    install_guard(root);tick=time.perf_counter();profile(torch);weights=SourceWeights(s);ps={k:weights.parameter('model.layers.0.mlp.'+name+'.weight') for k,name in [('G','gate_proj'),('U','up_proj'),('D','down_proj')]};enc=read(root/'encoder-candidate.json');fields={};actual=[]
    with torch.inference_mode():
        for c in ['00','10']:
            a,b=[enc['firstMLPFields'][policy+'-'+c] for policy in ['constant','tangent']]
            x=load(a['normalized'],'cuda')
            if a['normalized']['payloadSHA256']!=b['normalized']['payloadSHA256'] or a['residual']['payloadSHA256']!=b['residual']['payloadSHA256']:raise ValueError('Common-parent receiver differs between policies')
            write=F.linear(F.silu(F.linear(x,ps['G']))*F.linear(x,ps['U']),ps['D']);residual=load(a['residual'],'cuda');h1=residual+write
            fields[c]={'write':retain(root/'arrays/diagnostic'/c,'native-write',write),'H1':retain(root/'arrays/diagnostic'/c,'native-H1',h1)};actual.append({'case':c,'actualNormalizedPayloadSHA256':payload(x),'candidateNormalizedFieldSHA256':a['normalized']['sha256']})
    weights.unchanged()
    if sha(root/'candidate-seal.json')!=seal_sha:raise ValueError('Candidate seal changed')
    save(root/'native-write-diagnostic.json',{'schema':'tce-padding-native-write-diagnostic-v1','requestSHA256':r['requestSHA256'],'releasedAfterCandidateSealSHA256':seal_sha,'fields':fields,'actualInputs':actual,
        'sourceParameterReads':weights.reads,'sourceAssetChecks':weights.checks,'work':{'originalMLPEvaluations':2,'originalLinears':6,'nominalLinearMACs':2*512*3*2560*9728},'workerSecondsAfterImports':time.perf_counter()-tick,
        'scope':'Paid original512row MLP evaluated after all4images seal. Own first receiver shared exactly between policies; no native value enters primary serving.'})
