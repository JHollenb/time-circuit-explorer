"""Current read/product sketch before the original output map; no output fitting."""
from row_output import PARAMETER, geometry
SOURCE_NAMES = ['product', 'attentionRead', 'gate', 'text', 'image']


def work(dim=3072, mlp=9216, text=512, image=1024, rank=128):
    n = text+image; width = dim+mlp
    candidate = {'sketch': n*width*rank, 'projectInputs': n*rank*width, 'compressedOriginalOut': 2*rank*width*dim, 'reconstructUpdate': n*rank*dim}
    diagnostic = {'sketchReconstruction': n*rank*rank, 'basisGram': n*rank*rank}
    inherited = 2*n*dim*rank+4*dim*rank*mlp+2*n*rank*mlp
    return {'candidate': candidate, 'candidateMACs': sum(candidate.values()), 'diagnostic': diagnostic, 'diagnosticMACs': sum(diagnostic.values()),
        'assayMACs': sum(candidate.values())+sum(diagnostic.values()), 'inherited82GateUpFormationMACs': inherited,
        'prospectiveComposedProducerMACs': inherited+sum(candidate.values()), 'matched82FullOutMACs': n*width*dim,
        'additionalQRApproxFLOPs': 4*n*rank*rank-8*rank**3/3,
        'scope': 'New sketch/output products and post-exit diagnostics separate. Composed count retains historical82 GU formation; source/attention/learned maps and QR/scalar/data work remain paid. No full composed producer or serving speed measured.'}


def abis(dim=3072, mlp=9216, text=512, image=1024, rank=128):
    from row_output import abis as previous
    result = previous(dim, mlp, text, image, rank)
    result.update(sketch={'shape': [1, text+image, rank], 'dtype': 'torch.float64'}, sketchSigns={'shape': [2, dim+mlp, rank], 'dtype': 'torch.float64'})
    return result


def signs(width, rank, stream):
    import numpy as np
    raw = np.random.PCG64(np.random.SeedSequence([20261107, 85, stream])).random_raw((width, rank))
    return raw, np.where((raw & np.uint64(1)) == 0, -1., 1.).astype(np.float64)


def compute(source, weight, matmul, qr, join_observer, sketch_observer, text=512, rank=128):
    import torch
    n = source['product'].shape[1]; dim = weight.shape[0]; mlp = source['product'].shape[-1]
    expected = {'product': ([1, n, mlp], torch.bfloat16), 'attentionRead': ([1, n, dim], torch.bfloat16), 'gate': ([1, 1, dim], torch.bfloat16),
        'text': ([1, text, dim], torch.bfloat16), 'image': ([1, n-text, dim], torch.bfloat16)}
    if set(source) != set(expected) or any(list(v.shape) != expected[k][0] or v.dtype != expected[k][1] for k, v in source.items()) or weight.shape != (dim, dim+mlp) or weight.dtype != torch.bfloat16 or not 0 < rank <= min(text, n-text):
        raise ValueError('Current-operand source ABI differs')
    joined = torch.cat([source['attentionRead'], source['product']], -1); join_observer(joined)
    joined64 = joined.double(); w = weight.double().T
    kept = {name: [] for name in ['rowBasis', 'qrR', 'columnSigns', 'projectedInputs', 'compressedWrites', 'sketch', 'sketchSigns']}; updates = []
    for stream, (lo, hi) in enumerate([(0, text), (text, n)]):
        raw, omega = signs(dim+mlp, rank, stream)
        omega = torch.from_numpy(omega).to(joined.device); sketch_observer(stream, raw, omega)
        f = matmul('sketch'+str(stream), joined64[0, lo:hi], omega)
        u, r = qr('rowQR'+str(stream), f)
        sign = torch.where(torch.diagonal(r) < 0, -torch.ones(rank, dtype=r.dtype, device=r.device), torch.ones(rank, dtype=r.dtype, device=r.device))
        u = u*sign; r = sign[:, None]*r
        k = matmul('projectInputs'+str(stream), u.T, joined64[0, lo:hi])
        b = matmul('compressedOut'+str(stream), k, w)
        y = matmul('reconstructUpdate'+str(stream), u, b)
        for name, value in zip(kept, [u, r, sign, k, b, f, omega]):
            kept[name].append(value)
        updates.append(y)
    complete = torch.cat(updates, 0)[None]; update = complete.float().bfloat16().contiguous()
    gated = source['gate']*update; incoming = torch.cat([source['text'], source['image']], 1); exited = incoming+gated
    result = {name: (torch.cat(values, 0)[None] if name in ['rowBasis', 'sketch'] else torch.stack(values)) for name, values in kept.items()}
    result.update(updateFP64=complete, preResidualUpdate=update, gatedWrite=gated, exit=exited, textOut=exited[:, :text], imageOut=exited[:, text:])
    trace = {'BF16JoinedInput': geometry(joined), 'admittedUpdate': geometry(update), 'gatedWrite': geometry(gated), 'exit': geometry(exited),
        'layoutAuthority': 'Same source-derived contiguous update geometry as82/83; no numerical equivalence implied.',
        'dataWork': {'joinedBF16Elements': joined.numel(), 'joinedBF16ToFP64Elements': joined.numel(), 'originalOutBF16ToFP64Elements': weight.numel(),
            'rawPCG64Words': 2*(dim+mlp)*rank, 'FP64SketchSignElements': 2*(dim+mlp)*rank, 'canonicalBasisMultiplyElements': n*rank, 'canonicalRMultiplyElements': 2*rank*rank,
            'completeFP64ToFP32Elements': complete.numel(), 'finalFP32ToBF16Elements': complete.numel(), 'gateMultiplyElements': gated.numel(), 'incomingConcatElements': incoming.numel(), 'residualAddElements': exited.numel()},
        'scope': 'Eight FP64 products and two current-operand QR calls. Fixed unscaled PCG64 sketches; no output fit/rank/seed adjustment. One final double admission then separate BF16 gate/add. All source and original learned work paid.'}
    return result, trace
