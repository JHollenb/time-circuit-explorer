"""Select current row directions in the receiving gated-write metric."""
from row_output import PARAMETER, geometry

SOURCE_NAMES = ['product', 'attentionRead', 'gate', 'text', 'image']


def work(dim=3072, mlp=9216, text=512, image=1024, rank=128):
    n, width = text + image, dim + mlp
    candidate = {'pullbackSketch': width * dim * rank,
                 'sketch': n * width * rank, 'projectInputs': n * rank * width,
                 'compressedOriginalOut': 2 * rank * width * dim,
                 'reconstructUpdate': n * rank * dim}
    diagnostic = {'sketchReconstruction': n * rank * rank, 'basisGram': n * rank * rank}
    inherited = 2*n*dim*rank + 4*dim*rank*mlp + 2*n*rank*mlp
    return {'candidate': candidate, 'candidateMACs': sum(candidate.values()),
            'diagnostic': diagnostic, 'diagnosticMACs': sum(diagnostic.values()),
            'assayMACs': sum(candidate.values()) + sum(diagnostic.values()),
            'inherited82GateUpFormationMACs': inherited,
            'prospectiveComposedProducerMACs': inherited + sum(candidate.values()),
            'matched82FullOutMACs': n * width * dim,
            'additionalQRApproxFLOPs': 4*n*rank*rank - 8*rank**3/3,
            'scope': 'Shared learned pullback and eight sketch/output products paid. '
                     'Diagnostics/QR/scalar/data work separate; inherited GU composition '
                     'is prospective. Source, attention and original learned maps remain paid.'}


def abis(dim=3072, mlp=9216, text=512, image=1024, rank=128):
    from row_output import abis as previous
    result = previous(dim, mlp, text, image, rank)
    result.update(sketch={'shape': [1, text + image, rank], 'dtype': 'torch.float64'},
                  outputSigns={'shape': [dim, rank], 'dtype': 'torch.float64'},
                  gatedOutputSigns={'shape': [dim, rank], 'dtype': 'torch.float64'},
                  pulledBackSketch={'shape': [dim + mlp, rank], 'dtype': 'torch.float64'})
    return result


def signs(dim, rank):
    import numpy as np
    raw = np.random.PCG64(np.random.SeedSequence([20261107, 86])).random_raw((dim, rank))
    return raw, np.where((raw & np.uint64(1)) == 0, -1., 1.).astype(np.float64)


def compute(source, weight, matmul, qr, join_observer, sketch_observer, text=512, rank=128):
    import torch
    n, dim, mlp = source['product'].shape[1], weight.shape[0], source['product'].shape[-1]
    expected = {'product': ([1, n, mlp], torch.bfloat16),
                'attentionRead': ([1, n, dim], torch.bfloat16),
                'gate': ([1, 1, dim], torch.bfloat16),
                'text': ([1, text, dim], torch.bfloat16),
                'image': ([1, n-text, dim], torch.bfloat16)}
    if (set(source) != set(expected)
            or any(list(v.shape) != expected[k][0] or v.dtype != expected[k][1]
                   for k, v in source.items())
            or weight.shape != (dim, dim+mlp) or weight.dtype != torch.bfloat16
            or not 0 < rank <= min(text, n-text)):
        raise ValueError('Receiver-weighted source ABI differs')
    joined = torch.cat([source['attentionRead'], source['product']], -1)
    join_observer(joined)
    joined64, w = joined.double(), weight.double().T
    raw, omega_np = signs(dim, rank)
    omega = torch.from_numpy(omega_np).to(joined.device)
    gated_omega = source['gate'][0, 0].double()[:, None] * omega
    pulled = matmul('pullbackSketch', w, gated_omega)
    sketch_observer(raw, omega, gated_omega, pulled)
    kept = {name: [] for name in ['rowBasis', 'qrR', 'columnSigns',
                                  'projectedInputs', 'compressedWrites', 'sketch']}
    updates = []
    for stream, (lo, hi) in enumerate([(0, text), (text, n)]):
        f = matmul('sketch'+str(stream), joined64[0, lo:hi], pulled)
        u, r = qr('rowQR'+str(stream), f)
        sign = torch.where(torch.diagonal(r) < 0,
                           -torch.ones(rank, dtype=r.dtype, device=r.device),
                           torch.ones(rank, dtype=r.dtype, device=r.device))
        u, r = u * sign, sign[:, None] * r
        k = matmul('projectInputs'+str(stream), u.T, joined64[0, lo:hi])
        b = matmul('compressedOut'+str(stream), k, w)
        y = matmul('reconstructUpdate'+str(stream), u, b)
        for name, value in zip(kept, [u, r, sign, k, b, f]):
            kept[name].append(value)
        updates.append(y)
    complete = torch.cat(updates, 0)[None]
    update = complete.float().bfloat16().contiguous()
    gated = source['gate'] * update
    incoming = torch.cat([source['text'], source['image']], 1)
    exited = incoming + gated
    result = {name: (torch.cat(values, 0)[None] if name in ['rowBasis', 'sketch']
                     else torch.stack(values)) for name, values in kept.items()}
    result.update(outputSigns=omega, gatedOutputSigns=gated_omega, pulledBackSketch=pulled,
                  updateFP64=complete, preResidualUpdate=update, gatedWrite=gated,
                  exit=exited, textOut=exited[:, :text], imageOut=exited[:, text:])
    trace = {'BF16JoinedInput': geometry(joined), 'admittedUpdate': geometry(update),
             'gatedWrite': geometry(gated), 'exit': geometry(exited),
             'layoutAuthority': 'Same source-derived contiguous update geometry as82/85; '
                                'no numerical equivalence implied.',
             'dataWork': {'joinedBF16Elements': joined.numel(),
                          'joinedBF16ToFP64Elements': joined.numel(),
                          'originalOutBF16ToFP64Elements': weight.numel(),
                          'rawPCG64Words': dim*rank, 'FP64OutputSignElements': dim*rank,
                          'gateBF16ToFP64Elements': dim, 'gateSignMultiplyElements': dim*rank,
                          'canonicalBasisMultiplyElements': n*rank,
                          'canonicalRMultiplyElements': 2*rank*rank,
                          'completeFP64ToFP32Elements': complete.numel(),
                          'finalFP32ToBF16Elements': complete.numel(),
                          'gateMultiplyElements': gated.numel(),
                          'incomingConcatElements': incoming.numel(),
                          'residualAddElements': exited.numel()},
             'scope': 'One shared learned pullback plus eight FP64 products/two QR. '
                      'Output-coordinate PCG64 signs and current gate are fixed sources. '
                      'No target/basis fitting or search. Final double admission and '
                      'separate BF16 gate/add; all original learned work paid.'}
    return result, trace
