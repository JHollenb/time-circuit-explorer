"""One receiver-conditioned linear component on authenticated paid operands."""

SOURCE_NAMES = ['activation', 'upReference', 'upMean', 'selection_support', 'weightD', 'gate']
COMPARISON_NAMES = ['centeredProduct', 'centeredOutput', 'policyOutput', 'totalOutput']


def abis(dim=3072, mlp=9216, text=512, image=1024):
    return {name: {'shape': shape, 'dtype': 'torch.float64'} for name, shape in {
        'activationMean': [1, 2, mlp], 'linearProduct': [1, text+image, mlp],
        'linearOutput': [1, text+image, dim], 'gatedLinear': [1, text+image, dim],
        'gatedLinearMean': [1, 2, dim], 'gatedLinearCentered': [1, text+image, dim],
    }.items()}


def compute(source, linear, text=512):
    import torch
    a = source['activation'].double(); u = source['upReference']; n = a.shape[1]
    def mean(x): return torch.cat([x[:, :text].mean(1, keepdim=True), x[:, text:].mean(1, keepdim=True)], 1)
    def expand(x): return torch.cat([x[:, :1].expand(-1, text, -1), x[:, 1:].expand(-1, n-text, -1)], 1)
    ma = mean(a)
    product = expand(ma) * (u-expand(source['upMean']))
    product.index_fill_(-1, source['selection_support'], 0)
    y = linear('meanGateDown', product, source['weightD'])
    gated = source['gate'].double()*y; mg = mean(gated)
    return {'activationMean': ma, 'linearProduct': product, 'linearOutput': y,
            'gatedLinear': gated, 'gatedLinearMean': mg, 'gatedLinearCentered': gated-expand(mg)}
