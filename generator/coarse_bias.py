"""One explicit requested-to-consumed attention-bias conversion."""


def consumed_bias(requested,query):
    if str(requested.dtype)!='torch.float32' or str(query.dtype)!='torch.bfloat16' or len(query.shape)!=4 or list(requested.shape)!=[1,1,1,query.shape[1]]:
        raise ValueError('Unsupported coarse attention-bias conversion')
    return requested.to(query.dtype)
