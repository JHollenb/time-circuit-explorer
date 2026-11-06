"""One scoped numerical policy; always restore the caller's original flag."""
from contextlib import contextmanager


@contextmanager
def projection_policy(policy, is_kv):
    import torch
    if policy not in ('original', 'alternative'):
        raise ValueError('Unknown accumulation policy')
    before = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        if policy == 'alternative' and is_kv:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = before
