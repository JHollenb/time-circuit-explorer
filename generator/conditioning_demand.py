"""Conditioning layout and signed consumer-error accounting."""
import numpy as np


def parameter_from_words(words, shape, device):
    import torch
    # Parameters need version counters even when their caller is in inference mode.
    with torch.inference_mode(False):
        return torch.from_numpy(np.array(words, dtype=np.uint16, copy=True)).view(torch.bfloat16).reshape(shape).to(device)


def pack(states):
    import torch
    selected = [states[i] for i in (9, 18, 27)]
    if any(x.ndim != 3 or x.shape != selected[0].shape for x in selected):
        raise ValueError('Selected states must share B,T,D geometry')
    joined = torch.stack(selected, dim=1)
    b, count, length, width = joined.shape
    return joined.permute(0, 2, 1, 3).reshape(b, length, count * width)


def alignment(old, new, baseline):
    old_error = np.asarray(old, dtype=np.float64) - baseline
    new_error = np.asarray(new, dtype=np.float64) - baseline
    change = old_error - new_error
    energy = lambda x: float(np.sum(x * x))
    a, b, c = energy(old_error), energy(new_error), energy(change)
    dot = float(np.sum(old_error * new_error)); cross = 2 * float(np.sum(new_error * change))
    return {'oldErrorEnergy': a, 'newErrorEnergy': b, 'errorChangeEnergy': c,
            'oldNewErrorDot': dot, 'oldNewErrorCosine': dot / np.sqrt(a * b) if a and b else None,
            'twiceNewErrorDotChange': cross, 'energyIdentityRemainder': a - b - c - cross,
            'scalars': int(old_error.size)}
