"""Integer RGB comparison without unsigned subtraction or hidden clipping."""
import numpy as np


def png_account(full, old, new):
    arrays = [np.asarray(a) for a in (full, old, new)]
    if any(a.dtype != np.uint8 or a.shape != arrays[0].shape for a in arrays) or arrays[0].ndim != 3 or arrays[0].shape[-1] != 3:
        raise ValueError('Matched H,W,3 uint8 PNG pixels required')
    full, old, new = [a.astype(np.float64) for a in arrays]
    a, b = old - full, new - full
    old_pixel = np.mean(a * a, axis=-1); new_pixel = np.mean(b * b, axis=-1)
    result = {'pixels': int(old_pixel.size), 'channels': int(a.size),
        'pixelSquaredErrorImproved': int(np.sum(new_pixel < old_pixel)), 'pixelSquaredErrorWorsened': int(np.sum(new_pixel > old_pixel)),
        'pixelSquaredErrorUnchanged': int(np.sum(new_pixel == old_pixel))}
    for label, delta in [('old', a), ('new', b)]:
        absolute = np.abs(delta)
        result[label] = {'mae255': float(np.mean(absolute)), 'rms255': float(np.sqrt(np.mean(delta * delta))),
            'maxAbs255': int(absolute.max()), 'changedChannels': int(np.count_nonzero(delta)),
            'changedPixels': int(np.count_nonzero(np.any(delta != 0, axis=-1))),
            'absoluteChannelErrorCounts': np.bincount(absolute.astype(np.int64).ravel(), minlength=256).tolist()}
    return result
