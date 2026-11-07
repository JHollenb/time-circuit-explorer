"""Source-position selection, key-read closure, and demanded-row reconstruction."""
import numpy as np


def selection(mask, count=16):
    mask=np.asarray(mask)
    if mask.ndim!=1 or not len(mask) or not np.isin(mask,[0,1]).all() or not mask.any() or count<2:
        raise ValueError('Invalid active mask or endpoint budget')
    active=np.flatnonzero(mask);padding=np.flatnonzero(~mask.astype(bool))
    q=padding if len(padding)<=count else padding[np.arange(count)*(len(padding)-1)//(count-1)]
    selected=np.sort(np.concatenate([active,q])).astype(np.int64)
    expansion=np.empty(len(mask),dtype=np.int64);where={int(p):i for i,p in enumerate(selected)}
    for p in range(len(mask)):
        representative=p if mask[p] else int(q[np.argmin(np.abs(q-p))])
        expansion[p]=where[representative]
    return selected,expansion


def visibility(full, selected, mask):
    full=np.asarray(full);selected=np.asarray(selected);mask=np.asarray(mask,dtype=bool)
    if full.dtype!=np.bool_ or full.shape!=(len(mask),len(mask)) or selected.ndim!=1 or not len(selected) or np.any(selected<0) or np.any(selected>=len(mask)) or np.any(selected[1:]<=selected[:-1]):
        raise ValueError('Invalid native visibility geometry')
    retained=np.zeros(len(mask),dtype=bool);retained[selected]=True
    if np.any(mask & ~retained) or np.any(full[:,~mask]) or np.any(full[selected][:,~retained]):
        raise ValueError('Retained queries do not have a closed active key bank')
    compact=full[np.ix_(selected,selected)].copy()
    if not compact.any(axis=1).all():raise ValueError('A query has no eligible native key')
    return compact


def error_terms(small, reference, selected, expansion):
    small=np.asarray(small,dtype=np.float64);reference=np.asarray(reference,dtype=np.float64)
    selected=np.asarray(selected,dtype=np.int64);expansion=np.asarray(expansion,dtype=np.int64)
    if small.shape[0]!=len(selected) or reference.shape[0]!=len(expansion) or small.shape[1:]!=reference.shape[1:]:
        raise ValueError('Conditioning selection geometry differs')
    finite=(small-reference[selected])[expansion];reconstruction=reference[selected][expansion]-reference
    total=small[expansion]-reference
    return finite,reconstruction,total


def energy_account(finite,reconstruction,total):
    energy=lambda v:float(np.sum(v*v))
    closure=total-finite-reconstruction
    return {'finiteExecutionEnergy':energy(finite),'reconstructionEnergy':energy(reconstruction),
        'twiceCrossTerm':float(2*np.sum(finite*reconstruction)),'totalErrorEnergy':energy(total),
        'closureRMS':float(np.sqrt(np.mean(closure*closure))),'closureMaxAbs':float(np.max(np.abs(closure)))}
