"""Own-prefix capability guard and signed two-clock policy algebra."""
from pathlib import Path
import os
import sys
import numpy as np
from encoder_atoms import read


def interaction(nn,nh,hn,hh):
    xs=[np.asarray(x,dtype=np.float64) for x in (nn,nh,hn,hh)]
    if any(x.shape!=xs[0].shape or not np.isfinite(x).all() for x in xs):
        raise ValueError('Invalid complete policy fields')
    return (xs[0]-xs[1])-(xs[2]-xs[3])


def finite_commit(z,next_z,velocity,gap):
    z,next_z,velocity=[np.asarray(x,dtype=np.float64) for x in (z,next_z,velocity)]
    if z.shape!=next_z.shape or z.shape!=velocity.shape:
        raise ValueError('Finite commit geometry differs')
    return next_z-z-float(gap)*velocity


def install_guard(root):
    root=Path(root).resolve();s=read(root/'inputs.json')
    denied={str(Path(x).resolve()) for x in [root/'targets.json',*s['forbiddenTargets']]}
    allowed={str(Path(v['path']).resolve()) for fs in s['prefix'].values() for v in fs.values()}
    allowed.add(str(Path(s['ids']['path']).resolve()))
    for row in s['clockTable']['rows']:
        allowed.update(str(Path(v['path']).resolve()) for v in [*row['fields'].values(),*row['clockOperands'].values()])
    parents=[Path(x).resolve() for x in s['targetRoots']]
    model=Path(s['modelRoot']).resolve();assets={str((model/n).resolve()) for n in s['imageAssets']}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve();name=str(p)
        if name in denied or (any(p.is_relative_to(x) for x in parents) and name not in allowed):
            raise PermissionError('Historical future unavailable before all candidate images seal')
        if (p.is_relative_to(model) or p.suffix in ('.safetensors','.pt','.pth')) and name not in assets:
            raise PermissionError('Only declared image-consumer assets are available')
    sys.addaudithook(audit)
    for p in [root/'targets.json',s['guardCanary']]:
        try:open(p,'rb')
        except PermissionError:continue
        raise RuntimeError('Historical future guard inactive')
