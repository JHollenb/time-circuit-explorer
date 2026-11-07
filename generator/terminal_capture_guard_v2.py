"""Source-only input guard with narrowly owned immutable LocalStore output links."""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import sys

NUMERICAL={'.npy','.npz','.safetensors','.pt','.pth'}


def quarantine(root,allowed,denied,output_store):
    root=Path(root).resolve();store=Path(output_store).resolve()
    if not store.is_relative_to(root) or store.exists():raise ValueError('Output StateCut store must be fresh and inside this request')
    allowed={str(Path(p).resolve()) for p in allowed};denied={str(Path(p).resolve()) for p in denied}
    if allowed&denied:raise ValueError('Source/denied inputs overlap')
    state={'successfulFileOpenCounts':{},'ownStoreMissingProbes':{},'ownStoreLinkedFiles':[]};created=set();linked=set()
    def owned_blob(p):return p.parent==store/'blobs' and re.fullmatch('[0-9a-f]{64}\\.safetensors',p.name) is not None
    def audit(event,args):
        if event=='os.link':
            src,dst=[Path(os.fsdecode(p)).resolve() for p in args[:2]]
            if owned_blob(dst):
                if str(src) not in created or src.parent!=store/'blobs':raise PermissionError('Output blob must come from its own new local temporary')
                linked.add(str(dst));state['ownStoreLinkedFiles'].append(str(dst))
            return
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve();key=str(p);mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR))
        if key in denied:raise PermissionError('Sealed comparison unavailable to capture')
        if writing and p.is_relative_to(root):created.add(key)
        elif key in allowed:state['successfulFileOpenCounts'][key]=state['successfulFileOpenCounts'].get(key,0)+1
        elif p.suffix in NUMERICAL and key not in created:
            if owned_blob(p) and key in linked:return
            if owned_blob(p) and not p.exists():state['ownStoreMissingProbes'][key]=state['ownStoreMissingProbes'].get(key,0)+1
            else:raise PermissionError('Undeclared numerical capture input')
    sys.addaudithook(audit)
    for p in denied:
        try:open(p,'rb')
        except PermissionError:continue
        raise ValueError('Target denial inactive')
    return state


@contextmanager
def local_store_only(root):
    """Keep this small output store local; do not hydrate a configured shared pool."""
    try:
        import saturn_pub.shared_store as shared
    except ModuleNotFoundError as error:
        if error.name!='saturn_pub.shared_store':raise
        yield
        return
    root=Path(root).resolve();original=shared.publish_configured
    def publish(path,raw):
        return None if Path(path).resolve().is_relative_to(root) else original(path,raw)
    shared.publish_configured=publish
    try:yield
    finally:shared.publish_configured=original
