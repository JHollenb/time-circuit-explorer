"""Actual audit hook and LocalStore publication are exercised in a fresh process."""
import subprocess
import sys
from pathlib import Path


def test_guard_allows_only_own_atomic_store_and_preserves_denial(tmp_path):
    code=r'''
import sys,os
from pathlib import Path
sys.path.insert(0,str(Path('generator').resolve()))
from terminal_capture_guard import quarantine
from terminal_capture_cut import persist
from test_terminal_capture_cut import fixture
from saturn_pub.core import StateCut
root=Path(sys.argv[1]).resolve();target=root/'target.npy';target.write_bytes(b'sealed')
unknown=root/'old.safetensors';unknown.write_bytes(b'unlisted')
os.environ['MDB_SHARED_TENSOR_STORE']=str(root/'external-pool-must-not-open')
guard=quarantine(root,[],[target],root/'terminal-store')
state=fixture();original=StateCut('model',{},'source',None,state)
record=persist(state,original,root/'terminal-store','request','single',19,original.fingerprint)
assert record['roundtripExact'] and guard['ownStoreLinkedFiles'] and guard['ownStoreMissingProbes']
assert not (root/'external-pool-must-not-open').exists()
for path in [target,unknown]:
 try:path.read_bytes()
 except PermissionError:pass
 else:raise AssertionError('Existing target/unlisted file escaped guard')
print('own-store-only')
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path.resolve())],cwd=Path(__file__).resolve().parent.parent,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='own-store-only'
