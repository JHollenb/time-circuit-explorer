"""The exact frozen worker wheel must pass storage mechanics before a real prefix."""
import json
from pathlib import Path
import subprocess
import sys


def test_exact_worker_wheel_persistence_and_denial(tmp_path):
    root=Path(__file__).resolve().parent.parent;wheel=root/'artifacts/source-reader-v1/payload/saturn_pub-0.4.1-py3-none-any.whl'
    (tmp_path/'request.json').write_text(json.dumps({'requestSHA256':'tiny-package-check'}))
    code='''import sys
from pathlib import Path
sys.path.insert(0,sys.argv[2]);sys.path.insert(1,str(Path('generator').resolve()))
import saturn_pub.store
assert '.whl/' in saturn_pub.store.__file__
from terminal_capture_mechanics import execute
r=execute(sys.argv[1])
assert not r['sharedStoreModuleAvailable'] and not r['externalPoolOpened']
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path.resolve()),str(wheel)],cwd=root,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'TERMINAL_CAPTURE_WHEEL_MECHANICS_OK' in result.stdout
