import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from encoder_atoms import save,sha,identity
from padding_mlp_study_v2 import OWNED


class PaddingGuardOrder(unittest.TestCase):
    def test_actual_frozen_reader_authenticates_before_value_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for n in OWNED:shutil.copyfile(Path(__file__).parent/n,root/n)
            save(root/'targets.json',{'metadata':'hash before value exclusion'});save(root/'inputs.json',{'forbiddenTargets':[],'excludedSources':[],'lexicalStore':'unused'})
            r={'sourceHashes':{'targets.json':sha(root/'targets.json')},'manifestSHA256':'unused'};r['requestSHA256']=identity(r);save(root/'request.json',r)
            code='import sys;sys.path.insert(0,".");from pathlib import Path;import encoder_atoms;from padding_mlp_study_v2 import reader\ndef prefix(*args):\n try:open("targets.json","rb")\n except PermissionError:raise SystemExit(0)\n raise AssertionError("target guard missing")\nencoder_atoms.Prefix=prefix;reader(Path("."))'
            subprocess.run([sys.executable,'-I','-c',code],cwd=root,check=True,capture_output=True,timeout=20)


if __name__=='__main__':unittest.main()
