"""Tiny mechanics only; real historical-image scoring runs through mrun."""
from pathlib import Path
import subprocess
import sys
import unittest

class ResponseChartMechanics(unittest.TestCase):
    def test_anchor_roundtrip_and_quadratic_recovery_and_exclusion(self):
        source='''import numpy as np
from response_chart import evaluate,encode,guard
a=np.array([[[[3.,9.,12.]]],[[[4.,11.,15.]]],[[[19.,23.,30.]]]],dtype=np.float32)
for i,x in enumerate([0,.25,1]):
 for mode in ['piecewise','quadratic']:
  assert np.array_equal(encode(evaluate(a,x,mode)),encode(a[i]))
q=np.array([[[[x*x,2*x*x+3*x,5.]] ]for x in [0,.25,1]],dtype=np.float32)
assert np.allclose(evaluate(q,.1,'quadratic'),[[[.01,.32,5.]]],atol=1e-6)
try:import torch
except ImportError:pass
else:raise AssertionError('Neural import accepted')
guard()
try:open('evaluation/forbidden.png','rb')
except PermissionError:pass
else:raise AssertionError('Target read accepted')
'''
        subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,check=True,capture_output=True,text=True)

    def test_single_image_cli_from_tiny_synthetic_bundle(self):
        source='''import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
with tempfile.TemporaryDirectory() as d:
 p=Path(d);bundle=p/'chart.npz';image=p/'output.png'
 np.savez_compressed(bundle,**{'synthetic':np.array([[[[0,0,0]]],[[[100,50,0]]],[[[200,100,0]]]],dtype=np.uint8)})
 cmd=[sys.executable,'render_response.py','--bundle',str(bundle),'--context','synthetic','--alpha','.1','--output',str(image)]
 result=subprocess.run(cmd,check=True,capture_output=True,text=True)
 assert np.array_equal(np.asarray(Image.open(image)),[[[40,20,0]]])
 assert json.loads(result.stdout)['neuralEvaluations']==0
 assert subprocess.run(cmd,capture_output=True).returncode!=0
'''
        subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,check=True,capture_output=True,text=True)

if __name__=='__main__':unittest.main()
