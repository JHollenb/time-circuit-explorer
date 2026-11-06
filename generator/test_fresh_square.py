import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from encoder_atoms import save,identity,sha
from paired_fields import metrics
from fresh_square_study import CASES,PROMPTS,OWNED,factorial,candidate_guard,execute


class FreshSquareMechanics(unittest.TestCase):
    def request(self,root):
        r={'sourceHashes':{}};r['requestSHA256']=identity(r);save(root/'request.json',r);return r

    def test_factorial_recovers_edges_and_mixed_term(self):
        base=np.array([2.,-3.]);color=np.array([5.,2.]);relation=np.array([-7.,4.]);joint=np.array([3.,-9.])
        f=factorial({'00':base,'10':base+color,'01':base+relation,'11':base+color+relation+joint})
        for name,value in [('color-left',color),('color-right',color+joint),('relation-base',relation),('relation-swapped',relation+joint),('mixed',joint)]:np.testing.assert_array_equal(f[name],value)

    def test_png_response_is_signed_before_subtraction(self):
        f=factorial({'00':np.array([255],np.uint8),'10':np.array([0],np.uint8),'01':np.array([250],np.uint8),'11':np.array([5],np.uint8)})
        self.assertEqual(float(f['color-left'][0]),-255.);self.assertEqual(float(f['mixed'][0]),10.)

    def test_additive_control_has_no_interaction_and_zero_denominator_is_visible(self):
        f=factorial({'00':np.array([0.]),'10':np.array([2.]),'01':np.array([3.]),'11':np.array([5.])})
        np.testing.assert_array_equal(f['mixed'],[0.]);m=metrics(np.array([1.]),f['mixed']);self.assertIsNone(m['relativeRms']);self.assertEqual(m['rms'],1.)
        with self.assertRaises(ValueError):factorial({'00':[0.],'10':[2.],'01':[3.],'11':[5.,6.]})

    def test_raw_prompts_follow_fixed_factorial_without_length_matching(self):
        p={x['id']:x['prompt'] for x in PROMPTS};self.assertEqual(list(p),CASES)
        self.assertEqual(p['01'],p['00'].replace('left','right'));self.assertEqual(p['11'],p['10'].replace('left','right'))
        self.assertIn('orange ceramic pitcher',p['00']);self.assertIn('teal metal watering can',p['00']);self.assertIn('teal ceramic pitcher',p['10']);self.assertIn('orange metal watering can',p['10'])

    def test_reference_guard_denies_future_values_in_isolated_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in OWNED:shutil.copyfile(Path(__file__).parent/name,root/name)
            code='import sys;sys.path.insert(0,".");from pathlib import Path;from fresh_square_study import candidate_guard,reference_paths;candidate_guard(Path("."),"candidate","test");paths=reference_paths(Path("."));n=0\nfor p in paths:\n try:open(p,"wb")\n except PermissionError:n+=1\nassert n==len(paths);assert not {"torch","diffusers","transformers"}.intersection(sys.modules)'
            subprocess.run([sys.executable,'-I','-c',code],cwd=root,check=True,capture_output=True,timeout=20)

    def test_reference_requires_request_bound_complete_candidate_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=self.request(root)
            with self.assertRaises(ValueError):candidate_guard(root,'full','test')
            seal={'lane':'candidate','requestSHA256':r['requestSHA256']}
            for key,name in [('readerSHA256','reader-seal.json'),('encoderSHA256','encoder-candidate.json'),('imagesSHA256','images-candidate.json'),('noiseSealSHA256','noise-seal.json'),('clockTableSHA256','clock-table.json')]:save(root/name,{'item':name});seal[key]=sha(root/name)
            save(root/'candidate-seal.json',seal);candidate_guard(root,'full','test')
            (root/'encoder-candidate.json').write_text('{}')
            with self.assertRaises(ValueError):candidate_guard(root,'full','test')

    def test_all_candidate_phases_complete_before_reference_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.request(root);events=[]
            def child(cmd,**kw):events.append((cmd[3],cmd[-1] if '--lane' in cmd else None))
            def seal(path,lane):events.append(('seal',lane))
            with patch('fresh_square_study.subprocess.run',side_effect=child),patch('fresh_square_study.seal',side_effect=seal),patch('fresh_square_study.sha',return_value='sealed'),patch('fresh_square_study.score') as score:execute(root);score.assert_called_once()
            self.assertEqual(events,[('noise',None),('reader',None),('encoder','candidate'),('images','candidate'),('seal','candidate'),('encoder','full'),('images','full'),('seal','full')])

    def test_failed_candidate_does_not_release_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.request(root);events=[]
            def child(cmd,**kw):
                events.append((cmd[3],cmd[-1] if '--lane' in cmd else None))
                if cmd[3]=='images':raise RuntimeError('partial candidate')
            with patch('fresh_square_study.subprocess.run',side_effect=child),patch('fresh_square_study.seal') as seal,patch('fresh_square_study.score') as score:
                with self.assertRaises(RuntimeError):execute(root)
                seal.assert_not_called();score.assert_not_called()
            self.assertFalse(any(lane=='full' for _,lane in events))


if __name__=='__main__':unittest.main()
