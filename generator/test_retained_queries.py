import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
import numpy as np
from retained_queries import selection,visibility,error_terms,energy_account


class RetainedQueries(unittest.TestCase):
    def test_endpoint_rule_and_nearest_ties_original_addresses(self):
        mask=np.array([1,1,0,0,0,0,0,0,0],dtype=bool);s,r=selection(mask,3)
        np.testing.assert_array_equal(s,[0,1,2,5,8]);np.testing.assert_array_equal(s[r],[0,1,2,2,5,5,5,8,8])
        s,r=selection(np.array([1,0,0,0,0,0],dtype=bool),2)
        self.assertEqual(s[r][3],1)

    def test_small_or_unpadded_domain_keeps_all(self):
        for mask in [np.ones(7,dtype=bool),np.array([1,0,1,0],dtype=bool)]:
            s,r=selection(mask);np.testing.assert_array_equal(s,np.arange(len(mask)));np.testing.assert_array_equal(r,s)
        with self.assertRaises(ValueError):selection(np.zeros(4,dtype=bool))

    def test_key_bank_refuses_padding_or_missing_active_keys(self):
        mask=np.array([1,1,0,0,0],dtype=bool);s,_=selection(mask,2);full=np.tri(5,dtype=bool)&mask[None,:]
        np.testing.assert_array_equal(visibility(full,s,mask),full[np.ix_(s,s)])
        bad=full.copy();bad[4,3]=True
        with self.assertRaises(ValueError):visibility(bad,s,mask)
        with self.assertRaises(ValueError):visibility(full,s[s!=1],mask)

    def test_multilayer_nonlinear_trajectory_commutes_with_closed_selection(self):
        rng=np.random.default_rng(7);n=13;mask=np.arange(n)<5;s,r=selection(mask,3);full=np.tri(n,dtype=bool)&mask[None,:];small=visibility(full,s,mask)
        h=rng.normal(size=(n,8));hs=h[s].copy();positions=np.arange(n)*.31;weights=[rng.normal(size=(8,8))*.2 for _ in range(4)]
        def block(x,m,pos):
            q=x@weights[0]+np.sin(pos)[:,None];k=x@weights[1];v=x@weights[2];scores=q@k.T;scores=np.where(m,scores,-np.inf);scores-=scores.max(axis=1)[:,None];a=np.exp(scores);a/=a.sum(axis=1)[:,None];z=x+a@v;return z+np.tanh(z@weights[3])
        for _ in range(4):h=block(h,full,positions);hs=block(hs,small,positions[s]);np.testing.assert_allclose(hs,h[s],atol=1e-13,rtol=1e-13)
        wrong=block(h[s],small,np.arange(len(s))*.31);correct=block(h[s],small,positions[s]);self.assertGreater(np.linalg.norm(wrong-correct),.01)

    def test_reconstruction_telescope_keeps_signed_cross_term(self):
        mask=np.array([1,0,0,0,0,0],dtype=bool);s,r=selection(mask,2);ref=np.arange(24,dtype=float).reshape(6,4);own=ref[s]+np.array([0,2,-1])[:,None]
        f,g,total=error_terms(own,ref,s,r);np.testing.assert_array_equal(f+g,total);np.testing.assert_array_equal(g[mask],0)
        a=energy_account(f,g,total);self.assertEqual(a['closureMaxAbs'],0);self.assertAlmostEqual(a['finiteExecutionEnergy']+a['reconstructionEnergy']+a['twiceCrossTerm'],a['totalErrorEnergy'])

    def test_actual_lexical_bridge_authenticates_before_value_guard(self):
        from retained_query_study import OWNED
        from encoder_atoms import save,sha,identity
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for n in OWNED:shutil.copyfile(Path(__file__).parent/n,root/n)
            save(root/'targets.json',{});save(root/'inputs.json',{'forbiddenTargets':[],'lexicalStore':'unused'})
            request={'sourceHashes':{'targets.json':sha(root/'targets.json')},'manifestSHA256':'unused'};request['requestSHA256']=identity(request);save(root/'request.json',request)
            code='import sys;sys.path.insert(0,".");import encoder_atoms;from retained_query_study import lexical\ndef prefix(*args):\n try:open("targets.json","rb")\n except PermissionError:raise SystemExit(0)\n raise AssertionError("target guard missing")\nencoder_atoms.Prefix=prefix;lexical(".")'
            subprocess.run([sys.executable,'-I','-c',code],cwd=root,capture_output=True,check=True,timeout=20)

    def test_all_images_release_before_scoring_and_failure_never_releases(self):
        import retained_query_study as study
        for fail in [False,True]:
            with tempfile.TemporaryDirectory() as temp:
                events=[]
                def child(cmd,**kwargs):
                    events.append(cmd[-2])
                    if fail and cmd[-2]=='images':raise subprocess.CalledProcessError(1,cmd)
                with patch.object(study,'read',return_value={}),patch.object(study,'verify'),patch.object(study.subprocess,'run',side_effect=child),patch.object(study,'seal',side_effect=lambda p:events.append('seal')),patch.object(study,'sha',return_value='sealed'),patch.object(study,'score',side_effect=lambda *a:events.append('score')):
                    if fail:
                        with self.assertRaises(subprocess.CalledProcessError):study.execute(temp)
                        self.assertEqual(events,['lexical','encoder','images'])
                    else:study.execute(temp);self.assertEqual(events,['lexical','encoder','images','seal','score'])


if __name__=='__main__':unittest.main()
