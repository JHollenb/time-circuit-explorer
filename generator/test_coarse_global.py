import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from encoder_atoms import save
from coarse_global import pool,lift,geometry,mask,check,split_error


class CoarseGlobal(unittest.TestCase):
    def test_cell_pool_lift_projection_and_fine_carry(self):
        a=(np.arange(1*1024*128,dtype=np.float64).reshape(1,1024,128)//128)%32
        projected=lift(pool(a));np.testing.assert_array_equal(pool(projected),pool(a))
        np.testing.assert_array_equal(lift(pool(projected)),projected)
        np.testing.assert_array_equal(pool(a-projected),0)
        self.assertTrue(np.any(a-projected))
        self.assertEqual(float(np.sum(projected*(a-projected))),0)
        with self.assertRaises(ValueError):pool(a[:,:512])
        with self.assertRaises(ValueError):lift(a)

    def test_error_split_separates_representation_producer_and_finite_terms(self):
        held=(np.arange(131072,dtype=np.float64).reshape(1,1024,128)//128)%8
        native=held*2+1;coarse=pool(native)+.25;raw=held+lift(coarse-pool(held));emitted=raw+.125
        f=split_error(coarse,held,native,emitted)
        np.testing.assert_array_equal(f['actualVelocityError'],f['coarseProducerError']+f['omittedFineInnovation']+f['finiteEmissionError'])
        self.assertEqual(float(np.sum(f['coarseProducerError']*f['omittedFineInnovation'])),0)
        np.testing.assert_array_equal(f['finiteEmissionError'],.125)
        exact=split_error(pool(native),held,native,held+lift(pool(native)-pool(held)))
        np.testing.assert_array_equal(exact['coarseProducerError'],0)
        self.assertTrue(np.any(exact['omittedFineInnovation']))

    def test_physical_grid_centers_and_changed_input_rejected(self):
        h,w=np.meshgrid(np.arange(32),np.arange(32),indexing='ij');ids=np.stack([h*0,h,w,h*0],axis=-1).reshape(1,1024,4)
        c=geometry(ids);self.assertEqual(c.dtype,np.float32)
        np.testing.assert_array_equal(c[0,0],[0,.5,.5,0]);np.testing.assert_array_equal(c[0,-1],[0,30.5,30.5,0])
        with self.assertRaises(ValueError):geometry(ids[:,::-1])

    def test_key_mass_matches_collapsed_identical_keys(self):
        logits=np.array([.25,-.5]);values=np.array([2.,7.]);full=np.array([logits[0],*([logits[1]]*4)])
        probability=np.exp(full-full.max());native=np.dot(probability,np.array([2.,7.,7.,7.,7.]))/probability.sum()
        coarse=logits+np.array([0.,np.log(4)]);p=np.exp(coarse-coarse.max());actual=np.dot(p,values)/p.sum()
        self.assertAlmostEqual(actual,native,places=14)
        m=mask();self.assertEqual(m.shape,(1,1,1,768));self.assertEqual(m.dtype,np.float32)
        np.testing.assert_array_equal(m[0,0,0,:512],0);np.testing.assert_array_equal(m[0,0,0,512:],np.float32(np.log(4)))

    def test_only_bound_coarse_mode_admitted(self):
        def a(shape,dtype,digest):return SimpleNamespace(shape=shape,dtype=dtype,digest=digest)
        contract={'numericalProfile':{'threads':4},'EShape':[1,512,7680],'dtype':'torch.bfloat16'}
        kw={n:a(shape,dtype,n) for n,shape,dtype in [('hidden_states',[1,256,128],'torch.bfloat16'),('encoder_hidden_states',[1,512,7680],'torch.bfloat16'),
            ('img_ids',[1,256,4],'torch.float32'),('txt_ids',[1,512,4],'torch.int64'),('timestep',[1],'torch.bfloat16')]}
        kw['joint_attention_kwargs']={'attention_mask':a([1,1,1,768],'torch.float32','mask')};expected={n:x.digest for n,x in kw.items() if n!='joint_attention_kwargs'};expected['attention_mask']='mask'
        with patch('coarse_global.payload',side_effect=lambda x:x.digest):
            check(kw,{'threads':4},contract,expected)
            for mutate in [lambda x:x.update(guidance=1),lambda x:x['joint_attention_kwargs'].update(scale=.5),
                           lambda x:setattr(x['img_ids'],'dtype','torch.int64'),lambda x:setattr(x['hidden_states'],'shape',[1,1024,128]),
                           lambda x:setattr(x['joint_attention_kwargs']['attention_mask'],'digest','wrong')]:
                bad=copy.deepcopy(kw);mutate(bad)
                with self.assertRaises(ValueError):check(bad,{'threads':4},contract,expected)
        from flux_clock import check_mode
        with self.assertRaises(ValueError):check_mode(kw,{'threads':4},contract)

    def test_all_candidates_seal_before_target_scoring_and_failure_stops_release(self):
        import coarse_global_study as study
        for failure in [False,True]:
            with tempfile.TemporaryDirectory() as temp:
                events=[];save(Path(temp)/'request.json',{})
                def child(*args,**kwargs):
                    events.append('images')
                    if failure:raise RuntimeError('failed candidate')
                def sealed(root):events.append('seal');(Path(root)/'candidate-seal.json').write_text('{}')
                with patch.object(study,'verify'),patch.object(study.subprocess,'run',side_effect=child),patch.object(study,'seal',side_effect=sealed),patch.object(study,'score',side_effect=lambda *args:events.append('targets')):
                    if failure:
                        with self.assertRaises(RuntimeError):study.execute(temp)
                        self.assertEqual(events,['images'])
                    else:study.execute(temp);self.assertEqual(events,['images','seal','targets'])

    def test_workers_import_without_neural_runtime(self):
        code='import sys;sys.path.insert(0,'+repr(str(Path(__file__).parent))+');import coarse_global,coarse_global_images,coarse_global_study,coarse_global_images_v2,coarse_global_study_v2,coarse_bias;assert not {"torch","transformers","diffusers"}.intersection(sys.modules)'
        subprocess.run([sys.executable,'-I','-c',code],check=True,capture_output=True,timeout=20)

    def test_bias_conversion_requires_declared_request_and_query_geometry(self):
        from coarse_bias import consumed_bias
        class Requested:
            shape=[1,1,1,768];dtype='torch.float32'
            def to(self,dtype):return ('converted',dtype)
        q=SimpleNamespace(shape=[1,768,24,128],dtype='torch.bfloat16')
        self.assertEqual(consumed_bias(Requested(),q),('converted','torch.bfloat16'))
        for other in [SimpleNamespace(shape=[1,768,24,128],dtype='torch.float32'),SimpleNamespace(shape=[1,256,24,128],dtype='torch.bfloat16')]:
            with self.assertRaises(ValueError):consumed_bias(Requested(),other)
