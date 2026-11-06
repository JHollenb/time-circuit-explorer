import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from encoder_atoms import save,sha,identity
from padding_mlp import smooth_numpy,jacobian_numpy,smooth_jacobian_torch,shared_active,SourceWeights
from padding_mlp_study import execute,OWNED


class PaddingMLPMechanics(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(35);return rng.normal(size=5)*.2,rng.normal(size=(7,5))*.3,rng.normal(size=(7,5))*.3,rng.normal(size=(5,7))*.3

    def request(self,root):
        r={'sourceHashes':{}};r['requestSHA256']=identity(r);save(root/'request.json',r)

    def test_compiler_jacobian_against_independent_autograd(self):
        import torch
        x,G,U,D=self.fixture();xt,Gt,Ut,Dt=[torch.tensor(v,dtype=torch.float64) for v in [x,G,U,D]]
        expected=torch.autograd.functional.jacobian(lambda t:Dt@(torch.nn.functional.silu(Gt@t)*(Ut@t)),xt)
        value,J,_,_=smooth_jacobian_torch(xt,Gt,Ut,Dt)
        np.testing.assert_allclose(J.numpy(),expected.numpy(),rtol=3e-14,atol=3e-16);np.testing.assert_allclose(J.numpy(),jacobian_numpy(x,G,U,D),rtol=3e-14,atol=3e-16)
        np.testing.assert_allclose(value.numpy(),smooth_numpy(x,G,U,D),rtol=3e-14,atol=3e-16)

    def test_directional_remainder_contracts_quadratically(self):
        x,G,U,D=self.fixture();direction=np.array([1.,-2.,3.,-.5,2.]);b=smooth_numpy(x,G,U,D);J=jacobian_numpy(x,G,U,D)
        errors=[np.linalg.norm(smooth_numpy(x+h*direction,G,U,D)-(b+h*J@direction)) for h in [.001,.0005]]
        self.assertGreater(errors[0]/errors[1],3.9);self.assertLess(errors[0]/errors[1],4.1)
        self.assertGreater(np.linalg.norm(J[:,-1]),0.)

    def test_native_anchor_offset_and_zero_delta(self):
        x,G,U,D=self.fixture();J=jacobian_numpy(x,G,U,D).astype(np.float32);b=(smooth_numpy(x,G,U,D)+.125).astype(np.float32)
        np.testing.assert_array_equal(b+J@(x.astype(np.float32)-x.astype(np.float32)),b)
        self.assertGreater(np.linalg.norm(b-smooth_numpy(x,G,U,D)),0.)

    def test_shared_active_write_computes_once_and_refuses_changed_receiver(self):
        from paired_fields import payload
        cache={};x=np.arange(12).reshape(3,4);calls=[]
        compute=lambda:(calls.append(1) or x+1)
        a,first=shared_active(cache,'00',x,compute,payload);b,second=shared_active(cache,'00',x.copy(),compute,payload)
        self.assertIs(a,b);self.assertFalse(first);self.assertTrue(second);self.assertEqual(len(calls),1)
        with self.assertRaises(ValueError):shared_active(cache,'00',x+1,compute,payload)

    def test_source_embedding_row_bounds_and_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'text_encoder').mkdir();raw=np.arange(12,dtype='<u2').reshape(4,3);header=json.dumps({'model.embed_tokens.weight':{'dtype':'BF16','shape':[4,3],'data_offsets':[0,24]}}).encode();p=root/'text_encoder/tiny.safetensors';p.write_bytes(struct.pack('<Q',len(header))+header+raw.tobytes());save(root/'text_encoder/model.safetensors.index.json',{'weight_map':{'model.embed_tokens.weight':'tiny.safetensors'}})
            source={'lexicalSource':{'modelRoot':str(root)},'qwenShards':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'header_sha256':hashlib.sha256(header).hexdigest()}]};reader=SourceWeights(source)
            with patch('conditioning_demand.parameter_from_words',side_effect=lambda a,shape,device:a.reshape(shape).copy()):
                np.testing.assert_array_equal(reader.parameter('model.embed_tokens.weight',row=2),raw[2:3])
                with self.assertRaises(ValueError):reader.parameter('model.embed_tokens.weight',row=4)
            p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaises(ValueError):reader.unchanged()

    def test_target_exclusion_in_isolated_staged_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for n in OWNED:shutil.copyfile(Path(__file__).parent/n,root/n)
            save(root/'inputs.json',{'forbiddenTargets':[str(root/'native-target.npy')]})
            code='import sys;sys.path.insert(0,".");from pathlib import Path;from padding_mlp_study import install_guard;assert install_guard(Path("."));\ntry:open("native-target.npy","wb")\nexcept PermissionError:pass\nelse:raise AssertionError("target allowed")\nassert not {"torch","diffusers","transformers"}.intersection(sys.modules)'
            subprocess.run([sys.executable,'-I','-c',code],cwd=root,check=True,capture_output=True,timeout=20)

    def test_all_images_seal_before_paid_native_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.request(root);events=[]
            with patch('padding_mlp_study.subprocess.run',side_effect=lambda cmd,**k:events.append(cmd[3])),patch('padding_mlp_study.seal',side_effect=lambda p:events.append('seal')),patch('padding_mlp_study.sha',return_value='sealed'),patch('padding_mlp_study.score',side_effect=lambda *x:events.append('score')):execute(root)
            self.assertEqual(events,['compile','reader','encoder','images','seal','diagnostic','score'])

    def test_incomplete_image_phase_cannot_release_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.request(root)
            def child(cmd,**k):
                if cmd[3]=='images':raise RuntimeError('partial')
            with patch('padding_mlp_study.subprocess.run',side_effect=child),patch('padding_mlp_study.seal') as seal,patch('padding_mlp_study.score') as score:
                with self.assertRaises(RuntimeError):execute(root)
                seal.assert_not_called();score.assert_not_called()


if __name__=='__main__':unittest.main()
