import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import subprocess
import sys
import numpy as np
from last_velocity import forecast, coefficient
from encoder_atoms import floats, words, save


class LastVelocity(unittest.TestCase):
    def test_affine_trajectory_extrapolation_and_hold_are_distinct(self):
        sigmas = [4.,3.,2.,1.,0.]
        slope = np.array([[1.,-2.,4.]],np.float32); bias = np.array([[7.,8.,9.]],np.float32)
        v1 = sigmas[1]*slope+bias; v2 = sigmas[2]*slope+bias
        raw, emitted = forecast(v1,v2,sigmas,'secant')
        np.testing.assert_array_equal(raw,sigmas[3]*slope+bias)
        np.testing.assert_array_equal(floats(emitted),raw)
        held,hw = forecast(v1,v2,sigmas,'hold')
        np.testing.assert_array_equal(held,v2);np.testing.assert_array_equal(hw,words(v2))
        self.assertFalse(np.array_equal(raw,held))

    def test_actual_nonuniform_schedule_is_extrapolation(self):
        s=[1.,0.9580853581428528,0.8839818835258484,0.7174965739250183,0.]
        self.assertEqual(coefficient(s),2.2466599638045444)
        x=np.array([1.],np.float32);y=np.array([2.],np.float32)
        raw,_=forecast(x,y,s,'secant')
        self.assertEqual(float(raw[0]),float(np.float32(2)+np.float32(coefficient(s))))
        self.assertGreater(raw[0],3.)

    def test_bf16_hold_preserves_actual_words_including_signed_zero(self):
        w=np.array([0,32768,0x3f81,0xbf81,0x0080],dtype=np.uint16)
        x=floats(w);raw,out=forecast(x,x,[4,3,2,1,0],'hold')
        np.testing.assert_array_equal(out,w)
        self.assertEqual(raw.dtype,np.float32)

    def test_invalid_source_or_forecast_is_integrity_error(self):
        for sigmas in [[1,2,3,4,5],[4,3,3,1,0],[4,3,2,1],[4,3,2,np.nan,0]]:
            with self.assertRaises(ValueError):forecast([1],[2],sigmas,'secant')
        for x,y,p in [([1],[2,3],'secant'),([np.nan],[1],'hold'),([1],[1],'fit')]:
            with self.assertRaises(ValueError):forecast(x,y,[4,3,2,1,0],p)

    def test_guard_denies_targets_sibling_arrays_and_transformer(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);model=p/'model';parent=p/'parent';parent.mkdir();model.mkdir()
            allowed=parent/'v1.npy';allowed.write_bytes(b'prefix');(parent/'hidden.npy').write_bytes(b'target')
            (model/'vae').mkdir();(model/'vae/weights.safetensors').write_bytes(b'vae')
            save(p/'targets.json',{})
            save(p/'inputs.json',{'modelRoot':str(model),'decoderAssets':{'vae/weights.safetensors':'hash'},
                'parentRoot':str(parent),'prefix':{'00':{'velocity1':{'path':str(allowed)}}},
                'ids':{'path':str(parent/'ids.npy')},'forbiddenTargets':[]})
            code='import sys;sys.path.insert(0,'+repr(str(Path(__file__).parent))+');from last_velocity import install_guard;from pathlib import Path;install_guard('+repr(str(p))+')\n'
            code+='assert Path('+repr(str(allowed))+').read_bytes()==b"prefix"\n'
            code+='assert Path('+repr(str(model/'vae/weights.safetensors'))+').read_bytes()==b"vae"\n'
            for target in [p/'targets.json',parent/'hidden.npy',model/'transformer/diffusion_pytorch_model.safetensors']:
                code+='try:open('+repr(str(target))+',"rb")\nexcept PermissionError:pass\nelse:raise AssertionError("guard failed")\n'
            subprocess.run([sys.executable,'-I','-c',code],check=True,capture_output=True,timeout=20)

    def test_complete_image_seal_precedes_score_and_failure_stays_sealed_off(self):
        import last_velocity_study as study
        for failure in [False,True]:
            with tempfile.TemporaryDirectory() as temp:
                events=[]
                def child(cmd,**kwargs):
                    events.append(cmd[-2])
                    if failure and cmd[-2]=='decode':raise subprocess.CalledProcessError(1,cmd)
                with patch.object(study,'read',return_value={}),patch.object(study,'verify'),patch.object(study.subprocess,'run',side_effect=child),patch.object(study,'seal',side_effect=lambda p:events.append('seal')),patch.object(study,'sha',return_value='sealed'),patch.object(study,'score',side_effect=lambda *a:events.append('score')):
                    if failure:
                        with self.assertRaises(subprocess.CalledProcessError):study.execute(temp)
                        self.assertEqual(events,['forecast','decode'])
                    else:
                        study.execute(temp);self.assertEqual(events,['forecast','decode','seal','score'])

    def test_lazy_workers_need_no_neural_runtime_at_import(self):
        code='import sys;sys.path.insert(0,'+repr(str(Path(__file__).parent))+');import last_velocity,last_velocity_decode,last_velocity_study;assert not ({"torch","diffusers","transformers"} & set(sys.modules))'
        subprocess.run([sys.executable,'-I','-c',code],check=True,capture_output=True,timeout=20)

    def test_automatic_submission_omits_every_declared_resource_argument(self):
        import importlib
        import last_velocity_submit as adapter
        keys=['schema','requestSHA256','specimen','component','backend','device','dtype','batch_size','seq_len','height','width','task_family']
        request={k:'test' for k in keys}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);save(root/'request.json',request)
            with patch.object(adapter,'verify'),patch('run.command',return_value=['env','uv','python','-u','worker.py','--request','request.json']),patch.object(importlib.import_module('mrun.client.submit'),'launch',return_value='job-one') as launcher:
                adapter.submit(root)
                kwargs=launcher.call_args.kwargs
                self.assertFalse(set(kwargs)&{'ram_mb','vram_mb','cpu_threads','est_wall_s','disk_gb'})
                self.assertEqual(kwargs['needs'],{'cuda':True})
                self.assertTrue(kwargs['guarded']);self.assertTrue(kwargs['preflight']);self.assertFalse(kwargs['retry_on_kill'])
                self.assertEqual(kwargs['config']['requestSHA256'],'test')
                with self.assertRaises(ValueError):adapter.submit(root)


if __name__=='__main__':unittest.main()
