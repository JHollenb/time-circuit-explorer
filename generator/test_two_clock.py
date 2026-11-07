import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from encoder_atoms import save
from two_clock import interaction,finite_commit


class TwoClock(unittest.TestCase):
    def test_affine_reader_can_have_nonzero_policy_interaction(self):
        z2=np.array([4.,6.]);v1=np.array([-2.,3.]);v2=3*z2+np.array([1.,-1.]);gap2=-.25;gap3=-.5
        z3n=z2+gap2*v2;z3h=z2+gap2*v1
        f3=lambda z:2*z+5
        nn=z3n+gap3*f3(z3n);nh=z3n+gap3*v2
        hn=z3h+gap3*f3(z3h);hh=z3h+gap3*v1
        j=interaction(nn,nh,hn,hh)
        np.testing.assert_array_equal(j,gap3*(2*gap2-1)*(v2-v1))
        self.assertTrue(np.any(j))
        np.testing.assert_array_equal(interaction(z3n,z3n,z3h,z3h),0)

    def test_finite_remainders_close_interaction_without_hiding_rounding(self):
        z3n=np.array([.75,1.5]);z3h=np.array([1.,1.]);vels=[np.array(x) for x in [[.25,.5],[.5,.5],[.75,.25],[.25,.5]]];parents=[z3n,z3n,z3h,z3h];gap=-.5
        eps=[np.array([i*i/64.,-i*i/128.]) for i in range(4)]
        ys=[z+gap*v+e for z,v,e in zip(parents,vels,eps)]
        measured=[finite_commit(z,y,v,gap) for z,y,v in zip(parents,ys,vels)]
        np.testing.assert_array_equal(interaction(*ys),gap*interaction(*vels)+interaction(*measured))
        self.assertTrue(np.any(interaction(*measured)))

    def test_signed_pixels_cast_before_subtraction_and_bad_shapes_refused(self):
        x=[np.array([v],dtype=np.uint8) for v in [2,250,240,1]]
        np.testing.assert_array_equal(interaction(*x),[-487.])
        with self.assertRaises(ValueError):interaction([1],[1,2],[1],[1])
        with self.assertRaises(ValueError):interaction([1],[np.nan],[1],[1])

    def test_inherited_corner_binding_rejects_wrong_actual_receiver(self):
        from two_clock_study import qualify
        nn={'fields':{},'actualEFields':{},'actual':[]};enc={'fields':{}};forecast={'prefix':{},'fields':{}};nh={'actual':[],'fields':{}};source={'prefix':{}}
        for c in ['00','10']:
            fs={n:{'payloadSHA256':c+n} for n in ['latent2','latent3','velocity1','velocity2']}
            nn['fields'][c]=fs;e={'payloadSHA256':c+'E'};nn['actualEFields'][c]=e;enc['fields'][c]={'E':e}
            prefix={n:fs[n] for n in ['latent3','velocity1','velocity2']};forecast['prefix'][c]=prefix;source['prefix'][c]=prefix
            forecast['fields']['hold-'+c]={'velocity3':fs['velocity2']};nh['fields']['hold-'+c]={'latent4':{'payloadSHA256':c+'heldZ4'}}
            nh['actual'].append({'policy':'hold','case':c,'Z3PayloadSHA256':fs['latent3']['payloadSHA256'],'velocityPayloadSHA256':fs['velocity2']['payloadSHA256']})
            for step in [2,3]:nn['actual'].append({'kind':'denoiser-input','case':c,'step':step,'latentPayloadSHA256':fs['latent'+str(step)]['payloadSHA256'],'EPayloadSHA256':e['payloadSHA256']})
        self.assertEqual(len(qualify(nn,enc,forecast,nh,source)),2)
        bad=copy.deepcopy(nh);bad['actual'][0]['velocityPayloadSHA256']='wrong'
        with self.assertRaises(ValueError):qualify(nn,enc,forecast,bad,source)
        bad=copy.deepcopy(nn);bad['actual'][0]['EPayloadSHA256']='wrong'
        with self.assertRaises(ValueError):qualify(bad,enc,forecast,nh,source)

    def test_guard_allows_only_actual_prefix_and_image_model_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parent=root/'parent';parent.mkdir();model=root/'model';model.mkdir()
            for n in ['Z2.npy','v1.npy','E.npy','v2.npy','Z3native.npy']:(parent/n).write_bytes(b'x')
            (model/'transformer').mkdir();weight=model/'transformer/weights.safetensors';weight.write_bytes(b'allowed image weights')
            (model/'text_encoder').mkdir();qwen=model/'text_encoder/weights.safetensors';qwen.write_bytes(b'forbidden encoder')
            save(root/'targets.json',{});save(root/'inputs.json',{'modelRoot':str(model),'imageAssets':{'transformer/weights.safetensors':'hash'},'targetRoots':[str(parent)],
                'prefix':{'00':{k:{'path':str(parent/n)} for k,n in [('latent2','Z2.npy'),('velocity1','v1.npy'),('E','E.npy')]}},
                'ids':{'path':str(parent/'ids.npy')},'clockTable':{'rows':[]},'forbiddenTargets':[],'guardCanary':str(parent/'v2.npy')})
            code='import sys;sys.path.insert(0,'+repr(str(Path(__file__).parent))+');from two_clock import install_guard;install_guard('+repr(str(root))+')\n'
            for path in [parent/'Z2.npy',parent/'v1.npy',parent/'E.npy',weight]:code+='assert open('+repr(str(path))+',"rb").read()\n'
            for path in [parent/'v2.npy',parent/'Z3native.npy',qwen,root/'targets.json']:
                code+='try:open('+repr(str(path))+',"rb")\nexcept PermissionError:pass\nelse:raise AssertionError("future leaked")\n'
            subprocess.run([sys.executable,'-I','-c',code],check=True,capture_output=True,timeout=20)

    def test_all_four_images_seal_before_targets_and_failure_cannot_release(self):
        import two_clock_study as study
        for fail in [False,True]:
            with tempfile.TemporaryDirectory() as temp:
                events=[]
                def child(*args,**kwargs):
                    events.append('images')
                    if fail:raise subprocess.CalledProcessError(1,args[0])
                with patch.object(study,'read',return_value={}),patch.object(study,'verify'),patch.object(study.subprocess,'run',side_effect=child),patch.object(study,'seal',side_effect=lambda p:events.append('seal')),patch.object(study,'sha',return_value='sealed'),patch.object(study,'score',side_effect=lambda *a:events.append('score')):
                    if fail:
                        with self.assertRaises(subprocess.CalledProcessError):study.execute(temp)
                        self.assertEqual(events,['images'])
                    else:study.execute(temp);self.assertEqual(events,['images','seal','score'])

    def test_lazy_workers_import_without_model_runtime(self):
        code='import sys;sys.path.insert(0,'+repr(str(Path(__file__).parent))+');import two_clock,two_clock_images,two_clock_study;assert not ({"torch","diffusers","transformers"}&set(sys.modules))'
        subprocess.run([sys.executable,'-I','-c',code],check=True,capture_output=True,timeout=20)


if __name__=='__main__':unittest.main()
