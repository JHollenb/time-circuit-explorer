"""Small dependency-light execution mechanics checks; no model weights or forwards."""
import hashlib
import importlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from worker import ParameterStore,predict,validate_input,verify_sources
import run as runner


class GeneratorMechanics(unittest.TestCase):
    def test_guarded_collection_rejects_ordinary_admission_or_wrong_execution_host(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'request';request=runner.stage(root,[{'prompt':'a cone','seed':113}])
            seal={'sha256':'a'*64,'size_bytes':123}
            job={'state':'succeeded','assigned_host':'beast','cmd':runner.command(),'config':{**request,'retry_on_kill':False,'mrun_guarded_request_sha256':'b'*64},
                'needs':{'payload_custody_v2':True},'admission':{'claim_key':'test'},'payload_custody':{'required':True,'declared':seal,'sealed':seal,'executed':{**seal,'host':'beast'}},
                'result':{'returncode':0,'diagnostics':{'cwd':'/mnt/hdd8tb/mrun-work/job-proof'}}}
            for variant in [{**job,'admission':None},{**job,'payload_custody':{**job['payload_custody'],'required':False}},
                            {**job,'payload_custody':{**job['payload_custody'],'executed':{**seal,'host':'other-host'}}}]:
                with patch.object(runner,'status',return_value={'_job':variant,'jobID':'job-proof'}),patch.object(runner.subprocess,'run',side_effect=AssertionError('Unexpected collection attempt')):
                    with self.assertRaisesRegex(ValueError,'guarded admission|host differs'):runner.collect(root)

    def test_request_binds_one_finite_scheduler_submission_with_measured_cold_start_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'request'
            request=runner.stage(root,[{'prompt':'a cone','seed':113}])
            self.assertEqual(request['executionContract']['expectedWallSeconds'],1000)
            with patch.object(importlib.import_module('mrun.client.submit'),'launch',return_value='job-mechanics-only') as launch:
                result=runner.submit(root)
                self.assertEqual(result['jobID'],'job-mechanics-only')
                kwargs=launch.call_args.kwargs
                self.assertEqual(kwargs['timeout_s'],1500);self.assertEqual(kwargs['est_wall_s'],1000)
                self.assertTrue(kwargs['needs']['cuda']);self.assertTrue(kwargs['preflight']);self.assertFalse(kwargs['retry_on_kill'])
                self.assertTrue(kwargs['guarded'])
                with self.assertRaisesRegex(ValueError,'already has a job'):runner.submit(root)
                self.assertEqual(launch.call_count,1)

    def test_qualified_sources_and_parameter_only_bindings(self):
        manifest=verify_sources();root=Path(__file__).parent
        self.assertEqual(manifest['sourceCampaign'],'job-4350531b1dcd')
        binding=json.loads((root/'bindings.json').read_text())
        self.assertEqual(len(binding['final_blocks']),20)
        self.assertNotIn('origins',binding)
        self.assertEqual(len(json.loads((root/'parameter-artifacts.json').read_text())['artifacts']),169)

    def test_prompt_and_seed_limits(self):
        self.assertEqual(validate_input('  a cone  ',113),('a cone',113))
        for prompt,seed in [('',0),('x'*1001,0),('ok',-1),('ok',2**32),('ok',True)]:
            with self.assertRaises(ValueError):validate_input(prompt,seed)

    def make_store(self,root):
        values=np.array([[0x3f80,0xbf80,0x4000],[0x0000,0x8000,0x3f00]],dtype='<u2');raw=values.tobytes()
        header={'tiny.weight':{'dtype':'BF16','shape':[2,3],'data_offsets':[0,len(raw)]}}
        h=json.dumps(header).encode();path=root/'tiny.safetensors';path.write_bytes(struct.pack('<Q',len(h))+h+raw)
        manifest={'artifacts':[{'source':{'tensor':'tiny.weight','dtype':'BF16','shape':[2,3],
            'file':'tiny.safetensors','file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'data_offsets':[0,len(raw)],'payload_sha256':hashlib.sha256(raw).hexdigest()}}]}
        return ParameterStore(root,manifest),path,manifest

    def test_parameter_stream_preserves_native_bits_and_signed_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            store,_,_=self.make_store(Path(directory));value=store.load('tiny.weight')
            self.assertEqual(value.dtype,np.float32);self.assertEqual(value.tolist(),[[1.,-1.,2.],[0.,-0.,.5]])
            self.assertTrue(np.signbit(value[1,1]));self.assertFalse(value.flags.writeable)
            self.assertEqual(store.read_bytes,12);self.assertEqual(store.loads,1);self.assertIsNone(store.load(None))

    def test_mutated_checkpoint_and_bad_recipe_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store,path,manifest=self.make_store(root)
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):store.load('tiny.weight')
            with self.assertRaisesRegex(ValueError,'digest'):ParameterStore(root,manifest)
            store,path,manifest=self.make_store(root);manifest['artifacts'][0]['source']['data_offsets']=[0,2]
            bad=ParameterStore(root,manifest)
            with self.assertRaisesRegex(ValueError,'geometry'):bad.load('tiny.weight')

    def test_all_112_operations_continue_from_own_state_without_history_retention(self):
        events=[];received=[]
        def project(z,c,i,t,time,weights,**kwargs):
            received.append(z.copy());return{'image':np.repeat(z,2,axis=-1)+1,'text':np.ones((1,1,4)),
                'context':{'single_mod':np.zeros((1,12)),'temb':np.zeros((1,4)),'rope':(np.zeros((3,4)),np.zeros((3,4)))}}
        def joint(image,text,context,weights,**kwargs):return{'image':image+1,'text':text+1}
        def single(current,context,**kwargs):return{'output':current+1}
        def readout(image,context,**kwargs):return{'noise':image[:,:,:2]}
        def commit(z,v,s,ns):
            out=z+v;return{'latent':out,'latent_descriptor':{'sum':float(out.sum())},'dt_bf16':float(ns-s)}
        kernels={'flux_reentry_reference.py':{'predict_project':project,'predict_joint':joint},
            'flux_transition_reference.py':{'predict_transition':single},'flux_step_boundary_reference.py':{'predict_readout':readout,'predict_flow_commit':commit}}
        spec={'parameters':{},'heads':1,'layernorm_eps':1e-6,'qk_eps':1e-6,'eps':1e-6,'axes_dims_rope':[4],'rope_theta':2000,'timestep_channels':4}
        binding={'text_rows':1,'project':spec,'readout':spec,'final_blocks':{str(i):spec for i in range(20)},**{'joint'+str(i):spec for i in range(5)}}
        class Store:
            def weights(self,spec):return{}
        z,commits=predict(np.zeros((1,2,2)),None,None,None,np.arange(4),np.arange(5),binding,Store(),kernels,lambda kind,**e:events.append(e))
        self.assertEqual(len(events),112);self.assertEqual([e['completed'] for e in events],list(range(1,113)))
        # Each pass reads its own previous commit. The toy whole-pass law is z′=2z+26.
        self.assertEqual([float(z[0,0,0]) for z in received],[0,26,78,182])
        self.assertTrue(np.all(z==390));self.assertEqual(len(commits),4)


if __name__=='__main__':unittest.main()
