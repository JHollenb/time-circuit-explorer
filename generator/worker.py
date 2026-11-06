"""Prediction-only FLUX.2 Klein generator. No native transformer is constructed.

Run real jobs through run.py/mrun. Native text encoding and VAE decoding surround
the byte-pinned NumPy denoiser. Only final latent bits, PNGs, and measurements
are saved; working registers are released as the clock advances.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import runpy
import struct
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
PROFILE={'height':512,'width':512,'passes':4,'cpuThreads':4,'queryChunk':64,'projectionChunk':256,'neuronChunk':256}
REQUIREMENTS={'torch':'2.13.0','diffusers':'0.39.0','transformers':'5.14.1'}


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):result.update(block)
    return result.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)


def validate_input(prompt,seed):
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>1000:raise ValueError('Prompt must contain 1–1000 characters.')
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Seed must be an integer from 0 to 4294967295.')
    return prompt.strip(),seed


def verify_sources():
    manifest=json.loads((HERE/'kernel-manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if not (HERE/name).resolve().is_relative_to(HERE) or sha(HERE/name)!=digest:raise ValueError('Pinned source changed: '+name)
    return manifest


class ParameterStore:
    """Stream one exact checkpoint bank; never instantiate the native transformer."""
    def __init__(self,root,recipes,*,verified_files=None):
        self.root=Path(root).resolve();self.recipes={x['source']['tensor']:x for x in recipes['artifacts']}
        self.files={};self.loads=0;self.read_bytes=0
        for entry in self.recipes.values():
            src=entry['source'];path=(self.root/src['file']).resolve()
            if not path.is_relative_to(self.root):raise ValueError('Checkpoint path escapes model directory')
            if src['file'] in self.files:continue
            if not verified_files or verified_files.get(src['file'])!=src['file_sha256']:
                if sha(path)!=src['file_sha256']:raise ValueError('Checkpoint digest differs')
            with path.open('rb') as stream:
                prefix=stream.read(8)
                if len(prefix)!=8:raise ValueError('Truncated safetensors file')
                length=struct.unpack('<Q',prefix)[0]
                if not 0<length<=16<<20:raise ValueError('Invalid safetensors header')
                header=json.loads(stream.read(length))
            self.files[src['file']]={'path':path,'offset':8+length,'header':header,'signature':self.signature(path)}

    @staticmethod
    def signature(path):
        st=path.stat();return(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)

    def load(self,name):
        import numpy as np
        if name is None:return None
        src=self.recipes[name]['source'];file=self.files[src['file']];header=file['header'][name]
        if header['dtype']!='BF16' or header['shape']!=src['shape'] or header['data_offsets']!=src['data_offsets']:
            raise ValueError('Parameter recipe/header geometry differs: '+name)
        if self.signature(file['path'])!=file['signature']:raise ValueError('Checkpoint changed during prediction')
        start,end=src['data_offsets']
        if end-start!=2*int(np.prod(src['shape'])) or end+file['offset']>file['signature'][2]:raise ValueError('Invalid parameter bounds')
        with file['path'].open('rb') as stream:stream.seek(file['offset']+start);raw=stream.read(end-start)
        if len(raw)!=end-start or hashlib.sha256(raw).hexdigest()!=src['payload_sha256']:raise ValueError('Parameter payload digest differs: '+name)
        value=np.frombuffer(raw,dtype='<u2').astype(np.uint32);value<<=16
        value=value.view(np.float32).reshape(src['shape']);value.setflags(write=False)
        self.loads+=1;self.read_bytes+=len(raw);return value

    def weights(self,spec):return{name:self.load(ref) for name,ref in spec['parameters'].items()}


class Measurements:
    def __init__(self,torch):
        import psutil
        self.torch=torch;self.process=psutil.Process();self.stages=[];self.active=None
        self.done=threading.Event();self.peak_rss=0
        self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start()

    def sample(self):
        while not self.done.is_set():
            rss=self.process.memory_info().rss;self.peak_rss=max(self.peak_rss,rss)
            active=self.active
            if active is not None:active['peakRssBytes']=max(active['peakRssBytes'],rss)
            self.done.wait(.1)

    @contextmanager
    def stage(self,name):
        torch=self.torch
        if torch.cuda.is_initialized():torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        row={'stage':name,'peakRssBytes':self.process.memory_info().rss}
        wall,cpu=time.perf_counter(),time.process_time();self.active=row
        try:yield row
        finally:
            if torch.cuda.is_initialized():torch.cuda.synchronize()
            row.update(wallSeconds=time.perf_counter()-wall,cpuSeconds=time.process_time()-cpu,
                peakCudaAllocatedBytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
                peakCudaReservedBytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
            self.stages.append(row);self.active=None

    def close(self):self.done.set();self.thread.join(timeout=2)


def predict(latent,conditioning,img_ids,txt_ids,timesteps,sigmas,binding,store,kernels,event):
    """Own-state recurrence. No recording of full intermediate activations."""
    import numpy as np
    project_kernel=kernels['flux_reentry_reference.py'];single=kernels['flux_transition_reference.py'];boundary=kernels['flux_step_boundary_reference.py']
    commits=[]
    for step in range(4):
        spec=binding['project'];weights=store.weights(spec)
        result=project_kernel['predict_project'](latent,conditioning,img_ids,txt_ids,timesteps[step],weights,
            axes_dims_rope=spec['axes_dims_rope'],rope_theta=spec['rope_theta'],timestep_channels=spec['timestep_channels'],projection_chunk=256)
        del weights
        image,text,context=result['image'],result['text'],result['context'];del result
        event('transition',step=step,local=0,phase='project',completed=step*28+1)
        for index in range(5):
            spec=binding['joint'+str(index)];weights=store.weights(spec)
            result=project_kernel['predict_joint'](image,text,context,weights,heads=spec['heads'],layernorm_eps=spec['layernorm_eps'],qk_eps=spec['qk_eps'],
                query_chunk=64,projection_chunk=256,retain_fields=False)
            image,text=result['image'],result['text'];del result,weights
            event('transition',step=step,local=index+1,phase='joint',block=index,completed=step*28+index+2)
        current=np.concatenate((text,image),axis=1);del image,text
        for index in range(20):
            spec=binding['final_blocks'][str(index)];weights=store.weights(spec)
            result=single['predict_transition'](current,context['single_mod'],**weights,heads=spec['heads'],layernorm_eps=spec['layernorm_eps'],qk_eps=spec['qk_eps'],
                rope_cos=context['rope'][0],rope_sin=context['rope'][1],query_chunk=64,projection_chunk=256,neuron_chunk=256,
                retain_fields=False,text_rows=None,sample_indices=[0,31,511,512,543,767,1023,1535])
            current=result['output'];del result,weights
            event('transition',step=step,local=index+6,phase='single',block=index,completed=step*28+index+7)
        spec=binding['readout'];weights=store.weights(spec)
        result=boundary['predict_readout'](current[:,binding['text_rows']:],context['temb'],**weights,eps=spec['eps'],neuron_chunk=256,
            sample_indices=[0,31,32,255,511,512,767,1023],retain_fields=False)
        noise=result['noise'];del result,weights,current,context
        event('transition',step=step,local=26,phase='readout',completed=step*28+27)
        result=boundary['predict_flow_commit'](latent,noise,sigmas[step],sigmas[step+1]);latent=result['latent']
        commits.append({'step':step,'latent':result['latent_descriptor'],'dtBF16':result['dt_bf16']});del result,noise
        event('transition',step=step,local=27,phase='commit',completed=step*28+28)
        gc.collect()
    return latent,commits


def decode(pipe,latent,reference,img_ids,torch):
    import numpy as np
    # Match the qualified adapter's native latent layout and CPU BN-stat arithmetic.
    packed=torch.empty_strided(reference.shape,reference.stride(),dtype=reference.dtype,device='cuda')
    packed.copy_(torch.from_numpy(np.array(latent,copy=True)).to(device='cuda',dtype=torch.bfloat16))
    unpacked=pipe._unpack_latents_with_ids(packed,img_ids,512//(pipe.vae_scale_factor*2),512//(pipe.vae_scale_factor*2))
    mean=pipe.vae.bn.running_mean.detach().cpu().view(1,-1,1,1).to(unpacked)
    std=(pipe.vae.bn.running_var.detach().cpu().view(1,-1,1,1)+pipe.vae.config.batch_norm_eps).sqrt().to(unpacked)
    unpacked=pipe._unpatchify_latents(unpacked*std+mean)
    pixels=pipe.vae.decode(unpacked,return_dict=False)[0]
    if not bool(torch.isfinite(pixels).all()):raise ValueError('Native VAE returned nonfinite pixels')
    image=pipe.image_processor.postprocess(pixels,output_type='pil')[0]
    if image.size!=(512,512) or image.mode!='RGB':raise ValueError('Native VAE image geometry/mode differs')
    return image


def run(request_path):
    started=time.perf_counter();request=json.loads(Path(request_path).read_text());config=request['config']
    for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[variable]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    output=Path(config.get('output','prediction-result')).resolve();output.mkdir(parents=True,exist_ok=False)
    manifest=verify_sources()
    for name,expected in REQUIREMENTS.items():
        if importlib.metadata.version(name).split('+')[0]!=expected:raise ValueError('Qualified package version differs: '+name)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline,Flux2Transformer2DModel
    from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_timesteps,compute_empirical_mu
    from PIL import Image
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('This qualified encoder/VAE profile requires CUDA. Submit through mrun.')
    calls={'nativeDenoiser':0,'promptEncodings':0,'vaeDecodes':0,'predictedTransitions':0}
    def prohibited(*args,**kwargs):calls['nativeDenoiser']+=1;raise RuntimeError('Native denoiser execution is prohibited in prediction-only mode')
    Flux2Transformer2DModel.forward=prohibited
    events=(output/'events.jsonl').open('x')
    def event(kind,**values):
        row={'kind':kind,'workerSeconds':time.perf_counter()-started,**values};events.write(json.dumps(row,sort_keys=True)+'\n');events.flush()
        print('TCE_EVENT '+json.dumps(row,sort_keys=True),flush=True)
    measurement=Measurements(torch);root=Path(config['modelPath']).resolve()
    try:
        with measurement.stage('verify-checkpoint'):
            checkpoint_files=json.loads((HERE/'checkpoint-files.json').read_text())
            for name,digest in checkpoint_files.items():
                if sha(root/name)!=digest:raise ValueError('Pinned model file differs: '+name)
            recipes=json.loads((HERE/'parameter-artifacts.json').read_text());store=ParameterStore(root,recipes,verified_files=checkpoint_files)
        with measurement.stage('load-encoder-vae'):
            pipe=Flux2KleinPipeline.from_pretrained(str(root),transformer=None,torch_dtype=torch.bfloat16,local_files_only=True)
            if pipe.transformer is not None or not pipe.config.is_distilled or pipe.scheduler.config.stochastic_sampling:raise ValueError('Unsupported pipeline profile')
            pipe.set_progress_bar_config(disable=True)
            for component in (pipe.text_encoder,pipe.vae):component.eval().requires_grad_(False)
            encoder_type=type(pipe.text_encoder)
        binding=json.loads((HERE/'bindings.json').read_text())
        kernels={name:runpy.run_path(str(HERE/'kernels'/name)) for name in ['flux_reentry_reference.py','flux_transition_reference.py','flux_step_boundary_reference.py']}
        images=[];startup=list(measurement.stages);startup_wall=time.perf_counter()-started
        for index,specimen in enumerate(config['samples']):
            prompt,seed=validate_input(specimen['prompt'],specimen['seed']);sample_id=f'sample-{index:02d}'
            begin=time.perf_counter();stage_start=len(measurement.stages);reads=store.read_bytes;loads=store.loads
            event('sample-start',sample=sample_id,prompt=prompt,seed=seed)
            with measurement.stage(sample_id+'/encode'):
                if pipe.text_encoder is None:
                    pipe.text_encoder=encoder_type.from_pretrained(str(root/'text_encoder'),torch_dtype=torch.bfloat16,local_files_only=True)
                    pipe.text_encoder.eval().requires_grad_(False)
                pipe.vae.to('cpu');pipe.text_encoder.to('cuda')
                with torch.inference_mode():conditioning,txt_ids=pipe.encode_prompt(prompt,device=torch.device('cuda'),num_images_per_prompt=1,max_sequence_length=512)
                conditioning_cpu=conditioning.float().cpu().numpy();txt_ids_cpu=txt_ids.cpu().numpy();calls['promptEncodings']+=1
                del conditioning;pipe.text_encoder=None;gc.collect();torch.cuda.empty_cache()
            with measurement.stage(sample_id+'/prepare-noise-schedule'):
                with torch.inference_mode():
                    reference,img_ids=pipe.prepare_latents(1,32,512,512,torch.bfloat16,torch.device('cuda'),torch.Generator(device='cuda').manual_seed(seed))
                    schedule=None if getattr(pipe.scheduler.config,'use_flow_sigmas',False) else np.linspace(1.,.25,4)
                    timesteps,count=retrieve_timesteps(pipe.scheduler,4,torch.device('cuda'),sigmas=schedule,mu=compute_empirical_mu(image_seq_len=reference.shape[1],num_steps=4))
                    if count!=4:raise ValueError('Scheduler pass count differs')
                    latent=reference.float().cpu().numpy();image_ids_cpu=img_ids.cpu().numpy()
                    ts=timesteps.float().cpu().numpy();sigmas=pipe.scheduler.sigmas.float().cpu().numpy()
                origin={'latent':kernels['flux_reentry_reference.py']['descriptor'](latent),'conditioning':kernels['flux_reentry_reference.py']['descriptor'](conditioning_cpu),
                    'timesteps':ts.tolist(),'sigmas':sigmas.tolist(),'imageIDs':hashlib.sha256(image_ids_cpu.tobytes()).hexdigest(),'textIDs':hashlib.sha256(txt_ids_cpu.tobytes()).hexdigest()}
            def transition(kind,**values):calls['predictedTransitions']+=1;event(kind,sample=sample_id,**values)
            with measurement.stage(sample_id+'/prediction'):
                latent,commits=predict(latent,conditioning_cpu,image_ids_cpu,txt_ids_cpu,ts,sigmas,binding,store,kernels,transition)
            event('latent-ready',sample=sample_id,terminal=commits[-1]['latent'])
            with measurement.stage(sample_id+'/vae'):
                pipe.vae.to('cuda');pipe.vae.bn.to('cpu')
                with torch.inference_mode():image=decode(pipe,latent,reference,img_ids,torch)
                calls['vaeDecodes']+=1;pipe.vae.to('cpu');torch.cuda.empty_cache()
            with measurement.stage(sample_id+'/save'):
                image_path=output/(sample_id+'.png');image.save(image_path)
                latent_path=output/(sample_id+'.latent-bf16.npy');np.save(latent_path,(np.ascontiguousarray(latent).view(np.uint32)>>16).astype('<u2'),allow_pickle=False)
            sample={'id':sample_id,'prompt':prompt,'seed':seed,'width':512,'height':512,'passes':4,'transitions':112,'arithmetic':'Qualified NumPy FP64 reductions with explicit BF16 boundaries',
                'wallSeconds':time.perf_counter()-begin,'stages':measurement.stages[stage_start:],
                'origin':origin,'commits':commits,'parameterLoads':store.loads-loads,'parameterBytesRead':store.read_bytes-reads,
                'image':{'path':image_path.name,'sha256':sha(image_path),'bytes':image_path.stat().st_size},
                'latent':{'path':latent_path.name,'sha256':sha(latent_path),'bytes':latent_path.stat().st_size},
                'nativeDenoiserCalls':calls['nativeDenoiser'],'nativeTransformerLoaded':False,'nativeComparison':None}
            baseline=manifest['baseline']
            if prompt==baseline['prompt'] and seed==baseline['seed']:
                sample['baselineReproduction']={'initialExact':origin['latent']['bf16_le_sha256']==baseline['initialBF16'],
                    'conditioningExact':origin['conditioning']['bf16_le_sha256']==baseline['conditioningBF16'],
                    'terminalExact':commits[-1]['latent']['fp32_le_sha256']==baseline['terminalFP32'],
                    'pngExact':sample['image']['sha256']==baseline['pngSHA256'],'authority':'Previous sealed mathematical forecast, not native CUDA parity.'}
                # This is an execution-parity check for the extracted generator.
                if not all(sample['baselineReproduction'][key] for key in ['initialExact','conditioningExact','terminalExact','pngExact']):
                    save(output/(sample_id+'.json'),sample);raise ValueError('Extracted generator differs from its qualified baseline; result retained')
            save(output/(sample_id+'.json'),sample);images.append(sample)
            event('image-ready',sample=sample_id,image=sample['image'],seconds=sample['wallSeconds'])
            del latent,conditioning_cpu,txt_ids_cpu,image_ids_cpu,reference,img_ids,image;gc.collect()
        report={'schema':'tce-flux-prediction-only-v1','requestSHA256':request['requestSHA256'],'sourceManifestSHA256':sha(HERE/'kernel-manifest.json'),
            'profile':PROFILE,'startupStages':startup,'startupWallSeconds':startup_wall,'samples':images,'counts':calls,'nativeTransformerLoaded':False,
            'encoderRetention':'Released after encoding; subsequent samples reload the same native encoder. Encoder reload/transfers are included in the encode stage.',
            'workerWallSeconds':time.perf_counter()-started,'processCpuSeconds':time.process_time(),
            'peakRssBytes':max(measurement.peak_rss,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
            'peakCudaAllocatedBytes':max(row['peakCudaAllocatedBytes'] for row in measurement.stages),
            'peakCudaReservedBytes':max(row['peakCudaReservedBytes'] for row in measurement.stages),
            'environment':{'host':platform.node(),'platform':platform.platform(),'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),
                'versions':{name:importlib.metadata.version(name) for name in ['torch','diffusers','transformers','numpy','psutil','pillow']}},
            'measurementScope':'Wall time uses synchronized CUDA boundaries; process CPU time includes all threads. RSS sampled every 100 ms plus process high-water mark. CUDA allocation/reservation are PyTorch counters, not board power or total device residency. Startup, encoding/transfers, noise/schedule, CPU prediction, VAE/transfers, and save are separate.',
            'scope':'Prediction-only, full dense learned-weight computation. Native encoder and VAE retained. No native denoiser or reference image run; no speedup or new-prompt accuracy claim.'}
        save(output/'report.json',report);event('complete',images=len(images),workerSeconds=report['workerWallSeconds'])
        return report
    finally:measurement.close();events.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args()
    try:run(args.request)
    except Exception as error:
        print('TCE_FAILURE '+json.dumps({'type':type(error).__name__,'message':str(error)}),flush=True);raise
