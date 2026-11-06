import {$,assetJSON} from './components.js';
import {escapeHTML as esc} from './kernels.js';

const seconds=value=>Number.isFinite(value)?`${value.toFixed(2)} s`:'—';
const memory=value=>Number.isFinite(value)?`${(value/2**30).toFixed(2)} GiB`:'—';
const labels={encode:'Native text encoder', 'prepare-noise-schedule':'Initial noise + schedule',prediction:'Reconstructed denoiser',vae:'Native VAE',save:'Save PNG + latent'};
export function summarizeSample(sample){
  const stage=key=>sample.stages.find(x=>x.stage.endsWith('/'+key));
  return{prediction:stage('prediction')?.wallSeconds,cpu:stage('prediction')?.cpuSeconds,
    total:sample.wallSeconds,ram:Math.max(...sample.stages.map(x=>x.peakRssBytes)),cuda:Math.max(...sample.stages.map(x=>x.peakCudaAllocatedBytes)),
    stages:sample.stages.map(x=>({...x,key:x.stage.split('/').at(-1)}))};
}
export const shellQuote=value=>`'${String(value).replaceAll("'","'\\''")}'`;

export async function mountGenerator(signal){
  if(!document.getElementById('generator-styles')){const link=document.createElement('link');link.id='generator-styles';link.rel='stylesheet';link.href='./generator.css';document.head.append(link);}
  const [data,capabilities]=await Promise.all([assetJSON('./data/generator/benchmarks.json',signal),
    fetch('./api/generator/capabilities',{signal,cache:'no-store'}).then(r=>r.ok?r.json():{mode:'recorded'}).catch(()=>({mode:'recorded'}))]);
  if(signal.aborted)return()=>{};
  let selected=0,activeJob=null,pollTimer=null,polling=false,pending=null;
  const live=capabilities.mode==='live';
  const storageKey='tce-generator-request';
  const persist=()=>{try{if(activeJob||pending)sessionStorage.setItem(storageKey,JSON.stringify({activeJob,pending}));else sessionStorage.removeItem(storageKey);}catch{}};
  if(live){try{const saved=JSON.parse(sessionStorage.getItem(storageKey)||'null');if(/^[a-f0-9]{32}$/.test(saved?.activeJob))activeJob=saved.activeJob;else if(saved?.pending?.requestID)pending=saved.pending;}catch{}}
  const finish=()=>{activeJob=null;pending=null;persist();$('generator-submit').disabled=!live;$('generator-check').hidden=true;};
  const on=(id,event,fn)=>$(id).addEventListener(event,fn,{signal});
  $('generator-mode').textContent=live?'LIVE · LOCAL MRUN QUEUE':'RECORDED · MEASURED RUNS';
  $('generator-mode').classList.toggle('live',live);
  $('generator-submit').disabled=!live||Boolean(activeJob||pending);
  $('generator-mode-note').textContent=live?'Generate a new 512 × 512 image through the local mrun queue. One image runs at a time. Text encoding and VAE decoding use CUDA; the reconstructed denoiser runs on CPU.':
    'Explore actual prediction-only outputs and their measured compute. Start the local generator server to submit a new prompt; this hosted demonstration shows saved runs.';
  const samples=data.samples||[];
  $('generator-presets').innerHTML=samples.map((sample,i)=>`<button data-sample="${i}" class="${i===selected?'active':''}" aria-pressed="${i===selected}">${i===0?'Qualified scene':'New prompt'}<span>seed ${sample.seed}</span></button>`).join('');
  function render(sample,base,reportURL,{startup=0,recorded=true,jobID=''}={}){
    const stats=summarizeSample(sample);
    $('generator-result').innerHTML=`<figure class="generator-image"><img src="${base+sample.image.path}" alt="${esc(sample.prompt)}" width="512" height="512"><figcaption>${recorded?'Actual benchmark output':'New prediction-only output'} · seed ${sample.seed}<br>Predicted latent → native VAE · ${sample.transitions} reconstructed transitions</figcaption></figure><div class="generator-result-info"><div class="eyebrow">${recorded?'MEASURED GENERATION':'GENERATION COMPLETE'}</div><h3>From a prompt to its own predicted future.</h3><p class="generator-prompt">“${esc(sample.prompt)}”</p><div class="generator-stats"><div><strong>${seconds(stats.prediction)}</strong><span>CPU denoiser wall time</span></div><div><strong>${seconds(stats.total)}</strong><span>image time after setup</span></div><div><strong>${memory(stats.ram)}</strong><span>peak sampled process RAM</span></div><div><strong>${memory(stats.cuda)}</strong><span>peak CUDA allocation</span></div></div><div class="generator-stages">${stats.stages.map(stage=>`<div><span>${labels[stage.key]||esc(stage.key)}</span><i><b style="width:${Math.max(1,100*stage.wallSeconds/stats.total)}%"></b></i><strong>${seconds(stage.wallSeconds)}</strong></div>`).join('')}</div><p class="generator-measurement">CPU prediction: ${seconds(stats.cpu)} of process CPU time across four configured threads. Parameter reads: ${(sample.parameterBytesRead/1e9).toFixed(2)} GB. ${startup?`One-time setup: ${seconds(startup)}; first image including setup: ${seconds(startup+stats.total)}.`:'Setup was already loaded for this image.'}</p><div class="generator-proof"><span>● Native denoiser calls: ${sample.nativeDenoiserCalls}</span><span>● Native transformer loaded: ${sample.nativeTransformerLoaded?'yes':'no'}</span>${sample.baselineReproduction?`<span>● Prior forecast latent: ${sample.baselineReproduction.terminalExact?'exact reproduction':'differs'}</span><span>● Prior forecast PNG: ${sample.baselineReproduction.pngExact?'byte-identical':'differs'}</span>`:'<span>● New prompt; native accuracy comparison unmeasured</span>'}</div><p class="generator-links"><a href="${base+sample.image.path}" download>Download original PNG ↗</a><a href="${reportURL}" download>Measurement receipt ↗</a></p><span class="generator-job">${esc(jobID)}</span></div>`;
  }
  function select(index){
    selected=index;const sample=samples[index];if(!sample)return;
    $('generator-prompt').value=sample.prompt;$('generator-seed').value=sample.seed;
    for(const button of $('generator-presets').querySelectorAll('button')){const active=Number(button.dataset.sample)===index;button.classList.toggle('active',active);button.setAttribute('aria-pressed',active);}
    render(sample,'./data/generator/','./data/generator/report.json',{startup:index===0?data.startupWallSeconds:0,jobID:data.jobID});
  }
  function command(){
    $('generator-command').textContent=`../common/bin/common exec python generator/server.py\n# Open http://127.0.0.1:4177/?model=flux2-klein-4b#generator\n\n# Or submit directly:\n../common/bin/common exec python generator/run.py stage artifacts/my-prediction \\\n  --prompt ${shellQuote($('generator-prompt').value.trim())} --seed ${Number($('generator-seed').value)||0}\n../common/bin/common exec python generator/run.py submit artifacts/my-prediction`;
  }
  function progress(value){
    const event=value.progress,completed=value.completedTransitions||0;
    $('generator-progress').hidden=false;$('generator-progress-fill').style.width=`${completed/112*100}%`;
    $('generator-progress-text').textContent=event?.kind==='transition'?`Pass ${event.step+1} / 4 · ${event.phase}${event.block!==undefined?' '+event.block:''} · ${completed} / 112 transitions`:value.state==='preparing'?'Preparing the bounded prediction request…':value.state==='queued'?'Queued in mrun. Waiting for an available worker.':event?.kind==='latent-ready'?'Predicted latent ready. Rendering through the native VAE…':`${value.state} · ${event?.kind||'waiting for worker progress'}`;
  }
  async function poll(){
    if(signal.aborted||!activeJob||polling)return;polling=true;
    try{
      const response=await fetch(`./api/generator/jobs/${activeJob}`,{signal,cache:'no-store'}),value=await response.json();
      if(response.status===404)finish();
      if(!response.ok)throw new Error(value.error||'Could not read job status.');signal.throwIfAborted();progress(value);
      if(value.error)$('generator-status').textContent=value.error;
      if(value.state==='succeeded'&&value.report){const sample=value.report.samples[0];render(sample,`./api/generator/jobs/${activeJob}/`,`./api/generator/jobs/${activeJob}/report.json`,{startup:value.report.startupWallSeconds,recorded:false,jobID:value.jobID});$('generator-status').textContent='New prediction complete. The image and measurements came from the generator.';finish();return;}
      if(['failed','killed','cancelled','timeout','timed_out','rejected'].includes(value.state)){finish();throw new Error(value.error||'Generation ended without a complete result. The run is retained in the local ledger.');}
      pollTimer=setTimeout(poll,2500);
    }catch(error){if(!signal.aborted){$('generator-status').textContent=error.message;$('generator-check').hidden=!activeJob;}}
    finally{polling=false;}
  }
  on('generator-presets','click',e=>{const button=e.target.closest('[data-sample]');if(button)select(Number(button.dataset.sample));command();});
  on('generator-prompt','input',command);on('generator-seed','input',command);
  async function submitPending(){
    if(!live||activeJob||!pending)return;
    $('generator-submit').disabled=true;$('generator-status').textContent='Submitting a prediction-only request…';
    try{
      const response=await fetch('./api/generator/jobs',{method:'POST',signal,headers:{'Content-Type':'application/json'},body:JSON.stringify(pending)});
      const result=await response.json();if(!response.ok){if(response.status<500)finish();throw new Error(result.error||'Could not submit generation.');}signal.throwIfAborted();
      activeJob=result.id;pending=null;persist();progress(result);$('generator-status').textContent='The worker will compute all four passes and decode its predicted final latent. This takes several minutes.';poll();
    }catch(error){if(!signal.aborted){$('generator-status').textContent=error.message+(pending?' Check the existing request before trying another prompt.':'');$('generator-check').hidden=!pending;}}
  }
  on('generator-check','click',()=>{$('generator-check').hidden=true;pending?submitPending():poll();});
  on('generator-form','submit',event=>{
    event.preventDefault();if(!live||activeJob||pending)return;
    pending={prompt:$('generator-prompt').value.trim(),seed:Number($('generator-seed').value),requestID:crypto.randomUUID()};persist();submitPending();
  });
  if(samples.length)select(0);else $('generator-result').innerHTML='<p class="flux-note">The prediction-only benchmark is being collected. Results will appear here with their measurement receipt.</p>';
  $('generator-scope').textContent='The generator evaluates the full original learned-weight arithmetic; it is not a compact student. Image cards show per-image stage peaks; startup is separate. Parameter reads are logical bytes and may hit the OS cache. '+(data.measurementScope||'Wall time, process CPU time, sampled RAM and PyTorch CUDA allocations are measured separately.');
  if(data.jobWallSeconds)$('generator-scope').textContent+=` Two-image worker: ${seconds(data.workerWallSeconds)}. Scheduler execution including process overhead: ${seconds(data.jobWallSeconds)}. Queue wait is excluded.`;
  command();
  if(activeJob){$('generator-status').textContent='Resuming the existing generation…';poll();}
  else if(pending){$('generator-status').textContent='A submission is awaiting confirmation. Check the existing request to resume it.';$('generator-check').hidden=false;}
  return()=>{clearTimeout(pollTimer);};
}
