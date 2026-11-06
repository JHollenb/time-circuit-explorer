import {$,assetJSON} from './components.js';
import {renderResponse,responseWeights} from './response-kernel.js';

const base='./data/response/';
const percent=x=>`${(100*x).toFixed(1)}%`;
const ms=x=>`${(1000*x).toFixed(2)} ms`;
export async function mountResponse(signal){
  if(!document.getElementById('response-styles')){const link=document.createElement('link');link.id='response-styles';link.rel='stylesheet';link.href='./response.css';document.head.append(link);}
  const data=await assetJSON(base+'response.json',signal);signal.throwIfAborted();
  const on=(id,event,fn)=>$(id).addEventListener(event,fn,{signal});
  const canvas=$('response-canvas'),ctx=canvas.getContext('2d'),pixels=ctx.createImageData(data.width,data.height);
  const cache=new Map();let anchors=null,revision=0,frame=0,downloadURL=null;
  const dispose=()=>{revision++;cancelAnimationFrame(frame);if(downloadURL)URL.revokeObjectURL(downloadURL);};
  signal.addEventListener('abort',dispose,{once:true});
  $('response-context').replaceChildren(...data.contexts.map(c=>new Option(`Seed ${c.id.split('.')[0]} · branch ${c.id.split('.')[1]}`,c.id)));
  const timing=data.timing;
  $('response-measured-compute').textContent=ms(timing.medianComputeSeconds);
  $('response-measured-save').textContent=ms(timing.medianComputeAndSaveSeconds);
  $('response-measured-ram').textContent=`${(data.serving.peakRssBytes/2**20).toFixed(1)} MiB`;
  $('response-cost-detail').textContent=`Beast CPU · NumPy · 16 primary outputs. Median CPU arithmetic ${ms(timing.medianCpuSeconds)}. Bundle load ${ms(data.serving.loadSeconds)}; compilation ${data.compileSeconds.toFixed(3)} s; all 96 outputs including controls ${data.serving.wallSeconds.toFixed(3)} s; worker + scorer ${data.workerWallSeconds.toFixed(3)} s. Per-image time excludes startup, loading, staging and queue wait. One measured run, not a speedup comparison.`;
  $('response-history-cost').textContent=`Historical source: ${data.sourceCost.vaeImages} native VAE decodes, ${data.sourceCost.nativeSchedulerChecks} scheduler checks and ${data.sourceCost.loads} model load; earlier acquisition adds cost. The chart stores 12 calibration RGB images (${(data.bundleBytes/1e6).toFixed(2)} MB compressed), larger than their 3.03 MB of original PNGs.`;
  $('response-error-table').innerHTML=[.05,.1,.5].map(alpha=>{
    const find=m=>data.groups.find(r=>r.method===m&&r.alpha===alpha),p=find('piecewise'),n=find('nearest');
    return `<tr><th scope="row">${alpha.toFixed(2)}</th><td>${percent(p.meanEditRelativeRms)}</td><td>${percent(n.meanEditRelativeRms)}</td><td>${percent(p.meanInteractionRelativeRms)}</td></tr>`;
  }).join('');
  function draw(){
    frame=0;if(!anchors||signal.aborted)return;
    const alpha=Number($('response-dose').value),context=$('response-context').value;
    $('response-dose-value').textContent=alpha.toFixed(3);
    const begin=performance.now();renderResponse(anchors,alpha,pixels.data);const arithmetic=performance.now()-begin;
    const paint=performance.now();ctx.putImageData(pixels,0,0);const paintMs=performance.now()-paint;
    $('response-live-time').textContent=`${arithmetic.toFixed(2)} ms arithmetic · ${paintMs.toFixed(2)} ms canvas paint`;
    $('response-live-label').textContent=`Computed here · ${context} · dose ${alpha.toFixed(3)}`;
    const weights=responseWeights(alpha);$('response-equation').textContent=`Ŷ = ${weights[0].toFixed(3)} Y₀ + ${weights[1].toFixed(3)} Y₀.₂₅ + ${weights[2].toFixed(3)} Y₁`;
    for(const b of $('response-doses').querySelectorAll('button'))b.setAttribute('aria-pressed',Number(b.dataset.dose)===alpha);
    const held=data.held.find(x=>x.context===context&&x.alpha===alpha),anchor=data.anchors.indexOf(alpha);
    $('response-reference-image').hidden=!held;$('response-anchor-canvas').hidden=anchor<0;
    $('response-reference-empty').hidden=Boolean(held)||anchor>=0;
    if(held){
      $('response-reference-image').src=base+held.path;
      const score=data.scores.find(x=>x.context===context&&x.alpha===alpha);
      $('response-accuracy').textContent=`Historical held comparison · edit error ${percent(score.edit.relativeRms)} · edit cosine ${score.edit.cosine.toFixed(3)} · RGB MAE ${score.rgb.mae.toFixed(3)} / 255.`;
      $('response-reference-label').textContent='Historical native VAE output · opened only after the assay sealed';
    }else if(anchor>=0){
      const target=$('response-anchor-canvas'),targetCtx=target.getContext('2d'),image=targetCtx.createImageData(512,512);
      renderResponse(anchors,alpha,image.data);targetCtx.putImageData(image,0,0);
      $('response-accuracy').textContent='Calibration anchor · supplied to the chart. Exact reproduction here is not a held prediction.';
      $('response-reference-label').textContent='Supplied calibration image';
    }else{
      $('response-accuracy').textContent='New dose · no native reference in this dataset. Accuracy is unmeasured.';
      $('response-reference-label').textContent='No native target for this dose';
    }
    $('response-status').textContent='Rendered locally. Zero encoder, denoiser, VAE or neural-student evaluations.';
    $('response-download').disabled=false;
  }
  const schedule=()=>{if(!frame)frame=requestAnimationFrame(draw);};
  async function load(){
    const id=$('response-context').value,own=++revision;anchors=null;$('response-download').disabled=true;
    $('response-status').textContent='Loading this context’s three RGB calibration anchors…';
    try{
      const entry=data.contexts.find(c=>c.id===id);
      if(!cache.has(id)){
        const response=await fetch(base+entry.path,{signal});if(!response.ok)throw new Error('Could not load calibration pixels.');
        const compressed=await response.arrayBuffer();signal.throwIfAborted();
        const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
        if(await digest(compressed)!==entry.sha256)throw new Error('Calibration asset digest differs.');
        const raw=await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
        if(raw.byteLength!==entry.rawBytes||await digest(raw)!==entry.rawSHA256)throw new Error('Calibration pixels differ.');
        signal.throwIfAborted();cache.set(id,new Uint8Array(raw));
      }
      if(own!==revision||signal.aborted)return;anchors=cache.get(id);draw();
    }catch(error){if(own===revision&&!signal.aborted)$('response-status').textContent=error.message;}
  }
  on('response-context','change',load);on('response-dose','input',schedule);
  on('response-doses','click',e=>{const button=e.target.closest('[data-dose]');if(button){$('response-dose').value=button.dataset.dose;schedule();}});
  on('response-download','click',()=>{
    if(!anchors)return;if(frame){cancelAnimationFrame(frame);draw();}
    const revisionAtClick=revision,alpha=$('response-dose').value,context=$('response-context').value;
    canvas.toBlob(blob=>{if(!blob||signal.aborted||revision!==revisionAtClick)return;if(downloadURL)URL.revokeObjectURL(downloadURL);downloadURL=URL.createObjectURL(blob);const link=document.createElement('a');link.href=downloadURL;link.download=`response-${context}-a${alpha}.png`;document.body.append(link);link.click();link.remove();},'image/png');
  });
  // Load only when the visitor reaches the section, including direct hash links.
  const observer=new IntersectionObserver(entries=>{if(entries.some(e=>e.isIntersecting)){observer.disconnect();load();}},{rootMargin:'200px'});
  observer.observe($('zero-network'));signal.addEventListener('abort',()=>observer.disconnect(),{once:true});
  return()=>{observer.disconnect();dispose();};
}
