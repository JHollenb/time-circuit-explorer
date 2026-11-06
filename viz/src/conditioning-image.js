import {$,assetJSON} from './components.js';

export function mountConditioningImage(signal){
  if(!document.getElementById('conditioning-image-styles')){const link=document.createElement('link');link.id='conditioning-image-styles';link.rel='stylesheet';link.href='./conditioning-image.css';document.head.append(link);}
  const root='./data/conditioning-image/';
  let data=null,images={},failed=false;
  const canvas=$('consequence-canvas'),ctx=canvas.getContext('2d');
  const loadImage=name=>new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('Original image could not load.'));image.src=root+name+'.png';});
  const pixels=image=>{const c=document.createElement('canvas');c.width=512;c.height=512;const x=c.getContext('2d');x.drawImage(image,0,0);return x.getImageData(0,0,512,512).data;};
  let errorImages={};
  function render(){
    const divider=Number($('consequence-divider').value),branch=$('consequence-policy').value,isError=$('consequence-error').checked;
    $('consequence-divider-value').textContent=divider+'%';
    if(!data)return;
    if(isError){
      if(!errorImages[branch]){const a=pixels(images[branch]),b=pixels(images.full),out=ctx.createImageData(512,512);
        for(let i=0;i<a.length;i+=4){const v=Math.max(Math.abs(a[i]-b[i]),Math.abs(a[i+1]-b[i+1]),Math.abs(a[i+2]-b[i+2]));out.data[i]=v;out.data[i+1]=v;out.data[i+2]=v;out.data[i+3]=255;}errorImages[branch]=out;}
      ctx.putImageData(errorImages[branch],0,0);
    }else{
      ctx.drawImage(images[branch],0,0);ctx.save();ctx.beginPath();ctx.rect(0,0,512*divider/100,512);ctx.clip();ctx.drawImage(images.full,0,0);ctx.restore();
      if(divider>0&&divider<100){ctx.strokeStyle='white';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(512*divider/100,0);ctx.lineTo(512*divider/100,512);ctx.stroke();}
    }
    $('consequence-divider').disabled=isError;
    const label=branch==='old'?'Original compact':'New compact';canvas.setAttribute('aria-label',isError?`${label} absolute RGB error, fixed black-to-white scale 0–255`:`Full reference on left, ${label} on right`);
    $('consequence-caption').textContent=isError?'Largest absolute channel difference per pixel. Fixed black = 0, white = 255; neither branch is rescaled.':`Full reference ← divider → ${label.toLowerCase()}. These are original recorded images.`;
    $('consequence-download').href=root+branch+'.png';
    const png=data.report.png[branch],latent=data.report.accounts.find(x=>x.port==='latent4')[branch];
    $('consequence-metrics').innerHTML=`<div><strong>${png.rms255.toFixed(3)}</strong><span>PNG channel RMS · 0–255</span></div><div><strong>${png.mae255.toFixed(3)}</strong><span>PNG channel MAE · 0–255</span></div><div><strong>${(latent.relativeRms*100).toFixed(2)}%</strong><span>final latent error / full RMS</span></div><div><strong>${png.maxAbs255}</strong><span>largest PNG channel difference</span></div>`;
    const p=data.report.png;
    $('consequence-pixels').textContent=`New versus original compact: ${p.pixelSquaredErrorImproved.toLocaleString()} pixels improve, ${p.pixelSquaredErrorWorsened.toLocaleString()} worsen, ${p.pixelSquaredErrorUnchanged.toLocaleString()} are unchanged, by squared error to full. Lower RMS coexists with higher MAE.`;
    $('consequence-compute').textContent=`Three-image assay: ${data.scheduler.elapsed_s.toFixed(2)} s scheduler execution, ${data.report.workerSecondsAfterFrameworkImports.toFixed(2)} s worker body after imports. Peak sampled RSS ${(data.scheduler.peak_rss_mb/1000).toFixed(2)} GB; VRAM ${(data.scheduler.peak_vram_mb/1000).toFixed(2)} GB. Setup and instrument captures are included; branch warm-up differs. This is not a speedup comparison.`;
    $('consequence-status').textContent=`Recorded ${data.jobID} · seed 314159 · 512 × 512 · one noise draw · 12 denoiser calls · 3 VAE decodes.`;
  }
  async function load(){
    if(data||failed||signal.aborted)return;
    try{const [summary,full,old,newImage]=await Promise.all([assetJSON(root+'summary.json',signal),loadImage('full'),loadImage('old'),loadImage('new')]);
      if(signal.aborted)return;data=summary;images={full,old,new:newImage};render();
    }catch(error){if(!signal.aborted){failed=true;$('consequence-status').textContent=error.message;}}
  }
  for(const id of ['consequence-policy','consequence-error'])$(id).addEventListener('change',render,{signal});
  $('consequence-divider').addEventListener('input',render,{signal});
  const observer=new IntersectionObserver(entries=>{if(entries.some(e=>e.isIntersecting)){observer.disconnect();load();}},{rootMargin:'250px'});observer.observe($('conditioning-image'));
  signal.addEventListener('abort',()=>observer.disconnect(),{once:true});
  return()=>observer.disconnect();
}
