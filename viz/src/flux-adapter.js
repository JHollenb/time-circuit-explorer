import {$,assetJSON,mountTemplate,openDialog,renderWalkthrough,renderArchitecture,renderInspector,scrollToSection,Playback} from './components.js';
import {escapeHTML as esc} from './kernels.js';
import {address,boundaryRecord,euler,bf16,toyAttention,toyMLP} from './flux-kernels.js';

const fmt=(x,d=6)=>Number.isFinite(x)?Number(x.toPrecision(d)).toString():'unavailable';
const pct=x=>Number.isFinite(x)?`${(100*x).toFixed(4)}%`:'undefined';
const vector=xs=>`[${xs.map(x=>fmt(x,4)).join(', ')}]`;
const sourceURL='https://github.com/JHollenb/saturn-pub/blob/main/packages/mdb/';
const equations={
  project:['Open the pass','(X,Y,m) = Project(Zₛ,E,σₛ)','The carried latent and fixed conditioning create new working registers. Time supplies gates, scales, and shifts.'],
  joint:['Exchange, then respond','R = X + gA ⊙ A(X,Y); X′ = R + gM ⊙ M(ModLN(R))','Both streams attend together, then each stream’s MLP receives its attention-updated residual. The text stream follows the analogous law.'],
  single:['Parallel branches, shared write','H′ = H + g ⊙ Wₒ[Attention(H); SiLU(u(H)) ⊙ v(H)]','H = [text; image]. Q/K RMSNorm and RoPE precede unmasked attention. Attention and the MLP are parallel branches before their shared projection.'],
  readout:['Read only the image rows','Vₛ = Readout(H[512:],m)','The denoiser output has latent geometry. Reading it leaves the image and text registers intact. No RGB image is decoded here.'],
  commit:['Close the pass','Zₛ₊₁ = Q(FP32(Zₛ + Q(FP32(δₛ Vₛ))))','The committed latent becomes the next pass’s input. Image, text, modulation, and denoiser registers clear. The context bindings remain.' ]
};

export async function mount(pack,environment) {
  const {signal}=environment,root=pack.evidence.root;
  if(!document.getElementById('flux-styles')){const link=document.createElement('link');link.id='flux-styles';link.rel='stylesheet';link.href='./flux.css';document.head.append(link);}
  const [manifest,timeline]=await Promise.all([assetJSON(root+'manifest.json',signal),assetJSON(root+'timeline.json',signal)]);
  signal.throwIfAborted();
  await mountTemplate('./templates/flux.html',signal);
  const state={history:'full',cursor:0,side:'after',field:'image',token:0,feature:0,channel:0,node:'project',lesson:0,guided:true,errorMode:'relative_rms',speed:650,eulerIndex:0,eulerPrecision:'bf16',dose:0,query:0,logit:0,value:0,input:0,gate:0,outer:.5,endpointHistory:'full',...environment.state};
  let drawRevision=0,eulerData=null,endpointData=null,endpointLoaded=false,endpointRevision=0;
  const on=(id,type,fn)=>$(id).addEventListener(type,fn,{signal});
  const safe=fn=>(...args)=>Promise.resolve(fn(...args)).catch(error=>{if(!signal.aborted){$('flux-status').textContent=`Evidence could not load: ${error.message}`;console.error(error);}});
  const playback=new Playback({step:()=>{if(state.cursor>=111)return false;state.cursor++;state.side='after';state.node=nodeFor(address(state.cursor));render();return state.cursor<111;},delay:()=>state.speed,onToggle:running=>{if($('flux-play')){$('flux-play').textContent=running?'Ⅱ Pause':'▶ Run';$('flux-play').setAttribute('aria-pressed',running);}},signal});
  const nodeFor=a=>({project:'project',joint:'joint',single:'single',readout:'readout',commit:'commit'})[a.kind];
  const history=()=>timeline.histories[state.history];
  function originSample(){return state.history==='full'?manifest.initial.sample:timeline.histories.full.records[55].fields.latent.native;}
  function currentBoundary(){return boundaryRecord(history(),state.cursor,state.side);}
  function fieldsAtBoundary(){const record=currentBoundary();if(record)return record.fields;const sample=originSample();return{latent:{native:sample,predicted:sample,error:{rms:0,relative_rms:0,max_abs:0,sample_count:131072}}};}
  function stop(){playback.stop();}
  function setCursor(cursor){stop();state.cursor=Math.max(history().origin,Math.min(111,cursor));state.node=nodeFor(address(state.cursor));render();}
  function graph(){renderArchitecture($('architecture-graph'),pack.graph,{selected:state.node,footerLeft:'TEXT ROWS FIRST · IMAGE ROWS SECOND',footerRight:'4 PASSES · ONE SHARED CONTEXT'});const node=pack.graph.nodes.find(n=>n.id===state.node);$('component-inspector').innerHTML=renderInspector(node,{action:'flux-inspect'});}
  function lessons(){renderWalkthrough(pack.lessons,state.lesson,state.guided);}
  function chooseLesson(index){state.lesson=index;state.guided=true;lessons();const l=pack.lessons[index];setCursor(l.cursor<history().origin?history().origin+l.cursor%28:l.cursor);state.node=l.node;graph();scrollToSection(l.target);if(l.target==='euler')safe(loadEuler)();if(l.target==='endpoints')safe(loadEndpoints)();}
  function metadata(){
    $('flux-prompt').textContent=`“${manifest.prompt}”`;
    $('flux-bindings').innerHTML=`<div class="flux-bindings">${['SEED 113','4 PASSES','CUDA BF16','24 HEADS','d = 128','RoPE 4 × 32'].map(x=>`<span class="flux-pill">${x}</span>`).join('')}</div>`;
    $('flux-schedule').innerHTML=`<table class="flux-schedule-table"><thead><tr><th>Pass / source s</th><th>Time τ</th><th>σ → σ′</th><th>BF16 interval δ</th></tr></thead><tbody>${manifest.timesteps.map((t,s)=>`<tr><td>${s+1} / s=${s}</td><td>${fmt(t,9)}</td><td>${fmt(manifest.sigmas[s],8)} → ${fmt(manifest.sigmas[s+1],8)}</td><td>${fmt(euler(0,0,manifest.sigmas[s],manifest.sigmas[s+1]).dt,9)}</td></tr>`).join('')}</tbody></table><p class="flux-hash">Initial noise SHA-256<br>${manifest.noiseHash}<br>Conditioning SHA-256<br>${manifest.conditioningHash}<br>Model identity<br>${manifest.modelIdentity}</p><p class="flux-note">${esc(manifest.scope)}</p>`;
    const readings=[['FLUX-INITIAL-NOISE.md','A forecast from initial noise','The complete four-pass account'],['FLUX-MLP-STATE-LAW.md','The receiving-state MLP','Input-dependent finite responses'],['FLUX-PARALLEL-MERGE.md','Attention and MLP merge','How parallel branches write together'],['FLUX-STATE-CLOSURE.md','What belongs in the state?','Typed boundaries and continuation']];
    $('flux-readings').innerHTML=readings.map(([file,title,sub],i)=>`<a class="reading-card" href="${sourceURL}docs/${file}" target="_blank" rel="noreferrer"><span class="reading-index">0${i+1}</span><div><span class="reading-type">RESEARCH FIELD NOTE</span><h3>${title}</h3><p>${sub}</p></div><span class="reading-arrow">↗</span></a>`).join('');
  }
  function render(){
    if(signal.aborted)return;const a=address(state.cursor),record=currentBoundary(),fields=fieldsAtBoundary(),eq=equations[a.kind];
    $('flux-history').value=state.history;$('flux-pass').value=a.pass;$('flux-speed').value=state.speed;$('flux-scrub').value=a.local;
    for(const option of $('flux-pass').options)option.disabled=Number(option.value)*28<history().origin;
    $('flux-address').textContent=`PASS ${a.pass+1} / 4 · ${a.label.toUpperCase()} · ${state.side.toUpperCase()} · SOURCE ${a.cursor}`;
    for(const side of ['before','after']){$('flux-'+side).classList.toggle('active',state.side===side);$('flux-'+side).setAttribute('aria-pressed',state.side===side);}
    $('flux-prev').disabled=state.cursor===history().origin;$('flux-next').disabled=state.cursor===111;
    $('flux-depth').innerHTML=Array.from({length:28},(_,local)=>{const b=address(a.pass*28+local);return `<button data-depth="${local}" data-kind="${b.kind}" title="${b.label} · source ${b.cursor}" aria-label="${b.label}, source ${b.cursor}" ${local===a.local?'class="active" aria-current="step"':''}>${local===0?'P':local===26?'R':local===27?'Σ':b.block}</button>`;}).join('');
    $('flux-operation').innerHTML=`<div><h3>${eq[0]}</h3><p>${eq[2]}</p></div><div class="flux-equation">${esc(eq[1])}</div>`;
    const descriptions={latent:'Z · 1024×128',image:'X · 1024×3072',text:'Y · 512×3072',context:'m · pass-local',noise:'V · 1024×128'};
    $('flux-fields').innerHTML=Object.entries(descriptions).map(([id,label])=>`<span class="flux-pill ${fields[id]?'':'off'}">${fields[id]?'●':'○'} ${label} · ${fields[id]?'live':'not live'}</span>`).join('')+`<span class="flux-pill">E + IDs + schedule · bound</span>`;
    if(!fields[state.field])state.field='latent';
    for(const option of $('flux-field').options)option.disabled=!fields[option.value];$('flux-field').value=state.field;
    const field=fields[state.field],n=field.native,p=field.predicted;
    if(!n.rows.includes(state.token))state.token=n.rows[0];if(!n.features.includes(state.feature))state.feature=n.features[0];
    $('flux-token').innerHTML=n.rows.map(row=>`<option value="${row}">${row}</option>`).join('');$('flux-token').value=state.token;
    $('flux-feature').innerHTML=n.features.map(f=>`<option value="${f}">${f}</option>`).join('');$('flux-feature').value=state.feature;
    const ri=n.rows.indexOf(state.token),fi=n.features.indexOf(state.feature),nv=n.values[ri][fi],pv=p.values[ri][fi];
    const max=Math.max(1,...n.values.flat().map(Math.abs),...p.values.flat().map(Math.abs));
    const lane=(sample,title,forecast)=>`<article class="flux-lane ${forecast?'forecast':''}"><div class="lane-label"><span class="lane-dot ${forecast?'recovered':'native'}"></span>${title}<span class="lane-badge">RECORDED</span></div><h3>${fmt(sample.values[ri][fi],9)}</h3><p>${state.field.toUpperCase()} [${sample.shape.join(' × ')}]<br>token ${state.token} · feature ${state.feature} · BF16 values</p><div class="flux-coordinate-grid ${forecast?'forecast':''}" aria-label="Retained coordinate samples">${sample.values.flatMap((row,r)=>row.map((v,f)=>`<span class="${r===ri&&f===fi?'selected':''}" title="token ${sample.rows[r]}, feature ${sample.features[f]}: ${v}"><i style="--height:${Math.abs(v)/max*100}%"></i></span>`)).join('')}</div><p>${sample.values.length*sample.features.length} retained coordinates · display sample</p></article>`;
    $('flux-lanes').innerHTML=lane(n,'NATIVE EXECUTION',false)+lane(p,`${state.history==='full'?'FULL':'MATCHED'} MATHEMATICAL FORECAST`,true);
    const metric=field.error;
    $('flux-source').innerHTML=`Selected coordinate Δ (forecast − native): <strong>${fmt(pv-nv,9)}</strong> · full-field RMS: ${fmt(metric.rms,7)} · relative RMS: ${pct(metric.relative_rms)} · max |Δ|: ${fmt(metric.max_abs,7)}<br>Reduction over ${metric.sample_count?.toLocaleString()} scalars; the coordinate grid above is a separate sample.<br>${record?`Boundary after source ${record.cursor}, ${record.kind}${record.block!==null?' '+record.block:''}. Field last written at source ${n.sourceCursor}.`:`Observed origin Z${history().origin/28}; no forecast transitions have run.`}<br>Native field SHA-256: ${n.sha256}<br>Forecast field SHA-256: ${p.sha256||'origin binding (same supplied state)'}`;
    $('flux-status').textContent=`${state.history==='full'?'Full112: from actual initial Z₀.':'Matched56: starts at observed native Z₂ after 56 native transitions; earlier clocks are unavailable.'} ${record?.local===27?'Commit cleared the working registers. ':''}Playback follows source order, not worker wall time. ${state.side==='before'?'Inputs to the selected operation are shown.':'Outputs of the selected operation are shown.'}`;
    $('flux-error-mode').value=state.errorMode;$('flux-tile-channel').value=state.channel;
    graph();drawError();safe(drawTiles)();if(eulerData)drawEuler();
  }
  function drawError(){
    const key=state.errorMode,records=history().records,points=records.filter(r=>r.fields[state.field]&&Number.isFinite(r.fields[state.field].error[key]));
    const max=Math.max(1e-9,...points.map(r=>r.fields[state.field].error[key]??0));
    const x=c=>50+(c/111)*920,y=v=>135-v/max*105;
    const path=points.map((r,i)=>`${i&&r.cursor===points[i-1].cursor+1?'L':'M'}${x(r.cursor)},${y(r.fields[state.field].error[key]??0)}`).join(' ');
    const shown=state.cursor-(state.side==='before'?1:0);
    $('flux-error-scope').textContent=`${state.field.toUpperCase()} · ${key==='rms'?'absolute RMS in register units':'RMS(Δ) / RMS(native at the same clock)'} · full-field reductions; gaps mean field not live.`;
    $('flux-error-plot').innerHTML=`<path d="M50 22V135H970" fill="none" stroke="#e3e8ef"/>${[0,28,56,84,111].map(c=>`<path d="M${x(c)} 25V135" stroke="#edf0f5"/><text x="${x(c)}" y="155" fill="#8b99af" font-size="10" text-anchor="middle">${c}</text>`).join('')}<text x="44" y="32" text-anchor="end" font-size="10" fill="#8b99af">${key==='rms'?fmt(max,3):pct(max)}</text><text x="44" y="138" text-anchor="end" font-size="10" fill="#8b99af">0</text><path d="${path}" fill="none" stroke="#b59a66" stroke-width="2"/>${points.map(r=>`<circle cx="${x(r.cursor)}" cy="${y(r.fields[state.field].error[key]??0)}" r="2" fill="#b59a66"><title>source ${r.cursor}: ${r.fields[state.field].error[key]}</title></circle>`).join('')}<path d="M${x(Math.max(0,shown))} 20V136" stroke="#7b739c" stroke-dasharray="4 3"/><text x="970" y="168" text-anchor="end" font-size="9" fill="#9ba5b4">source transition index</text>`;
  }
  async function drawTiles(){
    const revision=++drawRevision,record=currentBoundary();
    const pass=record?record.pass:history().origin/28,which=record?.local===27?'out':'in',channel=state.channel;
    const tile=await assetJSON(`${root}tiles/${state.history}-${pass}.json`,signal).catch(e=>{if(!signal.aborted)throw e;});
    if(signal.aborted||revision!==drawRevision||!tile)return;
    const points=manifest.idGrid,max=manifest.latentScale;
    for(const lane of ['native','predicted']){
      const canvas=$(`flux-${lane}-tile`),ctx=canvas.getContext('2d'),values=tile.tiles[`${lane}-${which}`][channel];ctx.clearRect(0,0,256,256);
      values.forEach((v,i)=>{const t=Math.min(1,Math.abs(v)/max),base=v<0?[116,147,186]:[186,152,97];ctx.fillStyle=`rgb(${base.map(c=>Math.round(246+(c-246)*t)).join(',')})`;ctx.fillRect(points[i][1]*8,points[i][0]*8,8,8);});
      canvas.title=`${lane} Z${pass+(which==='out'?1:0)} channel ${channel}; fixed range ±${max}`;
      $(`flux-tile-${lane}-label`).textContent=`Z${pass+(which==='out'?1:0)} · channel ${channel}`;
    }
    $('flux-tile-info').innerHTML=`Z persists throughout the pass; transformer blocks update working registers.<br>32 × 32 positions from recorded IDs.<br>Fixed shared range: −${max} to +${max}.<br>Blue = negative, amber = positive.<br>Four complete retained channels available; no RGB decoding.`;
  }
  async function loadEuler(){
    if(!eulerData)eulerData=await assetJSON(root+'euler.json',signal);if(signal.aborted)return;
    drawEuler();$('euler-hostile').innerHTML=eulerData.hostile.map(h=>`<p class="flux-hash">δ BF16 ${h.dt_bf16}, FP32 ${fmt(h.dt_fp32,10)}; v=${h.edge.noise}<br>Native update ${h.edge.native}; BF16 interval update ${h.edge.bf16_dt}; FP32 interval update ${h.edge.fp32_dt}<br>${h.native_vs_fp32_dt_mismatches} / ${h.values} values differ with FP32 interval promotion.</p>`).join('');
  }
  function drawEuler(){
    const pass=address(state.cursor).pass,fixtures=eulerData.histories[state.history],fixture=fixtures.find(x=>x.pass===pass)||fixtures[0];
    state.eulerIndex=Math.min(state.eulerIndex,fixture.coordinates.length-1);
    $('euler-coordinate').innerHTML=fixture.coordinates.map((c,i)=>`<option value="${i}">${c.lane} · token ${c.row} / f${c.feature}</option>`).join('');$('euler-coordinate').value=state.eulerIndex;
    const c=fixture.coordinates[state.eulerIndex],v=state.eulerPrecision==='bf16'?bf16(c.v+state.dose):c.v+state.dose,result=euler(c.z,v,fixture.sigma,fixture.sigmaNext,state.eulerPrecision==='bf16');
    $('euler-precision').value=state.eulerPrecision;$('euler-dose').value=state.dose;$('euler-dose-label').textContent=state.dose===0?'Recorded value':`Recorded V + ${fmt(state.dose,3)}`;
    $('euler-values').innerHTML=[['Incoming Z',c.z],['Interval δ',result.dt],['Denoiser V',v],['Rounded update D',result.update],['Next latent Z′',result.next]].map(([label,value])=>`<div><span>${label}</span><strong>${fmt(value,10)}</strong></div>`).join('');
    $('euler-result').textContent=`Pass ${fixture.pass+1}, source s=${fixture.pass}; σ ${fmt(fixture.sigma,9)} → ${fmt(fixture.sigmaNext,9)}. ${state.dose!==0?'Custom operand: no recorded native comparison for this edited V.':state.eulerPrecision==='ideal'?`Ideal-arithmetic illustration; retained ${c.lane} result is ${c.expected}.`:`Live BF16 result ${result.next===c.expected?'exactly matches':'DIFFERS FROM'} retained ${c.lane} result ${c.expected}.`} The native and forecast V inputs can differ.`;
  }
  function drawToys(){
    const a=toyAttention(state.query,state.logit,state.value),m=toyMLP(state.input,state.gate);
    for(const [id,value]of [['toy-query',state.query],['toy-logit',state.logit],['toy-value',state.value],['toy-input',state.input],['toy-gate',state.gate],['toy-outer',state.outer]])$(id).value=value;
    for(const [id,value]of [['toy-logit-label',state.logit],['toy-value-label',state.value],['toy-input-label',m.input[0]],['toy-gate-label',state.gate],['toy-outer-label',state.outer]])$(id).textContent=fmt(value,3);
    $('toy-attention').innerHTML=`<div class="toy-weights">${a.weights.map((w,i)=>`<div><i style="height:${w*95}px"></i><strong>${w.toFixed(3)}</strong><small>${i<2?'T'+i:'I'+(i-2)}</small></div>`).join('')}</div><p class="flux-note">Attention weights sum to ${a.weights.reduce((x,y)=>x+y,0).toFixed(6)}. All six tokens are visible; there is no causal mask.</p><div class="flux-vector">Output ${vector(a.output)}</div>`;
    $('toy-mlp').innerHTML=`<div class="flux-vector">SiLU gates ${vector(m.gate)}<br>Product ${vector(m.product)}<br>Output ${vector(m.output)}</div><p class="flux-note">Expanded product and factored effective matrix agree to ${fmt(Math.max(...m.output.map((v,i)=>Math.abs(v-m.factored[i]))),3)}. The effective matrix is recomputed for this x, not a globally fixed linear map. This internal gate differs from the outer residual gate below.</p>`;
    const parallel=[...a.output,...m.output],write=Array.from({length:4},(_,i)=>parallel.reduce((sum,v,j)=>sum+v*Math.cos((i+1)*(j+2))*.25,0)),next=m.input.map((x,i)=>x+state.outer*write[i]);
    $('toy-merge').innerHTML=`<div class="flux-vector">Wₒ[A; M] = ${vector(write)}<br>H + g · write = ${vector(next)}</div>`;
  }
  async function loadEndpoints(){
    const revision=++endpointRevision;$('endpoint-load').textContent='Loading…';
    endpointData=endpointData||await assetJSON(root+'endpoints.json',signal);signal.throwIfAborted();
    const mode=state.endpointHistory,branch=mode+'.predicted',metric=endpointData.comparisons[branch];
    const native=await loadImage(root+'images/native.png'),predicted=await loadImage(root+`images/${branch}.png`);
    if(signal.aborted||revision!==endpointRevision)return;endpointLoaded=true;
    $('endpoint-load').textContent='Reload selected comparison';$('endpoint-history').value=mode;
    $('endpoint-content').innerHTML=`<div class="flux-endpoint-grid"><figure><img src="${native.src}" width="512" height="512" alt="Original native VAE image: green cone left of yellow cylinder"><figcaption>Native terminal output<span>Observed Z₄ → native VAE</span></figcaption></figure><figure><img src="${predicted.src}" width="512" height="512" alt="Original ${mode} forecast VAE image"><figcaption>${mode==='full'?'Full112':'Matched56'} forecast output<span>Predicted Z₄ → same native VAE</span></figcaption></figure><figure><canvas id="endpoint-error" width="512" height="512" role="img" aria-label="Absolute RGB channel error heatmap"></canvas><figcaption>Largest channel difference per pixel<span>Fixed range 0–42 · black to amber</span></figcaption></figure></div><div class="flux-metrics"><div><strong>${pct(metric.latent.relative_rms)}</strong><small>latent relative RMS · 131,072 scalars</small></div><div><strong>${metric.rgb.mae.toFixed(6)}</strong><small>RGB MAE on 0–255 channel scale</small></div><div><strong>${metric.rgb.max_abs}</strong><small>maximum absolute channel difference</small></div></div><p id="endpoint-pixel" class="flux-note" aria-live="polite"></p><p class="flux-note">Full RGB channel-error distribution, including zero. Bars use log(1 + count); hover for exact counts. All 786,432 channel values are included.</p><div class="flux-hist">${endpointData.distributions[branch].absolute_channel_error_counts.map((count,i)=>`<i style="height:${Math.log1p(count)/Math.log1p(786432)*100}%" title="|Δ| = ${i}: ${count.toLocaleString()} channels" aria-label="Error ${i}: ${count} channels"></i>`).join('')}</div><p class="flux-note">The rare tail remains visible. A small average error does not imply exact hidden states or pixels. Native/sham replay has zero latent and RGB discrepancy. Independently reconstructing the sealed program is a separate comparison: 0 mismatches across ${endpointData.independent.scalarValues.toLocaleString()} scalar values in ${endpointData.independent.fields} fields. Both forecasts share one seed and one native history; this does not establish cross-seed behavior or inference speedup.</p><p class="flux-hash"><a class="flux-download" href="${root}images/${branch}.png" download>Download original forecast PNG</a> · <a class="flux-download" href="${root}endpoints.json" download>Complete endpoint measurements</a></p>`;
    const canvas=$('endpoint-error'),ctx=canvas.getContext('2d'),scratch=document.createElement('canvas');scratch.width=scratch.height=512;const sc=scratch.getContext('2d',{willReadFrequently:true});sc.drawImage(native,0,0);const n=sc.getImageData(0,0,512,512);sc.drawImage(predicted,0,0);const p=sc.getImageData(0,0,512,512),out=ctx.createImageData(512,512);let worst={d:-1,x:0,y:0};
    for(let pixel=0;pixel<512*512;pixel++){const offset=pixel*4,delta=Math.max(...[0,1,2].map(c=>Math.abs(p.data[offset+c]-n.data[offset+c])));if(delta>worst.d)worst={d:delta,x:pixel%512,y:Math.floor(pixel/512)};const t=delta/42;out.data.set([Math.round(t*245),Math.round(t*166),Math.round(t*64),255],offset);}ctx.putImageData(out,0,0);
    const pixelText=(x,y)=>{const i=(y*512+x)*4,ns=Array.from(n.data.slice(i,i+3)),ps=Array.from(p.data.slice(i,i+3));$('endpoint-pixel').textContent=`Pixel (${x}, ${y}): native RGB ${vector(ns)}; forecast RGB ${vector(ps)}; Δ ${vector(ps.map((v,j)=>v-ns[j]))}. Worst absolute channel difference: ${worst.d} at (${worst.x}, ${worst.y}). Tap or move over the map to inspect.`;};pixelText(worst.x,worst.y);
    canvas.addEventListener('pointermove',event=>{const r=canvas.getBoundingClientRect();pixelText(Math.min(511,Math.max(0,Math.floor((event.clientX-r.left)/r.width*512))),Math.min(511,Math.max(0,Math.floor((event.clientY-r.top)/r.height*512))));},{signal});
  }
  function loadImage(src){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('Could not load original image'));image.src=src;signal.addEventListener('abort',()=>reject(signal.reason),{once:true});});}
  async function provenance(){const data=await assetJSON(root+'provenance.json',signal);if(signal.aborted)return;openDialog('SOURCE-BOUND EVIDENCE',`<h2>One recorded history. Explicit numerical scope.</h2><p>${esc(manifest.scope)}</p><p>${esc(data.reader)}</p><p>Exported ${Object.keys(data.sourceFiles).length} source identities and ${Object.keys(data.forecastLeaves).length} framed forecast leaves. New model forwards: ${data.newModelForwards}.</p><p>Audit SHA-256</p><code>${data.auditSHA256}</code><p>Native conditioning, the VAE, and the native CUDA trace are observed. The forecast reconstructs the denoiser trajectory. Full-size attention and MLP activations here are recorded; Euler math and synthetic sandboxes run locally.</p><a class="download-link" href="${root}provenance.json" download>Download provenance manifest ↗</a><p><a href="${sourceURL}qualification/flux_transition_reference.py" target="_blank" rel="noreferrer">Read the executable transition reference ↗</a></p>`);}

  on('lesson-nav','click',e=>{const b=e.target.closest('[data-lesson]');if(b)chooseLesson(+b.dataset.lesson);});
  on('next-lesson','click',()=>{if(state.lesson===pack.lessons.length-1){state.guided=false;lessons();}else chooseLesson(state.lesson+1);});
  on('guided-mode','click',()=>{state.guided=true;lessons();});on('explore-mode','click',()=>{state.guided=false;lessons();});
  on('architecture-graph','click',e=>{const n=e.target.closest('[data-node]');if(n){state.node=n.dataset.node;graph();}});
  on('architecture-graph','keydown',e=>{if(e.key==='Enter'||e.key===' '){const n=e.target.closest('[data-node]');if(n){e.preventDefault();state.node=n.dataset.node;graph();}}});
  on('component-inspector','click',e=>{if(e.target.closest('#flux-inspect')){const n=pack.graph.nodes.find(n=>n.id===state.node);openDialog(n.label,`<h2>${esc(n.label)}</h2><p>${esc(n.text)}</p><pre>${esc(n.equation)}</pre><p>${esc(n.dims)}</p><p>Real-arithmetic overview. The recorded CUDA BF16 profile includes finite rounding, RMSNorm, RoPE, and time modulation. Full block operands remain recorded; the sandbox uses explicitly synthetic weights.</p>`);}});
  on('flux-play','click',()=>{if(state.cursor===111)setCursor(history().origin);playback.toggle();});on('flux-reset','click',()=>setCursor(history().origin));on('flux-prev','click',()=>setCursor(state.cursor-1));on('flux-next','click',()=>setCursor(state.cursor+1));
  on('flux-depth','click',e=>{const b=e.target.closest('[data-depth]');if(b)setCursor(address(state.cursor).pass*28+Number(b.dataset.depth));});
  on('flux-scrub','input',e=>setCursor(address(state.cursor).pass*28+Number(e.target.value)));on('flux-pass','change',e=>setCursor(Number(e.target.value)*28+address(state.cursor).local));
  on('flux-history','change',e=>{state.history=e.target.value;setCursor(Math.max(state.cursor,history().origin));});on('flux-speed','change',e=>{state.speed=Number(e.target.value);});
  for(const side of ['before','after'])on('flux-'+side,'click',()=>{stop();state.side=side;render();});
  for(const [id,key]of [['flux-field','field'],['flux-token','token'],['flux-feature','feature'],['flux-tile-channel','channel'],['flux-error-mode','errorMode']])on(id,'change',e=>{state[key]=['token','feature','channel'].includes(key)?Number(e.target.value):e.target.value;render();});
  on('euler-coordinate','change',safe(async e=>{state.eulerIndex=Number(e.target.value);await loadEuler();}));on('euler-precision','change',safe(async e=>{state.eulerPrecision=e.target.value;await loadEuler();}));on('euler-dose','input',safe(async e=>{state.dose=Number(e.target.value);await loadEuler();}));on('euler-reset','click',safe(async()=>{state.dose=0;await loadEuler();}));
  for(const [id,key]of [['toy-query','query'],['toy-logit','logit'],['toy-value','value'],['toy-input','input'],['toy-gate','gate'],['toy-outer','outer']])on(id,id==='toy-query'?'change':'input',e=>{state[key]=Number(e.target.value);drawToys();});
  on('endpoint-load','click',safe(loadEndpoints));on('endpoint-history','change',safe(async e=>{state.endpointHistory=e.target.value;if(endpointLoaded)await loadEndpoints();}));on('flux-provenance','click',safe(provenance));
  $('sources-button').onclick=safe(provenance);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();},{signal});
  const observer=new IntersectionObserver(entries=>{if(entries.some(e=>e.isIntersecting)){observer.disconnect();safe(loadEuler)();}},{rootMargin:'200px'});observer.observe($('euler'));signal.addEventListener('abort',()=>observer.disconnect(),{once:true});
  metadata();lessons();drawToys();render();$('endpoint-history').value=state.endpointHistory;
  return{getState:()=>({...state}),dispose(){stop();drawRevision++;endpointRevision++;observer.disconnect();$('sources-button').onclick=null;}};
}
