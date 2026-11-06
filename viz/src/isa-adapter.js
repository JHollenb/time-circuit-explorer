import {validatePack} from './models.js';
import {wordLaw, trajectory, scanRow, scanTrajectory, writeContributions, conditionalMixer, postMLP, linear, maxError, escapeHTML as esc} from './kernels.js';
import {assetJSON,mountTemplate,openDialog,scrollToSection,renderWalkthrough,renderArchitecture as drawArchitecture,renderInspector,Playback} from './components.js';

const $ = id => document.getElementById(id);
let model, lifecycle;
const state = {lesson:0,guided:true,node:'scan2',view:'scan',row:8,channel:0,mode:0,dose:1,mlpDelta:0,phase:3,scenario:'carry',instruction:0,hex:true,playing:false,math:false,h:false,conv:false};
const initialState={...state};
const phases = [ ['Context','inputs'],['Route','operands'],['Form update','writer'],['Carry','scan state'],['Read','consumer'],['Commit','next input'] ];
let carry, loops, frontier, scan, readings, playback, loadingScan, provenance;
const history = [];
const precision = x => Math.abs(x)<1e-4 && x!==0 ? x.toExponential(3) : Number(x).toFixed(5);
const short = x => Math.abs(x)>10000||Math.abs(x)<.001&&x!==0 ? Number(x).toExponential(2) : Number(x).toFixed(4);
const word = x => state.hex?'0x'+BigInt(x).toString(16).padStart(16,'0'):BigInt(x).toString();
const vecNorm=x=>Math.sqrt(x.reduce((a,b)=>a+b*b,0));
function json(url) {return assetJSON(url,lifecycle.signal);}
function toast(message) {$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,3400);}
function stop() {state.playing=false;playback?.stop();if($('play')){$('play').innerHTML='<span>▶</span> Run';$('play').setAttribute('aria-label','Play execution');}}
function setView(view) {
  state.view=view;
  $('scan-tab').classList.toggle('active',view==='scan');$('mlp-tab').classList.toggle('active',view==='mlp');
  $('scan-tab').setAttribute('aria-pressed',view==='scan');$('mlp-tab').setAttribute('aria-pressed',view==='mlp');
  const dose=$('dose');dose.min=view==='scan'?'.25':'-2';dose.max='2';dose.step='.05';dose.value=view==='scan'?state.dose:state.mlpDelta;
  dose.closest('.control-group').querySelector('label').firstChild.textContent=view==='scan'?'CLOCK MULTIPLIER ':'FEATURE 0 SHIFT ';
  const labels=dose.closest('.control-group').querySelectorAll('.range-labels>span');labels[0].textContent=view==='scan'?'0.25×':'−2';labels[1].textContent=view==='scan'?'2×':'+2';
  $('channel').disabled=view==='mlp';$('state-mode').disabled=view==='mlp';
  renderComparison();
}
function selectNode(id) {
  state.node=id;
  const node=model.graph.nodes.find(n=>n.id===id);
  if(node?.phase!==undefined)state.phase=node.phase;
  if(id==='scan2')setView('scan');if(id==='post2')setView('mlp');
  renderArchitecture();renderMachine();
}
function selectLesson(index,scroll=true) {
  stop();history.length=0;state.lesson=index;
  const l=model.lessons[index];state.node=l.node;state.row=l.row;state.phase=l.phase;
  state.dose=1;state.mlpDelta=0;
  if(index===3){state.scenario='symbolic-00';state.instruction=0;state.phase=0;state.row=0;}
  else if(index!==4){state.scenario='carry';state.instruction=0;}
  setView(index===2?'mlp':'scan');
  renderLesson();renderArchitecture();renderMachine();renderComparison();
  if(scroll)scrollToSection(l.target);
}
function renderLesson() {
  renderWalkthrough(model.lessons,state.lesson,state.guided);
}
function renderArchitecture() {
  drawArchitecture($('architecture-graph'),model.graph,{selected:state.node,
    edgePath(a,b,s,t){if(a==='decoder'&&['heads','flags'].includes(b)){const y=b==='heads'?24:292;return `M${s.x+s.w/2},${b==='heads'?s.y:s.y+s.h} V${y} H${t.x+t.w/2} V${b==='heads'?t.y:t.y+t.h}`;}},
    footerLeft:'INPUTS → LIVE FACTORS → STATE → READOUT',footerRight:`BIT ROW ${String(state.row).padStart(2,'0')} / 63`});
  const n=model.graph.nodes.find(n=>n.id===state.node);
  $('component-inspector').innerHTML=renderInspector({...n,typeLabel:n.kind==='mlp'?'MULTILAYER PERCEPTRON':n.kind==='scan'?'SELECTIVE STATE SPACE':n.kind.toUpperCase()},{explore:['scan2','post2'].includes(n.id)});
}
function spark(values,selected,color='#7392bf') {
  const width=460,height=65,pad=6,min=Math.min(...values),max=Math.max(...values),range=max-min||1;
  const x=i=>pad+i/(values.length-1)*(width-pad*2),y=v=>height-pad-(v-min)/range*(height-pad*2);
  return `<svg class="native-spark" viewBox="0 0 ${width} ${height}" aria-label="Recorded state trajectory across 64 bit rows"><path d="M6 58H454" stroke="#edf0f5" fill="none"/><path d="${values.map((v,i)=>(i?'L':'M')+x(i).toFixed(2)+' '+y(v).toFixed(2)).join(' ')}" stroke="${color}" stroke-width="1.4" fill="none"/><path d="M${x(selected)} 3V62" stroke="${color}" opacity=".25" stroke-dasharray="2 3"/><circle cx="${x(selected)}" cy="${y(values[selected])}" r="3" fill="${color}" stroke="white" stroke-width="1.5"/></svg>`;
}
function valueNode(label,value,sub) {return `<div class="value-node"><span class="value-label">${label}</span><span class="value-box">${short(value)}</span><span class="value-sub">${sub}</span></div>`;}
function contributionPlot(rows,end) {
  const terms=writeContributions(rows,end),maximum=Math.max(...terms.map(t=>Math.abs(t.value)),1e-10),width=460;
  return `<div class="history-unroll"><h4>Every earlier write, carried into this row</h4><svg viewBox="0 0 460 92" role="img" aria-label="Each bar is one writer contribution multiplied by its subsequent retention factors"><path d="M4 43H456" stroke="#d9deea"/>${terms.map((t,i)=>{const w=448/terms.length,h=Math.abs(t.value)/maximum*30;return `<rect x="${6+i*w}" y="${t.value>=0?43-h:43}" width="${Math.max(1,w-2)}" height="${Math.max(.4,h)}" rx="1" fill="${t.value>=0?'#b89b63':'#9a8dbb'}"><title>Row ${t.row}: write ${t.writer.toPrecision(5)} × retention ${t.transport.toPrecision(5)} = ${t.value.toPrecision(5)}</title></rect>`;}).join('')}<text x="5" y="86" font-size="8" fill="#a0a9b9" font-family="monospace">writer at row 0</text><text x="344" y="86" font-size="8" fill="#a0a9b9" font-family="monospace">row ${end} → Hₜ</text></svg><p>Each bar is Wₖ multiplied by every later D before the selected row. Their signed sum is ${short(terms.reduce((s,t)=>s+t.value,0))}. Earlier writes can persist even though only the current H needs to be carried.</p></div>`;
}
function heatSVG(groups,colors=['#7192bd','#a08cc7','#78a798']) {
  let html=''; const width=430,cols=[18,168,318],cell=5.7,gap=2;
  groups.forEach((g,index)=>{
    const max=Math.max(...g.values.map(Math.abs),1e-8);
    g.values.forEach((v,i)=>{const cx=cols[index]+(i%8)*(cell+gap),cy=32+Math.floor(i/8)*(cell+gap);html+=`<rect x="${cx}" y="${cy}" width="${cell}" height="${cell}" rx="1" fill="${colors[index]}" opacity="${.16+.84*Math.min(1,Math.abs(v)/max)}"><title>${esc(g.title)}[${i}] = ${v}</title></rect>`;});
    html+=`<text x="${cols[index]}" y="17" class="mlp-stage-title">${esc(g.title)}</text><text x="${cols[index]}" y="142" class="mlp-stage-label">${g.values.length} values</text>`;
    if(index<groups.length-1)html+=`<path d="M${cols[index]+67} 64C${cols[index]+94} 64 ${cols[index]+118} 64 ${cols[index+1]-13} 64" fill="none" stroke="#dfe5ef"/><path d="M${cols[index+1]-17} 61l4 3-4 3" fill="none" stroke="#dfe5ef"/>`;
  });
  return `<svg class="mlp-visual" viewBox="0 0 ${width} 150" role="img" aria-label="Recorded activation tiles for 48 normalized inputs, 96 GELU responses, and 48 update outputs">${html}</svg>`;
}
async function ensureScan() {
  if(scan)return scan;
  if(!loadingScan)loadingScan=json(model.evidence.scan).then(d=>{scan=d;return d;}).finally(()=>{loadingScan=null;});
  return loadingScan;
}
function renderComparison() {
  $('bit-row').value=state.row;$('row-value').textContent=`Bit ${String(state.row).padStart(2,'0')} / 63`;
  $('trace-location').textContent=`CARRY-DEPTH-0 / ${state.view==='scan'?'MIXER':'MLP'} 3 / ROW ${String(state.row).padStart(2,'0')}`;
  $('dose-value').textContent=state.view==='scan'?`${state.dose.toFixed(2)}×`:`${state.mlpDelta>=0?'+':''}${state.mlpDelta.toFixed(2)}`;
  $('math-details').hidden=!state.math;$('math-button').innerHTML=`${state.math?'Hide':'Show'} equations <span>${state.math?'−':'+'}</span>`;
  if(!scan){const signal=lifecycle.signal;ensureScan().then(()=>{if(!signal.aborted)renderComparison();}).catch(error=>{if(!signal.aborted)$('comparison-status').textContent=error.message;});return;}
  const row=state.row;
  const localMixer=state.view==='scan'?conditionalMixer(scan,row,state.dose).output:[...scan.post.input[row]];
  if(state.view==='mlp')localMixer[0]+=state.mlpDelta;
  const localPost=postMLP(localMixer,scan.weights).output;
  const priorBit=carry?Number((BigInt(carry.before[7])>>BigInt(row))&1n):0;
  const localLogit=linear(scan.weights['reg_out.weight'],scan.weights['reg_out.bias'],localPost)[7]+8*priorBit-4;
  const nativeLogit=scan.nativeRegs[7][row];
  $('consumer-bridge').innerHTML=`<div class="bridge-label">FOLLOW THE CONSUMER <small>same bit row · feature 0 shown</small></div><div class="bridge-flow">${[
    ['Mixer output',scan.mixer[row][0],localMixer[0]],['Post MLP + residual',scan.post['residual.output'][row][0],localPost[0]],['R7 output logit',nativeLogit,localLogit]
  ].map(([label,n,r],i)=>`${i?'<span class="bridge-arrow">→</span>':''}<div class="bridge-item"><span>${label}</span><strong class="native-ink">${short(n)}</strong><strong class="recovered-ink">${short(r)}</strong></div>`).join('')}<span class="bridge-arrow">→</span><div class="bridge-bit"><span>decoded bit</span><b>${Number(nativeLogit>0)} <i>/</i> ${Number(localLogit>0)}</b></div></div>`;
  if(state.view==='scan') {
    const base=scanRow(scan,row,state.channel,state.mode,1),computed=scanTrajectory(scan,state.channel,state.mode,row,state.dose),changed={...scanRow(scan,row,state.channel,state.mode,state.dose),...computed[row]};
    $('native-title').textContent='A selective state update';
    $('recovered-title').textContent='Retain + write = next state';
    $('native-description').textContent=`Mixer 3, channel ${state.channel}, mode ${state.mode}. A measured coordinate inside a 96 × 4 state.`;
    $('recovered-description').textContent='The same source-qualified recurrence, evaluated here from the captured live factors.';
    $('native-visual').innerHTML=`<div class="value-nodes">${valueNode('Hₜ₋₁',base.previous,'previous state')}<span class="operator">→</span>${valueNode('Dₜ · H + Wₜ',base.native,'recorded next state')}</div>${spark(scan.h.map(r=>r[state.channel][state.mode]),row)}<div class="plot-label">RECORDED H ACROSS BIT POSITIONS &nbsp; 0 → 63</div>`;
    $('recovered-visual').innerHTML=`<div class="sum-view"><div class="sum-term"><small>Dₜ × Hₜ₋₁</small><strong>${short(changed.retained)}</strong><em>retained</em></div><span class="operator">+</span><div class="sum-term"><small>dtₜ × uₜ × Bₜ</small><strong>${short(changed.writer)}</strong><em>written</em></div><span class="operator">=</span><div class="sum-term result"><small>Hₜ</small><strong>${short(changed.h)}</strong><em>next state</em></div></div>${spark(computed.map(r=>r.h),row,'#b59253')}<div class="plot-label">COMPUTED FROM H₋₁ = 0 · CAPTURED FACTORS · 64 ROWS</div>`;
    $('native-foot').innerHTML=`<span>dt = ${short(base.dt)}</span><span>CPU · FP32 · recorded</span>`;
    $('recovered-foot').innerHTML=`<span>retention = ${short(changed.decay)}</span><span>${state.dose===1?'baseline reconstruction':'conditional row intervention'}</span>`;
    $('math-details').innerHTML=`<code>A = −exp(a_log)<br>Dₜ = exp(dtₜ A) &nbsp; Wₜ = (dtₜ uₜ) Bₜ<br>Hₜ = Dₜ ⊙ Hₜ₋₁ + Wₜ<br>Hₜ = Σₖ₌₀ᵗ ( ∏ⱼ₌ₖ₊₁ᵗ Dⱼ ) Wₖ &nbsp; [H₋₁ = 0]<br>rₜ = Σₛ Hₜ Cₜ<br>mixerₜ = Xₜ + Wout[((rₜ + d ⊙ uₜ) ⊙ Gₜ)] + bout</code><p>At this coordinate: exp(${precision(changed.dt)} × ${precision(changed.a)}) × ${precision(changed.previous)} + ${precision(changed.writer)} = ${precision(changed.h)}. The multiplier changes both decay and writing at the selected row. The amber curve recomputes this coordinate from zero through all 64 rows, with u, B, C, G and other dt values captured. The consumer strip uses a local row calculation from captured previous H. These are conditional calculations, not a full-model counterfactual. G is the model’s SiLU gate; the post MLP uses GELU.</p>${contributionPlot(computed,row)}`;
    const err=Math.abs(base.h-base.native);
    $('comparison-status').innerHTML=state.dose===1?`<span class="status-dot"></span>Baseline local reconstruction · absolute difference ${err.toExponential(2)} at this coordinate. Native parallel scan and sequential equations associate FP32 operations differently.`:`<span class="status-dot" style="background:#b59253"></span>Custom ${state.dose.toFixed(2)}× clock · live local calculation. The native lane stays at its recorded baseline; no native result is available for this setting.`;
  } else {
    const x=[...scan.post.input[row]];x[0]+=state.mlpDelta;
    const result=postMLP(x,scan.weights),baseline=postMLP(scan.post.input[row],scan.weights);
    const nativeOut=scan.post['residual.output'][row];
    $('native-title').textContent='A measured MLP response';$('recovered-title').textContent='One function. A different receiving state.';
    $('native-description').textContent='Post MLP 3 at the selected bit row. Every tile represents one recorded activation.';
    $('recovered-description').textContent='Exported weights, normalization, and GELU run locally. Shift input feature 0 to explore the response.';
    $('native-visual').innerHTML=heatSVG([{title:'Layer norm',values:scan.post['layer_norm.output'][row]},{title:'GELU',values:scan.post['gelu.output'][row]},{title:'MLP update',values:scan.post.output[row]}]);
    const bars=arr=>`<div class="mlp-bars">${arr.slice(0,32).map(v=>`<i style="height:${5+Math.min(Math.abs(v)/Math.max(...arr.map(Math.abs),1e-8),1)*35}px"></i>`).join('')}</div>`;
    $('recovered-visual').innerHTML=`<div class="mlp-strip"><div><strong>48</strong><span>normalized features</span>${bars(result.norm)}</div><span class="operator">→</span><div><strong>96</strong><span>GELU responses</span>${bars(result.activated)}</div><span class="operator">→</span><div><strong>48</strong><span>update features</span>${bars(result.update)}</div></div><div class="bit-equation mlp-lane-foot"><span>X</span><span>+</span><span>MLP(LN(X))</span><span>=</span><span class="formula-result">Y</span></div><div class="plot-label" style="text-align:center;margin-top:12px">BAR PREVIEW: FIRST 32 FEATURES · HEIGHT = MAGNITUDE</div>`;
    $('native-foot').innerHTML=`<span>‖update‖₂ = ${short(vecNorm(scan.post.output[row]))}</span><span>recorded GELU</span>`;
    $('recovered-foot').innerHTML=`<span>‖update‖₂ = ${short(vecNorm(result.update))}</span><span>feature 0 shift: ${state.mlpDelta.toFixed(2)}</span>`;
    $('math-details').innerHTML=`<code>n = γ ⊙ (X − mean(X)) / √(var(X)+10⁻⁵) + β<br>a = W₁n + b₁<br>g = GELU(a) = ½a[1 + erf(a/√2)]<br>Δ = W₂g + b₂<br>Y = X + Δ</code><p>The 48 → 96 → 48 parameter matrices come from the same frozen checkpoint. This browser uses double precision and a small erf approximation (maximum erf error about 1.5×10⁻⁷). Normalization couples every input feature to the others, so moving one coordinate can change many responses. Only this MLP is recomputed.</p>`;
    $('comparison-status').innerHTML=state.mlpDelta===0?`<span class="status-dot"></span>Baseline MLP reconstruction · maximum absolute difference across 48 output features: ${maxError(baseline.output,nativeOut).toExponential(2)}.`:`<span class="status-dot" style="background:#b59253"></span>Feature 0 shifted by ${state.mlpDelta.toFixed(2)} · maximum output change ${maxError(result.output,baseline.output).toFixed(5)}. Custom MLP input; no recorded native comparison.`;
  }
}
function currentContext() {
  if(!carry||!loops)return null;
  if(state.scenario==='carry') {
    const words=carry.before.map(BigInt),after=[...words];const r=wordLaw(words[7]);after[7]=r.word;
    return {nativeBefore:words,nativeAfter:carry.after.map(BigInt),predBefore:words,predAfter:after,
      flagsBefore:carry.flagsBefore,nativeFlags:carry.flagsAfter,predFlags:[0,0,Number((r.word>>63n)&1n),0],
      offset:0,length:4,route:7,rows:r.rows,raw:carry.rawWindowHex,count:1,source:'carry-depth-0',matchScope:'One recorded instruction; compact response-law comparison.'};
  }
  const c=loops.cases.find(c=>c.id===state.scenario);
  const computed=trajectory(c.initial,c.initial.program_hex,4),index=state.instruction,step=c.steps[index],r=computed[index];
  const prior=index?c.steps[index-1].native_transition:null;
  return {nativeBefore:(prior?prior.register_words:c.initial.register_words_before).map(BigInt),nativeAfter:step.native_transition.register_words.map(BigInt),
    predBefore:r.before.words,predAfter:r.after.words,flagsBefore:prior?prior.flag_bits_in_input_order:c.initial.flags_before,
    predFlagsBefore:r.before.flags,nativeFlags:step.native_transition.flag_bits_in_input_order,predFlags:r.after.flags,
    offset:r.before.offset,length:4,route:r.after.route,rows:r.after.rows,raw:r.after.window,count:4,source:c.id,
    matchScope:'Two four-instruction paths: 8/8 recorded steps match registers, flags, and length. Flag hypotheses remain non-unique.'};
}
function renderMachine() {
  const ctx=currentContext();if(!ctx)return;
  $('scenario').value=state.scenario;$('instruction-value').textContent=`Instruction ${state.instruction+1} / ${ctx.count}`;
  $('machine-row').textContent=`${String(state.row).padStart(2,'0')} / 63`;
  $('instruction-steps').innerHTML=Array.from({length:ctx.count},(_,i)=>`<button data-instruction="${i}" class="${state.instruction===i?'active':''}" aria-label="Select instruction ${i+1}" ${state.instruction===i?'aria-current="step"':''}>${String(i+1).padStart(2,'0')}</button>`).join('');
  $('phase-track').innerHTML=phases.map(([label,sub],i)=>`<button data-phase="${i}" class="phase-step ${state.phase===i?'active':i<state.phase?'done':''}" aria-label="Conceptual phase: ${label}" ${state.phase===i?'aria-current="step"':''}><span class="phase-circle">${i<state.phase?'✓':String(i+1).padStart(2,'0')}</span><span class="phase-name">${label}</span><span class="phase-sub">${sub}</span></button>`).join('');
  const begin=Math.max(0,Math.min(52,state.row-5)),visible=ctx.rows.slice(begin,begin+12);
  $('bit-tape').innerHTML=`<span></span>${visible.map(r=>`<span class="bit-index">${String(r.i).padStart(2,'0')}</span>`).join('')}${[['input','x'],['state','q'],['output','y']].map(([label,key])=>`<span class="bit-label">${label}</span>${visible.map(r=>`<span class="bit ${r[key]?'one':''} ${r.i===state.row?'active':''} ${key==='y'?'output':''}" title="Bit ${r.i}, ${key}=${r[key]}">${r[key]}</span>`).join('')}`).join('')}`;
  const r=ctx.rows[state.row];
  $('bit-equation').innerHTML=`<span>y = <b>${r.q}</b> ⊕ <b>${r.x}</b> = <span class="formula-result">${r.y}</span></span><span style="color:#c2c9d4">/</span><span>q′ = <b>${r.q}</b> · <b>${r.x}</b> = <span class="formula-result">${r.next}</span></span>`;
  const committed=state.phase===5, native=committed?ctx.nativeAfter:ctx.nativeBefore, predicted=committed?ctx.predAfter:ctx.predBefore;
  const parity=native.every((v,i)=>v===predicted[i])&&(committed?ctx.nativeFlags.every((v,i)=>v===ctx.predFlags[i]):ctx.flagsBefore.every((v,i)=>v===(ctx.predFlagsBefore||ctx.flagsBefore)[i]));
  $('state-parity').textContent=parity?'MATCH':'DIFFERENCE';
  $('registers').innerHTML=native.map((v,i)=>`<div class="register-cell ${i===ctx.route?'selected':''} ${committed&&v!==ctx.nativeBefore[i]?'changed':''}"><div class="register-top"><span class="register-name">R${i}</span><span class="register-marker">${committed&&v!==ctx.nativeBefore[i]?'UPDATED':i===ctx.route?'ROUTED':'PRESERVED'}</span></div><div class="register-values"><span class="native-ink"><i>N</i>${word(v)}</span><span class="recovered-ink"><i>R</i>${word(predicted[i])}</span></div></div>`).join('');
  const nativeFlags=committed?ctx.nativeFlags:ctx.flagsBefore,predFlags=committed?ctx.predFlags:(ctx.predFlagsBefore||ctx.flagsBefore);
  $('state-metadata').innerHTML=`<span>FLAGS <strong class="native-ink">${nativeFlags.join(' ')}</strong><strong class="recovered-ink">${predFlags.join(' ')}</strong></span><span>BYTE OFFSET <strong>${ctx.offset+(committed?ctx.length:0)}</strong></span><span>FETCH <strong>${esc(ctx.raw.slice(0,8).match(/../g).join(' '))}</strong></span><span>${committed?'COMMITTED → NEXT INPUT':'PENDING · BEFORE COMMIT'}</span>`;
  const notes=[`Context: ${ctx.source}. Native trace and recovered program receive their own state.`,`Route: raw window bits 16–18 select register slot ${ctx.route} on this tested window family.`,`Form update: live factors determine both the retention and the writer.`,`Inner clock: bit ${state.row}. The compact q and the full neural H are different state descriptions.`,`Read: native logits decode into words and flags; the recovered program emits its own prediction.`,`Commit: registers, flags, and byte offset become the next call’s inputs. Internal scan and convolution histories reset.`];
  $('machine-status').textContent=notes[state.phase];
  $('previous').disabled=history.length===0;
  const contextLabel=state.scenario==='carry'?'RECORDED DETAIL · CARRY-DEPTH-0':`RECORDED DETAIL · CARRY-DEPTH-0 (OUTER LOOP: ${state.scenario==='symbolic-00'?'PATH A':'PATH B'})`;
  $('trace-location').title=contextLabel;
  $('trace-location').setAttribute('aria-label',contextLabel);
}
function snapshot() {return {phase:state.phase,row:state.row,instruction:state.instruction,node:state.node};}
function stepForward() {
  const ctx=currentContext();if(!ctx)return;
  history.push(snapshot());
  if(state.phase<3)state.phase++;
  else if(state.phase===3&&state.row<63)state.row++;
  else if(state.phase<5)state.phase++;
  else if(state.instruction<ctx.count-1){state.instruction++;state.phase=0;state.row=0;}
  else {history.pop();stop();toast('Execution complete. Reset or choose another context.');return;}
  state.node=state.phase===0?'decoder':state.phase===1?'routing':state.phase<=3?'scan2':state.phase===4?'post2':'heads';
  renderArchitecture();renderMachine();renderComparison();
}
function run() {
  playback.start();
}
function resetExecution() {stop();history.length=0;state.row=0;state.phase=0;state.instruction=0;state.node='inputs';renderArchitecture();renderMachine();renderComparison();}
function renderFrontier() {
  if(!frontier)return;
  $('toggle-h').setAttribute('aria-pressed',state.h);$('toggle-conv').setAttribute('aria-pressed',state.conv);
  const cell=frontier.cells.find(c=>c.h===Number(state.h)&&c.conv===Number(state.conv));
  $('frontier-chart').innerHTML=frontier.cells.map(c=>`<div class="frontier-bar-row ${c===cell?'selected':''}"><span>${c.h?'H':'—'} + ${c.conv?'conv':'—'}</span><div class="frontier-bar-track"><div class="frontier-bar ${c.logit>0?'positive':''}" style="width:${Math.max(.7,Math.abs(c.logit)/8*85)}%"></div></div><span class="frontier-value">${c.logit>0?'+':''}${c.logit.toFixed(4)}</span></div>`).join('');
  $('frontier-result').innerHTML=`<strong>${cell.logit>0?'+':''}${cell.logit.toFixed(5)}</strong><span>R7 · bit 32 logit<b>${cell.bit?'Bit flips to 1 · joint frontier':'Bit remains 0'}</b></span>`;
  $('exception-list').innerHTML=frontier.exceptions.map(e=>`<div class="exception-row">${esc(e.query_id)} · cut ${e.cut} · output bit ${e.output_bit}<br>Recovered ${e.program_bit} / native ${e.native_bit} · native logit +${e.native_logit.toFixed(6)}<br>Prefix ${esc(e.prefix_hex)} · suffix ${e.suffix_value}</div>`).join('');
}
async function showSources() {
  const signal=lifecycle.signal;
  try {
    provenance ||= await json(model.evidence.provenance);
    if(signal.aborted)return;
    openDialog('EVIDENCE & PROVENANCE',`<h2>Every measurement has a source.</h2><p>This is the frozen R3 Mamba direct-effect ISA checkpoint, captured on CPU in FP32. The website runs small local calculations; it runs no model forwards.</p><pre>Checkpoint\n${esc(provenance.checkpoint)}\n\nNew model forwards: 0\nNative observations: recorded FP32\nBrowser local mathematics: double precision</pre><p>Source-derived equations, finite recovered response laws, and illustrative playback phases have different scopes. The conceptual phase track is a teaching map, not a measurement of native wall-clock timing. The scan and MLP panels use the fixed carry-depth-0 context; switching the outer-loop context changes the state-machine panel.</p><a class="download-link" href="./data/provenance.json" download>Download the hashed source manifest ↧</a><p>The exporter preserves the original research artifacts. The compact law’s three held-out exceptions and non-unique flag hypotheses remain visible.</p>${Object.entries(provenance.sources).map(([path,hash])=>`<div class="source-row"><p>${esc(path)}</p><small>SHA-256 ${hash}</small></div>`).join('')}<p>Teaching references: <a href="https://ciechanow.ski/gears/" target="_blank" rel="noreferrer">Ciechanowski: Gears ↗</a> · <a href="https://www.redblobgames.com/pathfinding/a-star/introduction.html" target="_blank" rel="noreferrer">Red Blob Games: A* ↗</a> · <a href="https://poloclub.github.io/transformer-explainer/" target="_blank" rel="noreferrer">Transformer Explainer ↗</a></p>`);
  } catch(error){if(!signal.aborted)toast(error.message);}
}
async function showReading(id) {
  const signal=lifecycle.signal;
  const reading=readings.find(r=>r.id===id);
  openDialog('FROM THE FIELD NOTES',`<h2>${esc(reading.title)}</h2><p>Loading the original research note…</p>`);
  try {
    const response=await fetch(reading.path,{signal});if(!response.ok)throw new Error('The research note could not be loaded.');
    const text=await response.text();
    if(signal.aborted)return;
    const guides={
      'state-math':{lesson:1,text:'A model advances a state under a fixed parameter law. Context determines which operands are available, how they are read, and what an update will change. The useful question is which parts of that state must cross the next boundary so that future computations can continue.',equation:'Sₜ₊₁ = Fθ(Sₜ, uₜ)',scope:'The full native state is sufficient by construction. A smaller operational state becomes a useful explanation only when it preserves the future computations we care about.'},
      'mlp':{lesson:2,text:'An MLP has fixed weights, but its finite response depends on the state it receives. Layer normalization couples coordinates; GELU changes the effective response of each expanded feature. A good local account tracks the input, the update, the receiving state, and the downstream consumer together.',equation:'Y = X + W₂ GELU(W₁ LN(X) + b₁) + b₂',scope:'A local derivative describes an infinitesimal response. A finite change can cross nonlinear regions and needs its own calculation. Try shifting one input feature in the live MLP view.'},
      'carry':{lesson:1,text:'The carry example connects a behavioral change to its formation, writer, carrier, and consumer. A visible output is the end of a chain: earlier rows shape the available information, state carries it forward, and a later read turns it into an output change.',equation:'earlier writes → retained state → readout → MLP → output bit',scope:'A neural state coordinate is not automatically a semantic carry bit. The explorer shows a source-qualified recurrence alongside the separate, compact Boolean response hypothesis.'},
      'symbols':{lesson:3,text:'A dynamic symbol table attaches names and addresses to computations observed in a particular model and context. Its useful entries include live operands, state transitions, consumer bindings, and the evidence that a frozen prediction survived a later native comparison.',equation:'context + address + operands + law + consumer → prediction',scope:'Source-derived equations, measured response laws, and semantic names are different levels of understanding. The compact ISA law has finite successes and retained exceptions; the full parameter lowering has a different scope.'},
    };
    const guide=guides[id];
    openDialog('FROM THE FIELD NOTES',`<h2>${esc(reading.title)}</h2><p>${esc(guide.text)}</p><pre>${esc(guide.equation)}</pre><p>${esc(guide.scope)}</p><button class="dark-button" id="reading-explore" data-chapter="${guide.lesson}">Explore this idea →</button><p>Original source: <code>${esc(reading.source)}</code></p><a class="download-link" href="${reading.path}" download>Download original Markdown ↧</a><details class="original-note"><summary>Read the original source note</summary><p>Original mathematical notation and Obsidian backlinks are preserved below.</p><pre class="blog-text">${esc(text.replace(/^---\n[\s\S]*?\n---\n/,''))}</pre></details>`);
    $('reading-explore').onclick=()=>{$('detail-dialog').close();selectLesson(guide.lesson);};
  } catch(error){if(!signal.aborted)toast(error.message);}
}
function setupEvents() {
  $('lesson-nav').addEventListener('click',e=>{const b=e.target.closest('[data-lesson]');if(b)selectLesson(Number(b.dataset.lesson));},{signal:lifecycle.signal});
  $('next-lesson').onclick=()=>{if(state.lesson<4)selectLesson(state.lesson+1);else{state.guided=false;renderLesson();scrollToSection('architecture');toast('Explore mode. Select components, scrub time, and change the context.');}};
  $('guided-mode').onclick=()=>{state.guided=true;renderLesson();};$('explore-mode').onclick=()=>{state.guided=false;renderLesson();};
  const graphClick=e=>{const node=e.target.closest('[data-node]');if(node)selectNode(node.dataset.node);};
  $('architecture-graph').addEventListener('click',graphClick,{signal:lifecycle.signal});$('architecture-graph').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();graphClick(e);}},{signal:lifecycle.signal});
  $('component-inspector').addEventListener('click',e=>{if(e.target.closest('#inspect-computation')){const n=model.graph.nodes.find(n=>n.id===state.node);if(['scan2','post2'].includes(n.id)){setView(n.id==='scan2'?'scan':'mlp');scrollToSection('computation');}else openDialog('COMPONENT / SOURCE-DERIVED',`<h2>${esc(n.label)}</h2><p>${esc(n.text)}</p><pre>${esc(n.equation)}</pre><p>${esc(n.dims)}</p><p>The live numerical playground is anchored at Mixer 3 and Post MLP 3 in carry-depth-0. This component is shown in the native architecture with its own source-qualified shape and equation.</p>`);}});
  $('scan-tab').onclick=()=>{state.node='scan2';setView('scan');renderArchitecture();};$('mlp-tab').onclick=()=>{state.node='post2';setView('mlp');renderArchitecture();};
  $('bit-row').oninput=e=>{stop();state.row=Number(e.target.value);state.phase=3;renderArchitecture();renderComparison();renderMachine();};
  $('dose').oninput=e=>{if(state.view==='scan')state.dose=Number(e.target.value);else state.mlpDelta=Number(e.target.value);renderComparison();};
  $('reset-dose').onclick=()=>{state.dose=1;state.mlpDelta=0;$('dose').value=state.view==='scan'?1:0;renderComparison();};
  $('channel').onchange=e=>{const x=Number(e.target.value);if(!Number.isInteger(x)||x<0||x>95){e.target.value=state.channel;toast('Channel must be an integer from 0 to 95.');return;}state.channel=x;renderComparison();};
  $('state-mode').onchange=e=>{state.mode=Number(e.target.value);renderComparison();};$('math-button').onclick=()=>{state.math=!state.math;renderComparison();};
  $('play').onclick=()=>{if(state.playing)stop();else{state.playing=true;$('play').innerHTML='<span>Ⅱ</span> Pause';$('play').setAttribute('aria-label','Pause execution');run();}};
  $('next').onclick=()=>{stop();stepForward();};$('previous').onclick=()=>{stop();const prev=history.pop();if(prev){Object.assign(state,prev);renderArchitecture();renderMachine();renderComparison();}};
  $('reset').onclick=resetExecution;
  $('scenario').onchange=e=>{state.scenario=e.target.value;resetExecution();};
  $('instruction-steps').onclick=e=>{const b=e.target.closest('[data-instruction]');if(b){stop();history.length=0;state.instruction=Number(b.dataset.instruction);state.phase=0;state.row=0;renderMachine();renderArchitecture();renderComparison();}};
  $('phase-track').onclick=e=>{const b=e.target.closest('[data-phase]');if(b){stop();history.push(snapshot());state.phase=Number(b.dataset.phase);if(state.phase>=4)state.row=63;state.node=state.phase===0?'decoder':state.phase===1?'routing':state.phase<=3?'scan2':state.phase===4?'post2':'heads';renderArchitecture();renderMachine();renderComparison();}};
  $('format-button').onclick=()=>{state.hex=!state.hex;renderMachine();};
  $('sandbox-form').onsubmit=e=>{e.preventDefault();try{const input=$('sandbox-word').value.trim();if(!/^(?:0x[0-9a-f]+|[0-9]+)$/i.test(input))throw new Error('Use a decimal or 0x-prefixed unsigned 64-bit integer.');const x=BigInt(input),r=wordLaw(x);$('sandbox-result').textContent=`${x} → ${r.word}   (q starts at 1; 64 Boolean transitions)`;}catch(error){$('sandbox-result').textContent=error.message;}};
  $('toggle-h').onclick=()=>{state.h=!state.h;renderFrontier();};$('toggle-conv').onclick=()=>{state.conv=!state.conv;renderFrontier();};
  $('sources-button').onclick=showSources;
  $('reading-list').onclick=e=>{const b=e.target.closest('[data-reading]');if(b)showReading(b.dataset.reading);};
  $('about-models').onclick=()=>openDialog('MODEL PACKS',`<h2>Same explorer. Different computation.</h2><p>The model pack supplies its component graph, clocks, state fields, evidence, lessons, and calculation kernels. The shared explorer draws that graph and coordinates playback. The explorer ships R3 Mamba ISA and FLUX.2 Klein packs.</p><pre>ModelPack v1\n  id, name, checkpoint, kernel\n  graph: nodes, edges, groups, geometry\n  clocks, stateFields\n  lessons\n  evidence: recorded data URLs</pre><p>Attention and diffusion packs introduce different topology, state, and clocks. They need their own evidence and kernel binding; they are not automatically interchangeable with ISA numerical calculations.</p><p>Source and extension instructions are included in the project README. No weights, model server, accounts, or GPU are needed for this demonstration.</p>`);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();},{signal:lifecycle.signal});
}
async function init() {
  renderLesson();renderArchitecture();setupEvents();
  try {
    [carry,loops,frontier,readings]=await Promise.all([json(model.evidence.carry),json(model.evidence.loop),json(model.evidence.frontier),json('./data/readings.json')]);
    if(lifecycle.signal.aborted)return;
    $('reading-list').innerHTML=readings.map((r,i)=>`<button class="reading-card" data-reading="${r.id}"><span class="reading-index">0${i+1} / RESEARCH NOTE</span><h3>${esc(r.title)}</h3><span class="reading-type">OPEN FIELD NOTE</span><span class="reading-arrow">↗</span></button>`).join('');
    renderMachine();renderFrontier();await ensureScan();if(!lifecycle.signal.aborted)renderComparison();
  }catch(error){if(lifecycle.signal.aborted)return;$('comparison-status').innerHTML=`<span class="error-notice">Evidence could not load. ${esc(error.message)}. Reload to retry.</span>`;stop();}
}
export async function mount(pack,environment) {
  const signal=environment.signal;
  await mountTemplate('./templates/isa.html',signal);
  model=validatePack(pack);lifecycle=environment;Object.assign(state,initialState,environment.state||{},{playing:false});history.length=0;
  playback=new Playback({step:()=>{stepForward();return state.playing;},delay:()=>Number($('speed').value),onToggle:active=>{state.playing=active;if($('play')){$('play').innerHTML=active?'<span>Ⅱ</span> Pause':'<span>▶</span> Run';$('play').setAttribute('aria-label',active?'Pause execution':'Play execution');}},signal});
  await init();
  if(!signal.aborted){setView(state.view);$('channel').value=state.channel;$('state-mode').value=state.mode;}
  return {getState(){return {...state,playing:false};},dispose(){stop();clearTimeout(toast.timer);$('toast').hidden=true;$('sources-button').onclick=null;}};
}
