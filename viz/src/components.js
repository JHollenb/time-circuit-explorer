import {escapeHTML as esc} from './kernels.js';

export const $=id=>document.getElementById(id);
const texts=new Map();
const jsons=new Map();
async function cached(url,kind,signal) {
  const cache=kind==='json'?jsons:texts;
  if(cache.has(url))return cache.get(url);
  const response=await fetch(url,{signal});
  if(!response.ok)throw new Error(`Could not load ${url} (${response.status})`);
  const value=await response[kind]();cache.set(url,value);return value;
}
export const assetJSON=(url,signal)=>cached(url,'json',signal);
export async function mountTemplate(url,signal) {
  const html=await cached(url,'text',signal);
  signal?.throwIfAborted();$('main').innerHTML=html;$('main').setAttribute('aria-busy','false');
}
export function openDialog(label,html) {
  $('dialog-label').textContent=label;$('dialog-content').innerHTML=html;
  if(!$('detail-dialog').open)$('detail-dialog').showModal();
}
export function installDialog() {
  $('close-dialog').onclick=()=>$('detail-dialog').close();
  $('detail-dialog').addEventListener('click',e=>{if(e.target===$('detail-dialog')){
    const r=$('detail-dialog').getBoundingClientRect();
    if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('detail-dialog').close();
  }});
}
export function scrollToSection(id) {
  $(id)?.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
}
export function renderWalkthrough(lessons,active,guided) {
  $('lesson-nav').innerHTML=lessons.map((l,i)=>`<button data-lesson="${i}" class="${i===active?'active':''}" ${i===active?'aria-current="step"':''}><span class="lesson-number">${String(i+1).padStart(2,'0')}</span>${esc(l.short)}</button>`).join('');
  const l=lessons[active];
  $('lesson-kicker').textContent=l.kicker;$('lesson-heading').textContent=l.heading;$('lesson-text').textContent=l.text;
  $('next-lesson').innerHTML=active===lessons.length-1?'Explore freely <span>↗</span>':'Next chapter <span>→</span>';
  for(const [id,on]of [['guided-mode',guided],['explore-mode',!guided]]){$(id).classList.toggle('active',on);$(id).setAttribute('aria-pressed',on);}
  document.querySelector('.lesson-caption').hidden=!guided;
}
export function renderArchitecture(svg,graph,{selected,edgePath,footerLeft='',footerRight=''}={}) {
  const nodes=new Map(graph.nodes.map(n=>[n.id,n]));
  const colors={input:['#f9fafc','#e1e6ef'],mlp:['#eef7f2','#cde2d6'],scan:['#f2eff9','#d5cce9'],route:['#eef4fc','#cbdaf1'],head:['#eef4fc','#cbdaf1'],attention:['#f2eff9','#d5cce9'],scheduler:['#fcf6eb','#e4d5b6'],merge:['#f2eff9','#d5cce9'],context:['#eef4fc','#cbdaf1']};
  svg.setAttribute('viewBox',`0 0 ${graph.width} ${graph.height}`);
  const edges=graph.edges.map(([a,b,options={}])=>{
    const s=nodes.get(a),t=nodes.get(b);let path=options.path||edgePath?.(a,b,s,t);
    if(!path){
      if(s.x===t.x)path=`M${s.x+s.w/2},${s.y+s.h} V${t.y}`;
      else if(s.x>t.x)path=`M${s.x+s.w/2},${s.y+s.h} V${t.y-10} H${t.x+t.w/2} V${t.y}`;
      else if(s.y>t.y+50)path=`M${s.x+s.w},${s.y+s.h/2} H${s.x+s.w+9} V${t.y+t.h/2} H${t.x}`;
      else path=`M${s.x+s.w},${s.y+s.h/2} C${s.x+s.w+20},${s.y+s.h/2} ${t.x-20},${t.y+t.h/2} ${t.x},${t.y+t.h/2}`;
    }
    const active=a===selected||b===selected;
    return `<path class="graph-edge ${active?'highlight':''}" d="${path}" ${options.dashed?'stroke-dasharray="4 4"':''} marker-end="url(#arrow-${active?'active':'default'})"/>`;
  });
  svg.innerHTML=`<defs>${[['default','#cdd5e2'],['active','#9282c4']].map(([name,color])=>`<marker id="arrow-${name}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L8 4L0 8" fill="${color}"/></marker>`).join('')}</defs>
    ${(graph.groups||[]).map(g=>`<rect x="${g.x}" y="${g.y}" width="${g.w}" height="${g.h}" rx="10" fill="#fcfbfe" stroke="#e7e2f0" stroke-dasharray="3 3"/><text x="${g.x+15}" y="${g.y+20}" font-size="6.5" fill="#a094b6" letter-spacing=".7" font-family="monospace">${esc(g.label)}</text>`).join('')}
    ${edges.join('')}
    ${graph.nodes.map(n=>{const c=colors[n.kind]||colors.input;return `<g class="graph-node ${selected===n.id?'selected':''}" data-node="${n.id}" role="button" tabindex="0" aria-label="Inspect ${esc(n.label)}" aria-pressed="${selected===n.id}"><rect class="node-box" x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="7" fill="${c[0]}" stroke="${c[1]}"/><text class="node-title" x="${n.x+11}" y="${n.y+21}">${esc(n.label)}</text><text class="node-sub" x="${n.x+11}" y="${n.y+35}">${esc(n.sub||'')}</text>${n.h>60?`<text class="node-math" x="${n.x+11}" y="${n.y+54}">${esc(n.math||'')}</text>`:''}${selected===n.id?`<circle cx="${n.x+n.w-9}" cy="${n.y+10}" r="2.5" fill="#9482bd"/>`:''}</g>`;}).join('')}
    <text x="20" y="${graph.height-16}" font-size="7" fill="#a2adbe" font-family="monospace">${esc(footerLeft)}</text><text x="${graph.width-210}" y="${graph.height-16}" font-size="7" fill="#9a8eb3" font-family="monospace">${esc(footerRight)}</text>`;
}
export function renderInspector(node,{explore=false,action='inspect-computation'}={}) {
  return `<div><span class="inspector-type">${esc(node.typeLabel||node.kind.toUpperCase())}</span><h3>${esc(node.label)}</h3><span class="dimensions">${esc(node.dims)}</span></div><p>${esc(node.text)}</p><button class="inspect-link" id="${action}">${explore?'Explore this computation':'Inspect the equation'} <span>↗</span></button><div class="inspector-symbol">${esc(node.equation)}</div>`;
}
export class Playback {
  constructor({step,delay,onToggle=()=>{},signal}){Object.assign(this,{step,delay,onToggle});this.running=false;signal?.addEventListener('abort',()=>this.stop(),{once:true});}
  start(){if(this.running)return;this.running=true;this.onToggle(true);this.tick();}
  tick(){if(!this.running)return;if(this.step()===false){this.stop();return;}if(this.running)this.timer=setTimeout(()=>this.tick(),this.delay());}
  stop(){clearTimeout(this.timer);if(this.running){this.running=false;this.onToggle(false);}}
  toggle(){this.running?this.stop():this.start();}
}
