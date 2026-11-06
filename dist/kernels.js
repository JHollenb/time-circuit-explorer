// Small, source-qualified calculations. No neural inference runtime is loaded.
export const U64 = (1n << 64n) - 1n;
export function wordLaw(word) {
  const x = BigInt(word);
  if (x < 0n || x > U64) throw new RangeError('Enter an unsigned 64-bit word.');
  let q = 1, y = 0n;
  const rows = [];
  for (let i = 0; i < 64; i++) {
    const bit = Number((x >> BigInt(i)) & 1n), out = q ^ bit, next = q * bit;
    rows.push({i, x:bit, q, y:out, next});
    y |= BigInt(out) << BigInt(i); q = next;
  }
  return {word:y, rows};
}
export function routeFromWindow(hex) {
  if (!/^[0-9a-f]{6,}$/i.test(hex) || hex.length % 2) throw new Error('Invalid raw byte window.');
  return parseInt(hex.slice(4,6),16) & 7;
}
export function advance(state, programHex) {
  const window = programHex.slice(state.offset*2).slice(0,30).padEnd(30,'0');
  const route = routeFromWindow(window);
  const result = wordLaw(state.words[route]);
  const words = state.words.map(BigInt); words[route] = result.word;
  return {words, flags:[0,0,Number((result.word >> 63n) & 1n),0], offset:state.offset+4, route, window, rows:result.rows};
}
export function trajectory(initial, programHex, horizon = 4) {
  let state = {words:initial.register_words_before.map(BigInt), flags:[...initial.flags_before], offset:0};
  const steps=[];
  for (let i=0;i<horizon;i++) { const before = state; state = advance(state, programHex); steps.push({before,after:state}); }
  return steps;
}
export function dot(w,x) { let y=0; for(let i=0;i<w.length;i++) y+=w[i]*x[i]; return y; }
export function linear(w,b,x) { return w.map((row,i)=>dot(row,x)+b[i]); }
export function erf(x) {
  // Abramowitz-Stegun 7.1.26; max absolute erf approximation error ~1.5e-7.
  const sign=x<0?-1:1, v=Math.abs(x), t=1/(1+0.3275911*v);
  const p=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t;
  return sign*(1-p*Math.exp(-v*v));
}
export function gelu(x) {return 0.5*x*(1+erf(x/Math.SQRT2));}
export function postMLP(x,weights) {
  const mean=x.reduce((a,b)=>a+b,0)/x.length;
  const variance=x.reduce((a,b)=>a+(b-mean)**2,0)/x.length;
  const norm=x.map((v,i)=>(v-mean)/Math.sqrt(variance+1e-5)*weights['post.2.0.weight'][i]+weights['post.2.0.bias'][i]);
  const pre=linear(weights['post.2.1.weight'],weights['post.2.1.bias'],norm);
  const activated=pre.map(gelu);
  const update=linear(weights['post.2.3.weight'],weights['post.2.3.bias'],activated);
  return {norm, pre, activated, update, output:x.map((v,i)=>v+update[i])};
}
export function scanRow(data,row,channel=0,mode=0,scale=1) {
  const a=data.a[channel][mode], dt=data.dt[row][channel]*scale;
  const previous=row?data.h[row-1][channel][mode]:0;
  const decay=Math.exp(dt*a), writer=dt*data.u[row][channel]*data.b[row][mode];
  return {previous, a, dt, decay, writer, retained:decay*previous, h:decay*previous+writer, native:data.h[row][channel][mode]};
}
export function scanTrajectory(data,channel=0,mode=0,editRow=-1,scale=1) {
  let h=0;const rows=[];
  for(let row=0;row<64;row++) {
    const dt=data.dt[row][channel]*(row===editRow?scale:1);
    const decay=Math.exp(dt*data.a[channel][mode]);
    const writer=dt*data.u[row][channel]*data.b[row][mode];
    const previous=h;h=decay*h+writer;
    rows.push({h,previous,decay,writer,retained:decay*previous});
  }
  return rows;
}
export function writeContributions(rows,end) {
  let transport=1;const contributions=[];
  for(let k=end;k>=0;k--) {
    contributions.unshift({row:k,transport,writer:rows[k].writer,value:transport*rows[k].writer});
    transport*=rows[k].decay;
  }
  return contributions;
}
export function conditionalMixer(data,row,scale=1) {
  // The selected row only. Its previous H and all upstream factors are captured.
  const states=[], z=[];
  for(let c=0;c<96;c++) {
    const hs=Array.from({length:4},(_,s)=>scanRow(data,row,c,s,scale).h);
    states.push(hs);
    z.push((dot(hs,data.c[row])+data.d[c]*data.u[row][c])*data.g[row][c]);
  }
  const projected=linear(data.wOut,data.outBias,z);
  return {states,output:projected.map((v,i)=>v+data.residual[row][i])};
}
export function maxError(a,b) {return Math.max(...a.map((v,i)=>Math.abs(v-b[i])));}
export function escapeHTML(s) {return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
