// Scalar teaching kernels. These do not execute the full FLUX transformer.
const float=new Float32Array(1),bits=new Uint32Array(float.buffer);
export function bf16(value) {
  float[0]=value;const u=bits[0];
  if((u&0x7f800000)===0x7f800000)return float[0];
  bits[0]=(u+0x7fff+((u>>>16)&1))&0xffff0000;
  return float[0];
}
export function euler(z,v,sigma,sigmaNext,rounded=true) {
  if(!rounded){const dt=sigmaNext-sigma,update=dt*v;return{dt,update,next:z+update};}
  const dt=bf16(Math.fround(Math.fround(sigmaNext)-Math.fround(sigma)));
  const update=bf16(Math.fround(dt*v));
  return{dt,update,next:bf16(Math.fround(z+update))};
}
export function address(cursor) {
  if(!Number.isInteger(cursor)||cursor<0||cursor>111)throw new RangeError('FLUX cursor must be 0–111.');
  const pass=Math.floor(cursor/28),local=cursor%28;
  const kind=local===0?'project':local<6?'joint':local<26?'single':local===26?'readout':'commit';
  const block=kind==='joint'?local-1:kind==='single'?local-6:null;
  return{cursor,pass,local,kind,block,label:kind==='joint'?`Joint ${block}`:kind==='single'?`Single ${block}`:{project:'Projection',readout:'Image readout',commit:'Euler commit'}[kind]};
}
export function boundaryRecord(history,cursor,side) {
  const target=cursor-(side==='before'?1:0);
  if(target<history.origin)return null;
  return history.records[target-history.origin]??null;
}
export const silu=x=>x/(1+Math.exp(-x));
export function softmax(xs) {const m=Math.max(...xs),exp=xs.map(x=>Math.exp(x-m)),sum=exp.reduce((a,b)=>a+b,0);return exp.map(x=>x/sum);}
// Six explicitly synthetic tokens, four features. Fixed matrices, no learned model weights.
export const toyTokens=Array.from({length:6},(_,t)=>Array.from({length:4},(_,f)=>Math.sin((t+1)*(f+1)*.7)));
export function toyAttention(query=0,bias=0,valueShift=0) {
  const q=toyTokens[query],keys=toyTokens.map(x=>[x[1],-x[0],x[3],x[2]]);
  const values=toyTokens.map((x,i)=>x.map((v,j)=>v+(i===4&&j===0?valueShift:0)));
  const logits=keys.map((k,i)=>k.reduce((s,v,j)=>s+v*q[j],0)/2+(i===4?bias:0));
  const weights=softmax(logits),output=Array.from({length:4},(_,j)=>values.reduce((s,v,i)=>s+weights[i]*v[j],0));
  return{logits,weights,values,output};
}
export function toyMLP(inputShift=0,gateShift=0) {
  const input=[.4+inputShift,-.7,.2,.9];
  const Wg=Array.from({length:8},(_,i)=>Array.from({length:4},(_,j)=>Math.sin((i+1)*(j+2))*.6));
  const Wv=Array.from({length:8},(_,i)=>Array.from({length:4},(_,j)=>Math.cos((i+2)*(j+1))*.6));
  const Wo=Array.from({length:4},(_,i)=>Array.from({length:8},(_,j)=>Math.sin((i+2)*(j+3))*.35));
  const mul=(w,x)=>w.map(row=>row.reduce((s,v,j)=>s+v*x[j],0));
  const u=mul(Wg,input).map(x=>x+gateShift),v=mul(Wv,input),gate=u.map(silu),product=gate.map((g,i)=>g*v[i]);
  const output=mul(Wo,product);
  // Equivalent factorization Wo diag(SiLU(Wg x)) Wv, for the CURRENT receiving state.
  const effective=Wo.map(row=>Wv[0].map((_,j)=>row.reduce((s,w,k)=>s+w*gate[k]*Wv[k][j],0)));
  return{input,u,v,gate,product,output,factored:mul(effective,input),effective};
}
