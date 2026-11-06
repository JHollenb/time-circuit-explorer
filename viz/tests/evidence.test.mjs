import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,statSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {gzipSync} from 'node:zlib';
import {wordLaw,routeFromWindow,trajectory,scanRow,scanTrajectory,writeContributions,conditionalMixer,postMLP,linear,maxError,U64} from '../src/kernels.js';
import {isa,validatePack} from '../src/models.js';
import {ModelHost} from '../src/model-host.js';

const read=n=>JSON.parse(readFileSync(new URL(`../public/data/${n}.json`,import.meta.url)));
const scan=read('scan'),loop=read('loop'),carry=read('carry'),frontier=read('frontier');

test('64-bit Boolean law handles carries, overflow, and values beyond JS integer precision',()=>{
  for(const x of [0n,1n,511n,19088895n,9007199254740993n,U64]) assert.equal(wordLaw(x).word,(x+1n)&U64);
  assert.equal(wordLaw(511n).rows[8].next,1);assert.equal(wordLaw(511n).rows[9].next,0);
  assert.throws(()=>wordLaw(-1),RangeError);assert.throws(()=>wordLaw(U64+1n),RangeError);
});
test('raw byte routing uses low three bits of byte 2 in the tested family',()=>{
  for(let r=0;r<8;r++) assert.equal(routeFromWindow(`4883c${r}01`),r);
  assert.throws(()=>routeFromWindow('xx'));
});
test('all eight independently computed own-state steps match saved native states and sealed predictions',()=>{
  for(const c of loop.cases){
    const computed=trajectory(c.initial,c.initial.program_hex,4);
    for(let i=0;i<4;i++){
      const ref=c.steps[i].native_transition,sealed=c.predictions[i].transition;
      assert.deepEqual(computed[i].after.words,ref.register_words.map(BigInt));
      assert.deepEqual(computed[i].after.words,sealed.register_words.map(BigInt));
      assert.deepEqual(computed[i].after.flags,ref.flag_bits_in_input_order);
      assert.equal(computed[i].after.offset,ref.predicted_next_byte_offset);
      if(i)assert.deepEqual(computed[i].before.words,computed[i-1].after.words);
    }
  }
});
test('carry example decodes the saved native logits and matches the compact word law',()=>{
  assert.equal(carry.carryDepth,9);assert.equal(carry.before[7],'19088895');
  assert.equal(wordLaw(carry.before[7]).word,BigInt(carry.after[7]));
  carry.registerLogits.forEach((reg,j)=>{let w=0n;reg.forEach((v,i)=>{if(v>0)w|=1n<<BigInt(i)});assert.equal(w,BigInt(carry.after[j]));});
});
test('every recorded final-scan coordinate agrees with the local source equation within FP32 rounding',()=>{
  let error=0;
  for(let r=0;r<64;r++)for(let c=0;c<96;c++)for(let m=0;m<4;m++){
    const v=scanRow(scan,r,c,m);error=Math.max(error,Math.abs(v.h-v.native));
    // Source values are FP32; compare to double precision with FP32 accumulation allowance.
    assert.ok(Math.abs(v.h-v.native)<3e-6+Math.abs(v.native)*2e-6);
  }
  console.log('Observed scan-coordinate maximum absolute difference:',error);
});
test('all 64 mixer readouts and residual MLP rows reproduce recorded outputs',()=>{
  let mixerError=0,mlpError=0;
  for(let r=0;r<64;r++){
    mixerError=Math.max(mixerError,maxError(conditionalMixer(scan,r).output,scan.mixer[r]));
    mlpError=Math.max(mlpError,maxError(postMLP(scan.post.input[r],scan.weights).output,scan.post['residual.output'][r]));
  }
  assert.ok(mixerError<1e-5);assert.ok(mlpError<5e-6);
  console.log('Observed final-mixer / MLP maximum absolute differences:',mixerError,mlpError);
});
test('a clock multiplier changes both retention and writer, and MLP perturbation changes the response',()=>{
  const a=scanRow(scan,8,0,0,1),b=scanRow(scan,8,0,0,1.5);
  assert.notEqual(a.decay,b.decay);assert.equal(b.writer/a.writer,1.5);
  const x=scan.post.input[9],changed=[...x];changed[0]+=1;
  assert.ok(maxError(postMLP(x,scan.weights).output,postMLP(changed,scan.weights).output)>.1);
});
test('expanded writer products factor back into the sequential state, with clock edits carried forward',()=>{
  for(const c of [0,17,95])for(const m of [0,3]){
    const rows=scanTrajectory(scan,c,m,8,1.5);
    for(const end of [0,8,31,63])assert.ok(Math.abs(writeContributions(rows,end).reduce((s,t)=>s+t.value,0)-rows[end].h)<1e-10);
    const original=scanTrajectory(scan,c,m);assert.equal(original[7].h,rows[7].h);assert.notEqual(original[9].h,rows[9].h);
  }
});
test('the live consumer bridge includes the native register copy term',()=>{
  for(let row=0;row<64;row++){
    const h=postMLP(conditionalMixer(scan,row).output,scan.weights).output;
    const projected=linear(scan.weights['reg_out.weight'],scan.weights['reg_out.bias'],h);
    for(let reg=0;reg<8;reg++){
      const input=Number((BigInt(carry.before[reg])>>BigInt(row))&1n);
      assert.ok(Math.abs(projected[reg]+8*input-4-scan.nativeRegs[reg][row])<3e-5);
    }
  }
});
test('the four retained frontier cells preserve the interaction and all native exceptions',()=>{
  const cell=(h,c)=>frontier.cells.find(x=>x.h===h&&x.conv===c).logit;
  assert.ok(cell(0,0)<0&&cell(0,1)<0&&cell(1,0)<0&&cell(1,1)>0);
  assert.ok(Math.abs(cell(1,1)-cell(1,0)-cell(0,1)+cell(0,0)-3.432219505)<1e-6);
  assert.equal(frontier.exceptions.length,3);assert.equal(frontier.exactWords,93);
  frontier.exceptions.forEach(x=>assert.notEqual(x.program_bit,x.native_bit));
});
test('model packs accept a different topology and different clock dimensions',()=>{
  const fixture={schema:1,id:'fixture',kernel:'fixture-only',clocks:[{id:'step',count:3}],stateFields:['latent'],graph:{width:200,height:100,groups:[],nodes:[{id:'denoise'},{id:'schedule'}],edges:[['denoise','schedule']]}};
  assert.equal(validatePack(fixture),fixture);assert.equal(validatePack(isa),isa);
  assert.throws(()=>validatePack({...fixture,graph:{...fixture.graph,edges:[['missing','denoise']]}}));
});
test('the model host swaps family adapters and disposes the previous model',async()=>{
  const fixture={schema:1,id:'fixture',kernel:'fixture-only',adapter:'fixture',clocks:[{id:'step',count:3}],stateFields:['latent'],graph:{width:200,height:100,groups:[],nodes:[{id:'denoise'},{id:'schedule'}],edges:[['denoise','schedule']]}};
  const events=[];
  const host=new ModelHost([isa,fixture],async pack=>({mount(p,environment){assert.equal(pack.id,p.id);events.push([p.id,p.graph.nodes.length,p.clocks[0].count]);return{dispose(){events.push(['dispose',p.id]);}};}}));
  await host.select(isa.id);await host.select(fixture.id);
  assert.equal(host.current.id,'fixture');assert.deepEqual(events[1],['dispose',isa.id]);assert.deepEqual(events[2],['fixture',2,3]);host.dispose();
  await assert.rejects(()=>host.select('missing'));
});
test('core transfer stays below 1 MB compressed and evidence export has source hashes',()=>{
  const files=['index.html','style.css','app.js','isa-adapter.js','model-host.js','models.js','kernels.js','data/scan.json','data/carry.json','data/loop.json','data/frontier.json','data/readings.json'];
  const bytes=files.reduce((sum,n)=>sum+gzipSync(readFileSync(new URL(n.startsWith('data/')?`../public/${n}`:`../src/${n}`,import.meta.url))).length,0);
  assert.ok(bytes<1_000_000);console.log('Core gzip bytes:',bytes);
  const p=read('provenance');assert.equal(p.checkpoint,isa.checkpoint);assert.equal(p.newModelForwards,0);assert.equal(Object.keys(p.sources).length,14);
  for(const hash of Object.values(p.sources))assert.match(hash,/^[a-f0-9]{64}$/);
});
