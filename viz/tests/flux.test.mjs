import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,readdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {gzipSync} from 'node:zlib';
import {bf16,euler,address,boundaryRecord,softmax,toyAttention,toyMLP} from '../src/flux-kernels.js';
import {flux,isa,registry,validatePack} from '../src/models.js';
import {ModelHost} from '../src/model-host.js';

const root=new URL('../public/data/flux-klein/',import.meta.url);
const read=name=>JSON.parse(readFileSync(new URL(name+'.json',root)));
const manifest=read('manifest'),timeline=read('timeline'),fixtures=read('euler'),endpoints=read('endpoints'),provenance=read('provenance');

test('FLUX pack binds the accepted specimen and a distinct topology and clocks',()=>{
  assert.equal(validatePack(flux),flux);assert.deepEqual(registry.map(p=>p.id),[isa.id,flux.id]);
  assert.equal(manifest.auditSHA256,'314fe2d2c63aedf1392b87f0c599a417bfc241e54fac8a9f127d868c834f3115');
  assert.deepEqual(manifest.geometry,{latent:[1,1024,128],conditioning:[1,512,7680],image:[1,1024,3072],text:[1,512,3072],heads:24,headDim:128,mlpWidth:9216,ropeAxes:[32,32,32,32]});
  assert.equal(manifest.seed,113);assert.equal(manifest.passes,4);assert.equal(provenance.newModelForwards,0);
});
test('all 112 addresses and before/after boundaries join the correct source clocks',()=>{
  for(const [mode,h]of Object.entries(timeline.histories)){
    assert.equal(h.origin,mode==='full'?0:56);assert.equal(h.records.length,112-h.origin);
    for(const r of h.records){const a=address(r.cursor);assert.equal(a.pass,r.pass);assert.equal(a.local,r.local);assert.equal(a.kind,r.kind);assert.equal(a.block,r.block);assert.equal(boundaryRecord(h,r.cursor,'after'),r);assert.equal(boundaryRecord(h,r.cursor,'before'),r.cursor===h.origin?null:h.records[r.cursor-h.origin-1]);}
  }
  assert.equal(timeline.histories.full.records[55].kind,'commit');assert.equal(timeline.histories.full.records[111].kind,'commit');
  assert.equal(address(55).pass,1);assert.equal(address(111).pass,3);assert.throws(()=>address(112));
  assert.equal(boundaryRecord(timeline.histories.matched,55,'after'),null);
});
test('working fields have source-qualified lifetimes and commits clear temporaries',()=>{
  for(const h of Object.values(timeline.histories))for(const r of h.records){
    assert.ok(r.fields.latent);assert.equal(Boolean(r.fields.context),r.local<27);assert.equal(Boolean(r.fields.noise),r.local===26);
    assert.equal(Boolean(r.fields.image),r.local<27);assert.equal(Boolean(r.fields.text),r.local<27);
    if(r.local===26){assert.equal(r.fields.image.native.sourceCursor,r.cursor-1);assert.equal(r.fields.text.predicted.sourceCursor,r.cursor-1);}
    for(const [field,lanes]of Object.entries(r.fields))if(field!=='context'){
      assert.equal(lanes.native.measurement,'coordinate-sample');assert.equal(lanes.predicted.measurement,'coordinate-sample');
      assert.equal(lanes.error.sample_count,lanes.native.shape.reduce((a,b)=>a*b,1));
      assert.deepEqual(lanes.native.shape,lanes.predicted.shape);assert.deepEqual(lanes.native.rows,lanes.predicted.rows);assert.deepEqual(lanes.native.features,lanes.predicted.features);
      for(const sample of [lanes.native,lanes.predicted]){assert.ok(provenance.sourceFiles[sample.source.path]);if(sample.source.offset!==undefined)assert.ok(provenance.forecastLeaves[sample.source.sha256]);else assert.equal(provenance.sourceFiles[sample.source.path],sample.source.sha256);}
    }
  }
});
test('each forecast pass consumes its own preceding latent, with native-only matched origin',()=>{
  for(const [mode,h]of Object.entries(timeline.histories))for(let index=0;index<h.records.length;index+=28){
    const entry=h.records[index].fields.latent,previous=index===0?(mode==='full'?manifest.initial.sample:timeline.histories.full.records[55].fields.latent.native):h.records[index-1].fields.latent.predicted;
    assert.deepEqual(entry.predicted.values,previous.values);
    for(let j=0;j<27;j++)assert.deepEqual(h.records[index+j].fields.latent.predicted.values,entry.predicted.values);
  }
  assert.notDeepEqual(timeline.histories.full.records[55].fields.latent.predicted.values,timeline.histories.matched.records[0].fields.latent.predicted.values);
});
test('BF16 rounds ties to even and preserves signed zero, infinities, and subnormals',()=>{
  assert.equal(bf16(1+1/256),1);assert.equal(bf16(1+3/256),1+2/128);assert.equal(bf16(-1-1/256),-1);
  assert.ok(Object.is(bf16(-0),-0));assert.ok(Object.is(bf16(-(2**-140)),-0));assert.equal(bf16(2**-133),2**-133);
  assert.equal(bf16(Infinity),Infinity);assert.equal(bf16(-Infinity),-Infinity);assert.ok(Number.isNaN(bf16(NaN)));
});
test('all 192 retained native/forecast Euler coordinates reproduce exact recorded results',()=>{
  let count=0;
  for(const [mode,passes]of Object.entries(fixtures.histories))for(const pass of passes)for(const c of pass.coordinates){const result=euler(c.z,c.v,pass.sigma,pass.sigmaNext);assert.equal(result.dt,pass.dtBF16);assert.equal(result.next,c.expected);const lane=c.lane==='native'?'native':'predicted',records=timeline.histories[mode].records,readout=records.find(r=>r.cursor===pass.pass*28+26),commit=records.find(r=>r.cursor===pass.pass*28+27);const value=field=>field.values[field.rows.indexOf(c.row)][field.features.indexOf(c.feature)];assert.equal(value(readout.fields.latent[lane]),c.z);assert.equal(value(readout.fields.noise[lane]),c.v);assert.equal(value(commit.fields.latent[lane]),c.expected);count++;}
  assert.equal(count,192);
});
test('hostile scalar evidence distinguishes BF16 and FP32 interval promotion',()=>{
  for(const h of fixtures.hostile){assert.equal(bf16(Math.fround(h.dt_fp32)),h.dt_bf16);assert.equal(bf16(Math.fround(h.dt_bf16*h.edge.noise)),h.edge.native);assert.equal(bf16(Math.fround(h.dt_fp32*h.edge.noise)),h.edge.fp32_dt);assert.notEqual(h.edge.native,h.edge.fp32_dt);assert.equal(h.native_vs_bf16_dt_mismatches,0);}
});
test('toy attention normalizes stably; keys control weights and values control the read',()=>{
  assert.deepEqual(softmax([10000,10000]),[.5,.5]);
  for(let query=0;query<6;query++)for(const bias of [-5,0,5]){const a=toyAttention(query,bias,0),b=toyAttention(query,bias,1);assert.ok(Math.abs(a.weights.reduce((s,x)=>s+x,0)-1)<1e-14);assert.deepEqual(a.weights,b.weights);assert.ok(Math.abs(b.output[0]-a.output[0]-a.weights[4])<1e-14);assert.deepEqual(a.output.slice(1),b.output.slice(1));}
  assert.ok(toyAttention(0,5).weights[4]>toyAttention(0,-5).weights[4]);
});
test('SiLU product expansion equals the receiving-state effective matrix',()=>{
  for(const input of [-2,0,2])for(const gate of [-2,0,2]){const m=toyMLP(input,gate);m.output.forEach((v,i)=>assert.ok(Math.abs(v-m.factored[i])<1e-14));}
  assert.notDeepEqual(toyMLP(0).effective,toyMLP(1).effective);
});
test('latent tiles use all 1024 recorded positions and the same fixed paired scale',()=>{
  assert.equal(new Set(manifest.idGrid.map(x=>x.join(','))).size,1024);assert.ok(manifest.idGrid.every(([y,x])=>x>=0&&x<32&&y>=0&&y<32));
  for(const mode of ['full','matched'])for(let pass=mode==='full'?0:2;pass<4;pass++){
    const tile=read(`tiles/${mode}-${pass}`);assert.equal(tile.pass,pass);
    for(const channels of Object.values(tile.tiles)){assert.equal(channels.length,4);channels.forEach(values=>{assert.equal(values.length,1024);assert.ok(values.every(x=>Math.abs(x)<=manifest.latentScale));});}
  }
});
test('complete RGB histograms retain the rare tail and reproduce endpoint measurements',()=>{
  for(const [mode,max]of [['full',42],['matched',10]]){const key=mode+'.predicted',counts=endpoints.distributions[key].absolute_channel_error_counts,metric=endpoints.comparisons[key].rgb;assert.equal(counts.findLastIndex(x=>x>0),max);assert.equal(counts.reduce((a,b)=>a+b,0),786432);assert.equal(counts.reduce((a,b,i)=>a+b*i,0)/786432,metric.mae);assert.equal(metric.max_abs,max);assert.ok(counts[0]>0);assert.ok(counts[max]>0);}
  assert.equal(endpoints.comparisons.sham.rgb.max_abs,0);assert.equal(endpoints.independent.scalarValues,740327424);
});
test('every generated artifact, including byte-copied original PNGs, matches its manifest digest',()=>{
  for(const [path,digest]of Object.entries(provenance.artifactHashes))assert.equal(createHash('sha256').update(readFileSync(new URL(path,root))).digest('hex'),digest,path);
  assert.ok(Object.keys(provenance.sourceFiles).length>=236);assert.equal(Object.keys(provenance.forecastLeaves).length,199);
});
test('host restores separate model sessions and cancels stale asynchronous mounts',async()=>{
  const events=[],load=async pack=>({async mount(p,env){events.push('mount:'+p.id);return{getState:()=>({cursor:p.id===isa.id?8:55}),dispose:()=>events.push('dispose:'+p.id)};}});
  const host=new ModelHost([isa,flux],load);await host.select(isa.id);await host.select(flux.id);await host.select(isa.id);assert.deepEqual(host.sessions.get(isa.id),{cursor:8});assert.deepEqual(host.sessions.get(flux.id),{cursor:55});host.dispose();
  let started,release;const begun=new Promise(r=>started=r),pending=new Promise(r=>release=r);
  const racing=new ModelHost([isa,flux],async()=>({async mount(p,env){if(p.id===isa.id){started();await pending;env.signal.throwIfAborted();}return{getState:()=>({}),dispose(){}};}}));
  const stale=racing.select(isa.id);await begun;const fresh=racing.select(flux.id);release();await Promise.all([stale,fresh]);assert.equal(racing.current.id,flux.id);racing.dispose();
});
test('FLUX initial assets and every lazy detail stay inside transfer budgets',()=>{
  const src=new URL('../src/',import.meta.url),initial=['index.html','app.js','models.js','flux-model.js','model-host.js','components.js','kernels.js','style.css','flux.css','flux-adapter.js','flux-kernels.js','templates/flux.html','generator-demo.js','generator.css'];
  const generator=new URL('../public/data/generator/',import.meta.url),example=JSON.parse(readFileSync(new URL('benchmarks.json',generator)));
  const generatorBytes=gzipSync(readFileSync(new URL('benchmarks.json',generator))).length+(example.samples[0]?readFileSync(new URL(example.samples[0].image.path,generator)).length:0);
  const bytes=initial.reduce((s,p)=>s+gzipSync(readFileSync(new URL(p,src))).length,0)+['manifest','timeline'].reduce((s,p)=>s+gzipSync(readFileSync(new URL(p+'.json',root))).length,0)+generatorBytes;
  console.log('FLUX initial gzip bytes:',bytes);assert.ok(bytes<1_000_000);
  const paths=['provenance.json','endpoints.json','euler.json',...readdirSync(new URL('tiles/',root)).map(p=>'tiles/'+p)];for(const p of paths)assert.ok(gzipSync(readFileSync(new URL(p,root))).length<2_000_000,p);
});
