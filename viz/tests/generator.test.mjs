import test from 'node:test';
import assert from 'node:assert/strict';
import {summarizeSample,shellQuote} from '../src/generator-demo.js';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';

test('generator summaries keep prediction CPU cost distinct from elapsed time and memory',()=>{
  const sample={wallSeconds:12,stages:[{stage:'sample-00/encode',wallSeconds:2,cpuSeconds:1,peakRssBytes:100,peakCudaAllocatedBytes:90},
    {stage:'sample-00/prediction',wallSeconds:10,cpuSeconds:35,peakRssBytes:200,peakCudaAllocatedBytes:20}]};
  const summary=summarizeSample(sample);
  assert.deepEqual([summary.prediction,summary.cpu,summary.total,summary.ram,summary.cuda],[10,35,12,200,90]);
  assert.deepEqual(summary.stages.map(x=>x.key),['encode','prediction']);
});
test('copyable prompt command keeps quotes and shell expressions literal',()=>{
  assert.equal(shellQuote("a'b $(secret) `secret`"),"'a'\\''b $(secret) `secret`'");
});

test('published generator examples are complete measured executions with original PNG bytes',()=>{
  const root=new URL('../public/data/generator/',import.meta.url);
  const read=name=>JSON.parse(readFileSync(new URL(name,root)));
  const data=read('benchmarks.json'),provenance=read('provenance.json');
  const digest=name=>createHash('sha256').update(readFileSync(new URL(name,root))).digest('hex');
  assert.equal(data.samples.length,2);assert.equal(data.counts.predictedTransitions,224);
  assert.equal(data.counts.nativeDenoiser,0);assert.equal(data.nativeTransformerLoaded,false);
  assert.ok(data.jobWallSeconds>=data.workerWallSeconds);
  assert.equal(createHash('sha256').update(readFileSync(new URL('../../generator/worker.py',import.meta.url))).digest('hex'),provenance.sourceAudit.inputs['worker.py'].sha256);
  assert.equal(provenance.payloadCustody.executed,provenance.executedPayloadSeal);
  assert.equal(digest('worker-report.json'),provenance.originalReportSHA256);
  for(const [name,sha]of Object.entries(provenance.publicArtifactSHA256))assert.equal(digest(name),sha,name);
  const [baseline,fresh]=data.samples;
  assert.deepEqual(Object.values(baseline.baselineReproduction).filter(x=>typeof x==='boolean'),[true,true,true,true]);
  assert.notEqual(fresh.seed,baseline.seed);assert.notEqual(fresh.prompt,baseline.prompt);assert.equal(fresh.nativeComparison,null);
  for(const sample of data.samples){
    assert.equal(sample.transitions,112);assert.equal(digest(sample.image.path),sample.image.sha256);
    assert.equal(provenance.originalResultInventory.files[sample.image.path].sha256,sample.image.sha256);
    const summary=summarizeSample(sample);assert.ok(summary.prediction>0);assert.ok(summary.cpu>0);assert.ok(summary.total>summary.prediction);
    assert.ok(sample.parameterBytesRead>7e9);assert.ok(sample.stages.every(x=>x.wallSeconds>=0&&x.cpuSeconds>=0&&x.peakRssBytes>0));
  }
});
