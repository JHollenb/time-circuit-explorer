import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {renderResponse,responseWeights,roundEven} from '../src/response-kernel.js';

const root=new URL('../public/data/response/',import.meta.url);
const read=name=>JSON.parse(readFileSync(new URL(name,root)));
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
test('browser primary pixels reproduce all 16 original sealed NumPy outputs',()=>{
  const data=read('response.json');
  for(const context of data.contexts){
    const compressed=readFileSync(new URL(context.path,root)),anchors=gunzipSync(compressed);
    assert.equal(digest(compressed),context.sha256);assert.equal(digest(anchors),context.rawSHA256);
    const rgba=new Uint8ClampedArray(data.width*data.height*4),rgb=new Uint8Array(data.width*data.height*3);
    for(const row of data.expectedPrimaryPixels.filter(x=>x.context===context.id)){
      renderResponse(anchors,row.alpha,rgba);
      for(let p=0,i=0;p<rgba.length;p+=4){rgb[i++]=rgba[p];rgb[i++]=rgba[p+1];rgb[i++]=rgba[p+2];assert.equal(rgba[p+3],255);}
      assert.equal(digest(rgb),row.rgbSHA256,`${row.context} @ ${row.alpha}`);
    }
  }
});
test('public browser evidence retains original source identity, costs and held outcomes',()=>{
  const data=read('response.json'),report=read('report.json'),provenance=read('provenance.json');
  assert.equal(digest(readFileSync(new URL('report.json',root))),data.reportSHA256);
  assert.equal(data.requestSHA256,report.requestSHA256);assert.equal(data.held.length,12);
  assert.equal(data.contexts.length,4);assert.equal(data.expectedPrimaryPixels.length,16);
  assert.equal(data.timing.primaryImages,16);assert.equal(data.physicalWork.newDenoiserForwards,0);assert.equal(data.physicalWork.newVAEForwards,0);
  assert.equal(data.sourceCost.vaeImages,24);assert.ok(data.serving.wallSeconds>0);
  assert.equal(provenance.payloadCustody.declared.sha256,provenance.payloadCustody.executed.sha256);
  for(const row of data.held)assert.equal(digest(readFileSync(new URL(row.path,root))),row.sha256);
  for(const [name,hash] of Object.entries(read('files.json')))assert.equal(digest(readFileSync(new URL(name,root))),hash,name);
  assert.ok(data.groups.filter(x=>x.method==='transfer').every(x=>x.meanInteractionRelativeRms>1));
});
test('browser rounding preserves ties to even and disallows out-of-domain doses',()=>{
  assert.deepEqual([.5,1.5,2.5,3.5,-1,256].map(roundEven),[0,2,2,4,0,255]);
  for(const x of [-.01,1.01,NaN,Infinity])assert.throws(()=>responseWeights(x),RangeError);
});
