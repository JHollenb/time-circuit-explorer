// Matches the measured NumPy float32 order, including nearest-even uint8 rounding.
export function responseWeights(alpha){
  if(!Number.isFinite(alpha)||alpha<0||alpha>1)throw new RangeError('Dose must be between 0 and 1.');
  return (alpha<=.25?[1-4*alpha,4*alpha,0]:[0,(1-alpha)/.75,(alpha-.25)/.75]).map(Math.fround);
}
export function roundEven(value){
  const v=Math.max(0,Math.min(255,value)),lo=Math.floor(v),fraction=v-lo;
  return fraction===.5?(lo%2?lo+1:lo):Math.round(v);
}
export function renderResponse(anchors,alpha,rgba){
  const channels=rgba.length/4*3;
  if(anchors.length!==channels*3)throw new RangeError('Three RGB anchors required.');
  const [a,b,c]=responseWeights(alpha),f=Math.fround;
  for(let i=0,p=0;i<channels;i++,p++){
    if(i&&i%3===0){rgba[p++]=255;}
    const sum=f(f(f(a*anchors[i])+f(b*anchors[i+channels]))+f(c*anchors[i+2*channels]));
    rgba[p]=roundEven(sum);
  }
  rgba[rgba.length-1]=255;return rgba;
}
