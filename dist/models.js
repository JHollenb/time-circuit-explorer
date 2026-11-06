// Model packs own topology, clocks, state vocabulary, lessons, and kernel binding.
// The renderer traverses groups/nodes/edges; it does not assume transformer layers.
export const isa = {
  schema:1, id:'r3-mamba-isa', name:'R3 Mamba · ISA', family:'Selective state-space model',
  checkpoint:'9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e',
  kernel:'isa-r3', adapter:'./isa-adapter.js', clocks:[{id:'instruction',label:'Instruction',count:4},{id:'row',label:'Bit row',count:64}],
  stateFields:['register words','flags','byte offset','scan H','convolution history'],
  graph:{width:1080,height:324,
    groups:[{id:'blocks',x:425,y:38,w:424,h:234,label:'3 REPEATED BLOCKS · 64 ROWS × 48 FEATURES'}],
    nodes:[
      {id:'inputs',kind:'input',x:16,y:116,w:133,h:82,label:'Instruction + state',sub:'120 bits · 8 × 64 bits',math:'window, registers, flags',text:'The supplied interface contains a 15-byte instruction window, eight 64-bit register words, and four flag bits. These are inputs to the learned computation, not a supplied arithmetic answer.',dims:'120 + 512 + 4 input bits',equation:'z = decoder(2w − 1)',phase:0},
      {id:'decoder',kind:'mlp',x:184,y:62,w:171,h:68,label:'Instruction decoder',sub:'MLP · 120 → 256 → 256',math:'GELU · learned weights',text:'Two learned affine maps with GELU form the instruction context. The same parameters run on every call; their live operands depend on the current window.',dims:'120 → 256 → 256',equation:'z = GELU(W₁ GELU(W₀(2w−1)+b₀)+b₁)',phase:0},
      {id:'routing',kind:'route',x:184,y:170,w:171,h:68,label:'Route + form rows',sub:'Reads · positions · projection',math:'context selects operands',text:'Learned destination/source and immediate reads combine with register bits, position features, and decoder context to form 64 rows. The recovered route bits 16–18 predict register choice on the finite tested window family.',dims:'64 rows × 48 features',equation:'X = project(registers, reads, positions, z)',phase:1},
      ...[0,1,2].flatMap(i=>[
        {id:`scan${i}`,kind:'scan',x:445+i*137,y:92,w:119,h:76,label:`Mixer ${i+1}`,sub:'Conv → selective scan',math:'H: 96 × 4',text:`Mixer ${i+1} normalizes and projects its live row, applies a causal convolution, forms input-dependent factors, advances H, and reads through C and G. Each of the three mixers retains its own H and two-row convolution history within this call.`,dims:'48 → 96 channels · 4 modes',equation:'Hₜ = exp(dtₜ A) ⊙ Hₜ₋₁ + (dtₜ uₜ) ⊙ Bₜ',phase:3},
        {id:`post${i}`,kind:'mlp',x:445+i*137,y:190,w:119,h:54,label:`Post MLP ${i+1}`,sub:'48 → 96 → 48',math:'normalize · GELU · add',text:`Post MLP ${i+1} consumes its mixer output. Layer normalization and GELU make the finite response depend on the receiving state. The residual path adds the learned update to that state.`,dims:'LayerNorm · 48 → 96 → 48',equation:'Y = X + W₂ GELU(W₁ LN(X)+b₁)+b₂',phase:4}
      ]),
      {id:'heads',kind:'head',x:886,y:82,w:176,h:74,label:'Readout heads',sub:'Registers · immediate · store',math:'11 output heads in total',text:'Learned output projections emit register-bit logits, immediate and store values. Decoder-side heads also emit destination, source, width, length, syscall, and store enable. The register head adds an explicit copy of the input bits. Positive register logits decode to bit 1.',dims:'8 × 64 register logits + auxiliary outputs',equation:'register logits = 8 register_bits − 4 + Wreg Y + breg',phase:4},
      {id:'flags',kind:'mlp',x:886,y:192,w:176,h:60,label:'Flag MLP',sub:'500 → 128 → 4',math:'context + pooled rows',text:'The flag network combines decoder context, flag inputs, pooled hidden rows, and query reads. A learned MLP is followed by an explicit input-flag carry term. Compact flag rules remain finite, non-unique hypotheses.',dims:'500 → 128 → 4',equation:'flags = W₂ GELU(W₁v+b₁)+b₂ + 8 flagsin − 4',phase:5}
    ],
    edges:[['inputs','decoder'],['inputs','routing'],['decoder','routing'],['routing','scan0'],['scan0','post0'],['post0','scan1'],['scan1','post1'],['post1','scan2'],['scan2','post2'],['post2','heads'],['post2','flags'],['decoder','heads'],['decoder','flags']]
  },
  lessons:[
    {id:'architecture',title:'Meet the architecture',short:'The model',target:'architecture',node:'scan2',row:8,phase:3,kicker:'01 / THE WHOLE MACHINE',heading:'One instruction. Many small computations.',text:'Start with the physical model. The highlighted mixer carries information between bit positions; the post MLP changes what the next component receives. Select any component to open its computation.'},
    {id:'scan',title:'Unfold a time circuit',short:'The circuit',target:'computation',node:'scan2',row:8,phase:3,kicker:'02 / INSIDE A ROW',heading:'Keep something. Write something. Read it.',text:'A time circuit unfolds a context-dependent update through time. Move the bit clock: the same parameters meet new operands, retain part of the previous H, and write a new contribution.'},
    {id:'mlp',title:'Open the MLP',short:'The MLP',target:'computation',node:'post2',row:9,phase:4,kicker:'03 / THE RECEIVING STATE',heading:'The MLP responds to the state it receives.',text:'Watch 48 features expand to 96 GELU responses and contract to a 48-feature update. The residual then adds this update back. A changed input can change the response without changing any weight.'},
    {id:'machine',title:'Advance the state',short:'The state',target:'machine',node:'heads',row:63,phase:5,kicker:'04 / FROM ONE CALL TO THE NEXT',heading:'The output becomes the next input.',text:'The model and the recovered program each continue from their own outputs. Registers, flags, and the byte offset persist across instructions. The internal scan starts again from zero.'},
    {id:'frontier',title:'Find the missing context',short:'The boundary',target:'frontier',node:'scan2',row:31,phase:3,kicker:'05 / WHERE THE SIMPLE STORY BREAKS',heading:'What has to travel with the state?',text:'The compact two-state law predicts 93 of 96 held words. A retained exception leads to a richer frontier: recurrent state and convolution history interact. Toggle the two transplants to see why both matter.'}
  ],
  evidence:{carry:'./data/carry.json',scan:'./data/scan.json',loop:'./data/loop.json',frontier:'./data/frontier.json',provenance:'./data/provenance.json'},
};
export const registry = [isa];
export function validatePack(p) {
  if(p.schema!==1 || !p.id || !p.graph || !p.clocks?.length || !p.stateFields?.length || !p.kernel) throw new Error('Incomplete model pack.');
  const ids=new Set(p.graph.nodes.map(n=>n.id));
  if(ids.size!==p.graph.nodes.length || p.graph.edges.some(e=>!ids.has(e[0])||!ids.has(e[1]))) throw new Error('Invalid model graph.');
  return p;
}
