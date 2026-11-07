"""Post-primary native A attention sham and swapped-V local B consumer."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import load, retain, payload
from value_innovation import IN, DOWN, WO


def execute(root):
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from flux_clock_study import verify, source_checks, assets_unchanged
    from complete_masked_study import profile
    from flux_clock import header, selected_load, snapshot
    from first_consumer import make_local, path
    from mdb.titan import Frame, ActSpec, ActRegistry, ProvidedFingerprint
    root = Path(root).resolve()
    request, s, targets = [read(root / n) for n in ['request.json', 'inputs.json', 'targets.json']]
    verify(root, request)
    out = root / 'response-result'
    seal = sha(out / 'primary-seal.json')
    primary = read(out / 'primary.json')
    if read(out / 'primary-seal.json')['primarySHA256'] != sha(out / 'primary.json'):
        raise ValueError('Native reference requires complete sealed candidate')
    start, cpu = time.perf_counter(), time.process_time()
    numerical = profile(torch, 4)
    if numerical != s['numericalProfile']:
        raise ValueError('Reference profile differs')
    model_root, assets = source_checks(s, diffusers)
    entries, head = header(model_root / 'transformer/diffusion_pytorch_model.safetensors')
    with torch.device('meta'):
        model = make_local(torch, native)
        del model.x_embedder
    loaded = selected_load(model, model_root / 'transformer/diffusion_pytorch_model.safetensors', sorted([WO, IN, DOWN]), entries, 'cuda')
    if {x['name']: x for x in loaded} != {x['name']: x for x in s['selectedParameters'] if x['name'] in [WO, IN, DOWN]}:
        raise ValueError('Reference three-matrix admission differs')
    model.eval().requires_grad_(False)
    block = model.transformer_blocks[0]
    def restore(item):
        x = load(item, 'cuda')
        stride = item.get('nativeStride', list(x.stride()))
        offset = item.get('nativeStorageOffset', 0)
        size = offset + 1 + sum((n - 1) * v for n, v in zip(x.shape, stride))
        backing = torch.empty(size, dtype=x.dtype, device=x.device)
        result = backing.as_strided(x.shape, stride, offset)
        result.copy_(x)
        if payload(result) != item['payloadSHA256']:
            raise ValueError('Reference layout restoration differs')
        return result
    acq = primary['acquisition']
    q, k, v, image = [restore(acq[n]) for n in ['Q', 'K', 'V', 'embedding']]
    mod = load(s['modImage'], 'cuda')
    parts = native.Flux2Modulation.split(mod, 2)
    vb = torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device=v.device)
    vb.copy_(v)
    j, kk = request['event']['keyIndices']
    vb[:, j], vb[:, kk] = v[:, kk], v[:, j]
    immutable = {**dict(model.named_parameters()), 'Q': q, 'K': k, 'V': v, 'swappedV': vb, 'embedding': image, 'modulation': mod}
    def census():
        return {n: (x.data_ptr(), x._version, payload(x)) for n, x in immutable.items()}
    before = census()
    rng_cpu, rng_cuda = torch.get_rng_state().clone(), [x.clone() for x in torch.cuda.get_rng_state_all()]
    names = {x.data_ptr(): n for n, x in model.named_parameters()}
    old_linear, old_attention = torch.nn.functional.linear, native.dispatch_attention_fn
    linears, attention = [], []
    references = {'A': {}, 'B': {}}
    phase, active = 'A-sham', references['A']
    def capture(name, x):
        if name in active:
            raise ValueError('Repeated native reference boundary')
        active[name] = retain(root / 'arrays/reference' / phase, name, x)
    def linear(x, w, bias=None):
        name = names.get(w.data_ptr())
        if name is None or bias is not None:
            raise ValueError('Reference affine outside selected path')
        linears.append({'phase': phase, 'parameter': name, 'inputShape': list(x.shape), 'inputStride': list(x.stride()), 'inputStorageOffset': x.storage_offset(), 'weightShape': list(w.shape),
                        'dtype': str(w.dtype), 'device': w.device.type, 'nominalMACs': x.numel() // x.shape[-1] * w.numel()})
        return old_linear(x, w, bias)
    def attend(xq, xk, xv, **kwargs):
        y = old_attention(xq, xk, xv, **kwargs)
        attention.append({'phase': phase, 'queryShape': list(xq.shape), 'keyShape': list(xk.shape), 'valueShape': list(xv.shape), 'queryStride': list(xq.stride()), 'keyStride': list(xk.stride()), 'valueStride': list(xv.stride()),
                          'requestedBackend': 'auto', 'outputPayloadSHA256': payload(y), 'nominalQKAVMACs': 9663676416})
        return y
    def carrier(xv):
        h = native.dispatch_attention_fn(q, k, xv, attn_mask=None, backend=None, parallel_config=None).flatten(2, 3).to(q.dtype)
        capture('H0', h)
        geometry = s['carrierGeometry']
        size = geometry['storageOffset'] + 1 + sum((n - 1) * v for n, v in zip(geometry['shape'], geometry['stride']))
        backing = torch.empty(size, dtype=h.dtype, device=h.device)
        layout = backing.as_strided(geometry['shape'], geometry['stride'], geometry['storageOffset'])
        layout.copy_(h)
        projection = block.attn.to_out[1](block.attn.to_out[0](layout))
        write = parts[0][2] * projection
        capture('projection', projection)
        capture('gatedCarrier', write)
        return write
    hooks = [block.ff.linear_in.register_forward_hook(lambda m, args, y: capture('fused', y)),
             block.ff.act_fn.gate_fn.register_forward_hook(lambda m, args, y: capture('activation', y)),
             block.ff.act_fn.register_forward_hook(lambda m, args, y: capture('product', y))]
    registry = ActRegistry()
    spec = ActSpec('tce.swapped_value_native_reference', 'v1', reads=('Q', 'K', 'V', 'embedding', 'modulation'), writes=('imageExit',),
                   write_kinds={'imageExit': 'tensor'}, write_abis={'imageExit': {'shape': [1, 1024, 3072], 'dtype': 'torch.bfloat16'}}, numerical_contract='post-seal-source-native-attention-output-projection-local-MLP')
    def operation(values):
        write = carrier(values['V'])
        result = path(block, values['embedding'], write, parts, capture)
        return {'imageExit': result}, {'capturedImageExitPayloadSHA256': payload(result), 'primarySealSHA256': seal}
    registry.register(spec, 'post-seal-local-B', operation)
    torch.cuda.reset_peak_memory_stats()
    try:
        torch.nn.functional.linear, native.dispatch_attention_fn = linear, attend
        with torch.inference_mode():
            a_write = carrier(v)
            a_residual = image + a_write
            capture('residualCommit', a_residual)
            sham = {
                'H0Exact': references['A']['H0']['payloadSHA256'] == s['expectedH0APayloadSHA256'],
                'carrierExact': references['A']['gatedCarrier']['payloadSHA256'] == s['carrierA']['payloadSHA256'],
                'residualExact': references['A']['residualCommit']['payloadSHA256'] == targets['A']['residualCommit']['payloadSHA256'],
                'embeddingExact': payload(image) == s['expectedEmbeddingPayloadSHA256'],
                'modulationExact': payload(mod) == s['modImage']['payloadSHA256'],
                'localParameterSourceProfileExact': True,
            }
            inherited = all(sham.values())
            if inherited:
                references['A'].update({n: x for n, x in targets['A'].items() if n not in references['A']})
            else:
                phase = 'A-new-local'
                # Reuse the actual sham carrier/residual; no extra attention or Wo.
                del references['A']['residualCommit']
                path(block, image, a_write, parts, capture)
            phase, active = 'B-new-local', references['B']
            frame_values = {'Q': q, 'K': k, 'V': vb, 'embedding': image, 'modulation': mod}
            frame = Frame.from_values(frame_values, kinds={n: 'tensor' for n in frame_values}, abis={n: {'shape': list(x.shape), 'dtype': str(x.dtype), 'stride': list(x.stride())} for n, x in frame_values.items()},
                                      slot_fingerprints={n: ProvidedFingerprint.content_sha256(payload(x)) for n, x in frame_values.items()}, program_fingerprint=request['requestSHA256'], metadata={'primarySealSHA256': seal, 'postSeal': True})
            result, evidence = registry.run(spec.identity, 'post-seal-local-B', frame)
            binding = {'sourceFrame': frame.to_dict(), 'outputFrame': result.to_dict(), 'evidence': evidence.to_dict()}
            if payload(result.values['imageExit']) != active['imageExit']['payloadSHA256']:
                raise ValueError('Native typed output differs')
    finally:
        torch.nn.functional.linear, native.dispatch_attention_fn = old_linear, old_attention
        for hook in hooks:
            hook.remove()
    torch.cuda.synchronize()
    if census() != before or snapshot(torch) != numerical:
        raise ValueError('Native reference immutable closure differs')
    if not torch.equal(rng_cpu, torch.get_rng_state()) or not all(torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all())):
        raise ValueError('Reference RNG changed')
    total = sum(x['nominalMACs'] for x in linears) + sum(x['nominalQKAVMACs'] for x in attention)
    if len(attention) != 2 or len(linears) != (4 if inherited else 6) or total != 125627793408 + (0 if inherited else 86973087744):
        raise ValueError('Native reference physical work differs')
    if sha(out / 'primary-seal.json') != seal:
        raise ValueError('Primary seal changed during reference')
    assets_unchanged(assets)
    save(out / 'reference.json', {'schema': 'tce-value-innovation-reference-v1', 'fields': references, 'sham': sham, 'inheritedA': inherited, 'frame': binding, 'primarySealSHA256': seal})
    save(out / 'reference-seal.json', {'referenceSHA256': sha(out / 'reference.json'), 'primarySealSHA256': seal, 'requestSHA256': request['requestSHA256']})
    save(out / 'reference-execution.json', {'loadedParameters': loaded, 'linears': linears, 'attention': attention, 'nominalReferenceMACs': total, 'parameterInputRNGClosure': True, 'numericalProfile': numerical, 'header': head,
                                           'workerWallSecondsAfterImports': time.perf_counter() - start, 'workerCPUSecondsAfterImports': time.process_time() - cpu, 'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'peakCUDAReservedBytes': torch.cuda.max_memory_reserved(),
                                           'scope': 'New native A attention sham and B local reference only, post-candidate seal/process exit. Conditional extra A MLP is counted if sham does not qualify inherited outputs.'})
    print('VALUE_INNOVATION_REFERENCE_SEALED', sha(out / 'reference-seal.json'), 'inheritedA', inherited, flush=True)


if __name__ == '__main__':
    execute(sys.argv[1])
