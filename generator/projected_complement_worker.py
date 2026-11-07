"""Distinct CPU compiler and target-denied projected-cache actor processes."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import load, retain, payload
from sparse_text_kv import PREFIX, normalize, project, headnorm, heads, quarantine
from projected_complement import WO, ACTOR_PARAMETERS, project_cache, projected_read, load_selected, fields


def execute(phase, root):
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from complete_masked_study import profile
    from flux_clock_study import verify, source_checks, assets_unchanged
    from flux_clock import header, snapshot
    root = Path(root).resolve()
    r, s = read(root / 'request.json'), read(root / 'inputs.json')
    verify(root, r)
    out = root / 'response-result'
    out.mkdir(exist_ok=phase != 'compile')
    compiled = None if phase == 'compile' else read(out / 'compiler.json')
    if compiled is not None:
        seal = read(out / 'compiler-seal.json')
        if seal['compilerSHA256'] != sha(out / 'compiler.json') or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Complete projected compiler release differs')
    denied = [str(root / 'targets.json'), s['parentPrimaryJSON'], s['parentPrimarySealJSON'], *s['deniedArrays']]
    assets_allowed = [str(Path(s['modelRoot']) / n) for n in s['imageAssets']]
    if phase == 'compile':
        allowed = [s['N']['path'], s['parentCompiler']['path'], s['parentCompilerSeal']['path'], *assets_allowed]
    else:
        allowed = [x['path'] for x in fields(compiled['actorInputs'])] + assets_allowed
        denied += [s['N']['path']]
    accesses = quarantine(root, allowed, denied)
    started, cpu = time.perf_counter(), time.process_time()
    numerical = profile(torch, 4)
    if numerical != s['numericalProfile']:
        raise ValueError('Original numerical profile differs')
    model, assets = source_checks(s, diffusers)
    checkpoint = model / 'transformer/diffusion_pytorch_model.safetensors'
    entries, header_record = header(checkpoint)
    names = [WO] if phase == 'compile' else ACTOR_PARAMETERS
    weights, loaded = load_selected(checkpoint, names, entries)
    original = {x['source']['tensor']: x['source'] for x in s['originalParameterArtifacts']}
    for item in loaded:
        e = original[item['name']]
        if item['payloadSHA256'] != e['payload_sha256'] or item['shape'] != e['shape'] or item['dataOffsets'] != e['data_offsets']:
            raise ValueError('Original selected retrieval identity differs')
    wo = weights[WO].double()
    initial = {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()}
    wo_before = (wo.data_ptr(), wo._version, payload(wo))
    rng = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if phase == 'actors' else None
    if phase == 'actors':
        torch.cuda.reset_peak_memory_stats()
    linears, contractions, timings = [], [], []
    old_linear = torch.nn.functional.linear
    arm = None
    def linear(x, w, bias=None):
        extra = {}
        if w.untyped_storage().data_ptr() == wo.untyped_storage().data_ptr():
            name = WO
            extra = {'head': w.storage_offset() // 128, 'storageOffset': w.storage_offset(), 'physicalColumnSlice': True}
            if tuple(w.shape) != (3072, 128):
                raise ValueError('Expected physical Wo head block')
        else:
            matches = [n for n, v in weights.items() if v is w]
            if len(matches) != 1 or matches[0] not in [PREFIX + 'add_k_proj.weight', PREFIX + 'add_v_proj.weight']:
                raise ValueError('Undeclared learned projection')
            name = matches[0]
        if bias is not None:
            raise ValueError('Unexpected affine bias')
        linears.append({'phase': phase, 'arm': arm, 'parameter': name, 'inputShape': list(x.shape), 'inputStride': list(x.stride()),
                        'weightShape': list(w.shape), 'weightStride': list(w.stride()), 'dtype': str(x.dtype), 'device': x.device.type,
                        'nominalMACs': x.numel() // x.shape[-1] * w.numel(), **extra})
        return old_linear(x, w, bias)
    def keep(group, name, x):
        return retain(root / 'arrays' / phase / group, name, x)
    result = {'actors': {}} if phase == 'actors' else {}
    try:
        torch.nn.functional.linear = linear
        with torch.inference_mode():
            if phase == 'compile':
                for record in [s['parentCompiler'], s['parentCompilerSeal']]:
                    if sha(record['path']) != record['sha256']:
                        raise ValueError('Original compiler bytes changed')
                parent, seal = read(s['parentCompiler']['path']), read(s['parentCompilerSeal']['path'])
                if seal['compilerSHA256'] != s['parentCompiler']['sha256'] or parent['actorInputs']['n'] != s['N']:
                    raise ValueError('Original N-to-compiler seal join differs')
                n = load(s['N'], 'cpu')
                n_before = payload(n)
                tick = time.perf_counter()
                nt = project_cache(n, wo, linear)
                timings.append({'phase': 'project-complement-only', 'wallSeconds': time.perf_counter() - tick})
                result.update(actorInputs={**s['actorInputs'], 'projectedN': keep('admitted', 'projectedN', nt)}, sourceN=s['N'],
                    parentCompilerSHA256=s['parentCompiler']['sha256'], parentCompilerSealSHA256=s['parentCompilerSeal']['sha256'],
                    projectedCacheLogicalBytes=nt.numel() * 8 + 2 * 24 * 1024 * 8)
                if payload(n) != n_before or result['projectedCacheLogicalBytes'] != 604372992:
                    raise ValueError('Compiler input/cache geometry changed')
            else:
                from mdb.titan import Frame, ActSpec, ActRegistry, ProvidedFingerprint
                values = {n: load(x, 'cpu' if n in ['projectedN', 'm', 'z'] else 'cuda') for n, x in compiled['actorInputs'].items()}
                input_before = {n: payload(x) for n, x in values.items()}
                registry = ActRegistry()
                reads = ['Q', 'm', 'z', 'projectedN', 'image', 'gate', 'txtMod', 'ropeCos', 'ropeSin', 'textRows', 'Kweight', 'Vweight', 'KnormWeight', 'WoFP64']
                spec = ActSpec('tce.projected_complement', 'v1', reads=tuple(reads), writes=('R_attention',),
                    write_kinds={'R_attention': 'tensor'}, write_abis={'R_attention': {'shape': [1, 1024, 3072], 'dtype': 'torch.bfloat16'}},
                    numerical_contract='own-fourrow-native-KV-positive-projected-FP64-complement-fixed-headsum-native-gate-residual')
                active = {}
                def operation(v):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    xt, _ = normalize(v['textRows'], v['txtMod'])
                    k = headnorm(weights, 'norm_added_k', project(weights, 'add_k_proj', xt, [],).unflatten(-1, (24, 128)))
                    k = native.apply_rotary_emb(k, (v['ropeCos'], v['ropeSin']), sequence_dim=1)
                    value = project(weights, 'add_v_proj', xt, []).unflatten(-1, (24, 128))
                    account = projected_read(heads(v['Q']), heads(k), heads(value), v['m'], v['z'], v['projectedN'], v['WoFP64'], linear)
                    contractions.extend([{'arm': arm, 'kind': 'selected-QK', 'nominalMACs': 24 * 1024 * 4 * 128},
                                         {'arm': arm, 'kind': 'selected-projected-AV', 'nominalMACs': 24 * 1024 * 4 * 3072}])
                    output = account['projectionFP64'].reshape(1, 1024, 3072).float().bfloat16().to('cuda')
                    carrier = v['gate'] * output
                    residual = v['image'] + carrier
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - tick
                    timings.append({'phase': 'actor-arithmetic-only-before-capture', 'arm': arm, 'wallSeconds': elapsed})
                    captures = {'K': k, 'V': value, **account, 'attentionReturn': output, 'carrier': carrier, 'R_attention': residual}
                    active.update({n: keep(arm, n, x) for n, x in captures.items()})
                    return {'R_attention': residual}, {'capturedPayloadSHA256': payload(residual), 'arithmeticWallSeconds': elapsed,
                                                      'projectedNumeratorBytes': v['projectedN'].numel() * 8, 'intermediateHeadRounding': False}
                registry.register(spec, 'projected-positive-fixed-query-read', operation)
                for arm in ['QQQ', 'PQP']:
                    active = {}
                    fv = {n: values[n] for n in reads if n in values}
                    fv.update(textRows=values[arm + '.rows'], Kweight=weights[PREFIX + 'add_k_proj.weight'], Vweight=weights[PREFIX + 'add_v_proj.weight'],
                              KnormWeight=weights[PREFIX + 'norm_added_k.weight'], WoFP64=wo)
                    frame = Frame.from_values(fv, kinds={n: 'tensor' for n in fv},
                        abis={n: {'shape': list(x.shape), 'dtype': str(x.dtype), 'stride': list(x.stride())} for n, x in fv.items()},
                        slot_fingerprints={n: ProvidedFingerprint.content_sha256(payload(x)) for n, x in fv.items()}, program_fingerprint=r['requestSHA256'],
                        metadata={'arm': arm, 'support': r['support'], 'familyPoint': r['familyPoint']})
                    after, evidence = registry.run(spec.identity, 'projected-positive-fixed-query-read', frame)
                    if payload(after.values['R_attention']) != active['R_attention']['payloadSHA256']:
                        raise ValueError('Actual typed output differs')
                    result['actors'][arm] = {'fields': active, 'sourceFrame': frame.to_dict(), 'outputFrame': after.to_dict(), 'evidence': evidence.to_dict()}
                if input_before != {n: payload(x) for n, x in values.items()}:
                    raise ValueError('Actor admitted input changed')
                result['compilerSealSHA256'] = sha(out / 'compiler-seal.json')
    finally:
        torch.nn.functional.linear = old_linear
    if initial != {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()} or wo_before != (wo.data_ptr(), wo._version, payload(wo)) or snapshot(torch) != numerical:
        raise ValueError('Selected parameter/profile closure differs')
    if not torch.equal(rng, torch.get_rng_state()) or (rng_cuda is not None and not all(torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all()))):
        raise ValueError('RNG changed')
    assets_unchanged(assets)
    total = sum(x['nominalMACs'] for x in linears + contractions)
    if total != (9663676416 if phase == 'compile' else 855638016) or len(linears) != (24 if phase == 'compile' else 52):
        raise ValueError('Actual physical projection/contraction ledger differs')
    name = 'compiler' if phase == 'compile' else 'primary'
    save(out / (name + '.json'), result)
    save(out / (name + '-seal.json'), {'requestSHA256': r['requestSHA256'], name + 'SHA256': sha(out / (name + '.json'))})
    save(out / (name + '-execution.json'), {'loadedParameters': loaded, 'linears': linears, 'contractions': contractions, 'nominalMatrixMACs': total,
        'scalarWorkPerActor': None if phase == 'compile' else {'cacheMultiplyAddPairs': 75497472, 'perHeadDivisions': 75497472, 'interHeadAdditions': 72351744},
        'numericalProfile': numerical, 'header': header_record, 'assets': assets, 'parameterInputRNGClosure': True,
        'WoFP64': {'shape': list(wo.shape), 'payloadSHA256': wo_before[2], 'bytes': wo.numel() * 8, 'sourceBF16SHA256': initial[WO][2]},
        'wallSecondsAfterImports': time.perf_counter() - started, 'CPUSecondsAfterImports': time.process_time() - cpu, 'arithmeticTimings': timings,
        'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated() if phase == 'actors' else 0,
        'peakCUDAReservedBytes': torch.cuda.max_memory_reserved() if phase == 'actors' else 0,
        'fileQuarantine': {'allowedFiles': sorted(allowed), 'deniedCanaries': denied, 'successfulFileOpenCounts': accesses,
                           'scope': 'Trusted-code read allowlist; old N and all target/old-head fields unavailable to actors.'}})
    print('PROJECTED_COMPLEMENT_' + name.upper() + '_SEALED', sha(out / (name + '-seal.json')), flush=True)


if __name__ == '__main__':
    execute(sys.argv[1], sys.argv[2])
