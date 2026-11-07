"""Separate paid cache compilation and target-denied four-row serving processes."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import load, retain, payload
from sparse_text_kv import (PARAMETERS, ACTOR_PARAMETERS, PREFIX, resolve, rows, complement, update,
                            load_weights, normalize, project, headnorm, heads, quarantine)


def fields(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield value
        else:
            for child in value.values():
                yield from fields(child)
    elif isinstance(value, list):
        for child in value:
            yield from fields(child)


def execute(phase, root):
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from complete_masked_study import profile
    from flux_clock_study import verify, source_checks, assets_unchanged
    from flux_clock import header, snapshot
    root = Path(root).resolve()
    request, s, meta = [read(root / n) for n in ['request.json', 'inputs.json', 'family-metadata.json']]
    verify(root, request)
    out = root / 'response-result'
    out.mkdir(exist_ok=phase != 'compile')
    denied = [str(root / 'targets.json'), *s['deniedTargets']]
    common = [str(Path(s['modelRoot']) / n) for n in s['imageAssets']]
    compiler = read(out / 'compiler.json') if phase == 'actors' else None
    if phase == 'compile':
        allowed = [x['path'] for x in fields(s['writerPorts'])] + [s['image']['path']] + list(meta['files']) + list(s['sourceMetadataFiles']) + common
    else:
        allowed = [x['path'] for x in fields(compiler['actorInputs'])] + common
        denied += [x['path'] for x in fields(s['writerPorts'])] + [s['image']['path']]
        denied += [x['path'] for x in fields(compiler['diagnostics'])]
    accesses = quarantine(root, allowed, denied)
    started, cpu = time.perf_counter(), time.process_time()
    numerical = profile(torch, 4)
    if numerical != s['numericalProfile']:
        raise ValueError('Original numerical profile differs')
    model, assets = source_checks(s, diffusers)
    checkpoint = model / 'transformer/diffusion_pytorch_model.safetensors'
    entries, head = header(checkpoint)
    names = [x for x in PARAMETERS if 'to_out.0' not in x] if phase == 'compile' else ACTOR_PARAMETERS
    weights, receipts = load_weights(checkpoint, names, entries)
    originals = {x['source']['tensor']: x['source'] for x in s['originalParameterArtifacts']}
    for r in receipts:
        e = originals[r['name']]
        if r['shape'] != e['shape'] or r['dataOffsets'] != e['data_offsets'] or r['payloadSHA256'] != e['payload_sha256']:
            raise ValueError('Selected original retrieval contract differs')
    initial_weights = {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()}
    rng, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    linears, contractions = [], []
    result = {'actorInputs': {}, 'diagnostics': {}} if phase == 'compile' else {'actors': {}}
    def keep(group, name, value):
        return retain(root / 'arrays' / phase / group, name, value)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        if phase == 'compile':
            from saturn_pub.store import LocalStore
            for file, digest in {**meta['files'], **s['sourceMetadataFiles']}.items():
                if sha(file) != digest:
                    raise ValueError('Source metadata/blob changed')
            selected = resolve(s['context'], meta['tokens'])
            if selected != s['supportResolution']:
                raise ValueError('Actual source address resolver differs')
            support = selected['support']
            context, _ = LocalStore(meta['pages']['context']['store'])._read_page(meta['pages']['context']['pageID'])
            ids, _ = LocalStore(meta['pages']['img_ids']['store'])._read_page(meta['pages']['img_ids']['pageID'])
            if meta['cut']['execution_point'] != request['familyPoint'] or meta['cut']['manifest']['model_identity'] != 'a09191c5ff813af77bf89d8e4b9a07237befb67c1aef3e722322d3491250fcd0':
                raise ValueError('Wrong source state invocation')
            source = {role: {n: load(v, 'cuda') for n, v in ports.items()} for role, ports in s['writerPorts'].items()}
            qsource, psource = source['Q'], source['P']
            for role, ports in source.items():
                if payload(rows(ports['text_in'], ports['text_attention_return_used'], ports['text_mlp_return_used'], ports['text_attention_gate'], ports['text_mlp_gate'])) != payload(ports['text_out']):
                    raise ValueError('Original source writer gate/add closure differs')
                for gate in ['text_attention_gate', 'text_mlp_gate']:
                    if payload(ports[gate]) != payload(qsource[gate]):
                        raise ValueError('Recipient gates differ across sources')
            raw = {}
            for arm, r, m in [('QQQ', qsource, qsource), ('PQP', psource, psource)]:
                raw[arm] = rows(r['text_in'][:, support], qsource['text_attention_return_used'][:, support], m['text_mlp_return_used'][:, support],
                                qsource['text_attention_gate'], qsource['text_mlp_gate'])
                endpoint = qsource['text_out'].clone()
                endpoint[:, support] = raw[arm]
                result['diagnostics'][arm + '.text'] = keep('diagnostics', arm + '.text', endpoint)
            image = load(s['image'], 'cuda')
            # Bind source array values to the actual parent cut's supplied image/text.
            payload_desc = {x['key']['value']: x['value'] for x in meta['cut']['manifest']['payload']['items']}
            if payload(image) != payload_desc['image']['sha256'] or payload(qsource['text_out']) != payload_desc['text']['sha256']:
                raise ValueError('Writer outputs do not bind to original joint1 cut')
            im, tx = context['img_mod'].to('cuda'), context['txt_mod'].to('cuda')
            rope = tuple(v.to('cuda') for v in context['rope'])
            xi, gate = normalize(image, im)
            xt, _ = normalize(qsource['text_out'], tx)
            q = headnorm(weights, 'norm_q', project(weights, 'to_q', xi, linears).unflatten(-1, (24, 128)))
            ki = headnorm(weights, 'norm_k', project(weights, 'to_k', xi, linears).unflatten(-1, (24, 128)))
            vi = project(weights, 'to_v', xi, linears).unflatten(-1, (24, 128))
            kt = headnorm(weights, 'norm_added_k', project(weights, 'add_k_proj', xt, linears).unflatten(-1, (24, 128)))
            vt = project(weights, 'add_v_proj', xt, linears).unflatten(-1, (24, 128))
            q = native.apply_rotary_emb(q, tuple(v[512:] for v in rope), sequence_dim=1)
            k = native.apply_rotary_emb(torch.cat([kt, ki], 1), rope, sequence_dim=1)
            v = torch.cat([vt, vi], 1)
            q64, k64, v64 = [heads(x) for x in [q, k, v]]
            m, z, n = complement(q64, k64, v64, support)
            contractions = [{'kind': 'baseline-all-key-QK', 'nominalMACs': 24 * 1024 * 1536 * 128},
                            {'kind': 'baseline-physical-all-key-AV-selected-weights-zero', 'nominalMACs': 24 * 1024 * 1536 * 128}]
            actor_inputs = {'Q': q, 'm': m, 'z': z, 'n': n, 'image': image, 'gate': gate, 'txtMod': tx,
                            'ropeCos': rope[0][support], 'ropeSin': rope[1][support], 'QQQ.rows': raw['QQQ'], 'PQP.rows': raw['PQP']}
            result['actorInputs'] = {name: keep('admitted', name, value) for name, value in actor_inputs.items()}
            result['diagnostics']['imageIDs'] = keep('diagnostics', 'imageIDs', ids)
            result['supportResolution'] = selected
            result['cacheLogicalBytes'] = sum(x.numel() * x.element_size() for x in [m, z, n])
            if result['cacheLogicalBytes'] != 25559040:
                raise ValueError('Cache geometry differs')
        else:
            from mdb.titan import Frame, ActSpec, ActRegistry, ProvidedFingerprint
            values = {name: load(item, 'cpu' if name in ['m', 'z', 'n'] else 'cuda') for name, item in compiler['actorInputs'].items()}
            before = {n: payload(v) for n, v in values.items()}
            registry = ActRegistry()
            reads = ['Q', 'm', 'z', 'n', 'image', 'gate', 'txtMod', 'ropeCos', 'ropeSin', 'textRows', 'Kweight', 'Vweight', 'Woweight', 'KnormWeight']
            spec = ActSpec('tce.sparse_text_kv', 'v1', reads=tuple(reads), writes=('R_attention',),
                write_kinds={'R_attention': 'tensor'}, write_abis={'R_attention': {'shape': [1, 1024, 3072], 'dtype': 'torch.bfloat16'}},
                numerical_contract='original-fourrow-KV-headnorm-rope-positive-fp64-complement-native-Wo-gate-residual')
            active = {}
            def operation(v):
                xt, _ = normalize(v['textRows'], v['txtMod'])
                k = headnorm(weights, 'norm_added_k', project(weights, 'add_k_proj', xt, linears).unflatten(-1, (24, 128)))
                k = native.apply_rotary_emb(k, (v['ropeCos'], v['ropeSin']), sequence_dim=1)
                value = project(weights, 'add_v_proj', xt, linears).unflatten(-1, (24, 128))
                account = update(heads(v['Q']), heads(k), heads(value), v['m'], v['z'], v['n'])
                contractions.extend([{'actor': arm, 'kind': name, 'nominalMACs': 24 * 1024 * 4 * 128} for name in ['selected-QK', 'selected-AV']])
                h = account['H_FP64'].permute(1, 0, 2).reshape(1, 1024, 3072).float().bfloat16().to('cuda')
                output = project(weights, 'to_out.0', h, linears)
                carrier = v['gate'] * output
                residual = v['image'] + carrier
                captures = {'K': k, 'V': value, **account, 'H_BF16': h, 'attentionReturn': output, 'carrier': carrier, 'R_attention': residual}
                active.update({name: keep(arm, name, x) for name, x in captures.items()})
                return {'R_attention': residual}, {'capturedPayloadSHA256': payload(residual), 'changedTextRows': request['support'], 'fullAttentionCalls': 0}
            registry.register(spec, 'four-text-row-positive-cache', operation)
            for arm in ['QQQ', 'PQP']:
                active = {}
                fv = {n: values[n] for n in reads if n in values}
                fv.update(textRows=values[arm + '.rows'], Kweight=weights[PREFIX + 'add_k_proj.weight'], Vweight=weights[PREFIX + 'add_v_proj.weight'],
                          Woweight=weights[PREFIX + 'to_out.0.weight'], KnormWeight=weights[PREFIX + 'norm_added_k.weight'])
                frame = Frame.from_values(fv, kinds={n: 'tensor' for n in fv},
                    abis={n: {'shape': list(x.shape), 'dtype': str(x.dtype), 'stride': list(x.stride())} for n, x in fv.items()},
                    slot_fingerprints={n: ProvidedFingerprint.content_sha256(payload(x)) for n, x in fv.items()}, program_fingerprint=request['requestSHA256'],
                    metadata={'arm': arm, 'support': request['support'], 'familyPoint': request['familyPoint']})
                after, evidence = registry.run(spec.identity, 'four-text-row-positive-cache', frame)
                if payload(after.values['R_attention']) != active['R_attention']['payloadSHA256']:
                    raise ValueError('Typed output capture differs')
                result['actors'][arm] = {'fields': active, 'sourceFrame': frame.to_dict(), 'outputFrame': after.to_dict(), 'evidence': evidence.to_dict()}
            if before != {n: payload(v) for n, v in values.items()}:
                raise ValueError('Admitted actor input changed')
            result['compilerSealSHA256'] = sha(out / 'compiler-seal.json')
    torch.cuda.synchronize()
    if initial_weights != {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()} or snapshot(torch) != numerical:
        raise ValueError('Selected weights/profile changed')
    if not torch.equal(rng, torch.get_rng_state()) or not all(torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all())):
        raise ValueError('RNG changed')
    assets_unchanged(assets)
    total = sum(x['nominalMACs'] for x in linears + contractions)
    if total != (48318382080 if phase == 'compile' else 19528679424):
        raise ValueError('Physical contraction census differs')
    name = 'compiler' if phase == 'compile' else 'primary'
    save(out / (name + '.json'), result)
    save(out / (name + '-seal.json'), {'requestSHA256': request['requestSHA256'], name + 'SHA256': sha(out / (name + '.json'))})
    save(out / (name + '-execution.json'), {'loadedParameters': receipts, 'linears': linears, 'contractions': contractions, 'nominalMACs': total,
        'numericalProfile': numerical, 'header': head, 'assets': assets, 'parameterInputRNGClosure': True,
        'wallSecondsAfterImports': time.perf_counter() - started, 'CPUSecondsAfterImports': time.process_time() - cpu,
        'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'peakCUDAReservedBytes': torch.cuda.max_memory_reserved(),
        'fileQuarantine': {'allowedFiles': sorted(allowed), 'deniedCanaries': denied, 'successfulFileOpenCounts': accesses,
                           'scope': 'Trusted Python audit hook; exact numeric read allowlist applies inside and outside worker root.'}})
    print('SPARSE_TEXT_KV_' + name.upper() + '_SEALED', sha(out / (name + '-seal.json')), flush=True)


if __name__ == '__main__':
    execute(sys.argv[1], sys.argv[2])
