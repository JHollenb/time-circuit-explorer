"""Isolated native-byte intake, knot-only prediction, then saved-only scoring."""
import argparse
import ast
import hashlib
import importlib.abc
import json
from pathlib import Path
import struct
import sys
import time

from encoder_atoms import read, save, sha, identity


class NoFrameworks(importlib.abc.MetaPathFinder):
    roots = {'torch', 'diffusers', 'transformers', 'jax', 'tensorflow', 'onnxruntime'}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.roots:
            raise RuntimeError('Saved-only spatial assay cannot import neural frameworks')


def raw_load(item):
    import numpy as np
    if sha(item['path']) != item['sha256']:
        raise ValueError('Saved field changed')
    a = np.load(item['path'], allow_pickle=False)
    if list(a.shape) != item['shape'] or hashlib.sha256(a.tobytes()).hexdigest() != item['payloadSHA256']:
        raise ValueError('Saved field geometry/payload changed')
    return a


def intake(root, r, s, out):
    import numpy as np
    from paired_fields import retain
    from spatial_norm import geometry
    files = {}
    declared = []
    for cut in s['sourceCuts'].values():
        item = cut['cutFile']
        if sha(item['path']) != item['sha256'] or read(item['path']) != cut['cut']:
            raise ValueError('Actual native cut differs')
        files[item['path']] = item['sha256']
        declared.append({x['key']['value']: x['value'] for x in cut['cut']['manifest']['payload']['items']}['img_ids'])
    if declared[0] != declared[1] or s['sourceCuts']['P']['cut']['slots']['img_ids'] != s['sourceCuts']['Q']['cut']['slots']['img_ids']:
        raise ValueError('P/Q native ID identity differs')
    slot = s['sourceCuts']['Q']['cut']['slots']['img_ids']
    page_path = Path(s['store']) / 'pages' / (slot + '.json')
    if sha(page_path) != slot:
        raise ValueError('Native ID page content address differs')
    page = read(page_path);files[str(page_path)] = slot
    if page['value'] != declared[0] or page['encoded']['kind'] != 'tensor':
        raise ValueError('Native page/manifest descriptor differs')
    blob_path = Path(s['store']) / 'blobs' / (page['tensor_blob'] + '.safetensors')
    blob = blob_path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != page['tensor_blob']:
        raise ValueError('Native ID blob content address differs')
    files[str(blob_path)] = page['tensor_blob']
    length = struct.unpack('<Q', blob[:8])[0];header = json.loads(blob[8:8+length])
    tensor = header[page['encoded']['name']]
    if tensor['dtype'] != 'I64' or tensor['shape'] != [1, 1024, 4] or declared[0]['dtype'] != 'torch.int64':
        raise ValueError('Native ID dtype/shape differs')
    start, end = tensor['data_offsets'];payload = blob[8+length+start:8+length+end]
    if hashlib.sha256(payload).hexdigest() != declared[0]['sha256'] or len(payload) != 32768:
        raise ValueError('Native ID payload identity differs')
    ids = np.frombuffer(payload, dtype='<i8').reshape(1, 1024, 4)
    knot_rows, strata = geometry(ids)
    # Authenticate the full file as bytes; never decode unsampled norm words here.
    source = s['qNorm'];data = Path(source['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != source['sha256'] or data[:8] != b'\x93NUMPY\x01\x00':
        raise ValueError('Saved norm NPY file identity/version differs')
    header_length = struct.unpack('<H', data[8:10])[0];offset = 10+header_length
    norm_header = ast.literal_eval(data[10:offset].decode('latin1').strip())
    if norm_header != {'descr': '<f8', 'fortran_order': False, 'shape': (1024, 24)} or len(data)-offset != 1024*24*8:
        raise ValueError('Saved norm byte geometry differs')
    if hashlib.sha256(data[offset:]).hexdigest() != source['payloadSHA256']:
        raise ValueError('Saved norm payload identity differs')
    segments = [{'row': int(j), 'offset': offset+int(j)*192, 'length': 192,
        'sha256': hashlib.sha256(data[offset+int(j)*192:offset+(int(j)+1)*192]).hexdigest()} for j in knot_rows]
    selected = b''.join(data[x['offset']:x['offset']+x['length']] for x in segments)
    knots = np.frombuffer(selected, dtype='<f8').reshape(256, 24)
    arrays = {'ids': retain(root / 'arrays/intake', 'ids', ids),
        'knots': retain(root / 'arrays/intake', 'knots', knots),
        'knotRows': retain(root / 'arrays/intake', 'knotRows', knot_rows),
        'strata': retain(root / 'arrays/intake', 'strata', strata)}
    save(out / 'intake.json', {'schema': 'tce-spatial-norm-intake-v1', 'requestSHA256': r['requestSHA256'],
        'fields': arrays, 'nativeFiles': files, 'nativeDescriptor': declared[0],
        'nativeDataOffset': 8+length+start, 'nativePayloadBytes': len(payload),
        'normSource': source, 'normHeaderSHA256': hashlib.sha256(data[:offset]).hexdigest(),
        'normDataOffset': offset, 'selectedSegments': segments,
        'strataRows': [int(np.sum(strata == a)) for a in range(3)],
        'scope': 'Actual native img_ids bytes and exactly256fixednorm rows; full norm byte hashing but unsampled words not numerically decoded.'})


def candidate(root, r, s, out):
    from paired_fields import retain
    from sparse_text_kv import quarantine
    from spatial_norm import reconstruct
    intake_receipt = read(out / 'intake.json')
    allowed = [x['path'] for x in intake_receipt['fields'].values()]
    denied = [s[n]['path'] for n in ['pNorm', 'qNorm', 'numerator', 'textLogits']]
    accesses = quarantine(root, allowed, denied)
    ids, knots, rows, strata = [raw_load(intake_receipt['fields'][n]) for n in ['ids', 'knots', 'knotRows', 'strata']]
    norm, squared, actual_rows, actual_strata = reconstruct(ids, knots)
    if not (rows == actual_rows).all() or not (strata == actual_strata).all():
        raise ValueError('Candidate physical geometry differs from byte intake')
    fields = {n: retain(root / 'arrays/candidate', n, value) for n, value in [('norm', norm), ('squaredNorm', squared)]}
    save(out / 'candidate.json', {'schema': 'tce-spatial-norm-candidate-v1', 'requestSHA256': r['requestSHA256'],
        'intakeSHA256': sha(out / 'intake.json'), 'fields': fields,
        'fileQuarantine': {'allowedFiles': sorted(allowed), 'deniedCanaries': denied, 'successfulFileOpenCounts': accesses},
        'arithmetic': {'knotSquares': 6144, 'interpolationMultiplications': 98304, 'interpolationAccumulatorAdds': 98304,
            'squareRoots': 24576, 'secondEpsilonAdditions': 0},
        'scope': 'Only current paid norm knots and authenticated physical IDs; no full current or previous norm, scores, raw keys, learned maps or consumer.'})
    save(out / 'candidate-seal.json', {'requestSHA256': r['requestSHA256'],
        'candidateSHA256': sha(out / 'candidate.json'), 'intakeSHA256': sha(out / 'intake.json')})


def diagnose(root, r, s, out):
    import numpy as np
    from paired_fields import load, metrics, retain
    from spatial_norm import score_diagnostics
    candidate_receipt, seal = read(out / 'candidate.json'), read(out / 'candidate-seal.json')
    if seal['candidateSHA256'] != sha(out / 'candidate.json') or seal['intakeSHA256'] != sha(out / 'intake.json'):
        raise ValueError('Exited candidate/intake seal differs')
    seal_hash = sha(out / 'candidate-seal.json')
    norm = load(candidate_receipt['fields']['norm'])
    intake_receipt = read(out / 'intake.json')
    ids, strata = [raw_load(intake_receipt['fields'][n]) for n in ['ids', 'strata']]
    pn, qn, numerator, text = [load(s[n]) for n in ['pNorm', 'qNorm', 'numerator', 'textLogits']]
    if pn.shape != qn.shape or qn.shape != (1024,24) or numerator.shape != (24,4,1024) or text.shape != (24,4,512):
        raise ValueError('Post-seal saved geometry differs')
    if not np.all(pn > 0) or not np.all(qn > 0):
        raise ValueError('Smooth reference norms not positive')
    def summary(mask, head=None):
        a, b, c = norm[mask], qn[mask], pn[mask]
        if head is not None:
            a, b, c = [x[:, head] for x in [a, b, c]]
        return {'cells': int(a.size), 'endpoint': metrics(a, b), 'response': metrics(a-c, b-c),
                'holdResponse': metrics(np.zeros_like(b), b-c), 'zeroTrueResponseCells': int(np.sum(b == c))}
    names = ['sampled', 'unsampledInterior', 'unsampledEdge']
    all_rows = np.ones(1024, dtype=bool)
    norm_summary = {'pooled': summary(all_rows), 'strata': {n: summary(strata == k) for k,n in enumerate(names)},
        'heads': [{'head': h, 'pooled': summary(all_rows,h),
                   'strata': {n: summary(strata == k,h) for k,n in enumerate(names)}} for h in range(24)]}
    ref_image, predicted_image = [numerator/(128**.5*x.T[:,None,:]) for x in [qn, norm]]
    reference, predicted = [np.concatenate((text, x),-1) for x in [ref_image, predicted_image]]
    scores = score_diagnostics(reference, predicted)
    arrays = {n: retain(root / 'arrays/diagnostics', n, value) for n,value in [
        ('referenceLogits', reference), ('candidateLogits', predicted), ('deltaLogits', scores['delta']),
        ('referenceProbability', scores['referenceProbability']), ('candidateProbability', scores['candidateProbability'])]}
    query_rows = []
    with (out / 'observations.jsonl').open('x') as stream:
        for j in range(1024):
            for h in range(24):
                stream.write(json.dumps({'panel': 'image-key-head', 'row': j, 'head': h,
                    'height': int(ids[0,j,1]), 'width': int(ids[0,j,2]), 'stratum': names[int(strata[j])],
                    'pNorm': float(pn[j,h]), 'qNorm': float(qn[j,h]), 'predictedNorm': float(norm[j,h]),
                    'actualResponse': float(qn[j,h]-pn[j,h]), 'predictedResponse': float(norm[j,h]-pn[j,h]),
                    'signedError': float(norm[j,h]-qn[j,h])}, sort_keys=True, allow_nan=False)+'\n')
        for h in range(24):
            for a, row in enumerate(r['support']):
                p, delta = scores['referenceProbability'][h,a], scores['delta'][h,a]
                mean = float(scores['mean'][h,a]);centered = delta-mean
                rec = {'panel': 'query-head', 'row': row, 'head': h,
                    'imageLogitEndpoint': metrics(predicted_image[h,a], ref_image[h,a]),
                    'unweightedDeltaRMS': float(np.sqrt(np.mean(delta*delta))), 'weightedDeltaMean': mean,
                    'weightedCenteredVariance': float(scores['variance'][h,a]), 'exactKL': float(scores['kl'][h,a]),
                    'directKLResidual': float(scores['directKLResidual'][h,a]),
                    'logPartitionDifference': float(scores['candidateLogPartition'][h,a]-scores['referenceLogPartition'][h,a]),
                    'textReferenceMass': float(p[:512].sum()),
                    'strata': {n: {'referenceMass': float(p[512:][strata == k].sum()),
                        'weightedCenteredVariance': float(np.sum(p[512:][strata == k]*centered[512:][strata == k]**2))}
                        for k,n in enumerate(names)}}
                query_rows.append(rec);stream.write(json.dumps(rec, sort_keys=True, allow_nan=False)+'\n')
    if seal_hash != sha(out / 'candidate-seal.json'):
        raise ValueError('Candidate seal changed during scoring')
    save(out / 'diagnostics.json', {'schema': 'tce-spatial-norm-capacity-result-v1', 'requestSHA256': r['requestSHA256'],
        'candidateSealSHA256': seal_hash, 'norms': norm_summary, 'queryHeads': query_rows, 'retainedFields': arrays,
        'observations': 24672, 'newMatrixMACs': 0, 'newOriginalProjections': 0, 'newConsumers': 0,
        'mechanics_status': 'complete', 'terminal_status': 'unassessed',
        'scope': 'One saved pair and fixed grid. Native current knots/conditioning/numerator/text terms remain paid. Score diagnostics only, no AV/MLP/next-query/image or measured producer saving. Arbitrary-prompt zero-neural generation unresolved.'})


if __name__ == '__main__':
    if NoFrameworks.roots.intersection(sys.modules):
        raise RuntimeError('Neural framework already imported')
    sys.meta_path.insert(0, NoFrameworks())
    p = argparse.ArgumentParser();p.add_argument('phase', choices=['intake', 'candidate', 'diagnose']);p.add_argument('root');a = p.parse_args()
    root = Path(a.root).resolve();r, s = read(root / 'request.json'), read(root / 'inputs.json')
    if identity({k: v for k,v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request identity differs')
    for n,h in r['sourceHashes'].items():
        if sha(root / n) != h:
            raise ValueError('Frozen worker/input source changed')
    globals()[a.phase](root, r, s, root / 'response-result')
