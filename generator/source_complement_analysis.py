"""Saved-only source/context contrasts after both isolated arms seal and exit."""
import json
from pathlib import Path
import numpy as np
from encoder_atoms import read, sha
from paired_fields import load, metrics


def effect_metrics(value, reference):
    m = metrics(value, reference)
    return {'effectRms': m['candidateRms'], 'referenceRms': m['referenceRms'],
            'relativeEffectRms': m['candidateRms'] / m['referenceRms'] if m['referenceRms'] else None,
            'cosineWithReference': m['cosine'], 'mean': float(np.mean(value)),
            'maxAbs': float(np.max(np.abs(value))), 'signedMeanProduct': float(np.mean(value * reference))}


def execute(root):
    root = Path(root)
    out = root / 'response-result'
    r, target = read(root / 'request.json'), read(root / 'targets.json')
    arms = {}
    for name in ['shared', 'primary', 'control']:
        seal = read(out / (name + '-seal.json'))
        if seal[name + 'SHA256'] != sha(out / (name + '.json')) or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Complete new phase seal differs')
        if name != 'shared':
            arms[name] = read(out / (name + '.json'))
    if arms['control']['primarySealSHA256'] != sha(out / 'primary-seal.json'):
        raise ValueError('Primary-before-control release differs')
    baseline = load(target['parentReader']['QQQ'])
    parent = load(target['parentReader']['PQP'])
    native_q, native_p = [load(target['nativeReader'][arm]) for arm in ['QQQ', 'PQP']]
    parent_delta, native_delta = parent - baseline, native_p - native_q
    ids = load(target['imageIDs']).astype(np.int64)[0]
    reader, sources, endpoints, responses, panels, rows = {}, {}, [], [], [], []
    source_metrics, row_metrics = [], []
    native_names = {'R': 'text_in', 'A': 'text_attention_return_used', 'U': 'text_after_attention',
                    'X': 'text_mlp_input', 'M': 'text_mlp_return_used', 'T': 'text_out'}
    for name, arm in arms.items():
        reader[name] = load(arm['recipient']['fields']['R_attention'])
        sources[name] = {n: load(x) for n, x in arm['source']['fields'].items()}
        delta = reader[name] - baseline
        for ref, ref_p in [('parent', parent), ('native', native_p)]:
            endpoints.append({'arm': name, 'reference': ref, **metrics(reader[name], ref_p)})
        for ref, reference in [('parent', parent_delta), ('native', native_delta)]:
            responses.append({'arm': name, 'reference': ref, **metrics(delta, reference)})
            for label, selection in [('all', np.ones(1024, dtype=bool)), ('left-grid', ids[:, 2] < 16), ('right-grid', ids[:, 2] >= 16)]:
                per = [metrics(delta[0, i], reference[0, i])['relativeRms'] for i in np.flatnonzero(selection)]
                finite = [x for x in per if x is not None]
                panels.append({'arm': name, 'reference': ref, 'panel': label, 'dependentQueries': int(selection.sum()),
                               **metrics(delta[:, selection], reference[:, selection]), 'medianRowRelativeError': float(np.median(finite)) if finite else None,
                               'rowsBetterThanZero': sum(x < 1 for x in finite), 'rowsWorseThanZero': sum(x > 1 for x in finite)})
        for i in range(1024):
            rows.append({'arm': name, 'imageQueryIndex': i, 'imageID': ids[i].tolist(),
                         'nativeResponse': metrics(delta[0, i], native_delta[0, i]),
                         'parentResponse': metrics(delta[0, i], parent_delta[0, i])})
        for field in ['A', 'U', 'X', 'M', 'T']:
            for ref, value in [('parent', load(target['parentSource'][field])),
                               ('native', load(target['nativeSource'][native_names[field]])[:, r['support']])]:
                source_metrics.append({'arm': name, 'field': field, 'reference': ref, **metrics(sources[name][field], value)})
        for ref, value in [('parent', load(target['parentRows'])), ('native', load(target['nativeRows'])[:, r['support']])]:
            row_metrics.append({'arm': name, 'reference': ref, **metrics(sources[name]['PQP'], value)})
    difference = reader['primary'] - reader['control']
    contrast_rows = [{'imageQueryIndex': i, 'imageID': ids[i].tolist(), **effect_metrics(difference[0, i], parent_delta[0, i])} for i in range(1024)]
    # This compares the context effect vector with the historical intervention
    # axis for size/direction only; it is not a prediction-error target.
    contrast = {'primaryMinusControl': effect_metrics(difference, parent_delta),
                'sourceDifferences': {n: effect_metrics(sources['primary'][n] - sources['control'][n], load(target['parentSource'][n])) for n in ['A', 'U', 'X', 'M', 'T']},
                'sourceRowDifference': effect_metrics(sources['primary']['PQP'] - sources['control']['PQP'], load(target['parentRows'])),
                'scope': 'Primary-minus-control effect vectors compared to historical response/source axes for size and orientation, not prediction-error targets.'}
    for filename, records in [('queries.jsonl', rows), ('context-effect.jsonl', contrast_rows)]:
        with (out / filename).open('x') as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
    executions = [read(out / (name + '-execution.json')) for name in ['shared', 'primary', 'control']]
    selected_identity = []
    for name, arm in arms.items():
        src = arm['source']['sourceFrame']['slots']
        selected_identity.append({'arm': name, **{n: src[n]['fingerprint'] for n in ['R', 'Q', 'K', 'V', 'txtMod']}})
    total = sum(e['nominalMatrixMACs'] for e in executions)
    if total != 1799356416:
        raise ValueError('Actual complete operation sum differs')
    return {'sharedSealSHA256': sha(out / 'shared-seal.json'), 'primarySealSHA256': sha(out / 'primary-seal.json'),
            'controlSealSHA256': sha(out / 'control-seal.json'), 'sourceMetrics': source_metrics, 'sourceRowMetrics': row_metrics,
            'endpoints': endpoints, 'responses': responses, 'queryPanels': panels, 'queryRecords': len(rows),
            'contextEffectRecords': len(contrast_rows), 'contextContrast': contrast, 'selectedOperandFingerprints': selected_identity,
            'comparisonBaseline': {'record': target['parentReader']['QQQ'], 'newlyComputed': False,
                                   'scope': 'Paid cycle54 QQQ baseline, opened only after both new arms seal and exit.'},
            'totalNewMatrixMACs': total, 'cacheLogicalBytes': 604372992,
            'scalarWorkPerReader': {'cacheMultiplyAddPairs': 75497472, 'perHeadDivisions': 75497472, 'interHeadAdditions': 72351744},
            'work': {'sourceAffines': 9, 'sourceAttentionCalls': 2, 'readerAffines': 52, 'readerOtherContractions': 4,
                     'newMLPs': 2, 'newFullTransformer': 0, 'newFullBanks': 0, 'newImages': 0,
                     'distinctBF16ParameterBytes': sum({x['name']: x['bytes'] for e in executions for x in e['loadedParameters']}.values()),
                     'processBF16ParameterBytes': sum(x['bytes'] for e in executions for x in e['loadedParameters'])}}
