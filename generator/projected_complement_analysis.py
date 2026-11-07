"""Compare sealed projected readers against both historical numerical targets."""
import json
from pathlib import Path
import numpy as np
from encoder_atoms import read, sha
from paired_fields import load, metrics, paired


def execute(root):
    root = Path(root)
    out = root / 'response-result'
    r = read(root / 'request.json')
    for name in ['compiler', 'primary']:
        seal = read(out / (name + '-seal.json'))
        if seal[name + 'SHA256'] != sha(out / (name + '.json')) or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Complete new release seal differs')
    target, primary, compiler = read(root / 'targets.json'), read(out / 'primary.json'), read(out / 'compiler.json')
    for name in ['parentPrimary', 'parentPrimarySeal']:
        if sha(target[name]['path']) != target[name]['sha256']:
            raise ValueError('Parent reference record changed')
    if read(target['parentPrimarySeal']['path'])['primarySHA256'] != target['parentPrimary']['sha256']:
        raise ValueError('Parent complete target release differs')
    own = {a: load(primary['actors'][a]['fields']['R_attention']) for a in ['QQQ', 'PQP']}
    references = {'family': {a: load(target['family'][a]) for a in own},
                  'parent': {a: load(target['parent'][a]['R_attention']) for a in own}}
    endpoints, responses, panels = [], [], []
    ids = load(target['imageIDs']).astype(np.int64)[0]
    rows = [{'imageQueryIndex': i, 'imageID': ids[i].tolist(), 'references': {}} for i in range(1024)]
    own_delta = own['PQP'] - own['QQQ']
    for name, ref in references.items():
        endpoints.extend({'reference': name, 'arm': a, **metrics(own[a], ref[a])} for a in own)
        responses.append({'reference': name, **paired(own['QQQ'], own['PQP'], ref['QQQ'], ref['PQP'])})
        delta = ref['PQP'] - ref['QQQ']
        for i, row in enumerate(rows):
            row['references'][name] = {'response': metrics(own_delta[0, i], delta[0, i]),
                                     **{a: metrics(own[a][0, i], ref[a][0, i]) for a in own}}
        for label, selection in [('all', np.ones(1024, dtype=bool)), ('left-grid', ids[:, 2] < 16), ('right-grid', ids[:, 2] >= 16)]:
            per = [x['references'][name]['response']['relativeRms'] for x, include in zip(rows, selection) if include and x['references'][name]['response']['relativeRms'] is not None]
            panels.append({'reference': name, 'panel': label, 'dependentQueries': int(selection.sum()), **metrics(own_delta[:, selection], delta[:, selection]),
                           'medianRowRelativeError': float(np.median(per)) if per else None, 'rowsBetterThanZero': sum(x < 1 for x in per),
                           'rowsWorseThanZero': sum(x > 1 for x in per)})
    with (out / 'queries.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    boundary_identity = []
    for arm in own:
        for name in ['K', 'V', 'scores', 'peak', 'cacheScale', 'selectedWeights', 'denominator']:
            a, b = primary['actors'][arm]['fields'][name], target['parent'][arm][name]
            # Open and validate both actual payloads after the new primary seal.
            load(a)
            load(b)
            boundary_identity.append({'arm': arm, 'field': name, 'payloadExact': a['payloadSHA256'] == b['payloadSHA256']})
    ce, pe = read(out / 'compiler-execution.json'), read(out / 'primary-execution.json')
    if sha(out / 'primary.json') != read(out / 'primary-seal.json')['primarySHA256']:
        raise ValueError('Primary changed during scoring')
    return {'compilerSealSHA256': sha(out / 'compiler-seal.json'), 'primarySealSHA256': sha(out / 'primary-seal.json'),
            'endpoints': endpoints, 'responses': responses, 'queryPanels': panels, 'queryRecords': len(rows), 'upstreamBoundaryIdentity': boundary_identity,
            'cacheLogicalBytes': compiler['projectedCacheLogicalBytes'], 'compilerMatrixMACs': ce['nominalMatrixMACs'], 'actorsMatrixMACs': pe['nominalMatrixMACs'],
            'totalIncrementalMatrixMACs': ce['nominalMatrixMACs'] + pe['nominalMatrixMACs'], 'historicalBaselineCompilerMatrixMACs': 48318382080,
            'scalarWorkPerActor': pe['scalarWorkPerActor'],
            'work': {'compilerAffines': len(ce['linears']), 'actorAffines': len(pe['linears']), 'actorOtherContractions': len(pe['contractions']),
                     'newFullTransformer': 0, 'newMLP': 0, 'newImages': 0, 'newNativeSDPA': 0,
                     'distinctBF16ParameterBytes': sum({x['name']: x['bytes'] for e in [ce, pe] for x in e['loadedParameters']}.values()),
                     'processBF16ParameterBytes': sum(x['bytes'] for e in [ce, pe] for x in e['loadedParameters']),
                     'WoFP64CopyBytesPerProcess': pe['WoFP64']['bytes']}}
