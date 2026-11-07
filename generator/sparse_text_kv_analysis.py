"""Archived target admission only after complete candidate seal and child exit."""
import json
from pathlib import Path
import numpy as np
from encoder_atoms import read, sha, save
from paired_fields import load, metrics, paired


def execute(root):
    root = Path(root)
    out = root / 'response-result'
    primary, compiler, seal = [read(out / n) for n in ['primary.json', 'compiler.json', 'primary-seal.json']]
    if seal['primarySHA256'] != sha(out / 'primary.json') or primary['compilerSealSHA256'] != sha(out / 'compiler-seal.json'):
        raise ValueError('Compiler/actor seal differs')
    targets = read(root / 'targets.json')
    native = {arm: load(targets['R_attention'][arm]) for arm in ['QQQ', 'PQP']}
    own = {arm: load(primary['actors'][arm]['fields']['R_attention']) for arm in ['QQQ', 'PQP']}
    text = []
    support = read(root / 'request.json')['support']
    for arm in ['QQQ', 'PQP']:
        a, b = load(compiler['diagnostics'][arm + '.text']), load(targets['text'][arm])
        text.append({'arm': arm, 'bitExact': bool(np.array_equal(a, b)), **metrics(a, b)})
    response = own['PQP'] - own['QQQ']
    truth = native['PQP'] - native['QQQ']
    ids = load(compiler['diagnostics']['imageIDs']).astype(np.int64)[0]
    query_rows = []
    with (out / 'queries.jsonl').open('x') as stream:
        for i in range(1024):
            row = {'imageQueryIndex': i, 'imageID': ids[i].tolist(), 'response': metrics(response[0, i], truth[0, i]),
                   'QQQ': metrics(own['QQQ'][0, i], native['QQQ'][0, i]), 'PQP': metrics(own['PQP'][0, i], native['PQP'][0, i])}
            query_rows.append(row)
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    panels = []
    # Physical image-grid halves are declared address panels, not semantic object labels.
    for label, selection in [('all', np.ones(1024, dtype=bool)), ('left-grid', ids[:, 2] < 16), ('right-grid', ids[:, 2] >= 16)]:
        per = [r['response']['relativeRms'] for r, include in zip(query_rows, selection) if include and r['response']['relativeRms'] is not None]
        panels.append({'panel': label, 'dependentQueries': int(selection.sum()), **metrics(response[:, selection], truth[:, selection]),
                       'medianRowRelativeError': float(np.median(per)) if per else None,
                       'rowsBetterThanZero': sum(x < 1 for x in per), 'rowsWorseThanZero': sum(x > 1 for x in per)})
    ce, pe = [read(out / n) for n in ['compiler-execution.json', 'primary-execution.json']]
    if sha(out / 'primary.json') != seal['primarySHA256']:
        raise ValueError('Primary changed during scoring')
    return {'primarySealSHA256': sha(out / 'primary-seal.json'), 'compilerSealSHA256': sha(out / 'compiler-seal.json'),
            'endpoints': [{'arm': arm, **metrics(own[arm], native[arm])} for arm in ['QQQ', 'PQP']],
            'response': paired(own['QQQ'], own['PQP'], native['QQQ'], native['PQP']), 'queryPanels': panels,
            'textEndpoints': text, 'queryRecords': len(query_rows), 'cacheLogicalBytes': compiler['cacheLogicalBytes'],
            'compilerMACs': ce['nominalMACs'], 'actorsMACs': pe['nominalMACs'], 'totalNewMACs': ce['nominalMACs'] + pe['nominalMACs'],
            'work': {'compilerAffines': len(ce['linears']), 'actorAffines': len(pe['linears']), 'compilerContractions': len(ce['contractions']),
                     'actorContractions': len(pe['contractions']), 'newNativeSDPA': 0, 'newFullTransformer': 0, 'newMLP': 0, 'newImages': 0,
                     'distinctParameterBytes': sum({x['name']: x['bytes'] for e in [ce, pe] for x in e['loadedParameters']}.values()),
                     'processParameterBytes': sum(x['bytes'] for e in [ce, pe] for x in e['loadedParameters'])}}
