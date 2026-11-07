"""Saved-only comparison after both producer and reader process exits."""
from pathlib import Path
from encoder_atoms import read, sha
from paired_fields import load, metrics
from projected_complement_analysis import execute as compare_readers


def execute(root):
    root = Path(root)
    out = root / 'response-result'
    r, targets = read(root / 'request.json'), read(root / 'targets.json')
    producer, seal = read(out / 'producer.json'), read(out / 'producer-seal.json')
    if seal['producerSHA256'] != sha(out / 'producer.json') or seal['requestSHA256'] != r['requestSHA256']:
        raise ValueError('Producer seal differs before source scoring')
    report = compare_readers(root)
    mapping = {'R': 'text_in', 'A': 'text_attention_return_used', 'U': 'text_after_attention',
               'X': 'text_mlp_input', 'M': 'text_mlp_return_used', 'T': 'text_out'}
    source_metrics = []
    for role in ['P', 'Q']:
        for own, native in mapping.items():
            candidate = load(producer['sources'][role]['fields'][own])
            reference = load(targets['sourcePorts'][role][native])[:, r['support']]
            source_metrics.append({'role': role, 'field': own, **metrics(candidate, reference)})
    text_metrics = []
    for arm in ['QQQ', 'PQP']:
        candidate = load(producer['splice']['fields'][arm])
        reference = load(targets['sourceTextTargets'][arm])[:, r['support']]
        text_metrics.append({'arm': arm, **metrics(candidate, reference)})
    se, pe = read(out / 'producer-execution.json'), read(out / 'primary-execution.json')
    report.update(producerSealSHA256=sha(out / 'producer-seal.json'), sourceMetrics=source_metrics,
                  selectedTextMetrics=text_metrics, sourceMatrixMACs=se['nominalMatrixMACs'],
                  totalIncrementalMatrixMACs=se['nominalMatrixMACs'] + pe['nominalMatrixMACs'],
                  historicalProjectedCompilerMatrixMACs=9663676416)
    report['work'].update(sourceAffines=len(se['linears']), sourceSDPA=len(se['attentions']), newNativeSDPA=2,
                          sourceMLPs=2, newMLP=2, newFullTransformer=0,
                          distinctBF16ParameterBytes=sum({x['name']: x['bytes'] for e in [se, pe] for x in e['loadedParameters']}.values()),
                          processBF16ParameterBytes=sum(x['bytes'] for e in [se, pe] for x in e['loadedParameters']))
    return report
