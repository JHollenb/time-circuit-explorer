"""Summarize published scalar rows, with no saved tensor/model loading."""
import json
from pathlib import Path
import shutil
import statistics

from encoder_atoms import read, save, sha
from publish_spatial_norm import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/spatial-norm-capacity';e = root / 'docs/results/spatial-norm-evidence'
    e.mkdir(parents=True, exist_ok=True)
    mechanics = audit(p);r = read(p / 'report.json');v = read(p / 'retained-array-verification.json')
    arrays = {x['path']:x for n in ['intake.json','candidate.json','report.json']
              for x in read(p/n).get('fields',read(p/n).get('retainedFields',{})).values()}
    def distribution(name):
        values = [x[name] for x in r['queryHeads']]
        return {'min':min(values),'median':statistics.median(values),'max':max(values),'mean':statistics.mean(values)}
    result = {'schema':'tce-spatial-norm-scalars-v1','reportSHA256':sha(p/'report.json'),
        'norms':{k:v for k,v in r['norms'].items() if k!='heads'},
        'normHeadsBetterThanHold':sum(x['pooled']['response']['relativeRms']<1 for x in r['norms']['heads']),
        'queryHeadDistributions':{n:distribution(n) for n in ['unweightedDeltaRMS','weightedDeltaMean',
            'weightedCenteredVariance','exactKL','directKLResidual','logPartitionDifference','textReferenceMass']},
        'strataWeightedMeans':{n:{k:statistics.mean(x['strata'][n][k] for x in r['queryHeads'])
            for k in ['referenceMass','weightedCenteredVariance']} for n in ['sampled','unsampledInterior','unsampledEdge']},
        'newArrays':len(arrays),'newArrayBytes':sum(x['bytes'] for x in arrays.values()),
        'fullByteFiles':len(v['files']),'fullByteCount':v['totalBytesVerified'],
        'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'localFreeBytesObserved':shutil.disk_usage(root).free,
        'scope':'Published scalar/address rows only. Current native knots paid; sampled equality is supplied capacity, not independent predictions. No consumer/image/generalization/serving certification.'}
    save(e/'mechanics.json',mechanics);save(e/'scalar-summary.json',result)
    return result


if __name__=='__main__':
    r=execute();print(json.dumps({k:r[k] for k in ['newArrays','newArrayBytes','fullByteFiles','fullByteCount','NPYPayloadWitnesses']},indent=2))
