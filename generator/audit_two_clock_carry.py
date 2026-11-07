"""Audit source/release/custody and summarize every published token observation."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from encoder_atoms import read,sha,identity


def audit(source):
    p=Path(source);r=read(p/'request.json');s=read(p/'inputs.json');report=read(p/'report.json');v=read(p/'retained-array-verification.json');prov=read(p/'provenance.json')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Published member differs')
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256'] or report['candidateSealSHA256']!=r['candidateSealSHA256'] or v['candidateSealSHA256']!=r['candidateSealSHA256']:raise ValueError('Source/release differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed source differs')
    items=list(s['primaryRecords'].values());new=read(p/'fields.json')
    for lanes in s['fields'].values():
        for fs in lanes.values():items+=list(fs.values())
    for fs in new.values():items+=list(fs.values())
    for item in items:
        if v['files'][item['path']]['sha256']!=item['sha256']:raise ValueError('Complete byte custody absent')
    for stream in prov['losslessStreamTransports']:
        if hashlib.sha256(gzip.decompress((p/stream['published']).read_bytes())).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless observations differ')
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()]
    if Counter(x['domain'] for x in rows)!={'00':1024,'10':1024,'color':1024} or len(rows)!=report['observations']:raise ValueError('Observation coverage differs')
    trends=[]
    for domain in ['00','10','color']:
        obs=[x for x in rows if x['domain']==domain];g=sorted(x['outgoingIncomingGain'] for x in obs if x['outgoingIncomingGain'] is not None)
        trends.append({'domain':domain,'tokenRows':len(obs),'zeroIncomingNormRows':sum(x['zeroIncomingNorm'] for x in obs),
            'gainBelowOneRows':sum(x<1 for x in g),'gainAboveOneRows':sum(x>1 for x in g),'gainExactlyOneRows':sum(x==1 for x in g),
            'definedGains':len(g),'gainMin':g[0] if g else None,'gainMedian':((g[(len(g)-1)//2]+g[len(g)//2])/2) if g else None,'gainMax':g[-1] if g else None,
            'parallelCancellationRows':sum(x['projection'] is not None and x['projection']['lambda']<0 for x in obs),
            'parallelReinforcementRows':sum(x['projection'] is not None and x['projection']['lambda']>0 for x in obs)})
    return {'schema':'tce-two-clock-carry-mechanics-v1','requestSHA256':r['requestSHA256'],'reportSHA256':sha(p/'report.json'),
        'candidateSealSHA256':r['candidateSealSHA256'],'ownedSourcePins':len(r['ownedSources']),'inputFields':16,
        'newArrays':sum(len(fs) for fs in new.values()),'observations':len(rows),'tokenTrends':trends,
        'scope':'Metadata/source/release/full-byte/lossless coverage; row summaries describe dependent observations and do not certify global contraction or image semantics.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');a=p.parse_args();print(json.dumps(audit(a.source),indent=2))
