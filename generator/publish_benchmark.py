"""Export authenticated prediction-only results into the website. No model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(run,target):
    receipt=json.loads((run/('collection-v2.json' if (run/'collection-v2.json').exists() else 'collection.json')).read_text());request=json.loads((run/'request.json').read_text());root=run/'result'
    report=json.loads((root/'report.json').read_text())
    if sha(root/'report.json')!=receipt['reportSHA256'] or report['requestSHA256']!=receipt['requestSHA256'] or request['requestSHA256']!=receipt['requestSHA256']:
        raise ValueError('Measurement receipt does not bind this report/request')
    if report['counts']['nativeDenoiser']!=0 or report['nativeTransformerLoaded'] or report['counts']['predictedTransitions']!=112*len(report['samples']):
        raise ValueError('Prediction-only execution census differs')
    target.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/'report.json',target/'worker-report.json')
    shutil.copyfile(run/'request.json',target/'request.json')
    for sample in report['samples']:
        image=sample['image'];path=root/image['path']
        if sha(path)!=image['sha256'] or receipt['remoteInventory']['files'][image['path']]['sha256']!=image['sha256']:raise ValueError('Original image bytes differ')
        shutil.copyfile(path,target/image['path'])
    raw=dict(report)
    raw.update(jobID=receipt['jobID'],requestSHA256=receipt['requestSHA256'],reportSHA256=receipt['reportSHA256'],payloadSHA256=receipt['payload']['sha256'],jobWallSeconds=receipt['schedulerResult']['elapsed_s'])
    (target/'report.json').write_text(json.dumps(raw,indent=2,sort_keys=True)+'\n')
    (target/'benchmarks.json').write_text(json.dumps({**raw,'schema':1},separators=(',',':'),sort_keys=True)+'\n')
    (target/'provenance.json').write_text(json.dumps({'schema':1,'jobID':receipt['jobID'],'requestSHA256':receipt['requestSHA256'],
        'payloadSHA256':receipt['payload']['sha256'],'originalReportSHA256':receipt['reportSHA256'],
        'executedPayloadSeal':receipt['executedPayloadSeal'],'payloadCustodyRequired':receipt['payloadCustodyRequired'],'payloadCustody':receipt.get('payloadCustody'),'sourceAudit':receipt['sourceAudit'],
        'originalResultInventory':receipt['remoteInventory'],
        'publicArtifactSHA256':{p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file() and p.name!='provenance.json'},
        'schedulerMeasuredResources':{k:v for k,v in receipt['schedulerResult'].items() if any(s in k for s in ['rss','vram','elapsed','wall','peak'])},
        'scope':'Original PNGs, raw worker measurements, original request, scheduler declared/sealed payload and authenticated post-run source/result hashes. Executed-seal availability is explicit; post-run hashes are not retroactive execution attestation. No browser or exporter model forwards.'},indent=2,sort_keys=True)+'\n')
    return{'images':len(report['samples']),'jobID':receipt['jobID'],'output':str(target)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'viz/public/data/generator')
    args=p.parse_args();print(json.dumps(publish(args.run.resolve(),args.output.resolve())))
