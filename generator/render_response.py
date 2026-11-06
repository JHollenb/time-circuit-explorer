"""Render one new dose from a compiled, context-bound RGB chart. No model."""
import argparse
import json
from pathlib import Path
import resource
import time

# Installs the neural-import exclusion before NumPy/Pillow load.
from response_chart import np, Image, encode, evaluate, guard, sha, write

def render(bundle,context,alpha,output):
    opened=guard();start=time.perf_counter()
    if output.exists() or Path(str(output)+'.json').exists():raise ValueError('Use a new output filename')
    with np.load(bundle,allow_pickle=False) as archive:
        if context not in archive.files:raise ValueError('Unknown bound context; no prompt/seed generalization')
        anchors=archive[context].astype(np.float32)
    begin,cpu=time.perf_counter(),time.process_time();pixels=encode(evaluate(anchors,alpha,'piecewise'))
    compute=time.perf_counter()-begin;cpu=time.process_time()-cpu
    output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(pixels).save(output)
    receipt={'schema':'tce-compiled-response-image-v1','context':context,'alpha':alpha,'method':'piecewise','bundleSHA256':sha(bundle),
             'pngSHA256':sha(output),'computeSeconds':compute,'cpuSeconds':cpu,'loadComputeAndSaveSeconds':time.perf_counter()-start,
             'peakRssBytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if __import__('sys').platform=='darwin' else 1024),
             'opened':sorted(set(opened)),'neuralEvaluations':0,'scope':'Calibration RGB interpolation on a supplied context/dose curve. No native accuracy measurement for a new dose.'}
    write(Path(str(output)+'.json'),receipt);print(json.dumps(receipt))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--context',required=True);p.add_argument('--alpha',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.bundle,a.context,a.alpha,a.output)
