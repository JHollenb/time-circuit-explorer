"""Plot published scalar errors with their actual increment/response denominators."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(source,scheduler,output,notebook):
    source=Path(source);scheduler=Path(scheduler);r=json.loads(source.read_text());job=json.loads(scheduler.read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,8));bg='#f8fafc';fig.patch.set_facecolor(bg)
    groups=['all','text','image'];x=np.arange(3)
    for ax,ports,title,ylabel in [(axes[0],['dynamicOwnIncrement','fixedOwnIncrement'],'Actual state updates become less accurate','Error / actual own-input increment RMS (%)'),
                                  (axes[1],['dynamicResponse','fixedResponse'],'Response error improves, with mixed rows','Response error / comparator response RMS (%)')]:
        cases=[(ports[0],'errorOverTargetRMS','Dynamic','#2563eb'),
               (ports[1],'errorOverTargetRMS','Frozen','#7c3aed'),
               (ports[0],'centeredErrorOverCenteredTargetRMS','Dynamic, centered','#93c5fd'),
               (ports[1],'centeredErrorOverCenteredTargetRMS','Frozen, centered','#c4b5fd')]
        maxima=[]
        for i,(port,key,label,color) in enumerate(cases):
            values=[100*r['panels'][g][port][key] if r['panels'][g][port][key] is not None else np.nan for g in groups]
            maxima.extend(v for v in values if np.isfinite(v))
            bars=ax.bar(x+(i-1.5)*.2,values,.2,label=label,color=color)
            ax.bar_label(bars,labels=[f'{v:.1f}' if np.isfinite(v) else 'undefined' for v in values],padding=4,fontsize=8)
        ax.set_ylim(0,max(maxima,default=1)*1.3);ax.set_ylabel(ylabel)
        ax.set_title(title,loc='left',fontsize=12,pad=14);ax.legend(frameon=False,ncol=2,fontsize=9,loc='upper left')
        ax.set_facecolor(bg);ax.set_xticks(x,['All rows','Text','Image']);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    fig.suptitle('Better response error, worse state updates',x=.055,y=.965,ha='left',fontsize=21,fontweight='bold',color='#0f172a')
    fig.text(.055,.91,'One parent-88 template / two existing inputs · clock 0 / single block 1 · 1,536 dependent rows',fontsize=11,color='#475569')
    fig.text(.055,.24,'Freezing improves response error on 1,216 rows / worsens 320; own-input error improves on 175 / worsens 1,361.',fontsize=10)
    fig.text(.055,.185,'Observed parent exit is byte-identical to the recorded exit (A0 = 0). Historical CUDA operand layout remains unattested.',fontsize=10)
    fig.text(.055,.13,f"Two evaluations: {r['work']['candidateMACs']/1e9:.3f} B matrix MACs; {job['result']['elapsed_s']:.2f} scheduler seconds. Original maps/templates/history paid.",fontsize=10)
    fig.text(.055,.075,'Dense comparator retains moment attention. Signed interactions matter; no original full-attention sensitivity or runtime-gain claim.',fontsize=10,color='#475569')
    fig.text(.055,.03,'No new generated image or arbitrary-prompt zero-neural capability. Figure uses published scalar JSON only.',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.07,right=.985,top=.81,bottom=.34,wspace=.28)
    path=output/'template-transport.png';fig.savefig(path,dpi=160,facecolor=bg);plt.close(fig);shutil.copyfile(path,notebook)
    receipt={'schema':'tce-template-transport-figure-v1','sourceReportSHA256':sha(source),'schedulerSHA256':sha(scheduler),
        'plotSourceSHA256':sha(__file__),'imageSHA256':sha(path),'notebookCopy':str(Path(notebook).resolve()),
        'scope':'Published scalar JSON only; actual-increment and comparator-response denominators remain distinct. No scientific arrays/models evaluated.'}
    with (output/'figure.json').open('x') as f:json.dump(receipt,f,indent=2)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('scheduler');p.add_argument('output');p.add_argument('notebook');a=p.parse_args()
    print(json.dumps(execute(a.source,a.scheduler,a.output,a.notebook)))
