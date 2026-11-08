"""Plot published scalar accounts; never load scientific tensors or models."""
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


def execute(source,supplement,output,notebook):
    source=Path(source);supplement=Path(supplement)
    r=json.loads(source.read_text());s=json.loads(supplement.read_text())
    if sha(source)!=s['sourceReportSHA256']:raise ValueError('Scalar supplement source differs')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,8));bg='#f8fafc';fig.patch.set_facecolor(bg)
    groups=['all','text','image'];x=np.arange(3)
    cases=[('E0OverDenseParentIncrementRMS','Parent input','#2563eb'),
           ('E1OverActualDenseOwnIncrementRMS','Own input','#64748b'),
           ('centeredE0OverDenseParentIncrementRMS','Parent, centered','#7c3aed'),
           ('centeredE1OverActualDenseOwnIncrementRMS','Own, centered','#c4b5fd')]
    for i,(key,label,color) in enumerate(cases):
        values=[100*s['panels'][g][key] for g in groups]
        bars=axes[0].bar(x+(i-1.5)*.2,values,.2,label=label,color=color)
        axes[0].bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=4,fontsize=8)
    axes[0].set_ylim(0,53);axes[0].set_ylabel('Error / actual comparator increment RMS (%)')
    axes[0].set_title('Local error stays large on both inputs',loc='left',fontsize=12,pad=14)
    axes[0].legend(frameon=False,ncol=2,fontsize=9,loc='upper left')
    cases=[('E0_energy','Parent error E0','#2563eb'),('Q_energy','Error change Q','#7c3aed'),
           ('twiceE0QInner','Signed cross term','#f43f5e'),('E1_energy','Own-input error E1','#64748b')]
    for i,(key,label,color) in enumerate(cases):
        values=[r['panels'][g]['signedAccount'][key]/1e6 for g in groups]
        bars=axes[1].bar(x+(i-1.5)*.2,values,.2,label=label,color=color)
        axes[1].bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4,fontsize=8)
    axes[1].set_ylim(-10.5,22.5);axes[1].set_ylabel('Signed squared norm (millions)')
    axes[1].set_title('E1 = E0 + Q: stable size hides changing direction',loc='left',fontsize=11,pad=14)
    axes[1].axhline(0,color='#94a3b8',linewidth=.7);axes[1].legend(frameon=False,ncol=2,fontsize=9,loc='upper left')
    for ax in axes:
        ax.set_facecolor(bg);ax.set_xticks(x,['All rows','Text','Image']);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    fig.suptitle('Similar error norms do not mean the same error',x=.055,y=.965,ha='left',fontsize=20,fontweight='bold',color='#0f172a')
    fig.text(.055,.91,'Missing dense-comparator corner · clock 0 / single block 1 · one input pair / 1,536 dependent rows',fontsize=11,color='#475569')
    fig.text(.055,.235,'Pooled: ||E1|| / ||E0|| = 1.009, but ||Q|| / ||E0|| = 0.718. Error-vector angle: 41.87 degrees.',fontsize=11)
    fig.text(.055,.18,'Q = reduced response - comparator response; ||Q|| / ||TN|| = 85.48%. Signed cancellation is not causal allocation.',fontsize=10)
    fig.text(.055,.125,'200.322 B matrix MACs + CPU scalar/data work; 23.90 scheduler seconds. Original learned maps/state/clocks remain paid.',fontsize=10)
    fig.text(.055,.07,'Comparator: original dense affine maps with frozen moment attention. No original full-attention sensitivity or global bound.',fontsize=10,color='#475569')
    fig.text(.055,.03,'No new image or arbitrary-prompt zero-neural capability. Each error uses its own actual comparator increment denominator.',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.07,right=.985,top=.81,bottom=.34,wspace=.28)
    path=output/'dense-corner.png';fig.savefig(path,dpi=160,facecolor=bg);plt.close(fig)
    shutil.copyfile(path,notebook)
    receipt={'schema':'tce-dense-corner-figure-v1','sourceReportSHA256':sha(source),'supplementSHA256':sha(supplement),'plotSourceSHA256':sha(__file__),'imageSHA256':sha(path),'notebookCopy':str(Path(notebook).resolve()),'scope':'Published scalar JSON only; no scientific arrays/model evaluations. Actual-increment denominators and signed accounts; no causal shares or image/zero-neural claim.'}
    with (output/'figure.json').open('x') as f:json.dump(receipt,f,indent=2)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('supplement');p.add_argument('output');p.add_argument('notebook');a=p.parse_args()
    print(json.dumps(execute(a.source,a.supplement,a.output,a.notebook)))
