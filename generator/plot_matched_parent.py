"""Plot public five-corner scalar summaries; no scientific array or model reads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def execute(source,output,notebook):
    source=Path(source);r=json.loads(source.read_text());output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,7));bg='#f8fafc';fig.patch.set_facecolor(bg);x=np.arange(3)
    groups=['all','text','image']
    cases=[('matchedIncrement','errorOverTargetRMS','Parent input','#2563eb'),
           ('ownIncrement','errorOverTargetRMS','Own input','#64748b'),
           ('matchedIncrement','centeredErrorOverCenteredTargetRMS','Parent, centered','#7c3aed'),
           ('ownIncrement','centeredErrorOverCenteredTargetRMS','Own, centered','#c4b5fd')]
    for i,(port,key,label,color) in enumerate(cases):
        values=[100*r['panels'][s][port][key] for s in groups]
        bars=axes[0].bar(x+(i-1.5)*.2,values,.2,label=label,color=color)
        axes[0].bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=4,fontsize=8)
    axes[0].set_ylim(0,54);axes[0].set_ylabel('Error / paid70 increment RMS (%)')
    axes[0].set_title('Changing the input does not remove the local bias',loc='left',fontsize=11,pad=14)
    axes[0].legend(frameon=False,fontsize=9,ncol=2,loc='upper left')
    cases=[('E0_energy','Same-parent error E0','#2563eb'),('U_energy','Response beyond carry U','#7c3aed'),
           ('twiceE0UInner','Signed cross term','#f43f5e'),('W_energy','Actual increment error W','#64748b')]
    for i,(key,label,color) in enumerate(cases):
        values=[r['panels'][s]['signedAccount'][key]/1e6 for s in groups]
        bars=axes[1].bar(x+(i-1.5)*.2,values,.2,label=label,color=color)
        axes[1].bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4,fontsize=8)
    axes[1].set_ylim(-6,27);axes[1].set_ylabel('Signed squared norm (millions)')
    axes[1].set_title('W = E0 + U: keep the cancellation visible',loc='left',fontsize=11,pad=14)
    axes[1].axhline(0,color='#94a3b8',linewidth=.7);axes[1].legend(frameon=False,fontsize=9,ncol=2,loc='upper left')
    for ax in axes:
        ax.set_facecolor(bg);ax.set_xticks(x,['All rows','Text','Image']);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    fig.suptitle('The same reduced block still errs on the parent state',x=.055,y=.96,ha='left',fontsize=19,fontweight='bold',color='#0f172a')
    fig.text(.055,.895,'One missing corner · identical frozen program / two inputs · clock0 single1 · 1,536 dependent rows',fontsize=11,color='#475569')
    fig.text(.055,.20,'Matched input: 1,398 increment rows improve / 138 worsen. Same-parent error combines approximation and finite slicing.',fontsize=10)
    fig.text(.055,.145,'Candidate response T includes carry D; U = T - D. T/D norm = 1.189. Signed components are not positive causal shares.',fontsize=10)
    fig.text(.055,.09,'94.625 B learned MACs + 4 QR calls (~492.132 M FLOPs) + CPU scalar/data analysis. Assay: 38.13 seconds.',fontsize=10)
    fig.text(.055,.04,'Paid70 dense-affine / moment-attention reference. No native-propagation attribution, new image or arbitrary-prompt zero-neural result.',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.065,right=.985,top=.79,bottom=.31,wspace=.25)
    path=output/'matched-parent.png';fig.savefig(path,dpi=160,facecolor=bg);plt.close(fig);shutil.copyfile(path,notebook)
    v={'schema':'tce-matched-parent-figure-v1','sourceReportSHA256':sha(source),'plotSourceSHA256':sha(__file__),'imageSHA256':sha(path),'notebookCopy':str(Path(notebook).resolve()),'scope':'Published scalar JSON only; no scientific arrays or model evaluations. Signed energy is not causal allocation.'}
    with (output/'figure.json').open('x') as f:json.dump(v,f,indent=2)
    return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('notebook');a=p.parse_args()
    print(json.dumps(execute(a.source,a.output,a.notebook)))
