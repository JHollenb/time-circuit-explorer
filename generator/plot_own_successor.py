"""Plot public scalar reports only; no model or scientific arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def execute(report_path, output, notebook):
    report_path=Path(report_path); r=json.loads(report_path.read_text())
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    bg='#f8fafc';fig.patch.set_facecolor(bg);x=np.arange(3)
    groups=['all','text','image']
    cases=[('incoming','Incoming discrepancy','#94a3b8'),
           ('outgoing','Own outgoing discrepancy','#2563eb'),
           ('holdOutgoing','Hold incoming state','#e11d48')]
    for i,(port,label,color) in enumerate(cases):
        values=[100*r['panels'][s][port]['errorOverTargetRMS'] for s in groups]
        bars=axes[0].bar(x+(i-1)*.25,values,.25,label=label,color=color)
        axes[0].bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4,fontsize=9)
    for i,(key,label,color) in enumerate([
            ('errorOverTargetRMS','Complete increment','#2563eb'),
            ('centeredErrorOverCenteredTargetRMS','Stream-centered increment','#7c3aed')]):
        values=[100*r['panels'][s]['increment'][key] for s in groups]
        bars=axes[1].bar(x+(i-.5)*.3,values,.3,label=label,color=color)
        axes[1].bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4,fontsize=9)
    for ax,title in zip(axes,['Does the next state move closer than holding?','How well is the actual next increment reproduced?']):
        ax.set_facecolor(bg);ax.set_xticks(x,['All rows','Text','Image'])
        ax.set_ylim(0,51);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
        ax.set_title(title,loc='left',fontsize=11,pad=16)
        ax.legend(frameon=False,fontsize=9,loc='upper left')
    axes[0].set_ylabel('Error / corresponding state target RMS (%)')
    axes[1].set_ylabel('Error / paid70 increment RMS (%)')
    fig.suptitle('A compressed state advances once — with growing error',
                 x=.055,y=.96,ha='left',fontsize=19,fontweight='bold',color='#0f172a')
    fig.text(.055,.895,'Actual cycle86 output → clock0 / single1 · one supplied prompt/seed · 1,536 dependent rows',
             color='#475569',fontsize=11)
    fig.text(.055,.20,'Every outgoing row beats holding. Image discrepancy still grows 10.20% → 14.43%; centered: 26.99% → 33.07%.',fontsize=10)
    fig.text(.055,.145,'Signed energy: 10.448 M incoming + 18.894 M increment − 3.333 M cross = 26.009 M outgoing.',fontsize=10)
    fig.text(.055,.09,'94.625 B candidate MACs + 0.755 B diagnostics; QR ≈492.132 M FLOPs additional. Assay: 53.43 seconds.',fontsize=10)
    fig.text(.055,.04,'Paid70 dense-affine / moment-attention comparison. Original maps and clocks paid. No new image or zero-neural capability.',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.07,right=.985,top=.79,bottom=.31,wspace=.26)
    path=output/'own-successor.png';fig.savefig(path,dpi=160,facecolor=bg);plt.close(fig)
    notebook=Path(notebook);shutil.copyfile(path,notebook)
    v={'schema':'tce-own-successor-figure-v1','sourceReportSHA256':sha(report_path),
       'plotSourceSHA256':sha(__file__),'imageSHA256':sha(path),'notebookCopy':str(notebook.resolve()),
       'scope':'Public scalar JSON visualization only, no scientific arrays or model evaluations. Different state/step denominators remain explicit.'}
    with (output/'figure.json').open('x') as f:json.dump(v,f,indent=2)
    return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('report');p.add_argument('output');p.add_argument('notebook');a=p.parse_args()
    print(json.dumps(execute(a.report,a.output,a.notebook)))

