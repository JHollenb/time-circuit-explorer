"""Only published scalar errors, with the paid composed arithmetic scope explicit."""
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


def execute(report_path,output,notebook):
    report_path=Path(report_path);r=json.loads(report_path.read_text());output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,6.3));fig.patch.set_facecolor('#f8fafc');x=np.arange(3);width=.33
    for ax,key,title in zip(axes,['errorOverTargetRMS','centeredErrorOverCenteredTargetRMS'],['Complete finite increment','Changing increment within each stream']):
        maximum=0.
        for index,(case,label,color) in enumerate([('old82vs73','Prior input-basis producer82','#64748b'),('vs73','New row-output producer83','#2563eb')]):
            values=[100*r['panels'][stream]['exitIncrement'][case][key] for stream in ['all','text','image']];maximum=max(maximum,max(values))
            bars=ax.bar(x+(index-.5)*width,values,width,color=color,label=label);ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=5,fontsize=10)
        ax.set_xticks(x,['All rows','Text','Image']);ax.set_ylim(0,max(1,maximum)*1.3);ax.set_ylabel('Error / dense73 target RMS (%)');ax.set_title(title,loc='left',fontsize=12,pad=14);ax.set_facecolor('#f8fafc');ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True);ax.tick_params(axis='x',length=0)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.055,.815),ncol=2,frameon=False,fontsize=10)
    fig.suptitle('Can the same input-derived row space carry the complete write?',x=.055,y=.97,ha='left',fontsize=18,fontweight='bold',color='#0f172a')
    fig.text(.055,.905,'One supplied current state · clock 0 / single block 0 · rank128 per stream · all1,536 dependent rows retained',fontsize=11,color='#475569')
    primary=r['panels']['all']['exitIncrement']['vs82']['errorOverTargetRMS'];counts=r['panels']['all']['exitIncrement']['rowsNewTotalVsOldTotal']
    fig.text(.055,.16,f'Increment difference vs prior82: {100*primary:.2f}% RMS. Against dense73: {counts["improve"]} rows improve / {counts["worsen"]} worsen / {counts["equal"]} tie.',fontsize=10)
    fig.text(.055,.105,'New output path: 12.684 B matrix MACs vs 57.982 B prior fullout; diagnostics +0.050 B, QR ≈95.071 M FLOPs additional.',fontsize=10)
    fig.text(.055,.055,'Prospective composed producer: 32.011 B MACs retains 19.327 B prior GU formation. Sources/maps paid; no image, runtime or zero-neural claim.',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.065,right=.985,top=.72,bottom=.29,wspace=.22)
    path=output/'row-output.png';fig.savefig(path,dpi=160,facecolor=fig.get_facecolor());plt.close(fig);notebook=Path(notebook);shutil.copyfile(path,notebook)
    value={'schema':'tce-current-row-output-figure-v1','sourceReportSHA256':sha(report_path),'plotSourceSHA256':sha(__file__),'imageSHA256':sha(path),'notebookCopy':str(notebook.resolve()),'scope':'Published scalar JSON visualization only, no scientific arrays or model evaluation.'}
    with (output/'figure.json').open('x') as f:json.dump(value,f,indent=2)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('report');parser.add_argument('output');parser.add_argument('notebook');args=parser.parse_args();print(json.dumps(execute(args.report,args.output,args.notebook),indent=2))
