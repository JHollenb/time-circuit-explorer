"""Published scalar reconstruction and consumer errors; no scientific arrays."""
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
    report_path=Path(report_path);report=json.loads(report_path.read_text());output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,6.4),gridspec_kw={'width_ratios':[1,1.5]});fig.patch.set_facecolor('#f8fafc')
    groups=[['input','gateRaw','upRaw'],['activation','product','preResidualUpdate','exitIncrement']]
    labels=[['Input Z','Gate G','Up U'],['SiLU(G)','MLP product','Full out write','Exit increment']]
    for ax,ports,ticks in zip(axes,groups,labels):
        maximum=0.;x=np.arange(len(ports));width=.24
        for j,(stream,color) in enumerate([('all','#64748b'),('text','#2563eb'),('image','#e76f36')]):
            values=[100*(report['inputReconstruction'][stream] if port=='input' else report['panels'][stream][port])['errorOverTargetRMS'] for port in ports]
            maximum=max(maximum,max(values));bars=ax.bar(x+(j-1)*width,values,width,color=color,label=stream.title())
            ax.bar_label(bars,labels=[f'{value:.1f}' for value in values],padding=4,fontsize=9)
        ax.set_xticks(x,ticks);ax.set_ylim(0,max(1,maximum)*1.28);ax.set_ylabel('Error / target RMS (%)');ax.set_facecolor('#f8fafc');ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True);ax.tick_params(axis='x',length=0)
    axes[0].set_title('What the current-input basis retains',loc='left',fontsize=12,pad=15)
    axes[1].set_title('What its unchanged native local consumer receives',loc='left',fontsize=12,pad=15)
    axes[0].legend(frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(-.02,1.22))
    fig.suptitle('A current-state basis must preserve the write, not just the input',x=.055,y=.97,ha='left',fontsize=19,fontweight='bold',color='#0f172a')
    fig.text(.055,.905,'One paid source73 state · clock 0 / single block 0 · 512 text + 1,024 image rows · rank128 per stream',fontsize=11,color='#475569')
    exits=report['panels'];text=' · '.join(f"{stream} {100*exits[stream]['exit']['errorOverTargetRMS']:.3f}%" for stream in ['all','text','image'])
    fig.text(.055,.165,'Full-exit error: '+text+'. The increment denominator removes the carried residual.',fontsize=10)
    fig.text(.055,.115,'Candidate: 77.309 B matrix MACs vs matched GU + full output 144.955 B (46.67% fewer); diagnostics add 0.705 B.',fontsize=10)
    fig.text(.055,.065,'QR ≈397.1 M FLOPs and scalar/data work are additional. Original source/maps remain paid; no runtime, trajectory or zero-neural image claim.',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.055,right=.985,top=.735,bottom=.29,wspace=.24)
    path=output/'gate-up-basis.png';fig.savefig(path,dpi=160,facecolor=fig.get_facecolor());plt.close(fig);notebook=Path(notebook);shutil.copyfile(path,notebook)
    value={'schema':'tce-current-state-gate-up-figure-v1','sourceReportSHA256':sha(report_path),'plotSourceSHA256':sha(__file__),'imageSHA256':sha(path),'notebookCopy':str(notebook.resolve()),
           'scope':'Published scalar JSON only; no scientific tensor/model calculation. Complete dependent rows remain in the linked evidence.'}
    with (output/'figure.json').open('x') as stream:json.dump(value,stream,indent=2)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('report');parser.add_argument('output');parser.add_argument('notebook');args=parser.parse_args()
    print(json.dumps(execute(args.report,args.output,args.notebook),indent=2))
