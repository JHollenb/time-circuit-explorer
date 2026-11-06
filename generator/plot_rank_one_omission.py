"""Plot published scalar descriptions; no model or tensor analysis."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from encoder_atoms import read,save,sha


def render(source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False);r=read(source/'report.json');regions=['all','active_both','padding_both'];names=['All rows','Active','Padding'];terms=['mass','linearValue','quadraticValue'];colors=['#91a0b9','#6b5bc4','#2e9d90'];labels=['Mass','Linear value','Quadratic value']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#273245','axes.labelcolor':'#374151','svg.hashsalt':'tce-rank-one-omission-v1'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4));fig.patch.set_facecolor('#fbfcfe');x=np.arange(3);width=.23
    pooled={region:next(z for z in r['scores'] if z.get('kind')=='paired' and z['region']==region and z['queryHead'] is None) for region in regions}
    for j,(term,label,color) in enumerate(zip(terms,labels,colors)):
        values=[100*pooled[region]['signedAlignmentOverInitialError'][term] for region in regions];bars=axes[0].bar(x+(j-1)*width,values,width,color=color,label=label)
        for b,value in zip(bars,values):axes[0].text(b.get_x()+width/2,value-1.7,f'{value:.1f}',ha='center',va='top',fontsize=8)
    axes[0].axhline(0,color='#64748b',lw=.7);axes[0].set_xticks(x,names);axes[0].set_ylim(-95,6);axes[0].set_ylabel('Signed alignment / initial error energy (%)');axes[0].set_title('Signed alignment with the initial head error',weight='bold',fontsize=11);axes[0].legend(frameon=False,fontsize=8,loc='lower left',ncol=3);axes[0].grid(axis='y',alpha=.15);axes[0].set_axisbelow(True)
    directional=[];finite=[]
    for region in regions:
        rows=[z for z in r['scores'] if z.get('kind')=='paired' and z['region']==region and z['queryHead'] is not None];directional.append(sum(z['signedInitialErrorAlignment']['linearValue']<0 for z in rows));finite.append(sum(z['signedInitialErrorAlignment']['linearValue']+z['termEnergies']['linearValue']<0 for z in rows))
    for offset,values,label,color in [(-.16,directional,'Infinitesimal positive dose','#b8aee0'),(.16,finite,'Unit algebraic addition','#6b5bc4')]:
        bars=axes[1].bar(x+offset,values,.3,label=label,color=color)
        for b,value in zip(bars,values):axes[1].text(b.get_x()+.15,value+.5,str(value),ha='center',fontsize=10)
    axes[1].set_xticks(x,names);axes[1].set_ylim(0,49);axes[1].set_ylabel('Heads with lower squared error (of 32)');axes[1].set_title('Linear-value benefit depends on dose',weight='bold',fontsize=11);axes[1].legend(frameon=False,fontsize=8,loc='upper left');axes[1].grid(axis='y',alpha=.15);axes[1].set_axisbelow(True)
    fig.suptitle('The omitted linear-value term carries useful signed structure',x=.035,ha='left',fontsize=15,weight='bold');fig.text(.035,.035,'One retained prompt pair. Saved-array head-space identities; isolated terms receive no BF16 re-quantization or original MLP/state consumer.',fontsize=9,color='#64748b');fig.tight_layout(rect=[.02,.11,.99,.9]);fig.savefig(out/'omission-comparison.svg',metadata={'Date':None});fig.savefig(out/'omission-comparison.png',dpi=150);plt.close(fig)
    p=out/'omission-comparison.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n');save(out/'figure.json',{'schema':'tce-rank-one-omission-figure-v1','sourceReportSHA256':sha(source/'report.json'),'rendererSHA256':sha(__file__),'matplotlib':matplotlib.__version__,'files':{p.name:sha(p) for p in [out/'omission-comparison.svg',out/'omission-comparison.png']},'scope':'Derived published scalar plot, not another experiment. Signed alignments are head-space path derivatives; unit addition uses E(e+t)-E(e)=2E[e*t]+E[t²], no isolated-term quantization or consumer.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('out');a=p.parse_args();render(a.source,a.out)
