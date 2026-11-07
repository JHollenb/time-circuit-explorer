"""Render published scalar representation floors and signed Gram; no real arrays."""
from pathlib import Path
import json
import shutil
from encoder_atoms import read,save,sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/own-state-gated-covariance';r=read(p/'report.json');out=root/'docs/results/own-state-gated-covariance-figure';out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(a,b)=plt.subplots(1,2,figsize=(11.8,5.5),facecolor='#fbfcfe',gridspec_kw={'width_ratios':[1.25,1]})
    for ax in [a,b]:ax.set_facecolor('#fbfcfe')
    names=['all','text','image'];x=np.arange(3)
    for offset,field,label,color in [(-.18,'oracleResidualOverOriginalRMS','Best supplied constant','#6e80ba'),(.18,'meanCovarianceResidualOverOriginalRMS','Covariance mean only','#5c9c89')]:
        values=[100*r['panels'][name][field] for name in names];bars=a.bar(x+offset,values,width=.32,label=label,color=color)
        for bar,value in zip(bars,values):a.text(bar.get_x()+bar.get_width()/2,value+1.5,f'{value:.1f}%',ha='center',fontsize=9)
    a.axhline(100,color='#9099ab',ls='--',lw=1);a.set_xticks(x,['All rows','Text','Image']);a.set_ylabel('Remaining RMS / reconstructed missing write (%)');a.set_ylim(0,max(115,a.get_ylim()[1]*1.15));a.set_title('What a stream constant can represent',loc='left',pad=15,fontweight='bold');a.legend(frameon=False,fontsize=8,loc='lower left');a.grid(axis='y',alpha=.12);a.set_axisbelow(True)
    g=np.asarray(r['panels']['all']['gram'])/r['panels']['all']['totalGatedErrorEnergy']*100;limit=float(np.abs(g).max()) or 1
    im=b.imshow(g,cmap='RdBu_r',vmin=-limit,vmax=limit);labels=['Stream mean','Row variation','Policy + assoc.'];b.set_xticks(range(3),labels,rotation=25,ha='right');b.set_yticks(range(3),labels)
    for i in range(3):
        for j in range(3):
            label=f'{g[i,j]:+.1f}%' if abs(g[i,j])>=.1 else f'{g[i,j]:+.2g}%'
            b.text(j,i,label,ha='center',va='center',color='white' if abs(g[i,j])>limit*.55 else '#263047',fontsize=10)
    b.set_title('Signed component inner products',loc='left',pad=15,fontweight='bold');fig.colorbar(im,ax=b,fraction=.046,pad=.05,label='% of total gated error energy')
    fig.suptitle('One own-state MLP: mean write, row variation and numerical policy',x=.055,y=.98,ha='left',fontsize=14,fontweight='bold')
    fig.text(.055,.055,'Fixed clock3 / single0 · 130.460 billion diagnostic MACs · newly paid FP64 Up and two Down projections',fontsize=9,color='#596378')
    fig.text(.055,.018,'Oracle representation floor, not a serving producer · signed cross terms retained · no native finite block-error or image-causality claim',fontsize=8,color='#596378')
    fig.subplots_adjust(left=.085,right=.96,top=.8,bottom=.24,wspace=.6)
    path=out/'own-state-gated-covariance.png';fig.savefig(path,dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
    notebook=root.parent/'obsidian/images/zero-neural-cycle76-own-state-gated-covariance.png';shutil.copyfile(path,notebook)
    save(out/'figure.json',{'schema':'tce-gated-covariance-figure-v1','reportSHA256':sha(p/'report.json'),'plotSourceSHA256':sha(__file__),'imageSHA256':sha(path),'notebookCopy':str(notebook),'scope':'Published scalar floor/Gram only; no model or numerical retained array loaded.'})
    return {'image':str(path),'sha256':sha(path)}


if __name__=='__main__':print(json.dumps(execute()))
