"""Published-scalar-only full/centered error and signed four-term Gram."""
from pathlib import Path
import json
import shutil
from encoder_atoms import read,save,sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/odd-gate';r=read(p/'report.json');out=root/'docs/results/odd-gate-figure-v2';out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(14.5,7.2),facecolor='#fbfcfe');a=fig.add_axes([.07,.33,.35,.47]);b=fig.add_axes([.62,.33,.28,.47]);a.set_facecolor('#fbfcfe');b.set_facecolor('#fbfcfe')
    names=['all','text','image'];xs=np.arange(3);maximum=100
    for offset,key,label,color in [(-.23,'baselineResidualOverTotalRMS','Linear baseline / full error','#8b96b3'),(0,'fullResidualOverTotalRMS','Linear + quadratic / full error','#637bc0'),(.23,'centeredResidualOverCenteredTotalRMS','Centered forecast / centered error','#4b9b86')]:
        values=[100*r['panels'][n][key] for n in names];maximum=max(maximum,*values);bars=a.bar(xs+offset,values,width=.21,color=color,label=label)
        for bar,v in zip(bars,values):a.text(bar.get_x()+bar.get_width()/2,v+maximum*(.055 if offset>.1 else .015),f'{v:.1f}%',ha='center',fontsize=8)
    a.set_xticks(xs,['All rows','Text','Image']);a.set_ylim(0,maximum*1.20);a.set_ylabel('Remaining RMS / named discrepancy (%)');a.axhline(100,color='#9099ab',lw=1,ls='--');a.grid(axis='y',alpha=.13);a.set_axisbelow(True);a.set_title('Full correction and row variation',loc='left',fontweight='bold',pad=16)
    gram=np.asarray(r['panels']['all']['gramAccounts']['full']['gram'])/r['panels']['all']['energies']['totalEnergy']*100;limit=float(np.max(np.abs(gram))) or 1
    im=b.imshow(gram,cmap='RdBu_r',vmin=-limit,vmax=limit);labels=['Linear','Quadratic','Remainder','Policy'];b.set_xticks(range(4),labels,rotation=30,ha='right');b.set_yticks(range(4),labels)
    for i in range(4):
        for j in range(4):
            v=gram[i,j];label=f'{v:+.1f}%' if abs(v)>=.1 else f'{v:+.1g}%';b.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if abs(v)>limit*.55 else '#263047')
    b.set_title('Signed output inner products',loc='left',fontweight='bold',pad=16);cb=fig.add_axes([.915,.35,.012,.40]);fig.colorbar(im,cax=cb,label='% of full discrepancy energy')
    fig.suptitle('One own-state MLP: a fixed quadratic SwiGLU component',x=.055,y=.965,ha='left',fontweight='bold',fontsize=15)
    fig.legend(*a.get_legend_handles_labels(),frameon=False,loc='lower left',bbox_to_anchor=(.065,.20),ncol=3,fontsize=9)
    fig.text(.055,.13,'Full scores include the residual mean. Centered scores use a different denominator and remain diagnostic.',fontsize=9,color='#596378')
    fig.text(.055,.085,'Fixed clock3 / single0 · paid original gate + Down: 86.973 billion MACs · inherited Up, linear correction and state remain paid',fontsize=9,color='#596378')
    fig.text(.055,.04,'Fresh FP64 gate reference is not native G · smooth-even / finite terms are product-space only · no new image or economical generator',fontsize=8,color='#596378')
    path=out/'odd-gate.png';fig.savefig(path,dpi=170,facecolor=fig.get_facecolor());plt.close(fig);copy=root.parent/'obsidian/images/zero-neural-cycle78-odd-gate-v2.png';shutil.copyfile(path,copy)
    save(out/'figure.json',{'schema':'tce-odd-gate-figure-v1','reportSHA256':sha(p/'report.json'),'plotSourceSHA256':sha(__file__),'imageSHA256':sha(path),'notebookCopy':str(copy),'scope':'Published scalar panels only; no real tensor or model loaded.'})
    return {'image':str(path),'sha256':sha(path)}


if __name__=='__main__':print(json.dumps(execute()))
