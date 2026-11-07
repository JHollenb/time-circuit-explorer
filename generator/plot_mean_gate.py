"""Published scalar-only picture of the full and centered correction accounts."""
from pathlib import Path
import json
import shutil
from encoder_atoms import read,save,sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/mean-gate';r=read(p/'report.json');out=root/'docs/results/mean-gate-figure';out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(a,b)=plt.subplots(1,2,figsize=(13.5,6.5),facecolor='#fbfcfe',gridspec_kw={'width_ratios':[1.3,1]})
    for ax in [a,b]:ax.set_facecolor('#fbfcfe')
    names=['all','text','image'];x=np.arange(3);maxvalue=100
    for offset,key,label,color in [(-.19,'fullResidualOverTotalRMS','Full correction error','#677dc3'),(.19,'centeredResidualOverCenteredTotalRMS','Centered variation error','#4d9b8d')]:
        values=[100*r['panels'][n][key] for n in names];maxvalue=max(maxvalue,*values);bars=a.bar(x+offset,values,width=.34,color=color,label=label)
        for bar,v in zip(bars,values):a.text(bar.get_x()+bar.get_width()/2,v+2,f'{v:.1f}%',ha='center',fontsize=9)
    a.axhline(100,color='#8892a8',ls='--',lw=1);a.set_ylim(0,maxvalue*1.18);a.set_xticks(x,['All rows','Text','Image']);a.set_ylabel('Remaining RMS / corresponding discrepancy (%)');a.grid(axis='y',alpha=.15);a.set_axisbelow(True);a.set_title('Mean activation × changing Up',loc='left',pad=15,fontweight='bold')
    gram=np.asarray(r['panels']['all']['gramAccounts']['centered']['gram'])/r['panels']['all']['centeredTotalEnergy']*100;limit=float(np.abs(gram).max()) or 1
    im=b.imshow(gram,cmap='RdBu_r',vmin=-limit,vmax=limit);labels=['Mean activation','Changing activation','Policy + assoc.'];b.set_xticks(range(3),labels,rotation=25,ha='right');b.set_yticks(range(3),labels)
    for i in range(3):
        for j in range(3):
            v=gram[i,j];label=f'{v:+.1f}%' if abs(v)>=.1 else f'{v:+.2g}%';b.text(j,i,label,ha='center',va='center',fontsize=10,color='white' if abs(v)>limit*.55 else '#263047')
    b.set_title('Centered signed inner products',loc='left',pad=15,fontweight='bold');fig.colorbar(im,ax=b,fraction=.046,pad=.05,label='% of centered total discrepancy energy')
    fig.suptitle('One own-state MLP: linear coefficients and changing-gate interaction',x=.06,y=.97,ha='left',fontsize=14,fontweight='bold')
    fig.legend(*a.get_legend_handles_labels(),frameon=False,loc='lower left',bbox_to_anchor=(.08,.14),ncol=2,fontsize=9)
    fig.text(.06,.09,'Full and centered bars use different denominators. Centering removes a paid oracle mean; it is not a produced correction.',fontsize=9,color='#596378')
    fig.text(.06,.045,'Fixed clock3 / single0 · one 43.487-billion-MAC Down · saved original Up/Down, activation and gates remain paid',fontsize=9,color='#596378')
    fig.text(.06,.015,'Changing term is subtraction-derived, with reassociation · no image, own-state continuation or economical compilation established',fontsize=8,color='#596378')
    fig.subplots_adjust(left=.085,right=.9,top=.80,bottom=.33,wspace=.78)
    image=out/'mean-gate.png';fig.savefig(image,dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
    copy=root.parent/'obsidian/images/zero-neural-cycle77-mean-gate.png';shutil.copyfile(image,copy)
    save(out/'figure.json',{'schema':'tce-mean-gate-figure-v1','reportSHA256':sha(p/'report.json'),'plotSourceSHA256':sha(__file__),'imageSHA256':sha(image),'notebookCopy':str(copy),'scope':'Published scalar panels only; no retained real tensor or model loaded.'})
    return {'image':str(image),'sha256':sha(image)}


if __name__=='__main__':print(json.dumps(execute()))
