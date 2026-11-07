"""Render published scalar bounds only; no scientific arrays are opened."""
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from encoder_atoms import read,save,sha


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/gated-even-bound';r=read(p/'report.json')
    records=[r['panels'][n] for n in ['all','text','image']]+r['retainedAdverseTextRows'];names=['All 1,536 rows','Text · 512 rows','Image · 1,024 rows','Text row 3','Text row 35']
    out=root/'docs/results/gated-even-bound-figure';out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.8));fig.patch.set_facecolor('#f7f9fc')
    colors=['#5360c9','#2d8d84','#bc7430','#2d8d84','#2d8d84']
    for ax in axes:ax.set_facecolor('#f7f9fc');ax.set_yticks(range(5),names);ax.set_ylim(4.65,-.65);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0,pad=8)
    for i,(v,c) in enumerate(zip(records,colors)):
        ratio=100*v['etaOverRemainingNormUpper'];axes[0].barh(i,ratio,color=c,height=.38);axes[0].text(ratio+1,i,f'{ratio:.2f}%',va='center',fontsize=10)
        lo,hi=v['quadraticEvenCosineInterval'];axes[1].plot([lo,hi],[i,i],color=c,lw=7,solid_capstyle='round');axes[1].plot([lo,hi],[i,i],'|',color=c,ms=13,mew=1.5)
        axes[1].text(hi+.028,i,f'{hi:+.3f}',va='center',fontsize=9,color=c)
    axes[0].set_xlim(0,62);axes[0].set_xlabel('Maximum non-H contribution / remaining norm (%)');axes[0].set_title('Uncertainty around the retained-H output',loc='left',pad=16,fontsize=11)
    axes[1].set_yticklabels([]);axes[1].set_xlim(-1.04,.23);axes[1].axvline(0,color='#758195',lw=1,ls='--');axes[1].set_xlabel('Allowed cosine with the quadratic output');axes[1].set_title('Text alignment stays negative; image sign is unresolved',loc='left',pad=16,fontsize=11)
    fig.suptitle('A conditional bound isolates text cancellation',x=.04,y=.98,ha='left',fontsize=18,fontweight='bold',color='#192539')
    fig.text(.04,.908,'Saved clock 3 / single 0 · zero new learned projections · H and F are never projected',fontsize=11,color='#536075')
    fig.text(.04,.06,'Intervals assume the stated FP64 reduction/contraction model; historical cuBLAS error bounds are not attested.',fontsize=10,color='#536075')
    fig.text(.04,.025,'One retained state, dependent rows. These bounds do not update a model state, generate an image or establish zero-neural serving.',fontsize=9,color='#536075')
    fig.subplots_adjust(left=.17,right=.975,bottom=.19,top=.80,wspace=.28)
    path=out/'gated-even-bound.png';fig.savefig(path,dpi=160);plt.close(fig)
    notebook=root.parent/'obsidian/images/zero-neural-cycle79-gated-even-bound.png';shutil.copyfile(path,notebook)
    save(out/'figure.json',{'schema':'tce-gated-even-bound-figure-v1','sourceReportSHA256':sha(p/'report.json'),'plotSourceSHA256':sha(__file__),'imageSHA256':sha(path),'notebookCopy':str(notebook),'scope':'Published scalar conditional bounds only; no new scientific-array/model computation.'})
    return {'path':str(path),'sha256':sha(path)}


if __name__=='__main__':print(execute())
