"""Plot complete published scalar comparisons; no real arrays opened."""
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from encoder_atoms import read,save,sha


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/rectifier-correction';r=read(p/'report.json');out=root/'docs/results/rectifier-correction-figure';out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    fig,axes=plt.subplots(1,3,figsize=(13,5.7));fig.patch.set_facecolor('#f7f9fc')
    methods=['zero','linear','quadratic','rectifier'];labels=['Zero correction','L only','L + quadratic','L + rectifier'];colors=['#bdc6d5','#69758b','#7a6ec7','#2d8d84']
    maximum=max(v for panel in r['panels'].values() for v in panel['fullResidualOverTotalRMS'].values())*100
    for ax,name in zip(axes,['all','text','image']):
        panel=r['panels'][name];values=[100*panel['fullResidualOverTotalRMS'][n] for n in methods]
        ax.set_facecolor('#f7f9fc');ax.barh(range(4),values,color=colors,height=.42);ax.set_yticks(range(4),labels if name=='all' else ['']*4);ax.set_ylim(3.65,-.65);ax.set_xlim(0,maximum*1.17+2);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0,pad=8)
        for i,v in enumerate(values):ax.text(v+maximum*.025,i,f'{v:.2f}%',va='center',fontsize=10)
        ax.set_title(f'{name.title()} · {panel["rows"]:,} rows',loc='left',pad=16,fontsize=12,fontweight='bold')
    fig.suptitle('Combine the activation pieces before projection',x=.035,y=.98,ha='left',fontsize=18,fontweight='bold',color='#192539')
    fig.text(.035,.91,'Same saved clock 3 / single 0 · one new rectifier + original Down · complete full-error comparison',fontsize=11,color='#536075')
    fig.supxlabel('Residual norm / reconstructed full missing-write norm (%)',y=.13,fontsize=11)
    fig.text(.035,.06,'Full residuals include their means. Centered diagnostics use a different denominator; no oracle mean is added.',fontsize=10,color='#536075')
    fig.text(.035,.025,'Original G/Up/L/maps/source formation remain paid. One retained state, dependent rows; no new image or zero-neural serving result.',fontsize=9,color='#536075')
    fig.subplots_adjust(left=.15,right=.975,bottom=.22,top=.80,wspace=.13)
    path=out/'rectifier-correction.png';fig.savefig(path,dpi=160);plt.close(fig)
    notebook=root.parent/'obsidian/images/zero-neural-cycle80-rectifier-correction.png';shutil.copyfile(path,notebook)
    save(out/'figure.json',{'schema':'tce-rectifier-correction-figure-v1','sourceReportSHA256':sha(p/'report.json'),'plotSourceSHA256':sha(__file__),'imageSHA256':sha(path),'notebookCopy':str(notebook),'scope':'Published scalar full-error comparison only; no new scientific-array or model processing.'})
    return {'path':str(path),'sha256':sha(path)}


if __name__=='__main__':print(execute())
