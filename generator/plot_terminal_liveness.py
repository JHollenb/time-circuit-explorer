"""Plot published scalar accounts only; no tensors, weights or model loading."""
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/terminal-liveness';r=read(p/'report.json');e=read(p/'execution.json')
    out=root/'docs/results/terminal-liveness-figure';out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    fig,(a,b)=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={'width_ratios':[1.1,1]},facecolor='#fbfcfe')
    for ax in (a,b):ax.set_facecolor('#fbfcfe')
    affine=[188863217664/1e9,e['affineMACs']/1e9];attention=[14495514624/1e9,e['attentionMACs']/1e9]
    a.barh([1,0],affine,color='#6e80ba',height=.48,label='Affines, including original head')
    a.barh([1,0],attention,left=affine,color='#73b9a8',height=.48,label='QK + AV')
    a.set_yticks([1,0],['Native account','Reduced execution']);a.set_xlim(0,235);a.set_xlabel('Billions of nominal model matrix MACs');a.set_title('Dead output work is physically removed',loc='left',pad=17)
    for y,total in [(1,r['referenceBlockAndHeadMatrixMACs']/1e9),(0,r['newMatrixMACs']/1e9)]:a.text(total+2,y,f'{total:.3f}',va='center',fontsize=10)
    a.legend(loc='lower left',bbox_to_anchor=(-.01,-.3),frameon=False,fontsize=8)
    values=[100*r['scores']['image']['endpoint']['relativeRms'],100*r['scores']['image']['nativeWrite']['relativeRms'],100*r['scores']['velocity']['endpoint']['relativeRms']]
    b.barh([2,1,0],values,color=['#96a4ce','#96a4ce','#c79778'],height=.48)
    b.set_yticks([2,1,0],['Image state','Native image write','Own final velocity']);b.set_xlim(0,.5);b.set_xlabel('RMS error / named native denominator (%)');b.set_title('Changed BF16 geometry has a measured cost',loc='left',pad=17)
    for y,value in zip([2,1,0],values):b.text(value+.009,y,f'{value:.3f}%',va='center',fontsize=10)
    fig.suptitle('One final FLUX block: 57.982 billion fewer matrix MACs',x=.05,ha='left',fontsize=15,fontweight='bold',y=.98)
    fig.text(.05,.025,'One saved source · all current K/V retained · original learned maps/head remain paid · no image or measured latency speedup',fontsize=8,color='#596378')
    fig.tight_layout(rect=[0,.09,1,.91],w_pad=3);image=out/'terminal-liveness.png';fig.savefig(image,dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
    notebook=root.parent/'obsidian/images/zero-neural-cycle65-terminal-liveness.png';shutil.copyfile(image,notebook)
    save(out/'figure.json',{'schema':'tce-terminal-liveness-figure-v1','reportSHA256':sha(p/'report.json'),'executionSHA256':sha(p/'execution.json'),
        'plotSourceSHA256':sha(__file__),'imageSHA256':sha(image),'notebookCopy':str(notebook),'scope':'Derived scalar plot of executed model matrix accounts and measured BF16 tensor discrepancy. Excludes toy7600MACs, normalization/softmax, identity I/O, nativeprefix/capture/failure. No imagequality or matchedlatency claim.'})
    return {'image':str(image),'sha256':sha(image)}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
