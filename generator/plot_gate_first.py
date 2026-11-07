"""Plot published scalar matrix accounts and consumer errors, no tensor math."""
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/gate-first-terminal';r=read(p/'report.json');e=read(p/'execution.json');out=root/'docs/results/gate-first-terminal-figure';out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(a,b)=plt.subplots(1,2,figsize=(12.4,5.7),facecolor='#fbfcfe',gridspec_kw={'width_ratios':[1.1,1]})
    for ax in (a,b):ax.set_facecolor('#fbfcfe')
    affine=[135712997376/1e9,e['affineMACs']/1e9];attention=[9663676416/1e9,e['attentionMACs']/1e9]
    a.barh([1,0],affine,color='#6e80ba',height=.48,label='Learned affines + original head')
    a.barh([1,0],attention,left=affine,color='#73b9a8',height=.48,label='Full image QK + AV')
    a.set_yticks([1,0],['Matched parent','Gate-first quarter']);a.set_xlim(0,175);a.set_xlabel('Billions of nominal matrix MACs');a.set_title('Physical terminal/head work',loc='left',pad=18,fontweight='bold')
    for y,total in [(1,r['matchedParentTerminalAndHeadMACs']/1e9),(0,r['newMatrixMACs']/1e9)]:a.text(total+2,y,f'{total:.3f}',va='center')
    fig.legend(*a.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.13,.11),frameon=False,fontsize=8,ncol=2)
    values=[100*r['scores']['image']['endpoint']['relativeRms'],100*r['scores']['image']['parentWrite']['relativeRms'],100*r['scores']['velocity']['endpoint']['relativeRms']]
    largest=max(values) or .1;b.barh([2,1,0],values,color=['#96a4ce','#96a4ce','#c79778'],height=.48);b.set_xlim(0,largest*1.4)
    b.set_yticks([2,1,0],['Image state','Parent image write','Own final velocity']);b.set_xlabel('RMS error / corresponding parent quantity (%)');b.set_title('Measured consumer discrepancy',loc='left',pad=18,fontweight='bold')
    for y,value in zip([2,1,0],values):b.text(value+largest*.025,y,f'{value:.3f}%',va='center')
    fig.suptitle('Current gates select MLP work: 43.487 billion fewer matrix MACs',x=.05,ha='left',y=.97,fontsize=15,fontweight='bold')
    fig.text(.05,.055,'One paid terminal state · full gate + 2,304/9,216 up/down channels · current K/V and original head retained',fontsize=9,color='#596378')
    fig.text(.05,.018,'Norm preparation, ranking and copies additionally paid · no new image, whole-trajectory quality or matched speedup claim',fontsize=8,color='#596378')
    fig.subplots_adjust(left=.13,right=.98,bottom=.28,top=.77,wspace=.7);image=out/'gate-first-terminal.png';fig.savefig(image,dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
    notebook=root.parent/'obsidian/images/zero-neural-cycle71-gate-first-terminal.png';shutil.copyfile(image,notebook)
    save(out/'figure.json',{'schema':'tce-gate-first-figure-v1','reportSHA256':sha(p/'report.json'),'executionSHA256':sha(p/'execution.json'),'plotSourceSHA256':sha(__file__),'imageSHA256':sha(image),'notebookCopy':str(notebook),
        'scope':'Published scalar work/endpoint/write/head errors, no weights or tensor-array calculation. Different endpoint denominators, source prefix and learned work paid.'})
    return {'image':str(image),'sha256':sha(image)}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
