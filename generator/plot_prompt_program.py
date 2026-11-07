"""Plot published trajectory scalars; no model or tensor-array computation."""
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/prompt-program';r=read(p/'report.json')
    out=root/'docs/results/prompt-program-figure';out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(a,b)=plt.subplots(1,2,figsize=(11.6,5.4),facecolor='#fbfcfe',gridspec_kw={'width_ratios':[1.15,1]})
    for ax in (a,b):ax.set_facecolor('#fbfcfe')
    for name,label,color in [('velocity','Own velocity','#6e80ba'),('latent','Own committed latent','#5c9c89'),('commitWrite','Own sampler write','#bd886c')]:
        a.plot([1,2,3,4],[100*x[name]['relativeRms'] for x in r['trajectory']],marker='o',lw=2,color=color,label=label)
    a.set_xticks([1,2,3,4]);a.set_xlabel('Sampler step');a.set_ylabel('RMS error / corresponding native quantity (%)');a.set_title('Finite discrepancy along its own state',loc='left',pad=16,fontweight='bold');a.grid(axis='y',alpha=.15);a.set_ylim(bottom=0)
    fig.legend(*a.get_legend_handles_labels(),frameon=False,fontsize=8,loc='lower left',bbox_to_anchor=(.085,.12),ncol=3)
    names=['vaeInput','decodedRaw','rgbFloat','pngRGB'];labels=['VAE input','Decoded tensor','Floating RGB','PNG pixels']
    values=[100*r['scores'][n]['metrics']['relativeRms'] for n in names]
    b.barh([3,2,1,0],values,color=['#6e80ba','#a2afcf','#8cbbaf','#bd886c'],height=.5)
    b.set_yticks([3,2,1,0],labels);b.set_xlabel('RMS error / native endpoint (%)');b.set_title('One actual final VAE decode',loc='left',pad=16,fontweight='bold')
    largest=max(values) or 1;b.set_xlim(0,largest*1.4)
    for y,value in zip([3,2,1,0],values):b.text(value+largest*.025,y,f'{value:.3f}%',va='center',fontsize=10)
    fig.suptitle('Depth-evolving text program across four own sampler steps',x=.06,ha='left',fontsize=15,fontweight='bold',y=.97)
    fig.text(.06,.055,'6.223 trillion compiler MACs + 629 MB K/V cache · 21.063 trillion first-image MACs · one supplied prompt/seed',fontsize=9,color='#596378')
    fig.text(.06,.018,'Actual candidate PNG and one existing native comparator · endpoint errors use different denominators · no matched speed or zero-neural claim',fontsize=8,color='#596378')
    fig.subplots_adjust(left=.09,right=.98,bottom=.28,top=.79,wspace=.6)
    image=out/'prompt-program.png';fig.savefig(image,dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
    notebook=root.parent/'obsidian/images/zero-neural-cycle68-prompt-program.png';shutil.copyfile(image,notebook)
    save(out/'figure.json',{'schema':'tce-prompt-program-figure-v1','reportSHA256':sha(p/'report.json'),'plotSourceSHA256':sha(__file__),
        'imageSHA256':sha(image),'notebookCopy':str(notebook),'scope':'Published scalar trajectory/endpoint plot only. No numerical arrays or model loaded; no new scientific comparison.'})
    return {'image':str(image),'sha256':sha(image)}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
