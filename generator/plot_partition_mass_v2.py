"""Plot published per-head scalar mass summaries; no arrays/model loaded."""
import gzip
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/partition-feedback';out=root/'docs/results/partition-mass-figure-v2';out.mkdir(parents=True,exist_ok=False)
    raw=gzip.decompress((p/'partition-observations.jsonl.gz').read_bytes());rows=[json.loads(x) for x in raw.splitlines()]
    fig,ax=plt.subplots(figsize=(10.6,4.7),facecolor='#fbfcfe');ax.set_facecolor('#fbfcfe')
    colors=['#6e80ba','#5c9c89','#bd886c','#9371a5']
    for step,color in enumerate(colors):
        points=[]
        for kind,count in [('joint',5),('single',19)]:
            for index in range(count):
                values=[x['mean'] for x in rows if (x['step'],x['kind'],x['index'])==(step,kind,index)]
                if len(values)!=24:raise ValueError('Incomplete mass summary')
                points.append(sum(values)/24)
        ax.plot(range(24),points,marker='.',color=color,lw=1.8,label='Clock '+str(step+1))
    ax.axvline(4.5,color='#a7afbf',ls=':',lw=1);ax.set_xticks([0,5,10,15,23],['Joint 0','Single 0','Single 5','Single 10','Single 18'])
    ax.set_ylim(0,1);ax.set_ylabel('Mean text attention mass α');ax.set_title('Own image keys change how much attention stays with text',loc='left',pad=16,fontweight='bold')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,ncol=4,loc='upper right')
    fig.text(.09,.035,'Means across 24 heads × 512 text queries; actual own states · no native mass comparison or causal count attribution',fontsize=8,color='#596378')
    fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.84);image=out/'partition-mass-v2.png';fig.savefig(image,dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
    notebook=root.parent/'obsidian/images/zero-neural-cycle69-partition-mass-v2.png';shutil.copyfile(image,notebook)
    save(out/'figure.json',{'schema':'tce-partition-mass-figure-v2','sourceStreamSHA256':sha(p/'partition-observations.jsonl.gz'),'plotSourceSHA256':sha(__file__),'imageSHA256':sha(image),'notebookCopy':str(notebook),
        'scope':'Plot of published scalar per-head mean masses, equal512querycounts. No numerical arrays/model or new scientific comparison.'})
    return {'image':str(image),'sha256':sha(image)}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
