"""Plot published report and scalar rows only; no model/tensor-array computation."""
import gzip
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/current-stream-mean-up';r=read(p/'report.json')
    rows=[json.loads(x) for x in gzip.decompress((p/'write-observations.jsonl.gz').read_bytes()).splitlines()]
    out=root/'docs/results/current-stream-mean-up-figure';out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(a,b)=plt.subplots(1,2,figsize=(11.6,5.6),facecolor='#fbfcfe')
    for ax in [a,b]:ax.set_facecolor('#fbfcfe')
    largest=0
    for shift,key,label,color in [(-.18,'priorErrorRelativeDenseWrite','Quarter channels','#a2afcf'),(.18,'ownErrorRelativeDenseWrite','Mean-Up completion','#5c9c89')]:
        values=[100*r['writePanels'][n][key] for n in ['all','text','image']];largest=max(largest,*values)
        a.bar([i+shift for i in range(3)],values,width=.32,color=color,label=label)
        for i,v in enumerate(values):a.text(i+shift,v+.9,f'{v:.2f}',ha='center',fontsize=9)
    a.set_xticks([0,1,2],['All rows','Text rows','Image rows']);a.set_ylim(0,max(1,largest)*1.4)
    a.set_ylabel('State difference / dense block increment (%)');a.set_title('Actual local write error',loc='left',pad=16,fontweight='bold');a.grid(axis='y',alpha=.15);a.set_axisbelow(True);a.legend(frameon=False,fontsize=9)
    for name,label,color in [('text','512 text rows','#6e80ba'),('image','1,024 image rows','#bd886c')]:
        part=[x for x in rows if x['stratum']==name and x['priorErrorRelativeDenseWrite'] and x['ownErrorRelativeDenseWrite']]
        b.scatter([x['priorErrorRelativeDenseWrite'] for x in part],[x['ownErrorRelativeDenseWrite'] for x in part],s=10,alpha=.48,color=color,label=label)
    positive=[v for x in rows for v in [x['priorErrorRelativeDenseWrite'],x['ownErrorRelativeDenseWrite']] if v and v>0]
    lo,hi=min(positive)*.7,max(positive)*1.3;b.plot([lo,hi],[lo,hi],ls='--',lw=1,color='#7e8798')
    b.set_xscale('log');b.set_yscale('log');b.set_xlim(lo,hi);b.set_ylim(lo,hi)
    b.set_xlabel('Quarter error / dense row increment');b.set_ylabel('Completion error / dense row increment')
    b.set_title('Every dependent row remains visible',loc='left',pad=16,fontweight='bold');b.legend(frameon=False,fontsize=8,loc='upper left')
    fig.suptitle('Two current stream means complete the omitted Up population',x=.06,ha='left',fontsize=15,fontweight='bold',y=.97)
    fig.text(.06,.09,'Same incoming state · unchanged quarter support · all current gates and the full output projection remain paid',fontsize=9,color='#596378')
    fig.text(.06,.04,'167.763 billion candidate MACs · saved dense moment-text comparator · below the dashed line improves · no image or zero-neural claim',fontsize=8,color='#596378')
    fig.subplots_adjust(left=.08,right=.98,bottom=.24,top=.79,wspace=.38)
    image=out/'current-stream-mean-up.png';fig.savefig(image,dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
    notebook=root.parent/'obsidian/images/zero-neural-cycle74-current-stream-mean-up.png';shutil.copyfile(image,notebook)
    save(out/'figure.json',{'schema':'tce-mean-up-figure-v1','reportSHA256':sha(p/'report.json'),'rowStreamSHA256':sha(p/'write-observations.jsonl.gz'),
        'plotSourceSHA256':sha(__file__),'imageSHA256':sha(image),'notebookCopy':str(notebook),'scope':'Published scalar comparisons only, dependent rows, common denominators. No tensor arrays/model or new scientific measurement.'})
    return {'image':str(image),'sha256':sha(image)}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
