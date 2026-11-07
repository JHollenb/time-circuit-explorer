"""Plot published scalar observations and nominal work; no model/tensor processing."""
import gzip
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from encoder_atoms import read,save,sha


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/source-reader';r=read(p/'report.json')
    out=root/'docs/results/source-reader-figure';out.mkdir(exist_ok=False)
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()]
    heads=[x for x in rows if x['field']=='nextQ' and x['panel']=='query-head']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f8fb','axes.facecolor':'white'})
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained');a,b,c=axes
    names=['H','A','M','T','nextQ'];x=np.arange(len(names));width=.34
    for shift,key,label,color in [(-width/2,'assistedResponse','Assisted allocation','#8592ae'),(width/2,'primaryResponse','Source-state reader','#269b8f')]:
        a.bar(x+shift,[r['scores'][n][key]['relativeRms']*100 for n in names],width=width,label=label,color=color)
    a.set(xticks=x,xticklabels=names,ylabel='Error / actual context response (%)',title='Own read → own MLP → own next query');a.legend(frameon=False,fontsize=8)
    for row,color in zip([4,11,12,13],['#5874ce','#2a9d8f','#cc6845','#a45ca6']):
        h=[v for v in heads if v['row']==row]
        b.scatter([v['assistedResponse']['relativeRms']*100 for v in h],[v['primaryResponse']['relativeRms']*100 for v in h],s=20,color=color,label=f'Query row {row}',alpha=.85)
    b.plot([0,50],[0,50],ls='--',color='#555',lw=1);b.set(xlim=(0,50),ylim=(0,50),xlabel='Assisted next-query response error (%)',ylabel='Source-state response error (%)',title='92 heads improve; 4 worsen');b.legend(frameon=False,fontsize=8)
    work=[14.495514624,.037748736,.018874368,.528482304,.415236096]
    c.barh(range(5),work,color=['#5874ce','#8592ae','#a45ca6','#269b8f','#cc6845'])
    c.set(yticks=range(5),yticklabels=['Full K','Selected Q','Joint QK','Deferred V','Consumer + nextQ'],xlabel='Billion nominal matrix MACs',title='Pay for a source-state interface');c.invert_yaxis()
    c.text(1,0,'14.496B',va='center',color='white');c.text(.7,3,'0.528B',va='center');c.set_xlim(0,15.4)
    fig.suptitle('A source-state reader reproduces native Q/K and advances its own state',fontsize=14,fontweight='bold')
    fig.supxlabel('One supplied native source pair; four demanded queries. Original prefixes, clock context and 13 learned tensors remain paid. No image or speedup claim.',fontsize=10)
    path=out/'source-reader.png';fig.savefig(path,dpi=160);plt.close(fig)
    shutil.copyfile(path,root.parent/'obsidian/images/zero-neural-cycle63-source-reader.png')
    save(out/'figure.json',{'reportSHA256':sha(p/'report.json'),'imageSHA256':sha(path),'plotSourceSHA256':sha(__file__),
        'scope':'Published per-query/head scalar errors and nominal matrix work only; no new model or scientific tensor processing.'})
    return str(path)


if __name__=='__main__':print(execute())
