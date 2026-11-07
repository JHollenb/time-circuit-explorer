"""Plot measured response rows and nominal value-production work."""
import gzip
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from encoder_atoms import read, save, sha


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/deferred-value';r=read(p/'report.json')
    out=root/'docs/results/deferred-value-figure';out.mkdir(exist_ok=False)
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()]
    heads=[x for x in rows if x['field']=='nextQ' and x['panel']=='query-head']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f8fb','axes.facecolor':'white'})
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    a,b,c=axes
    a.barh([1,0],[14.495514624+.018874368,.528482304],color=['#8592ae','#269b8f'])
    a.set(yticks=[1,0],yticklabels=['Full V + AV','Deferred V'],xlabel='Billion nominal matrix MACs',title='Form the demanded read instead')
    a.text(.6,0,'0.528B',va='center');a.text(14.3,1,'14.514B',va='center',ha='right',color='white');a.set_xlim(0,15.2)
    colors=['#5874ce','#2a9d8f','#cc6845','#a45ca6']
    for row,color in zip([4,11,12,13],colors):
        h=[x for x in heads if x['row']==row]
        b.scatter([x['retainedResponse']['relativeRms']*100 for x in h],[x['primaryResponse']['relativeRms']*100 for x in h],
            s=20,color=color,label=f'Query row {row}',alpha=.85)
    b.plot([10,52],[10,52],ls='--',color='#555',lw=1);b.set(xlim=(10,52),ylim=(10,52),
        xlabel='Supplied-V next-query response error (%)',ylabel='Deferred-V next-query response error (%)',title='56 heads improve; 40 worsen')
    b.legend(frameon=False,fontsize=8)
    names=['H','A','M','T','nextQ'];x=np.arange(5);width=.34
    for shift,key,label,color in [(-width/2,'retainedResponse','Supplied V','#8592ae'),(width/2,'primaryResponse','Deferred V','#269b8f')]:
        c.bar(x+shift,[r['scores'][n][key]['relativeRms']*100 for n in names],width=width,label=label,color=color)
    c.set(xticks=x,xticklabels=names,ylabel='Error / actual context response (%)',title='Errors change through the consumer');c.legend(frameon=False)
    fig.suptitle('Deferred values preserve a useful own-state response without reading a native value bank',fontsize=14,fontweight='bold')
    fig.supxlabel('Four demanded reads; allocation matches exactly. Original states, maps, duals and P norms remain paid. MACs are not wall-time speedups.',fontsize=10)
    path=out/'deferred-value.png';fig.savefig(path,dpi=160);plt.close(fig)
    shutil.copyfile(path,root.parent/'obsidian/images/zero-neural-cycle62-deferred-value.png')
    save(out/'figure.json',{'reportSHA256':sha(p/'report.json'),'imageSHA256':sha(path),'plotSourceSHA256':sha(__file__),
        'scope':'Published per-query/head scalar errors plus nominal original/deferred V work; no scientific arrays/model execution or image claim.'})
    return str(path)


if __name__=='__main__':print(execute())
