"""Publication figure from complete published scalar observations only."""
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
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/spatial-norm-capacity'
    report=read(p/'report.json');out=root/'docs/results/spatial-norm-figure';out.mkdir(exist_ok=False)
    rows=[json.loads(line) for line in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()]
    field=np.zeros((32,32));count=np.zeros((32,32))
    for x in rows:
        if x['panel']=='image-key-head':
            h,w=x['height'],x['width'];field[h,w]+=x['signedError']**2;count[h,w]+=1
    if not np.all(count==24):raise ValueError('Complete physical scalar field absent')
    field=np.sqrt(field/count)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f8fb',
        'axes.facecolor':'#ffffff','axes.titleweight':'bold'})
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),gridspec_kw={'width_ratios':[1,1.05,1.15]},layout='constrained')
    a,b,c=axes
    names=['sampled','unsampledInterior','unsampledEdge'];labels=['Supplied\nknots','Interior\n705 rows','Clamped edge\n63 rows']
    values=[report['norms']['strata'][n]['response']['relativeRms'] for n in names]
    a.bar(range(3),values,color=['#2a9d8f','#5874ce','#cc6845']);a.axhline(1,color='#444',ls='--',lw=1)
    a.set(xticks=range(3),xticklabels=labels,ylabel='Norm error / actual context change',title='Endpoint shape misses the update')
    for i,v in enumerate(values):a.text(i,v+1.5,f'{v:.2f}×',ha='center')
    a.text(.03,.92,'Hold = 1×',transform=a.transAxes,color='#444')
    image=b.imshow(field,cmap='magma',origin='upper');b.set(title='Norm error over the physical grid',xlabel='Native W',ylabel='Native H')
    fig.colorbar(image,ax=b,shrink=.75,label='RMS error across 24 heads')
    palette=['#5874ce','#2a9d8f','#cc6845','#a45ca6']
    for row,color in zip([4,11,12,13],palette):
        q=[x for x in report['queryHeads'] if x['row']==row]
        c.scatter([x['weightedCenteredVariance']/2 for x in q],[x['exactKL'] for x in q],s=22,alpha=.8,color=color,label=f'Query row {row}')
    c.plot([1e-4,3],[1e-4,3],ls='--',color='#444',lw=1)
    c.set(xscale='log',yscale='log',xlabel='½ attention-weighted score variance',ylabel='Exact KL(reference || candidate)',title='Attention sensitivity is heterogeneous')
    c.legend(frameon=False,fontsize=8,loc='upper left')
    fig.suptitle('A quarter-grid of exact key norms is insufficient on this saved pair',fontsize=15,fontweight='bold')
    fig.supxlabel('Fixed even 16×16 knots; coordinate 31 clamps to 30. Saved-only diagnostics; no new learned projections or consumers.',fontsize=10)
    path=out/'spatial-norm.png';fig.savefig(path,dpi=160);plt.close(fig)
    copy=root.parent/'obsidian/images/zero-neural-cycle61-spatial-norm.png';shutil.copyfile(path,copy)
    save(out/'figure.json',{'reportSHA256':sha(p/'report.json'),'imageSHA256':sha(path),'plotSourceSHA256':sha(__file__),
        'scope':'Published norm/score scalar rows only; exact sampled cells supplied, no activation/model execution or native consumer fidelity claim.'})
    return str(path)


if __name__=='__main__':print(execute())
