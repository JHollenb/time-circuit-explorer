"""Response-focused static figure from already-published scalar records."""
import gzip
import json
from pathlib import Path
from encoder_atoms import read, save, sha


def execute():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/stale-key-context'
    r,provenance=read(p/'report.json'),read(p/'provenance.json');rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#182b42'})
    fig,axes=plt.subplots(1,2,figsize=(12,5.8));fig.subplots_adjust(left=.08,right=.97,top=.77,bottom=.31,wspace=.3)
    fig.suptitle('Retained keys must predict the actual context response',x=.08,y=.97,ha='left',fontsize=18,fontweight='bold')
    fig.text(.08,.87,'Fixed primary: P keys + Q values. Diagnostic: Q keys + P values. Shared selected P inputs.',fontsize=11)
    names=['A','U','M','T','nextQ'];labels=['Attention','Residual','MLP','Own exit','Next query']
    for offset,role,label,color in [(-.18,'primaryResponse','Retained keys (primary)','#3976b7'),(.18,'diagnosticResponse','Retained values (diagnostic)','#bf7d30')]:
        values=[100*r['scores'][n][role]['relativeRms'] for n in names]
        axes[0].bar([i+offset for i in range(len(names))],values,width=.34,label=label,color=color)
    axes[0].set(xticks=range(len(names)),xticklabels=labels,ylabel='Error / actual context-response RMS (%)')
    axes[0].axhline(100,color='#788493',ls='--',lw=1);axes[0].legend(frameon=False,fontsize=8)
    heads=[x for x in rows if x['field']=='nextQ' and x['panel']=='query-head']
    for role,label,color in [('primaryResponse','Retained keys','#3976b7'),('diagnosticResponse','Retained values','#bf7d30')]:
        points=[(i,100*x[role]['relativeRms']) for i,x in enumerate(heads) if x[role]['relativeRms'] is not None]
        axes[1].scatter([x[0] for x in points],[x[1] for x in points],s=10,label=label,color=color,alpha=.75)
    axes[1].axhline(100,color='#788493',ls='--',lw=1,label='Zero response')
    axes[1].set(xlabel='All 96 selected row/head addresses',ylabel='Next-query response error (%)');axes[1].legend(frameon=False,fontsize=8)
    primary=r['scores']['nextQ']['primaryResponse']['relativeRms'];endpoint=r['scores']['nextQ']['primaryEndpoint']['relativeRms']
    fig.text(.08,.19,f'Primary next-query response error {100*primary:.2f}% · endpoint error {100*endpoint:.3f}% · denominators differ',fontsize=11)
    seconds=provenance['schedulerResult']['elapsed_s'];fig.text(.08,.12,f'905.970M new matrix MACs · {seconds:.2f}s instrumented assay · paid parent banks and diagonal trajectories',fontsize=10)
    fig.text(.08,.055,'This tests a possible key-reuse dependency. No fresh value producer, measured serving saving or arbitrary-prompt zero-neural image.',fontsize=9)
    out=root/'docs/results/stale-key-context-figure';out.mkdir(parents=True,exist_ok=False);file=out/'stale-key-context.png';fig.savefig(file,dpi=160);plt.close(fig)
    save(out/'figure.json',{'reportSHA256':sha(p/'report.json'),'plotSourceSHA256':sha(__file__),'imageSHA256':sha(file),'scope':'Published response errors and every final query-head address; no model/array execution.'})
    return str(file)


if __name__=='__main__':print(execute())
