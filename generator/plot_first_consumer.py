"""Stage response trends, finite-policy countertrends and query map."""
import argparse
import gzip
import json
from pathlib import Path
from encoder_atoms import np,read,save,sha


def execute(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from first_consumer import STAGES
    p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True);r=read(p/'report.json');rows=[json.loads(x) for x in gzip.decompress((p/'queries.jsonl.gz').read_bytes()).splitlines()];labels=['P','gP','residual','LN','scale','MLP in','gate','up','SiLU','product','down','MLP gate','exit'];x=np.arange(13)
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    for scope,color,label in [('all','#2563eb','All 1,024 queries'),('source-nonlocal','#d97706','944 source-nonlocal queries')]:
        values=[next(a for a in r['stageAccounts'] if a['stage']==s and a['scope']==scope) for s in STAGES]
        axes[0,0].plot(x,[a['cosine'] for a in values],'-o',color=color,label=label,markersize=4);axes[0,1].plot(x,[a['interactionOverR0'] for a in values],'-o',color=color,label=label,markersize=4)
    for ax in axes[0]:ax.set_xticks(x,labels,rotation=40,ha='right',fontsize=9);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9);ax.axvspan(7.5,9.5,color='#e9d5ff',alpha=.3)
    axes[0,0].set_ylabel('Cosine between receiver responses d₀ and d₁');axes[0,0].set_ylim(.75,1.01);axes[0,0].set_title('Alignment changes through the nonlinear channel product');axes[0,1].set_ylabel('‖d₁ − d₀‖ / ‖d₀‖ at that stage');axes[0,1].set_ylim(0,.72);axes[0,1].set_title('Receiver dependence is already substantial locally')
    for shift,receiver,scope,color,alpha,label in [(-1.5,'R0','all','#2563eb',1,'R0 all'),(-.5,'R1','all','#d97706',1,'R1 all'),(.5,'R0','source-nonlocal','#2563eb',.4,'R0 nonlocal'),(1.5,'R1','source-nonlocal','#d97706',.4,'R1 nonlocal')]:
        norm=next(a for a in r['factorAccounts'] if a['kind']=='normalization' and a['receiver']==receiver and a['scope']==scope);prod=next(a for a in r['factorAccounts'] if a['kind']=='product' and a['receiver']==receiver and a['scope']==scope)
        vals=[norm['nativePolicy']['relativeRms'],prod['nativeActivationPolicy']['relativeRms'],prod['nativeProductPolicy']['relativeRms']];axes[1,0].bar(np.arange(3)+shift*.18,100*np.array(vals),.18,color=color,alpha=alpha,label=label)
    axes[1,0].set_xticks(range(3),['LayerNorm response','SiLU response','Product response']);axes[1,0].set_ylabel('Native versus FP64-reference response error (%)');axes[1,0].set_title('Finite-policy discrepancies grow in weaker nonlocal responses');axes[1,0].legend(frameon=False,fontsize=8,ncol=2);axes[1,0].grid(axis='y',alpha=.2)
    qs=[a for a in rows if a['stage']=='product'];grid=np.array([a['cosine'] if a['cosine'] is not None else np.nan for a in qs]);im=axes[1,1].imshow(grid.reshape(32,32),cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
    for a in qs:
        if not a['sourceNonlocal']:i=a['imageQueryIndex'];axes[1,1].add_patch(Rectangle((i%32-.5,i//32-.5),1,1,fill=False,edgecolor='#22c55e',linewidth=.7))
    axes[1,1].set_title('Product response at actual query addresses');axes[1,1].set_xlabel('Image-query column');axes[1,1].set_ylabel('Image-query row');fig.colorbar(im,ax=axes[1,1],label='Signed per-query response cosine',shrink=.8)
    fig.suptitle('A common carrier change meets receiver-dependent nonlinear operands',fontsize=16,weight='bold');fig.supxlabel('One local image consumer, four supplied-state cells. Stage widths differ; every ratio uses its own response denominator.\nGreen outlines: original edit/halo. FP64 references retain native-policy differences. Original four-matrix work paid; no full-block-exit parity or image claim.',fontsize=10)
    dest=out/'first-image-consumer.png';fig.savefig(dest,dpi=145);plt.close(fig);save(out/'figure.json',{'schema':'tce-first-consumer-figure-v1','figureSHA256':sha(dest),'plotterSourceSHA256':sha(__file__),'reportSHA256':sha(p/'report.json'),'scope':'Published scalar/query records only. Upper axes display measured cosine range .75–1.01 and interaction0–.72; complete row distributions retained in lossless streams.'});return str(dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
