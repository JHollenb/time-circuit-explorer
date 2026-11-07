"""Scalar geometry and image-query-address maps; no retained array loading."""
import argparse
import gzip
import json
from pathlib import Path
from encoder_atoms import np,read,save,sha


def execute(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True);r=read(p/'report.json');rows=r['accounts'];queries=[json.loads(x) for x in gzip.decompress((p/'queries.jsonl.gz').read_bytes()).splitlines()]
    labels=[x['id'].replace('feedback-s','F ').replace('evaluation-s','E ').replace('.','\n') for x in rows];colors=['#2563eb' if x['population']=='feedback' else '#d97706' for x in rows]
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True);axes[0,0].bar(range(6),[100*x['residualEnergyFraction'] for x in rows],color=colors);axes[0,0].set_ylabel('Input energy outside source span (%)');axes[0,0].set_xticks(range(6),labels,fontsize=9);axes[0,0].set_title('Most supplied input energy lies in the source span')
    med=np.array([x['leverageMedian'] for x in rows]);lo=np.array([x['leverageMin'] for x in rows]);hi=np.array([x['leverageMax'] for x in rows]);axes[0,1].errorbar(range(6),med,yerr=[med-lo,hi-med],fmt='o',capsize=5,color='#475569',label='Median with observed min/max');axes[0,1].axhline(1,color='#dc2626',linestyle='--',label='Exact source reference');axes[0,1].set_yscale('log');axes[0,1].set_ylabel('Inverse-mode leverage (log scale)');axes[0,1].set_xticks(range(6),labels,fontsize=9);axes[0,1].set_title('Small singular scales magnify source coordinates');axes[0,1].legend(frameon=False,fontsize=9,loc='upper left')
    norm=LogNorm(vmin=1,vmax=max(x['leverage'] for x in queries));cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e2e8f0')
    for ax,key,title in [(axes[1,0],'feedback-s60673.base','Feedback 60673 base'),(axes[1,1],'evaluation-s60674.base','Evaluation 60674 base')]:
        grid=np.full(1024,np.nan)
        for q in queries:
            if q['id']==key:grid[q['imageQueryIndex']]=q['leverage']
        im=ax.imshow(grid.reshape(32,32),norm=norm,cmap=cmap,interpolation='nearest');ax.set_title(title+' · actual query addresses');ax.set_xlabel('Image-query column');ax.set_ylabel('Image-query row')
    fig.colorbar(im,ax=list(axes[1,:]),label='Leverage ||X V Σ⁻¹||² per query; shared scale',shrink=.85)
    for ax in axes[0]:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Source-span coverage does not bound this fixed map’s response',fontsize=16,weight='bold');fig.supxlabel('F = same-location feedback; E = top-left evaluation. Gray cells exclude the edit and halo.\nRanges are observed queries, not uncertainty intervals. Source leverage 1 is algebraic, not a new assay. No native images or new predictor.',fontsize=10)
    dest=out/'extension-geometry.png';fig.savefig(dest,dpi=145);plt.close(fig);save(out/'figure.json',{'schema':'tce-extension-geometry-figure-v1','figureSHA256':sha(dest),'plotterSourceSHA256':sha(__file__),'reportSHA256':sha(p/'report.json'),'scope':'Published scalar/query-address records only. Fixed base-role maps share one color scale; all six cohorts retain aggregate and complete query records.'});return str(dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
