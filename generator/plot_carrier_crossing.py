"""Plot published crossing scalars and actual query addresses, no tensor loads."""
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
    p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True);r=read(p/'report.json');rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()]
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);xx=np.arange(2);width=.24
    for shift,key,label,color in [(-1,'referenceRms','Receiver 0 response d₀','#2563eb'),(0,'candidateRms','Receiver 1 response d₁','#d97706'),(1,'rms','Interaction I = d₁ − d₀','#a855f7')]:
        axes[0,0].bar(xx+shift*width,[x[key] for x in r['accounts']],width,label=label,color=color)
    axes[0,0].set_xticks(xx,['All 1,024 queries','944 source-nonlocal queries']);axes[0,0].set_ylabel('Velocity response RMS');axes[0,0].set_title('Same supplied carrier pair, different responses');axes[0,0].legend(frameon=False,fontsize=9);axes[0,0].grid(axis='y',alpha=.2)
    cosine=np.array([x['cosine'] if x['cosine'] is not None else np.nan for x in rows]);nonlocal_rows=np.array([x['sourceNonlocal'] for x in rows]);interaction=np.array([x['rms'] for x in rows])
    axes[0,1].hist(cosine[np.isfinite(cosine)],bins=np.linspace(-1,1,41),color='#94a3b8',label='All query rows');axes[0,1].hist(cosine[nonlocal_rows & np.isfinite(cosine)],bins=np.linspace(-1,1,41),histtype='step',linewidth=2,color='#2563eb',label='Source-nonlocal subset');axes[0,1].axvline(0,color='#334155',linestyle='--',linewidth=1);axes[0,1].set_xlim(-1,1);axes[0,1].set_xlabel('Cosine of d₀ and d₁ within one query');axes[0,1].set_ylabel('Dependent query count');axes[0,1].set_title('Aggregate cosine: 0.0110 / 0.0125');axes[0,1].legend(frameon=False,fontsize=9)
    im=axes[1,0].imshow(cosine.reshape(32,32),cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest');fig.colorbar(im,ax=axes[1,0],label='Signed per-query response cosine',shrink=.8);axes[1,0].set_title('Direction depends on the receiving state')
    im=axes[1,1].imshow(interaction.reshape(32,32),cmap='magma',vmin=0,interpolation='nearest');fig.colorbar(im,ax=axes[1,1],label='Per-query interaction RMS',shrink=.8);axes[1,1].set_title('Interaction at actual image-query addresses')
    for ax in axes[1]:
        for i in np.flatnonzero(~nonlocal_rows):ax.add_patch(Rectangle((i%32-.5,i//32-.5),1,1,fill=False,edgecolor='#22c55e',linewidth=.7))
        ax.set_xlabel('Image-query column');ax.set_ylabel('Image-query row')
    fig.suptitle('A carrier change does not specify its downstream effect by itself',fontsize=16,weight='bold')
    fig.supxlabel('Four supplied-carrier interventions; hybrid states may be off the natural manifold. Green outlines mark the source edit/halo.\nOne donor pair × two receivers; rows are dependent, not independent trials. Four full native calls are paid; no images or cheap predictor.',fontsize=10)
    dest=out/'carrier-receiver-crossing.png';fig.savefig(dest,dpi=145);plt.close(fig)
    save(out/'figure.json',{'schema':'tce-carrier-crossing-figure-v1','figureSHA256':sha(dest),'plotterSourceSHA256':sha(__file__),'reportSHA256':sha(p/'report.json'),'scope':'Published all/944 scalar accounts and1024 actual-address query records; no complete tensor hydration or scientific threshold.'});return str(dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
