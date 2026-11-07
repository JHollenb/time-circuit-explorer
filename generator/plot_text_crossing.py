"""Signed energy decomposition and actual-address query maps from scalars."""
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
    p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True);r=read(p/'report.json');rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()];prior=read(p.parent/'carrier-crossing/report.json')
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);xx=np.arange(2);width=.25
    for shift,key,label,color in [(-1,'remainingReceiverEnergy','Remaining receiver term U²','#2563eb'),(0,'firstTextWriteEnergy','First text-write term T²','#d97706'),(1,'twiceRemainingText','Signed cross term 2⟨U,T⟩','#a855f7')]:
        axes[0,0].bar(xx+shift*width,[a[key]/a['originalInteractionEnergy'] for a in r['accounts']],width,label=label,color=color)
    axes[0,0].axhline(0,color='#475569',linewidth=.7);axes[0,0].axhline(1,color='#475569',linestyle='--',linewidth=.7,label='Their signed sum = 1');axes[0,0].set_xticks(xx,['All 1,024 queries','944 source-nonlocal']);axes[0,0].set_ylabel('Energy term / original interaction energy');axes[0,0].set_title('A substantial text effect partly cancels the remainder');axes[0,0].legend(frameon=False,fontsize=8);axes[0,0].grid(axis='y',alpha=.2)
    for shift,values,label,color in [(-1,[a['cosine'] for a in prior['accounts']],'Original d₀ versus d₁','#94a3b8'),(0,[a['responseR0TextR0Cosine'] for a in r['accounts']],'d₀ versus text-fixed d₁₀','#2563eb'),(1,[a['responseR1TextR0Cosine'] for a in r['accounts']],'d₁ versus text-fixed d₁₀','#d97706')]:axes[0,1].bar(xx+shift*width,values,width,label=label,color=color)
    axes[0,1].set_xticks(xx,['All 1,024 queries','944 source-nonlocal']);axes[0,1].set_ylim(-.05,1);axes[0,1].set_ylabel('Signed response cosine');axes[0,1].set_title('Fixing the first text write does not restore alignment');axes[0,1].legend(frameon=False,fontsize=8);axes[0,1].grid(axis='y',alpha=.2)
    cross=np.array([a['remainingTextCosine'] if a['remainingTextCosine'] is not None else np.nan for a in rows]);ratio=np.array([a['remainingOverOriginalInteraction'] if a['remainingOverOriginalInteraction'] is not None else np.nan for a in rows]);nonlocal_rows=np.array([a['sourceNonlocal'] for a in rows])
    im=axes[1,0].imshow(cross.reshape(32,32),cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest');fig.colorbar(im,ax=axes[1,0],label='Per-query cosine of U and T',shrink=.8);axes[1,0].set_title('Signed cancellation varies across the field')
    im=axes[1,1].imshow(ratio.reshape(32,32),cmap='viridis',vmin=0,interpolation='nearest');fig.colorbar(im,ax=axes[1,1],label='Per-query ‖U‖ / ‖I‖; full observed range',shrink=.8);axes[1,1].set_title('Remaining receiver difference at actual query addresses')
    for ax in axes[1]:
        for i in np.flatnonzero(~nonlocal_rows):ax.add_patch(Rectangle((i%32-.5,i//32-.5),1,1,fill=False,edgecolor='#22c55e',linewidth=.7))
        ax.set_xlabel('Image-query column');ax.set_ylabel('Image-query row')
    fig.suptitle('Holding one text write fixed leaves the main receiver dependence',fontsize=16,weight='bold');fig.supxlabel('I = U + T is an ordered signed decomposition, not additive causal percentages. Later text–image feedback stays live.\nSupplied hybrid states; two new full native calls paid. Green outlines mark the original source edit/halo. No fit, image or zero-neural generator.',fontsize=10)
    dest=out/'text-write-crossing.png';fig.savefig(dest,dpi=145);plt.close(fig);save(out/'figure.json',{'schema':'tce-text-crossing-figure-v1','figureSHA256':sha(dest),'plotterSourceSHA256':sha(__file__),'reportSHA256':sha(p/'report.json'),'parentReportSHA256':sha(p.parent/'carrier-crossing/report.json'),'scope':'Published scalar accounts and1024 actual-address query records. Full observed map ranges; no retained tensor hydration, independent-trial or causal-share claim.'});return str(dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
