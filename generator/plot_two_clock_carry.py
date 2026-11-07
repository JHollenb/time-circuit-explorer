"""Plot published scalar measurements; no scientific arrays or model runtime."""
import argparse
from pathlib import Path
from encoder_atoms import read,save,sha


def execute(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True);r=read(p/'report.json');rows=r['rows']
    labels=['Orange pitcher / teal can','Teal pitcher / orange can','Dependent color contrast']
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    for ax,values,title,unit in [
        (axes[0],[x['outgoingIncomingGain'] for x in rows],'Final latent difference / incoming difference','Gain'),
        (axes[1],[x['projection']['lambda'] for x in rows],'Write along incoming difference','Parallel coefficient'),
        (axes[2],[100*x['projection']['orthogonalWriteFractionOfEnergy'] for x in rows],'Write energy orthogonal to incoming difference','Percent')]:
        bars=ax.barh(range(3),values,color=['#0f766e','#d97706','#64748b']);ax.set_yticks(range(3),labels if ax is axes[0] else ['00','10','Color']);ax.invert_yaxis()
        ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4);ax.set_xlim(0,max(values)*1.3);ax.set_title(title,fontsize=10);ax.set_xlabel(unit);ax.grid(axis='x',alpha=.15)
    axes[0].axvline(1,color='#991b1b',linestyle='--',linewidth=1)
    axes[1].text(.02,-.25,'Cancellation would require a negative coefficient.',transform=axes[1].transAxes,fontsize=9)
    fig.suptitle('A better image comparison does not imply a contracting state update',fontsize=15,weight='bold')
    fig.text(.5,-.025,'Saved native NN/HN states; no new neural evaluation. Two known contexts, one seed; color is dependent evidence.',ha='center',fontsize=10)
    path=out/'two-clock-carry.png';fig.savefig(path,dpi=130,bbox_inches='tight');plt.close(fig)
    save(out/'figure.json',{'schema':'tce-two-clock-carry-figure-v1','figureSHA256':sha(path),'plotterSourceSHA256':sha(__file__),'sourceReportSHA256':sha(p/'report.json')})
    return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
