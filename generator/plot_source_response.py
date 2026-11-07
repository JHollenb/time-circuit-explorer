"""Published response-error and direction comparisons; no array/model loading."""
import argparse
from pathlib import Path
from encoder_atoms import read,save,sha


def execute(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True);report=read(p/'report.json');prov=read(p/'provenance.json')
    ids=[x['id'] for x in report['accounts'] if x['method']=='response-only'];fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,constrained_layout=True)
    for offset,(method,label,color) in enumerate([('original-writer','Endpoint + response objective','#94a3b8'),('response-only','Response-only objective','#2563eb')]):
        rows=[next(x for x in report['accounts'] if x['id']==key and x['method']==method) for key in ids];xs=[i+(offset-.5)*.32 for i in range(len(ids))]
        axes[0].bar(xs,[100*x['relativeRms'] for x in rows],width=.32,label=label,color=color)
        axes[1].plot(range(len(ids)),[x['cosine'] for x in rows],'o-',label=label,color=color)
    axes[0].axhline(100,color='#dc2626',linestyle='--',label='Zero response');axes[0].set_ylabel('Response error (% of target RMS)');axes[0].legend(frameon=False,ncols=3,fontsize=9)
    axes[1].axhline(0,color='#dc2626',linestyle='--');axes[1].set_ylabel('Response cosine');axes[1].set_xticks(range(len(ids)),[x.replace('discovery-s','Source ').replace('feedback-s','Feedback ').replace('evaluation-s','Evaluation ').replace('.','\n') for x in ids],fontsize=9)
    for ax in axes:
        ax.axvline(1.5,color='#cbd5e1');ax.axvline(3.5,color='#cbd5e1');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Response-only fitting improves source fit, but transfer still loses to zero',fontsize=15,weight='bold')
    s=prov['schedulerResult'];fig.supxlabel(f"{s['elapsed_s']:.2f} s · {s['peak_rss_mb']:.0f} MB RSS · 0 VRAM · no original-model/H0/image evaluations\n16 served learned affine products remain paid. Historical paired fields; seed and edit location are confounded.",fontsize=10)
    dest=out/'source-response.png';fig.savefig(dest,dpi=145);plt.close(fig)
    save(out/'figure.json',{'schema':'tce-source-response-figure-v1','figureSHA256':sha(dest),'plotterSourceSHA256':sha(__file__),'reportSHA256':sha(p/'report.json'),'scope':'Published scalar visualization, not new experiments.'})
    return str(dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
