"""Compare published scalar response records; no scientific-array loading."""
import argparse
from pathlib import Path
from encoder_atoms import read,save,sha


def execute(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True);r=read(p/'report.json');pr=read(p/'provenance.json')
    ids=[x['id'] for x in r['accounts'] if x['method']=='minimum-norm' and x['population']!='discovery'];fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,constrained_layout=True)
    for offset,(method,label,color) in enumerate([('original-writer','Original endpoint + response','#94a3b8'),('response-only','Fixed ridge .01','#2563eb'),('minimum-norm','All-mode minimum norm','#d97706')]):
        rows=[next(x for x in r['accounts'] if x['id']==key and x['method']==method) for key in ids];xs=[i+(offset-1)*.24 for i in range(len(ids))]
        axes[0].bar(xs,[100*x['relativeRms'] for x in rows],width=.24,label=label,color=color)
        axes[1].plot(range(len(ids)),[x['cosine'] for x in rows],'o-',label=label,color=color)
    axes[0].set_yscale('log');axes[0].set_ylim(80,3600);axes[0].set_yticks([100,300,1000,3000],['100','300','1,000','3,000']);axes[0].axhline(100,color='#dc2626',linestyle='--',label='Zero response')
    axes[0].set_ylabel('Response error (% of target RMS, log scale)');axes[0].legend(frameon=False,ncols=2,fontsize=10)
    axes[1].axhline(0,color='#dc2626',linestyle='--');axes[1].set_ylabel('Response cosine');axes[1].set_xticks(range(len(ids)),[x.replace('feedback-s','Feedback ').replace('evaluation-s','Evaluation ').replace('.','\n') for x in ids],fontsize=10)
    for ax in axes:ax.axvline(1.5,color='#cbd5e1');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Source interpolation does not recover the unseen response direction',fontsize=16,weight='bold')
    s=pr['schedulerResult'];fig.supxlabel(f"Source relative error: ~1.05e-13 on both fields (not plotted). Eight predictions use 2.970 billion learned affine MACs.\n{s['elapsed_s']:.2f} s · {s['peak_rss_mb']:.0f} MB RSS · 0 VRAM · no new original-model/H0/images · historical paired fields; seed/location confounded",fontsize=10)
    dest=out/'minimum-response.png';fig.savefig(dest,dpi=145);plt.close(fig)
    save(out/'figure.json',{'schema':'tce-minimum-response-figure-v1','figureSHA256':sha(dest),'plotterSourceSHA256':sha(__file__),'reportSHA256':sha(p/'report.json'),'scope':'Published scalar accounts only; source interpolation labeled separately; no new experiment or serving selection.'})
    return str(dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
