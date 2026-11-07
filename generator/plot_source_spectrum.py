"""Render the full published scalar spectrum without loading model arrays."""
import argparse
import gzip
import json
from pathlib import Path
from encoder_atoms import np,read,save,sha


def execute(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True);report=read(p/'report.json');s=report['summary'];prov=read(p/'provenance.json')
    rows=[json.loads(x) for x in gzip.decompress((p/'modes.jsonl.gz').read_bytes()).splitlines()];sigma=np.array([x['singularValue'] for x in rows]);target=np.array([x['targetEnergy'] for x in rows]);residual=np.array([x['ridgeResidualEnergy'] for x in rows]);position=np.arange(len(rows))
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,constrained_layout=True)
    axes[0].semilogy(position,np.where(sigma>0,sigma,np.nan),color='#2563eb',label='Singular value');axes[0].axhline(s['tau']**.5,color='#dc2626',linestyle='--',label='√(Nλ): response filter = ½');axes[0].set_ylabel('Singular value');axes[0].legend(frameon=False,fontsize=9)
    axes[1].semilogy(position,np.where(target>0,target,np.nan),color='#64748b',label='Target energy');axes[1].semilogy(position,np.where(residual>0,residual,np.nan),color='#2563eb',label='Ridge residual energy');axes[1].set_ylabel('Energy per source mode');axes[1].legend(frameon=False,fontsize=9)
    axes[2].plot(position,np.cumsum(target)/s['targetEnergy'],color='#64748b',label='Cumulative target energy');axes[2].plot(position,np.cumsum(residual)/s['targetEnergy'],color='#2563eb',label='Cumulative ridge residual energy');axes[2].set_ylabel('Fraction of complete target energy');axes[2].set_xlabel('Singular-direction index (descending σ; no modes selected)');axes[2].legend(frameon=False,fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('The fixed source spectrum separates ridge attenuation from reconstruction',fontsize=15,weight='bold');job=prov['schedulerResult']
    fig.supxlabel(f"One SVD · fixed λ = .01 · {s['rows']:,} × {s['features']:,} source matrix · {job['elapsed_s']:.2f} s / {job['peak_rss_mb']:.0f} MB RSS\nThin U is square; a small complement is not an information-sufficiency or transfer result. No new predictor or images.",fontsize=10)
    dest=out/'source-spectrum.png';fig.savefig(dest,dpi=140);plt.close(fig)
    save(out/'figure.json',{'schema':'tce-source-spectrum-figure-v1','figureSHA256':sha(dest),'plotterSourceSHA256':sha(__file__),'reportSHA256':sha(p/'report.json'),'scope':'Published scalar spectrum only; no new experiment or mode selection.'})
    return str(dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
