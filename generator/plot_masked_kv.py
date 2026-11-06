"""Derived row-work and own-state-error figure from the published assay."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from encoder_atoms import save, sha


def plot(source,destination):
    source=Path(source);out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s)for s in gzip.decompress((source/'observations.jsonl.gz').read_bytes()).splitlines()]
    values=sorted([r for r in rows if r['case']=='short'and r['index']==1 and r['channel']=='candidate-minus-native'],key=lambda r:r['position'])
    zero=sorted([r for r in rows if r['case']=='short'and r['index']==1 and r['channel']=='zero-minus-full'],key=lambda r:r['position'])
    active=sum(r['attended']for r in values)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    fig.suptitle('Masked projections can be omitted; finite geometry still matters',fontweight='bold',fontsize=14)
    axes[0].bar(['Full K/V','Compact K/V'],[512,active],color=['#94a3b8','#2563eb'],width=.55)
    axes[0].set(ylabel='Rows projected by each K/V linear',ylim=(0,570),title='Short raw prompt: 28 attended positions')
    for i,v in enumerate([512,active]):axes[0].text(i,v+12,str(v),ha='center',fontweight='bold')
    axes[0].text(.5,.48,'2.538 B nominal MACs omitted\n4.91% of block linear work',ha='center',transform=axes[0].transAxes,fontsize=10)
    x=[r['position']for r in values];ratios=[100*r['blockUpdate']['relativeRms'] if r['blockUpdate']['relativeRms']is not None else float('nan')for r in values]
    axes[1].axvspan(active-.5,511,color='#fef3c7',alpha=.65,label='Padding queries retained')
    axes[1].plot(x,ratios,color='#d97706',linewidth=1,label='Compact vs native H2 update')
    axes[1].plot([r['position']for r in zero],[100*r['blockUpdate']['relativeRms'] if r['blockUpdate']['relativeRms']is not None else float('nan')for r in zero],color='#0d9488',linewidth=1.5,label='Zero-masked vs full')
    axes[1].set(xlabel='Query / output-state row',ylabel='Error / native block-update RMS (%)',title='Same emitted H1; changed projection geometry',xlim=(0,511))
    axes[1].legend(fontsize=8,loc='upper right');axes[1].grid(alpha=.15)
    path=out/'masked-kv-boundary.png';fig.savefig(path,dpi=160);plt.close(fig)
    save(out/'figure.json',{'schema':'tce-masked-kv-derived-figure-v1','sourceSHA256':sha(source/'observations.jsonl.gz'),'figureSHA256':sha(path),
        'scope':'One short raw prompt, all512dependent query rows; projection arithmetic and retained-state errors, not measured speedup or image quality. All-attended control exact/no savings is in complete report.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();plot(a.source,a.destination)
