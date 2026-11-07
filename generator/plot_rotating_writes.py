"""Original routed images and same-state, paid-work comparison."""
import argparse
from pathlib import Path
from encoder_atoms import read,save,sha


def execute(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True);r=read(p/'report.json');prov=read(p/'provenance.json');originals={}
    fig=plt.figure(figsize=(16,11),constrained_layout=True);grid=fig.add_gridspec(3,4,height_ratios=[1,1,.6])
    for row,c in enumerate(['00','10']):
        for col,(policy,title) in enumerate([('full','Historical full encoding'),('HH','Hold previous velocity'),('routed','Full state / rotating writes'),('HN','Full same-state native read')]):
            name=policy+'-'+c+'.png';originals[name]=sha(p/name);ax=fig.add_subplot(grid[row,col]);ax.imshow(Image.open(p/name));ax.axis('off');ax.set_title(title+'\n'+('orange pitcher / teal can' if c=='00' else 'teal pitcher / orange can'),fontsize=10)
    ax=fig.add_subplot(grid[2,:2])
    for i,(name,label) in enumerate([('hold','Hold'),('routed','Rotating writes')]):
        rows=r['holdAccounts'] if name=='hold' else r['accounts']
        vals=[next(x['rms'] for x in rows if x['case']==c and x['port']=='pngRGB' and x['reference']=='HN') for c in ['00','10']]
        bars=ax.bar([j+(i-.5)*.3 for j in range(2)],vals,width=.3,label=label);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set_xticks(range(2),['00','10']);ax.set_ylabel('PNG RMS error vs same-state native read');ax.margins(y=.2);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    ax=fig.add_subplot(grid[2,2:]);ax.axis('off');s=prov['schedulerResult']
    vals=[next(x['relativeRms'] for x in r['pairedColorResponse'] if x['reference']=='HN' and x['port']==p0) for p0 in ['velocity3','latent4','pngRGB']]
    ax.text(0,1,'\n'.join([f"{s['elapsed_s']:.2f} s · RSS {s['peak_rss_mb']:.0f} MB · VRAM {s['peak_vram_mb']:.0f} MB",'2 routed denoisers · 2 paid VAE decodes · 2 new images',
        'Final-call affine MACs: 57.81% of full native read.',f"Signed color error vs HN: {vals[0]*100:.1f}% velocity / {vals[1]*100:.1f}% latent / {vals[2]*100:.1f}% PNG",'All1536 state rows/KV;768 selected queries/writes.',
        '50 unchanged-row witnesses;48 own-state handoff joins.', 'Four earlier full denoisers and encoding remain paid.', 'Known development pair; strict zero-neural goal open.']),va='top',fontsize=10,linespacing=1.5)
    fig.suptitle('Full readable state does not rescue this fixed sparse-write policy',fontsize=16,weight='bold');path=out/'rotating-writes.png';fig.savefig(path,dpi=130,bbox_inches='tight');plt.close(fig)
    save(out/'figure.json',{'schema':'tce-rotating-write-figure-v1','figureSHA256':sha(path),'plotterSourceSHA256':sha(__file__),'sourceReportSHA256':sha(p/'report.json'),'sourcePNGs':originals})
    return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
