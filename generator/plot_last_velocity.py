"""Original candidate and parent PNGs, with measured velocity/image error."""
import argparse
from pathlib import Path
from encoder_atoms import read,save,sha


def execute(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True)
    r=read(p/'report.json');prov=read(p/'provenance.json');images={}
    fig=plt.figure(figsize=(17,11),constrained_layout=True);grid=fig.add_gridspec(3,4,height_ratios=[1,1,.55])
    for row,case in enumerate(['00','10']):
        for col,(lane,title) in enumerate([('full','Historical full encoder'),('own','Own reduced-encoder parent'),('hold','Last velocity: hold'),('secant','Last velocity: secant')]):
            n=lane+'-'+case+'.png';images[n]=sha(p/n);ax=fig.add_subplot(grid[row,col]);ax.imshow(Image.open(p/n));ax.axis('off')
            ax.set_title(title+'\n'+('orange pitcher / teal can' if case=='00' else 'teal pitcher / orange can'),fontsize=10)
    ax=fig.add_subplot(grid[2,:2]);ports=['velocity3','latent4','pngRGB'];labels=['Last velocity','Final latent','PNG pixels']
    for i,policy in enumerate(['hold','secant']):
        v=[next(x['relativeRms'] for x in r['pairedColorResponse'] if x['port']==port and x['policy']==policy and x['reference']=='own') for port in ports]
        bars=ax.bar([j+(i-.5)*.35 for j in range(3)],[100*x for x in v],width=.35,label=policy)
        ax.bar_label(bars,labels=[f'{100*x:.1f}%' for x in v],padding=3,fontsize=9)
    ax.set_xticks(range(3),labels);ax.set_ylabel('Color-response error (%)');ax.margins(y=.25);ax.legend();ax.grid(axis='y',alpha=.2)
    ax=fig.add_subplot(grid[2,2:]);ax.axis('off');s=prov['schedulerResult']
    ax.text(0,1,'\n'.join([f"{s['elapsed_s']:.2f} s assay · RSS {s['peak_rss_mb']:.1f} MB · VRAM {s['peak_vram_mb']:.0f} MB",
        '0 new encoder/denoiser calls · 4 paid native VAE decodes',
        '6 historical denoisers remain charged own-prefix inputs.',
        'Fixed extrapolation a = 2.24666; one previous-velocity hold control.',
        'All four images seal before saved targets and exact scheduler replays.',
        'Potential 25% denoiser affine reduction; no measured speedup.',
        'Known-prefix replay, not arbitrary-prompt zero-neural generation.']),va='top',fontsize=10,linespacing=1.5)
    fig.suptitle('Can the preceding trajectory supply the final velocity?',fontsize=16,weight='bold')
    path=out/'last-velocity.png';fig.savefig(path,dpi=130);plt.close(fig)
    save(out/'figure.json',{'schema':'tce-last-velocity-figure-v1','figureSHA256':sha(path),'plotterSourceSHA256':sha(__file__),
        'sourceReportSHA256':sha(p/'report.json'),'sourcePNGs':images,
        'scope':'Four original new candidates beside their own and historical full originals. Metrics are development observations, not semantic certification.'})
    return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
