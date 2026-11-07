"""Original four-corner images and measured work/response tradeoffs."""
import argparse
from pathlib import Path
from encoder_atoms import read,save,sha


def execute(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=Path(source);out=Path(destination);out.mkdir(exist_ok=False,parents=True);r=read(p/'report.json');prov=read(p/'provenance.json');images={}
    fig=plt.figure(figsize=(20,11),constrained_layout=True);grid=fig.add_gridspec(3,5,height_ratios=[1,1,.55])
    for row,c in enumerate(['00','10']):
        for col,(lane,title) in enumerate([('full','Historical full encoding'),('NN','NN: four native calls'),('NH','NH: native then hold'),('HN','HN: hold then native'),('HH','HH: two holds')]):
            n=lane+'-'+c+'.png';images[n]=sha(p/n);ax=fig.add_subplot(grid[row,col]);ax.imshow(Image.open(p/n));ax.axis('off')
            ax.set_title(title+'\n'+('orange pitcher / teal can' if c=='00' else 'teal pitcher / orange can'),fontsize=10)
    ax=fig.add_subplot(grid[2,:3]);ports=['velocity3','latent4','pngRGB']
    for i,policy in enumerate(['NH','HN','HH']):
        vals=[next(x['relativeRms'] for x in r['pairedColorResponse'] if x['policy']==policy and x['reference']=='NN' and x['port']==port) for port in ports]
        bars=ax.bar([j+(i-1)*.25 for j in range(3)],[100*x for x in vals],width=.25,label=policy)
        ax.bar_label(bars,labels=[f'{100*x:.1f}%' for x in vals],padding=3,fontsize=9)
    ax.set_xticks(range(3),['Final velocity','Final latent','PNG pixels']);ax.set_ylabel('Color-response error (%)');ax.margins(y=.25);ax.legend();ax.grid(axis='y',alpha=.2)
    ax=fig.add_subplot(grid[2,3:]);ax.axis('off');s=prov['schedulerResult']
    ax.text(0,1,'\n'.join([f"{s['elapsed_s']:.2f} s · RSS {s['peak_rss_mb']:.1f} MB · VRAM {s['peak_vram_mb']:.0f} MB",
        '2 new denoisers · 4 paid VAE decodes · 4 new images',
        'Four earlier denoisers remain charged prefix inputs.',
        'HN and NH each use three calls; HH uses two.',
        'Shared own state; no historical-state refresh.',
        'Policy interaction, not isolated denoiser curvature.',
        'Known-prefix evidence; zero-neural goal remains open.']),va='top',fontsize=10,linespacing=1.5)
    fig.suptitle('Can the final reader recover an omitted earlier velocity read?',fontsize=16,weight='bold')
    path=out/'two-clock.png';fig.savefig(path,dpi=125);plt.close(fig)
    save(out/'figure.json',{'schema':'tce-two-clock-figure-v1','figureSHA256':sha(path),'plotterSourceSHA256':sha(__file__),
        'sourceReportSHA256':sha(p/'report.json'),'sourcePNGs':images,'scope':'Four new HH/HN originals beside inherited NN/NH/full originals; dependent development measurements, not semantic certification.'})
    return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
