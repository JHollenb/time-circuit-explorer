"""Original retained-query images, historical targets and measured consumer errors."""
import argparse
from pathlib import Path
from encoder_atoms import read,save,sha


def execute(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=Path(source);out=Path(destination);out.mkdir(parents=True,exist_ok=False);r=read(p/'report.json');prov=read(p/'provenance.json');enc=read(p/'encoder-candidate.json');images={}
    fig=plt.figure(figsize=(15,12),constrained_layout=True);grid=fig.add_gridspec(3,3,height_ratios=[1,1,.55])
    for row,c in enumerate(['00','10']):
        for col,(lane,title) in enumerate([('full','Historical full attention'),('top4','Historical top4 reader'),('candidate','New 53-query encoder')]):
            n=lane+'-'+c+'.png';images[n]=sha(p/n);ax=fig.add_subplot(grid[row,col]);ax.imshow(Image.open(p/n));ax.axis('off');ax.set_title(title+'\n'+('orange pitcher / teal can' if c=='00' else 'teal pitcher / orange can'),fontsize=10)
    ax=fig.add_subplot(grid[2,:2]);ports=['E','velocity0','latent4','pngRGB'];values=[next(x['relativeRms'] for x in r['pairedColorResponse'] if x['port']==k and x['reference']=='full' and x['region']=='all') for k in ports]
    bars=ax.bar(['Conditioning E','First velocity','Final latent','PNG pixels'],[100*(x or 0) for x in values],color='#4a8a98');ax.bar_label(bars,labels=['undefined' if x is None else f'{100*x:.1f}%' for x in values],padding=3);ax.set(ylabel='Signed color-response error vs full (%)');ax.margins(y=.2);ax.grid(axis='y',alpha=.2)
    ax=fig.add_subplot(grid[2,2]);ax.axis('off');s=prov['schedulerResult'];count=[x['retainedRows'] for x in enc['selectionRows']]
    lines=[f"{s['elapsed_s']:.2f} s · RSS {s['peak_rss_mb']:.1f} MB",f"VRAM {s['peak_vram_mb']:.0f} MB · retained rows {count}",
        '37 active + 16 fixed padding queries.', '27 complete native nonlinear blocks.', 'Nearest padding copy only at output taps.', '89.65% fewer encoder affine MACs.',
        '2 new images · 8 denoisers · 2 VAE.', 'Original weights and image work remain.', 'No speedup or zero-neural certificate.']
    ax.text(0,1,'\n'.join(lines),va='top',fontsize=10,linespacing=1.5);fig.suptitle('Keep nonlinear query trajectories; reconstruct the conditioning the image model reads',fontsize=15,weight='bold')
    path=out/'retained-queries.png';fig.savefig(path,dpi=140);plt.close(fig);save(out/'figure.json',{'schema':'tce-retained-query-figure-v1','figureSHA256':sha(path),'plotterSourceSHA256':sha(__file__),'sourceReportSHA256':sha(p/'report.json'),'sourcePNGs':images,'scope':'Two original new candidate PNGs beside four historical originals and measured signed-response errors; no synthetic illustration or semantic certificate.'});return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
