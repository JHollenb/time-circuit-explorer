"""Original four new images and four historical targets; measured MLP errors."""
import argparse
from pathlib import Path
from encoder_atoms import read,save,sha


def execute(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=Path(source);out=Path(destination);out.mkdir(parents=True,exist_ok=False);r=read(p/'report.json');prov=read(p/'provenance.json')
    fig=plt.figure(figsize=(16,12),constrained_layout=True);grid=fig.add_gridspec(3,4,height_ratios=[1,1,.55]);images={}
    lanes=[('full','Historical full attention'),('top4','Historical top4 reader'),('constant','New constant MLP'),('tangent','New tangent MLP')]
    for row,c in enumerate(['00','10']):
        for col,(lane,label) in enumerate(lanes):
            name=lane+'-'+c+'.png';ax=fig.add_subplot(grid[row,col]);ax.imshow(Image.open(p/name));ax.axis('off');images[name]=sha(p/name)
            colors='orange pitcher / teal can' if c=='00' else 'teal pitcher / orange can'
            ax.set_title(label+'\n'+colors,fontsize=10)
    ax=fig.add_subplot(grid[2,:2]);policies=['constant','tangent'];rows=[next(x for x in r['pairedColorResponse'] if x['port']=='pngRGB' and x['reference']==ref and x['policy']==policy) for policy in policies for ref in ['top4','full']]
    values=[100*x['relativeRms'] if x['relativeRms'] is not None else None for x in rows]
    bars=ax.bar(['Constant\nvs top4','Constant\nvs full','Tangent\nvs top4','Tangent\nvs full'],[v or 0 for v in values],color=['#bb9460']*2+['#4b8d99']*2)
    ax.bar_label(bars,labels=['undefined' if x is None else f'{x:.1f}%' for x in values],padding=3);ax.set(ylabel='Signed color-response error (%)');ax.margins(y=.2);ax.grid(axis='y',alpha=.2)
    ax=fig.add_subplot(grid[2,2:]);ax.axis('off');s=prov['schedulerResult'];lines=[f"Primary: {s['elapsed_s']:.2f} s · peak RSS {s['peak_rss_mb']:.1f} MB · VRAM {s['peak_vram_mb']:.0f} MB",
        '4 new own images · 16 original denoisers · 4 VAE decodes.', 'Native MLP on 37 active rows; 475 padding rows use fixed law.',
        'Tangent: 26.21 MB J; 32.37B nominal MAC saved per prompt.', 'Native b + J(x − x₀), followed by the original BF16 boundary.',
        'Compiled learned map; no measured speedup or zero-neural claim.', 'Two known prompts / one seed; original images shown.']
    ax.text(0,.98,'\n'.join(lines),va='top',fontsize=10,linespacing=1.5)
    fig.suptitle('Does a source-anchored local MLP law preserve the image future?',fontsize=16,weight='bold');path=out/'padding-mlp.png';fig.savefig(path,dpi=140);plt.close(fig)
    save(out/'figure.json',{'schema':'tce-padding-mlp-figure-v1','figureSHA256':sha(path),'plotterSourceSHA256':sha(__file__),'sourceReportSHA256':sha(p/'report.json'),'sourcePNGs':images,
        'scope':'Four new candidate PNGs, four historical full/top4 targets and measured color-response errors. No generated illustration or semantic certification.'});return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
