"""Original candidate/reference PNGs and measured fresh factorial responses."""
import argparse
from pathlib import Path
from encoder_atoms import read,save,sha


def execute(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=Path(source);out=Path(destination);out.mkdir(parents=True,exist_ok=False);r=read(p/'report.json');prov=read(p/'provenance.json');reader=read(p/'reader-seal.json')
    fig=plt.figure(figsize=(16,12),constrained_layout=True);grid=fig.add_gridspec(3,4,height_ratios=[1,1,.55]);cases=['00','10','01','11'];images={}
    for row,lane in enumerate(['candidate','full']):
        for col,c in enumerate(cases):
            name=lane+'-'+c+'.png';ax=fig.add_subplot(grid[row,col]);ax.imshow(Image.open(p/name));ax.axis('off');images[name]=sha(p/name)
            label='Frozen top4 candidate' if lane=='candidate' else 'Fresh full-attention reference'
            colors='orange pitcher / teal can' if c[0]=='0' else 'teal pitcher / orange can';side='left' if c[1]=='0' else 'right'
            ax.set_title(label+'\n'+colors+' · pitcher '+side,fontsize=10)
    ax=fig.add_subplot(grid[2,:2]);kinds=['color-left','color-right','relation-base','relation-swapped','mixed'];rows=[next(x for x in r['factorialResponses'] if x['port']=='pngRGB' and x['kind']==k and x['region']=='all') for k in kinds]
    values=[None if x['relativeRms'] is None else 100*x['relativeRms'] for x in rows]
    bars=ax.bar(['Color\nleft','Color\nright','Relation\nbase','Relation\nswapped','Mixed'],[x or 0 for x in values],color=['#4c8c9a']*4+['#b37a38']);ax.bar_label(bars,labels=['undefined' if x is None else f'{x:.1f}%' for x in values],padding=3,fontsize=9);ax.set(ylabel='Error / native signed-response RMS (%)');ax.margins(y=.2);ax.grid(axis='y',alpha=.2)
    ax=fig.add_subplot(grid[2,2:]);ax.axis('off');sched=prov['schedulerResult'];tokens=[x['activeTokens'] for x in reader['lexicalRows']]
    lines=[f"Fresh seed: {read(p/'request.json')['seed']} · active token rows: {tokens}",f"Measured assay: {sched['elapsed_s']:.2f} s · peak RSS {sched['peak_rss_mb']:.1f} MB · VRAM {sched['peak_vram_mb']:.0f} MB",
        '4 own candidate images sealed before 4 new references.',f"Paid work: {r['work']['nativeEncoderBlocks']} native encoder blocks + {r['work']['nativeDenoiserCalls']} denoisers + {r['work']['nativeVAEDecodes']} VAE decodes.",
        'Full K/V, score scans, learned lexical tables and compiled clocks remain.', 'One dependent prompt square; inspect object/color/relation failures.', 'Arbitrary-prompt zero-neural generation remains unresolved.']
    ax.text(0,.98,'\n'.join(lines),transform=ax.transAxes,va='top',fontsize=10,linespacing=1.5)
    fig.suptitle('Frozen first reader on fresh compositional prompts: candidate and native image futures',fontsize=16,weight='bold');path=out/'fresh-square.png';fig.savefig(path,dpi=140);plt.close(fig)
    save(out/'figure.json',{'schema':'tce-fresh-square-figure-v1','figureSHA256':sha(path),'plotterSourceSHA256':sha(__file__),'sourceReportSHA256':sha(p/'report.json'),'sourcePNGs':images,
        'scope':'Original eight freshly generated candidate/reference PNGs and signed-response measurements. No generated illustration, semantic score or zero-neural certificate.'});return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
