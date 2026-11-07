"""Original coarse-global outputs, source-fixed error split and paid work."""
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
        for col,(policy,title) in enumerate([('full','Historical full encoding'),('HH','Hold: detailed carry only'),('coarse','Coarse global innovation'),('HN','Full same-state native read')]):
            name=policy+'-'+c+'.png';originals[name]=sha(p/name);ax=fig.add_subplot(grid[row,col]);ax.imshow(Image.open(p/name));ax.axis('off');ax.set_title(title+'\n'+('orange pitcher / teal can' if c=='00' else 'teal pitcher / orange can'),fontsize=10)
    ax=fig.add_subplot(grid[2,:2]);names=['coarseProducerError','omittedFineInnovation','finiteEmissionError'];domains=['00','10','color']
    for i,name in enumerate(names):
        vals=[next(x for x in r['errorDecomposition'] if x['domain']==d and x.get('kind') is None)['termEnergy'][name] for d in domains]
        bars=ax.bar([j+(i-1)*.25 for j in range(3)],vals,width=.25,label={'coarseProducerError':'Coarse producer','omittedFineInnovation':'Omitted fine innovation','finiteEmissionError':'Finite emission'}[name]);ax.bar_label(bars,fmt='%.0f',padding=3,fontsize=8)
    ax.set_xticks(range(3),['00','10','Dependent color']);ax.set_ylabel('Velocity-error term energy');ax.margins(y=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,fontsize=8,frameon=False);ax.grid(axis='y',alpha=.2)
    ax=fig.add_subplot(grid[2,2:]);ax.axis('off');s=prov['schedulerResult']
    vals=[next(x['relativeRms'] for x in r['pairedColorResponse'] if x['reference']=='HN' and x['port']==p0) for p0 in ['velocity3','latent4','pngRGB']]
    ax.text(0,1,'\n'.join([f"{s['elapsed_s']:.2f} s · RSS {s['peak_rss_mb']:.1f} MB · VRAM {s['peak_vram_mb']:.0f} MB",'2 coarse denoisers · 2 paid VAE decodes · 2 new images',
        'Final-call affine MACs: 50.12% of full native read.',f"Signed color error vs HN: {vals[0]*100:.1f}% velocity / {vals[1]*100:.1f}% latent / {vals[2]*100:.1f}% PNG",'Four earlier full denoisers and encoding remain paid.',
        'Fixed pool/physical centers/image-key mass; no fit.', 'Requested log4 -> recorded BF16 kernel bias 1.3828125.', 'Known development pair; strict zero-neural goal open.']),va='top',fontsize=10,linespacing=1.5)
    fig.suptitle('Can a smaller global read produce the missing velocity innovation?',fontsize=16,weight='bold');path=out/'coarse-global.png';fig.savefig(path,dpi=130,bbox_inches='tight');plt.close(fig)
    save(out/'figure.json',{'schema':'tce-coarse-global-figure-v1','figureSHA256':sha(path),'plotterSourceSHA256':sha(__file__),'sourceReportSHA256':sha(p/'report.json'),'sourcePNGs':originals})
    return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
