"""Plot published scalars and original PNGs; no scientific array recomputation."""
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
        for col,(policy,title) in enumerate([('full','Historical full encoding'),('HH','Hold previous velocity'),('reader','One global read + writer'),('HN','Full same-state native read')]):
            name=policy+'-'+c+'.png';originals[name]=sha(p/name);ax=fig.add_subplot(grid[row,col]);ax.imshow(Image.open(p/name));ax.axis('off');ax.set_title(title+'\n'+('orange pitcher / teal can' if c=='00' else 'teal pitcher / orange can'),fontsize=10)
    ax=fig.add_subplot(grid[2,:2])
    for i,(method,label) in enumerate([('hold','Hold'),('candidate','Reader + writer')]):
        vals=[next(x['rms'] for x in r['accounts'] if x['id']==c and x['port']=='pngRGB' and x['method']==method) for c in ['00','10']]
        bars=ax.bar([j+(i-.5)*.3 for j in range(2)],vals,width=.3,label=label);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set_xticks(range(2),['00','10']);ax.set_ylabel('PNG RMS error vs same-state native read');ax.margins(y=.2);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    ax=fig.add_subplot(grid[2,2:]);ax.axis('off');s=prov['schedulerResult']
    vals=[next(x['relativeRms'] for x in r['pairedResponses'] if x['population']=='main' and x['method']=='candidate' and x['port']==port) for port in ['velocity3','latent4','pngRGB']]
    ax.text(0,1,'\n'.join([f"{s['elapsed_s']:.2f} s · RSS {s['peak_rss_mb']:.0f} MB · VRAM {s['peak_vram_mb']:.0f} MB",'46 paid first-joint reads · one closed-form fit · 30 writer products',
        'Served read + writer: 1.092% of full-call affine MACs.',f"Signed color error: {vals[0]*100:.1f}% velocity / {vals[1]*100:.1f}% latent / {vals[2]*100:.1f}% PNG",'18 discovery / 18 feedback / 8 old evaluation fields.',
        'Main held-v1 history differs from canonical discovery v2.', 'Two new original VAE decodes; earlier prefixes/encoding paid.', 'Known development pair; strict zero-neural goal open.']),va='top',fontsize=10,linespacing=1.5)
    fig.suptitle('One live global attention read feeds a shared velocity writer',fontsize=16,weight='bold');path=out/'source-native-reader.png';fig.savefig(path,dpi=130,bbox_inches='tight');plt.close(fig)
    save(out/'figure.json',{'schema':'tce-source-native-figure-v1','figureSHA256':sha(path),'plotterSourceSHA256':sha(__file__),'sourceReportSHA256':sha(p/'report.json'),'sourcePNGs':originals})
    return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(execute(a.source,a.destination))
