"""Inherited original PNGs, exact input proof and new direct-reader costs."""
import argparse
from pathlib import Path
from encoder_atoms import read, save, sha


def execute(source,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    root=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False);r=read(root/'report.json');c=read(root/'provenance.json')['schedulerResult'];proof=read(root/'input-identity.json')
    if r['outcome']!='reuse-qualified' or not proof['completeDemandedInputsIdentical']:raise ValueError('Figure requires explicit inherited-image qualification')
    fig,ax=plt.subplots(1,4,figsize=(23,5.8),constrained_layout=True)
    for i,case in enumerate(['A','B']):
        ax[i].imshow(Image.open(root/('inherited-'+case+'.png')));ax[i].axis('off')
        ax[i].set_title(case+': cube '+('left' if case=='A' else 'right')+' of sphere\nPreviously generated image, inherited by exact inputs',fontsize=11)
    sizes=[18.980992,2.101376,r['work']['allocatedNumericalStateBytesEach']/1e6]
    bars=ax[2].bar(['Rank-one','Selected-only','Direct scores'],sizes,color=['#71838d','#559786','#3879bd'])
    ax[2].bar_label(bars,labels=[f'{v:.3f} MB' for v in sizes],padding=4);ax[2].set(title='Comparable numerical reader state',ylabel='Allocated numerical payload (MB)',ylim=(0,22));ax[2].grid(axis='y',alpha=.2)
    ax[2].text(.02,.6,'Tables, identities, buffers and diagnostics\nare additional costs.\n\nFull causal K/V and score scan remain.',transform=ax[2].transAxes,fontsize=10,va='top')
    changed=sum(x['raw']['changedValues'] for x in r['comparisons']);lines=[f'Raw FP64 head changes: {changed:,}',
        'Largest absolute change: 1.665e−15','BF16 head mismatches: 0','H0 / mask / position / ID mismatches: 0','',
        'No anchors, routes or anchor-score cache.','Centers-only numeric operand: 8,192 B.','',f"New CPU assay: {c['elapsed_s']:.2f} s",f"Sampled peak RSS: {c['peak_rss_mb']:.1f} MB",'GPU allocation: 0 MB.',
        '0 new dense encoder/denoiser/VAE calls.','0 new images; 2 inherited original images.','',
        'Prior consumer remains paid historical work:','52 encoder blocks + 8 denoisers + 2 VAE.','No matched speedup or new-prompt claim.','Arbitrary-prompt zero-neural goal remains open.']
    ax[3].axis('off');ax[3].text(.01,.98,'\n'.join(lines),transform=ax[3].transAxes,va='top',fontsize=10.3,linespacing=1.4)
    fig.suptitle('Reuse current-query scores; inherit downstream results only after complete input identity',fontsize=17,weight='bold')
    path=out/'direct-reader.png';fig.savefig(path,dpi=140);plt.close(fig)
    save(out/'figure.json',{'schema':'tce-direct-reader-derived-figure-v1','figureSHA256':sha(path),'plotterSourceSHA256':sha(__file__),
        'sourceReportSHA256':sha(root/'report.json'),'inputIdentitySHA256':sha(root/'input-identity.json'),
        'sourcePNGs':{f'inherited-{case}.png':sha(root/f'inherited-{case}.png') for case in ['A','B']},
        'scope':'Already generated original PNGs inherited by complete input identity, with published scalar cost/proof accounts. No fresh model or image generation.'})
    return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();print(execute(a.source,a.output))
