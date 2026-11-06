"""Plot original native/rank/zero PNGs and signed response errors; no model work."""
import argparse
from pathlib import Path
from encoder_atoms import np, read, save, sha


def execute(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    root=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    report=read(root/'report.json');cost=read(root/'provenance.json')['schedulerResult']
    fig,ax=plt.subplots(2,4,figsize=(20,10),constrained_layout=True)
    for row,case in enumerate(['A','B']):
        for col,prefix in enumerate(['full','rank','zero']):
            ax[row,col].imshow(Image.open(root/(prefix+'-'+case+'.png')));ax[row,col].axis('off')
            ax[row,col].set_title(case+': cube '+('left' if case=='A' else 'right')+' of sphere\n'+{'full':'Previously sealed native reference','rank':'Previously sealed frozen-reader candidate','zero':'New zero-first-reader candidate'}[prefix],fontsize=10.5)
    ports=['H1','E','velocity0','latent4','pngRGB'];x=np.arange(len(ports));colors={'zero':'#b7a0c8','rank':'#559786'}
    pairs={(v['branch'],v['port']):v for v in report['pairedPromptResponse']}
    for branch,offset in [('zero',-.18),('rank',.18)]:
        ax[0,3].bar(x+offset,[100*pairs[branch,p]['relativeRms'] for p in ports],width=.36,label=branch,color=colors[branch])
    ax[0,3].set_xticks(x,ports,rotation=25);ax[0,3].set(title='Signed B−A response error',ylabel='RMS error / native response RMS (%)')
    ax[0,3].legend();ax[0,3].grid(axis='y',alpha=.2)
    zero=pairs['zero','pngRGB'];rank=pairs['rank','pngRGB']
    lines=['PNG response: a large native spatial change',f"Native response RMS: {zero['referenceRms']:.3f} /255",'',
        f"Zero reader error: {zero['rms']:.3f} ({100*zero['relativeRms']:.2f}%)",f"Rank reader error: {rank['rms']:.3f} ({100*rank['relativeRms']:.2f}%)",'',
        f"Entire new assay: {cost['elapsed_s']:.2f} s",f"Sampled peak VRAM: {cost['peak_vram_mb']:.0f} MB",'2 new candidates; no new references/noise draws.',
        'First contextual reader absent in zero arm.', 'Lexical residual and later learned work remain:', '52 encoder blocks + 8 denoisers + 2 VAE decodes.', '',
        'One known pair / one noise. No quality certificate.', 'No matched speedup or zero-neural image claim.']
    ax[1,3].axis('off');ax[1,3].text(.01,.98,'\n'.join(lines),va='top',transform=ax[1,3].transAxes,fontsize=10.2,linespacing=1.4)
    fig.suptitle('Does the contextual reader improve what the remaining model can already do?',fontsize=18,weight='bold')
    path=out/'zero-reader-image.png';fig.savefig(path,dpi=140);plt.close(fig)
    save(out/'figure.json',{'schema':'tce-zero-reader-derived-figure-v1','figureSHA256':sha(path),'plotterSourceSHA256':sha(__file__),
        'sourceReportSHA256':sha(root/'report.json'),'sourcePNGs':{prefix+'-'+case+'.png':sha(root/(prefix+'-'+case+'.png')) for prefix in ['full','rank','zero'] for case in ['A','B']},
        'scope':'Original PNG pixels and published signed-response/cost scalars. No local model evaluation or complete activation hydration.'})
    return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();print(execute(a.source,a.output))
