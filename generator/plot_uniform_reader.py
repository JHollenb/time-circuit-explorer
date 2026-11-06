"""Original uniform/rank/zero/native PNGs and published response/cost accounts."""
import argparse
from pathlib import Path
from encoder_atoms import np,read,save,sha


def execute(source,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    root=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    r=read(root/'report.json');cost=read(root/'provenance.json')['schedulerResult'];pairs={(x['branch'],x['port']):x for x in r['pairedPromptResponse']}
    fig,ax=plt.subplots(2,5,figsize=(25,10),constrained_layout=True)
    for row,case in enumerate(['A','B']):
        for col,branch in enumerate(['full','rank','uniform','zero']):
            ax[row,col].imshow(Image.open(root/(branch+'-'+case+'.png')));ax[row,col].axis('off')
            ax[row,col].set_title(case+': cube '+('left' if case=='A' else 'right')+' of sphere\n'+{'full':'Previously sealed native reference','rank':'Previously sealed rank-one state','uniform':'New uniform causal value state','zero':'Previously sealed zero reader'}[branch],fontsize=11)
    ports=['H1','E','velocity0','latent4','pngRGB'];x=np.arange(len(ports))
    for branch,offset,color in [('rank',-.25,'#559786'),('uniform',0,'#7766aa'),('zero',.25,'#c48e83')]:
        ax[0,4].bar(x+offset,[100*pairs[branch,p]['relativeRms'] for p in ports],width=.25,label=branch,color=color)
    ax[0,4].set_xticks(x,ports,rotation=25);ax[0,4].set(title='Signed B−A response error',ylabel='RMS error / native response RMS (%)');ax[0,4].legend();ax[0,4].grid(axis='y',alpha=.2)
    lines=['PNG response error / native response RMS','']
    for branch in ['rank','uniform','zero']:
        d=pairs[branch,'pngRGB'];lines.append(f"{branch:8} {100*d['relativeRms']:.2f}% (RMS {d['rms']:.3f})")
    w=r['readerWork'][0]
    lines+=['',f"Uniform sums: {w['sumPayloadBytes']:,} B",f"Count + clock: {w['countPayloadBytes']+w['clockPayloadBytes']} B",'Tables, input/output and diagnostics also paid.',
        'No Q/K scores, selection, index or quadratic tail.', '',f"Entire assay: {cost['elapsed_s']:.2f} s",f"Sampled peak VRAM: {cost['peak_vram_mb']:.0f} MB",'52 native encoder blocks + 8 denoisers + 2 VAE.',
        '2 new images; no new references/noise draws.', '', 'Known pair / one noise; no quality certificate.', 'Most learned work remains. No matched speedup.', 'Arbitrary-prompt zero-neural generation open.']
    ax[1,4].axis('off');ax[1,4].text(.01,.98,'\n'.join(lines),va='top',transform=ax[1,4].transAxes,fontsize=10.1,linespacing=1.4)
    fig.suptitle('Test a much smaller live contextual state through the actual image consumer',fontsize=19,weight='bold')
    path=out/'uniform-reader-image.png';fig.savefig(path,dpi=140);plt.close(fig)
    save(out/'figure.json',{'schema':'tce-uniform-reader-derived-figure-v1','figureSHA256':sha(path),'plotterSourceSHA256':sha(__file__),
        'sourceReportSHA256':sha(root/'report.json'),'sourcePNGs':{branch+'-'+case+'.png':sha(root/(branch+'-'+case+'.png')) for branch in ['full','rank','uniform','zero'] for case in ['A','B']},
        'scope':'Original image pixels and published state/response/cost scalars; no model evaluation or complete activation hydration.'})
    return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();print(execute(a.source,a.output))
