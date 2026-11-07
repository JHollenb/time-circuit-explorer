"""Publication links, frozen execution pins, log identity and planned cost."""
import ast
from pathlib import Path
import re
from encoder_atoms import read,save,sha
from publish_text_crossing import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/text-crossing';out=root/'docs/results/text-crossing-evidence';blog=root.parent/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md';target=out/'publication-qa.json'
    docs=[root/'README.md',root/'generator/README.md',root/'docs/first-text-write-crossing.md',root/'docs/plans/carrier-text-write-crossing.md',root/'docs/plans/first-image-consumer.md',out/'README.md',blog];broken=[];links=0;metadata=0;self_links=0
    for doc in docs:
        for name in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            name=name.split('#')[0].strip('<>')
            if not name or '://' in name or name.startswith(('mailto:','data:')):continue
            links+=1;file=(doc.parent/name).resolve()
            if file==target.resolve():self_links+=1
            elif not file.exists():broken.append({'source':str(doc),'target':name})
    for line in blog.read_text().split('---',2)[1].splitlines():
        if line.startswith('  - ../../'):
            metadata+=1
            if not (blog.parent/line[4:]).exists():broken.append({'source':str(blog),'target':line[4:]})
    if broken:raise ValueError(broken)
    progress=read(root/'docs/zero-neural-progress.json');mechanics=audit(p)
    if progress['researchCycle']!=47 or progress['status']!='active' or progress['findingsLog']['sha256']!=sha(blog):raise ValueError('Living-log/progress identity differs')
    figure=root/'docs/results/text-crossing-figure/text-write-crossing.png';copy=root.parent/'obsidian/images/tce-text-crossing-20261106/text-write-crossing.png'
    if sha(copy)!=sha(figure) or read(figure.parent/'figure.json')['plotterSourceSHA256']!=sha(root/'generator/plot_text_crossing.py'):raise ValueError('Figure identity differs')
    advisor=read(out/'advisor.json');drift=[]
    for item in advisor['blogSnapshots']:
        file=(out/item['path']).resolve()
        if file!=blog and sha(file)!=item['sha256']:drift.append({'path':item['path'],'reviewedSHA256':item['sha256'],'currentSHA256':sha(file)})
    for n,h in advisor['reviewedSources'].items():
        source=p/n if (p/n).exists() else root/'generator'/n
        if sha(source)!=h:raise ValueError('Advisor execution source differs')
    selected_bytes=2*(3072*128+3072*3072+18432*3072+3072*9216);macs=2*1024*3072*128+2*1024*3072*3072+4*1024*(3072*18432+9216*3072)
    if selected_bytes!=189530112 or macs!=368025010176 or progress['nextCut']['physicalWork']['nominalAffineMACs']!=macs:raise ValueError('Planned local geometry/cost differs')
    for file in (root/'generator').glob('*text_crossing*.py'):ast.parse(file.read_text(),filename=str(file))
    value={'schema':'tce-text-crossing-publication-qa-v1','markdownFiles':len(docs),'localLinks':links,'metadataLinks':metadata,'brokenLinks':broken,'selfLinksClosedOnSave':self_links,'blogSHA256':sha(blog),'figureSHA256':sha(figure),'notebookFigureByteIdentical':True,
        'executedSourcesUnchanged':mechanics['ownedSourcePins'],'researchCycle':47,'goalStatus':'active-unsolved','nextCut':'written-unexecuted-local-first-image-consumer','plannedSelectedParameterBytes':selected_bytes,'plannedAffineMACs':macs,'nextStorageReviewCycle':48,'advisorPostDrift':drift,
        'scope':'Local links, log/progress/figure/frozen source/seal/custody identities, Python syntax and next-plan integer geometry. This receipt link closes on exclusive save and final existence check. No scientific quality or generator certification.'}
    save(target,value)
    if not target.is_file():raise ValueError('QA self-link did not close')
    return value


if __name__=='__main__':print(execute())
