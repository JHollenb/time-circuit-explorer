"""Publication links, immutable-source pins and current living-log identity."""
import ast
from pathlib import Path
import re
from encoder_atoms import read,save,sha
from publish_carrier_crossing import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/carrier-crossing';out=root/'docs/results/carrier-crossing-evidence';blog=root.parent/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    docs=[root/'README.md',root/'generator/README.md',root/'docs/carrier-receiver-crossing.md',root/'docs/plans/carrier-receiver-crossing.md',root/'docs/plans/carrier-text-write-crossing.md',out/'README.md',blog]
    broken=[];links=0;metadata=0
    for doc in docs:
        content=doc.read_text()
        for target in re.findall(r'\]\(([^)]+)\)',content):
            target=target.split('#')[0].strip('<>')
            if not target or '://' in target or target.startswith(('mailto:','data:')):continue
            links+=1
            if not (doc.parent/target).exists():broken.append({'source':str(doc),'target':target})
    header=blog.read_text().split('---',2)[1]
    for line in header.splitlines():
        if line.startswith('  - ../../'):
            metadata+=1
            if not (blog.parent/line[4:]).exists():broken.append({'source':str(blog),'target':line[4:]})
    if broken:raise ValueError(broken)
    progress=read(root/'docs/zero-neural-progress.json');mechanics=audit(p)
    if progress['researchCycle']!=46 or progress['status']!='active' or progress['findingsLog']['sha256']!=sha(blog):raise ValueError('Living-log/progress identity differs')
    figure=root/'docs/results/carrier-crossing-figure/carrier-receiver-crossing.png';copy=root.parent/'obsidian/images/tce-carrier-receiver-20261106/carrier-receiver-crossing.png'
    if sha(copy)!=sha(figure) or read(figure.parent/'figure.json')['plotterSourceSHA256']!=sha(root/'generator/plot_carrier_crossing.py'):raise ValueError('Figure identity differs')
    advisor=read(out/'advisor.json');drift=[]
    for item in advisor['blogSnapshots']:
        file=(out/item['path']).resolve()
        if file!=blog and sha(file)!=item['sha256']:drift.append({'path':item['path'],'reviewedSHA256':item['sha256'],'currentSHA256':sha(file)})
    for n,h in advisor['reviewedSources'].items():
        source=p/n if (p/n).exists() else root/'generator'/n
        if sha(source)!=h:raise ValueError('Advisor execution source differs')
    for file in (root/'generator').glob('*carrier_crossing*.py'):ast.parse(file.read_text(),filename=str(file))
    value={'schema':'tce-carrier-crossing-publication-qa-v1','markdownFiles':len(docs),'localLinks':links,'metadataLinks':metadata,'brokenLinks':broken,'blogSHA256':sha(blog),'figureSHA256':sha(figure),'notebookFigureByteIdentical':True,
        'executedSourcesUnchanged':mechanics['ownedSourcePins'],'researchCycle':46,'goalStatus':'active-unsolved','nextCut':'written-unexecuted-two-native-calls-first-text-write-clamp','nextStorageReviewCycle':48,'advisorPostDrift':drift,
        'scope':'Local links, blog/progress/figure/source/seal/custody identities and Python syntax. Historical advisor blog snapshots retained; this living post intentionally updated after review. No scientific quality or generator certification.'}
    save(out/'publication-qa.json',value);return value


if __name__=='__main__':print(execute())
