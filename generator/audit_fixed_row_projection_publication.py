"""Final source, custody, scalar and synthesis checks; no scientific array loads."""
import json
from pathlib import Path
import re
from encoder_atoms import read, save, sha
from fixed_row_projection_study import OWNED
from publish_fixed_row_projection import audit


def execute():
    from mrun.client.api import Api
    root = Path(__file__).resolve().parent.parent
    public = root/'docs/results/fixed-row-projection'; evidence = root/'docs/results/fixed-row-projection-evidence'
    mechanics = audit(public); progress = read(root/'docs/zero-neural-progress.json')
    scheduler = read(public/'scheduler.json'); actual = Api().json('GET', '/api/jobs/'+scheduler['job_id'])
    if actual['state'] != 'succeeded' or actual['result']['returncode'] != 0 or actual['payload_custody'] != scheduler['payload_custody']:
        raise ValueError('Completed job or guarded custody differs')
    amendment = read(evidence/'admission-correction.json')
    if actual['config']['admissionCorrection'] != amendment or amendment != read(root/'artifacts/fixed-row-projection-v1/admission-correction.json'):
        raise ValueError('Effective submission/resource amendment differs')
    if sha(root/'generator/fixed_row_projection_submit_v2.py') != amendment['launcherSHA256'] or sha(public/'resource-basis.json') != amendment['frozenResourceBasisSHA256']:
        raise ValueError('Separately bound launcher/resource source differs')
    if {k: actual['reservation'][k] for k in ['ram_mb', 'vram_mb', 'cpu_threads', 'disk_gb', 'est_wall_s']} != {'ram_mb': 2048, 'vram_mb': 0, 'cpu_threads': 4, 'disk_gb': 2, 'est_wall_s': 60} or actual['timeout_s'] != 180:
        raise ValueError('Actual admission limits differ')
    refused = read(evidence/'submission-refusal.json')
    if refused['schedulerJobCreated'] or refused['scientificExecutionOccurred'] or not refused['frozenPayloadMembersUnchanged']:
        raise ValueError('Initial frontend refusal scope differs')
    if progress['status'] != 'active' or progress['researchCycle'] != 84 or progress['liveJobs'] or not progress['inFlightResearchCycle84']['noLiveJob']:
        raise ValueError('Active goal/current job bookkeeping differs')
    cut = progress['nextCut']; plan = root/'docs'/cut['plan']
    if cut['status'] != 'written-unexecuted' or not cut['noImplementation'] or not cut['noJobSubmitted'] or sha(plan) != cut['planSHA256']:
        raise ValueError('Next plan identity/execution boundary differs')
    tests = read(evidence/'tests.json')
    if tests['passed'] != 355 or tests['exitCode'] != 0:
        raise ValueError('Actual tiny test suite missing')
    for name in OWNED:
        if sha(root/'generator'/name) != tests['sourceHashes'][name]:
            raise ValueError('Tested science/helper changed: '+name)
    findings = read(evidence/'initial-findings.json')['findings'][0]
    if tests['sourceHashes']['publish_fixed_row_projection.py'] != findings['observedPublisherSHA256'] or sha(root/'generator/publish_fixed_row_projection.py') != findings['correctedPublisherSHA256']:
        raise ValueError('Initial/final publisher correction not preserved')
    for name, original in [('frozen-plan.md', 'fixed-row-projection-diagnostic.md'), ('implementation-notes.md', 'fixed-row-projection-implementation.md')]:
        if sha(public/name) != sha(root/'docs/plans'/original) or sha(public/name) != sha(root/'artifacts/fixed-row-projection-v1/payload'/name):
            raise ValueError('Frozen original contract differs')
    for name in ['advisor-prestage.json', 'advisor-admission.json', 'advisor-result.json', 'advisor-synthesis.json']:
        for path, digest in read(evidence/name)['reviewedSHA256'].items():
            if sha(root/path) != digest:
                raise ValueError('Reviewed source/result/synthesis changed: '+path)
    synthesis = read(evidence/'advisor-synthesis.json')
    if synthesis['decision'] != 'approved-exact-snapshots' or len(synthesis['reviewedSHA256']) != 4:
        raise ValueError('Final synthesis review missing')
    blog = root/progress['findingsLog']['path']
    if progress['findingsLog']['lastUpdatedCycle'] != 84 or sha(blog) != progress['findingsLog']['sha256']:
        raise ValueError('Living log identity differs')
    links = []
    for path in [root/'docs/fixed-row-projection.md', plan, evidence/'README.md', blog]:
        content = path.read_text()
        if path == blog:
            content = content.split('## Cycle 84:', 1)[1].split('## What would close the original goal', 1)[0]
        for target in re.findall(r'\]\(([^)]+)\)', content):
            if target.startswith(('https:', 'http:', '#')):
                continue
            resolved = path.parent/target.split('#', 1)[0]
            if not resolved.resolve().exists():
                raise ValueError('Broken local link: '+target)
            links.append(str(resolved.resolve()))
    figdir = root/'docs/results/fixed-row-projection-figure'; figure = read(figdir/'figure.json')
    if sha(figdir/'fixed-row-projection.png') != figure['imageSHA256'] or sha(root.parent/'obsidian/blog/images/fixed-row-projection-cycle84.png') != figure['imageSHA256'] or sha(root/'generator/plot_fixed_row_projection.py') != figure['plotSourceSHA256'] or read(evidence/'render-qa.json')['imageSHA256'] != figure['imageSHA256']:
        raise ValueError('Inspected scalar figure/source/copy differs')
    for command in ['run', 'verify']:
        if read(evidence/('saturn-'+command+'.json'))['returncode'] != 0:
            raise ValueError('Applied Saturn return code differs')
        output = json.loads((evidence/('saturn-'+command+'-stdout.txt')).read_text())
        if Path(output['target']) != root/'artifacts/fixed-row-projection-v1/result' or output['instrument_status'] != 'target-streams-only':
            raise ValueError('Instrument actual raw target differs')
    if read(evidence/'saturn-instrument.json')['records'] != 1536:
        raise ValueError('Applied instrument row coverage differs')
    storage = read(evidence/'storage.json'); review = read(root/'docs/results/storage-audit-cycle84/publication.json')
    if storage['newArrayCount'] != 23 or storage['newArrayFileBytes'] != 574417792 or storage['newArchivedFiles'] != 0 or storage['newDeletedFiles'] != 0 or review['totalRestorePointers'] != 93 or review['remaining100MBCandidates'] != 0:
        raise ValueError('New retention/scheduled storage review differs')
    for name, digest in review['files'].items():
        if sha(root/'docs/results/storage-audit-cycle84'/name) != digest:
            raise ValueError('Scheduled storage receipt changed')
    if progress['reviews']['lastFullNewestPostSurveyCycle'] != 84 or progress['reviews']['nextDueByCycle'] != 87 or progress['storageMaintenance']['lastReviewCycle'] != 84 or progress['storageMaintenance']['lastActualArchiveCycle'] != 78 or progress['storageMaintenance']['nextDueByCycle'] != 87:
        raise ValueError('Full review/archive history differs')
    survey = read(root/'docs/results/research-review-cycle84/advisor-survey.json')
    previous_blog = survey['reviewedWorkingSnapshots']['obsidian/blog/'+blog.name]
    if previous_blog != progress['reviews']['cycle83Snapshots'][progress['findingsLog']['path']] or previous_blog == sha(blog):
        raise ValueError('Historical surveyed blog/final synthesis distinction lost')
    request_rows = [json.loads(line) for line in (root.parent/'saturn/request.jsonl').read_text().splitlines() if line.strip()]
    if not any(row['handle'] == 'mrun-intent-first-cpu-thread-hints' for row in request_rows):
        raise ValueError('Observed runtime feature request missing')
    result = {'schema': 'tce-fixed-row-projection-publication-QA-v1', 'jobID': actual['job_id'], 'authoritativeState': actual['state'], 'mechanics': mechanics,
        'tests': 355, 'verifiedLocalLinks': len(links), 'reviewedSynthesisFiles': 4, 'goalStatus': 'active', 'nextCut': 'written-unexecuted',
        'admissionAmendmentBound': True, 'surveyedHistoricalBlogSHA256': previous_blog, 'blogSHA256': sha(blog), 'figureSHA256': figure['imageSHA256'], 'auditorSourceSHA256': sha(__file__),
        'scope': 'Actual source/custody/amended admission/byte/scalar/links/figure/survey/storage/goal-bookkeeping checks. No scientific-array numerical replay, image/serving/generalization/ideal-projector/zero-neural certification.'}
    save(evidence/'publication-qa.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
