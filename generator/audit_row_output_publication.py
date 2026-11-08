"""Publication checks against retained source, byte, scalar and advisory evidence."""
import json
from pathlib import Path
import re
from encoder_atoms import read, save, sha
from publish_row_output import audit


def execute():
    from mrun.client.api import Api
    root = Path(__file__).resolve().parent.parent
    public = root / 'docs/results/current-row-output'
    evidence = root / 'docs/results/current-row-output-evidence'
    mechanics = audit(public)
    progress = read(root / 'docs/zero-neural-progress.json')
    report = read(public / 'report.json')
    scheduler = read(public / 'scheduler.json')
    actual = Api().json('GET', '/api/jobs/' + scheduler['job_id'])
    if actual['state'] != 'succeeded' or actual['result']['returncode'] != 0 or actual['payload_custody'] != scheduler['payload_custody']:
        raise ValueError('Authoritative completed job or custody differs')
    if progress['status'] != 'active' or progress['researchCycle'] != 83 or progress['liveJobs']:
        raise ValueError('Goal scope/live state differs')
    if progress['inFlightResearchCycle83']['lastObservedState'] != 'succeeded' or not progress['inFlightResearchCycle83']['noLiveJob']:
        raise ValueError('Completed job bookkeeping differs')
    cut = progress['nextCut']
    if cut['status'] != 'written-unexecuted' or not cut['noJobSubmitted'] or not cut['noImplementation']:
        raise ValueError('Next diagnostic execution state differs')
    plan = root / 'docs' / cut['plan']
    if sha(plan) != cut['planSHA256'] or cut['matrixMACs'] != 1207959552 or cut['newLearnedMapApplications'] != 0:
        raise ValueError('Written diagnostic law/cost identity differs')
    if report['newImages'] != 0 or report['newContinuations'] != 0:
        raise ValueError('Image/continuation scope differs')
    tests = read(evidence / 'tests.json')
    if tests['passed'] != 353 or tests['exitCode'] != 0:
        raise ValueError('Actual mechanics suite missing')
    for name, digest in tests['sourceHashes'].items():
        if sha(root / 'generator' / name) != digest:
            raise ValueError('Tested source changed: ' + name)
    payload = root / 'artifacts/current-row-output-v1/payload'
    for name in ['frozen-plan.md', 'implementation-notes.md']:
        if sha(payload / name) != sha(public / name):
            raise ValueError('Executed input copy changed: ' + name)
    if sha(root / 'docs/plans/current-row-output-projection.md') != sha(public / 'frozen-plan.md'):
        raise ValueError('Original frozen plan differs')
    if sha(root / 'docs/plans/current-row-output-implementation.md') != sha(public / 'implementation-notes.md'):
        raise ValueError('Frozen implementation contract differs')
    result_review = read(evidence / 'advisor-result.json')
    for name, digest in result_review['reviewedSHA256'].items():
        if sha(root / name) != digest:
            raise ValueError('Reviewed scientific evidence changed: ' + name)
    advisor = read(evidence / 'advisor-synthesis.json')
    if advisor['decision'] != 'approved-exact-snapshots' or len(advisor['reviewedSHA256']) != 4:
        raise ValueError('Final exact synthesis review missing')
    for name, digest in advisor['reviewedSHA256'].items():
        if sha(root / name) != digest:
            raise ValueError('Reviewed synthesis changed: ' + name)
    blog = root / progress['findingsLog']['path']
    if sha(blog) != progress['findingsLog']['sha256'] or progress['findingsLog']['lastUpdatedCycle'] != 83:
        raise ValueError('Living log identity differs')
    links = []
    for path in [root / 'docs/current-row-output.md', plan, evidence / 'README.md', blog]:
        content = path.read_text()
        if path == blog:
            content = content.split('## Cycle 83:', 1)[1].split('## What would close the original goal', 1)[0]
        for target in re.findall(r'\]\(([^)]+)\)', content):
            if target.startswith(('http:', 'https:', '#')):
                continue
            file = target.split('#', 1)[0]
            if not (path.parent / file).resolve().exists():
                raise ValueError('Broken publication link: ' + target)
            links.append(str((path.parent / file).resolve()))
    figure = root / 'docs/results/current-row-output-figure'
    fig = read(figure / 'figure.json')
    if sha(figure / 'row-output.png') != fig['imageSHA256'] or sha(root.parent / 'obsidian/blog/images/current-row-output-cycle83.png') != fig['imageSHA256'] or sha(root / 'generator/plot_row_output.py') != fig['plotSourceSHA256']:
        raise ValueError('Inspected figure/copy/source differs')
    if read(evidence / 'render-qa.json')['imageSHA256'] != fig['imageSHA256']:
        raise ValueError('Figure inspection binding differs')
    instrument = read(evidence / 'saturn-instrument.json')
    if instrument['rawResultRunReturnCode'] != 0 or instrument['rawResultVerifyReturnCode'] != 0 or instrument['records'] != 1536:
        raise ValueError('Applied Saturn stream scope differs')
    for name in ['saturn-run.json', 'saturn-verify.json']:
        if read(evidence / name)['returncode'] != 0:
            raise ValueError('Actual instrument exit differs')
    for name in ['saturn-run-stdout.txt', 'saturn-verify-stdout.txt']:
        result = json.loads((evidence / name).read_text())
        if Path(result['target']) != root / 'artifacts/current-row-output-v1/result':
            raise ValueError('Instrument did not consume actual raw result')
    execution = read(public / 'execution.json')
    old = read(root / 'docs/results/current-state-gate-up/execution.json')
    old_join = next(p for p in old['products'] if p['name'] == 'outFull')['leftPayloadSHA256']
    byte_join = read(public / 'retained-array-verification.json')['actualBF16JoinWitness']['payloadSHA256']
    if old_join != execution['actualBF16Join']['payloadSHA256'] or old_join != byte_join:
        raise ValueError('Actual BF16 input differs from parent full affine')
    scalar = read(evidence / 'scalar-localization.json')
    if scalar['improvedGlobalRowsVs82Against73'] != [27, 1524] or [p['row'] for p in scalar['rowsLosingToNoUpdateAgainst73']] != [33]:
        raise ValueError('Retained scalar countertrends differ')
    storage = read(evidence / 'storage.json')
    if storage['newArrayCount'] != 15 or storage['newArrayFileBytes'] != 111759232 or storage['newArchivedFiles'] != 0 or storage['newDeletedFiles'] != 0:
        raise ValueError('New retention scope differs')
    if progress['storageMaintenance']['lastActualArchiveCycle'] != 78 or progress['storageMaintenance']['lastReviewCycle'] != 81 or progress['storageMaintenance']['nextDueByCycle'] != 84:
        raise ValueError('Historical/scheduled storage scope differs')
    if progress['reviews']['lastFullNewestPostSurveyCycle'] != 81 or progress['reviews']['nextDueByCycle'] != 84:
        raise ValueError('Periodic full survey scope differs')
    result = {'schema': 'tce-current-row-output-publication-QA-v1', 'jobID': actual['job_id'], 'authoritativeState': actual['state'], 'mechanics': mechanics,
        'tests': 353, 'verifiedLocalLinks': len(links), 'reviewedSynthesisFiles': 4, 'goalStatus': 'active', 'nextCut': 'written-unexecuted',
        'actualJoinEquals82Input': True, 'figureSHA256': fig['imageSHA256'], 'blogSHA256': sha(blog), 'auditorSourceSHA256': sha(__file__),
        'scope': 'Actual source/core publisher/test/custody/typed/work/scalar/bookkeeping/links/figure/applied-instrument/survey/storage checks. No scientific-array numerical replay, model/image quality, runtime/trajectory/generalization or zero-neural certification.'}
    save(evidence / 'publication-qa.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
