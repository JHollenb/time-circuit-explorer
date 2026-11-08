"""Publication/bookkeeping audit using sources, bytes and public scalars only."""
import json
from pathlib import Path
import re
from encoder_atoms import read, save, sha
from receiver_row_output_study import OWNED
from publish_receiver_row_output import audit


def execute():
    from mrun.client.api import Api
    root = Path(__file__).resolve().parent.parent
    public = root / 'docs/results/receiver-weighted-row-output'
    evidence = root / 'docs/results/receiver-weighted-row-output-evidence'
    mechanics = audit(public)
    progress = read(root / 'docs/zero-neural-progress.json')
    scheduler = read(public / 'scheduler.json')
    actual = Api().json('GET', '/api/jobs/' + scheduler['job_id'])
    if actual['state'] != 'succeeded' or actual['result']['returncode'] != 0:
        raise ValueError('Actual job did not succeed')
    if actual['payload_custody'] != scheduler['payload_custody']:
        raise ValueError('Actual guarded custody differs')
    custody = actual['payload_custody']
    if not custody['required'] or len({custody[k]['sha256'] for k in ['declared', 'sealed', 'executed']}) != 1:
        raise ValueError('Guarded payload seals missing or unequal')
    if {k: actual['reservation'][k] for k in ['ram_mb', 'vram_mb', 'cpu_threads', 'disk_gb', 'est_wall_s']} != {
            'ram_mb': 4096, 'vram_mb': 3072, 'cpu_threads': 4, 'disk_gb': 3, 'est_wall_s': 90} or actual['timeout_s'] != 300:
        raise ValueError('Actual declared resource exception differs')
    if progress['status'] != 'active' or progress['researchCycle'] != 86 or progress['liveJobs']:
        raise ValueError('Goal/current lease bookkeeping differs')
    if progress['inFlightResearchCycle86']['status'] != 'completed' or not progress['inFlightResearchCycle86']['noLiveJob']:
        raise ValueError('Completed cycle bookkeeping differs')
    cut = progress['nextCut']
    plan = root / 'docs' / cut['plan']
    if cut['status'] != 'written-unexecuted' or not cut['noImplementation'] or not cut['noJobSubmitted'] or sha(plan) != cut['planSHA256']:
        raise ValueError('Next plan/execution boundary differs')
    if cut['fullSurveyAndStorageReviewRequiredBeforeStagingCycle'] != 87:
        raise ValueError('Next scheduled full review condition differs')
    intake = read(evidence / 'successor-boundary-metadata.json')
    for path, digest in intake['metadataSHA256'].items():
        if sha(root / path) != digest:
            raise ValueError('Historical successor metadata differs: ' + path)
    for stream, join in intake['joins'].items():
        if len({join['parentSingle0']['payloadSHA256'], join['matched73DenseSingle0']['payloadSHA256'],
                join['single1RecordIncomingPayloadSHA256']}) != 1:
            raise ValueError('Planned parent/successor metadata join differs: ' + stream)
    if len(intake['single1Parameters']) != 4 or sum(item['bytes'] for item in intake['single1Parameters']) != 245367296:
        raise ValueError('Planned single1 parameter census differs')
    tests = read(evidence / 'tests-final.json')
    if tests['passed'] != 359 or tests['exitCode'] != 0:
        raise ValueError('Actual tiny mechanics tests missing')
    for name in OWNED:
        if sha(root / 'generator' / name) != tests['sourceHashes'][name]:
            raise ValueError('Tested executed source changed: ' + name)
    for name, original in [('frozen-plan.md', 'receiver-weighted-output-sketch.md'),
                           ('implementation-notes.md', 'receiver-weighted-output-sketch-implementation.md')]:
        if sha(public / name) != sha(root / 'docs/plans' / original) or sha(public / name) != sha(root / 'artifacts/receiver-weighted-row-output-v1/payload' / name):
            raise ValueError('Frozen historical contract differs')
    for name in ['advisor-prestage.json', 'advisor-result.json', 'advisor-synthesis.json']:
        for path, digest in read(evidence / name)['reviewedSHA256'].items():
            if sha(root / path) != digest:
                raise ValueError('Reviewed source/result/synthesis changed: ' + path)
    synthesis = read(evidence / 'advisor-synthesis.json')
    if synthesis['decision'] != 'approved-exact-snapshots' or len(synthesis['reviewedSHA256']) != 4:
        raise ValueError('Final exact synthesis review missing')
    initial = read(evidence / 'initial-findings.json')['findings'][0]
    original_tests = read(evidence / 'tests.json')
    final_tests = read(evidence / 'tests-final.json')
    changed = [name for name, digest in original_tests['sourceHashes'].items()
               if final_tests['sourceHashes'][name] != digest]
    if changed != ['receiver_row_output_study.py'] or initial['scientificLawChanged'] or initial['schedulerJobCreated']:
        raise ValueError('Pre-stage queue-note-only revision history differs')
    if initial['observedStudySHA256'] != original_tests['sourceHashes']['receiver_row_output_study.py'] or initial['correctedStudySHA256'] != final_tests['sourceHashes']['receiver_row_output_study.py'] or initial['correctedNoteCharacters'] != 343:
        raise ValueError('Original/final source and note boundary differ')
    blog = root / progress['findingsLog']['path']
    if progress['findingsLog']['lastUpdatedCycle'] != 86 or sha(blog) != progress['findingsLog']['sha256']:
        raise ValueError('Living log identity differs')
    links = []
    for path in [root / 'docs/receiver-weighted-row-output.md', plan, evidence / 'README.md', blog]:
        content = path.read_text()
        if path == blog:
            content = content.split('## Cycle 86:', 1)[1].split('## What would close the original goal', 1)[0]
        for target in re.findall(r'\]\(([^)]+)\)', content):
            if target.startswith(('https:', 'http:', '#')):
                continue
            resolved = (path.parent / target.split('#', 1)[0]).resolve()
            if not resolved.exists():
                raise ValueError('Broken local link: ' + target)
            links.append(str(resolved))
    figdir = root / 'docs/results/receiver-weighted-row-output-figure'
    figure = read(figdir / 'figure.json')
    if sha(figdir / 'receiver-row-output.png') != figure['imageSHA256'] or sha(root.parent / 'obsidian/blog/images/receiver-row-output-cycle86.png') != figure['imageSHA256']:
        raise ValueError('Figure/blob copy differs')
    if sha(root / 'generator/plot_receiver_row_output.py') != figure['plotSourceSHA256'] or sha(public / 'report.json') != figure['sourceReportSHA256'] or read(evidence / 'render-qa.json')['imageSHA256'] != figure['imageSHA256']:
        raise ValueError('Inspected figure/source differs')
    for command in ['run', 'verify']:
        if read(evidence / ('saturn-' + command + '.json'))['returncode'] != 0:
            raise ValueError('Applied Saturn instrument return code differs')
        value = json.loads((evidence / ('saturn-' + command + '-stdout.txt')).read_text())
        if Path(value['target']) != root / 'artifacts/receiver-weighted-row-output-v1/result' or value['instrument_status'] != 'target-streams-only':
            raise ValueError('Instrument actual raw target differs')
    if read(evidence / 'saturn-instrument.json')['records'] != 1536:
        raise ValueError('Instrument every-row coverage differs')
    storage = read(evidence / 'storage.json')
    if (storage['newArrayCount'], storage['newArrayFileBytes'], storage['newArchivedFiles'], storage['newDeletedFiles']) != (19, 133386624, 0, 0):
        raise ValueError('New retention differs')
    if progress['reviews']['lastFullNewestPostSurveyCycle'] != 84 or progress['reviews']['nextDueByCycle'] != 87:
        raise ValueError('Full survey history differs')
    if progress['storageMaintenance']['lastReviewCycle'] != 84 or progress['storageMaintenance']['lastActualArchiveCycle'] != 78 or progress['storageMaintenance']['nextDueByCycle'] != 87:
        raise ValueError('Full storage review/archive history differs')
    survey = read(root / 'docs/results/research-review-cycle84/advisor-survey.json')
    surveyed_blog = survey['reviewedWorkingSnapshots']['obsidian/blog/' + blog.name]
    if surveyed_blog != progress['reviews']['cycle83Snapshots'][progress['findingsLog']['path']] or surveyed_blog == sha(blog):
        raise ValueError('Historical survey/final blog distinction lost')
    result = {'schema': 'tce-receiver-weighted-row-output-publication-QA-v1', 'jobID': actual['job_id'],
              'authoritativeState': actual['state'], 'mechanics': mechanics, 'tests': 359,
              'verifiedLocalLinks': len(links), 'reviewedSynthesisFiles': 4,
              'goalStatus': 'active', 'nextCut': 'written-unexecuted',
              'surveyedHistoricalBlogSHA256': surveyed_blog, 'blogSHA256': sha(blog),
              'figureSHA256': figure['imageSHA256'], 'auditorSourceSHA256': sha(__file__),
              'scope': 'Actual source/custody/admission/byte/scalar/links/figure/review/storage/goal bookkeeping. No scientific-array numerical replay, quality/rank/trajectory/image/generalization/zero-neural certification.'}
    save(evidence / 'publication-qa.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
