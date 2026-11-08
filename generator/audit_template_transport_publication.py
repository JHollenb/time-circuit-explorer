"""Audit compact publication and scalar bookkeeping without scientific tensor replay."""
import gzip
import json
import math
from pathlib import Path
import re

from encoder_atoms import read, save, sha
from template_transport_study import OWNED
from publish_template_transport import audit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def execute():
    from mrun.client.api import Api
    root = Path(__file__).resolve().parent.parent
    public = root / 'docs/results/template-transport'
    evidence = root / 'docs/results/template-transport-evidence'
    mechanics = audit(public)
    require(mechanics == read(evidence / 'mechanics.json'), 'Recorded mechanics differs')
    progress = read(root / 'docs/zero-neural-progress.json')
    saved = read(public / 'scheduler.json')
    actual = Api().json('GET', '/api/jobs/' + saved['job_id'])
    require(actual['job_id'] == 'job-9ad2d0eb6dd0' and actual['state'] == 'succeeded'
            and actual['result']['returncode'] == 0, 'Authoritative completion differs')
    require(actual['payload_custody'] == saved['payload_custody'], 'Actual custody differs')
    custody = actual['payload_custody']
    require(custody['required'] and len({custody[k]['sha256'] for k in ['declared', 'sealed', 'executed']}) == 1
            and custody['executed']['sha256'] == '20cdb44d73b0dc791bac4a3fd4185f233051473eb834db3c4dbf646dc995af8d'
            and custody['executed']['size_bytes'] == 3231651, 'Guarded payload differs')
    require({k: actual['reservation'][k] for k in ['ram_mb', 'vram_mb', 'cpu_threads', 'disk_gb', 'est_wall_s']}
            == dict(ram_mb=6144, vram_mb=4096, cpu_threads=4, disk_gb=4, est_wall_s=90)
            and actual['timeout_s'] == 360, 'Allocation-based resource exception differs')
    require(tuple(actual['result'][k] for k in ['elapsed_s', 'peak_rss_mb', 'peak_vram_mb'])
            == (58.49, 2222.4, 2238.0), 'Observed resource scope differs')
    require(progress['status'] == 'active' and progress['researchCycle'] == 90 and not progress['liveJobs']
            and progress['inFlightResearchCycle90']['status'] == 'completed'
            and progress['inFlightResearchCycle90']['noLiveJob'], 'Active goal/completed cycle differs')
    cut = progress['nextCut']
    plan = root / 'docs' / cut['plan']
    require(cut['status'] == 'written-unexecuted' and cut['noImplementation'] and cut['noJobSubmitted']
            and cut['kind'] == 'complete-serving-construction-feasibility'
            and sha(plan) == cut['planSHA256'], 'Next construction/execution boundary differs')
    tests = read(evidence / 'tests.json')
    require(tests['passed'] == 372 and tests['exitCode'] == 0 and tests['seconds'] == 19.14
            and len(tests['sourceHashes']) == len(OWNED) == 42, 'Actual test/source census differs')
    for name in OWNED:
        require(sha(root / 'generator' / name) == tests['sourceHashes'][name], 'Tested source changed: ' + name)
    for name, original in [('frozen-plan.md', 'frozen-template-matched-pair.md'),
                           ('implementation-notes.md', 'template-transport-implementation.md')]:
        require(sha(public / name) == sha(root / 'docs/plans' / original)
                == sha(root / 'artifacts/template-transport-v1/payload' / name), 'Executed contract differs')
    pre = read(evidence / 'advisor-prestage.json')
    require(len(pre['sourceHashes']) == 10 and pre['decision'] == 'approved-exact-snapshots', 'Pre-stage review differs')
    for path, digest in {**pre['sourceHashes'], **pre['additionalReviewedSHA256']}.items():
        require(sha(root / path) == digest, 'Exact pre-stage member changed: ' + path)
    for name in ['advisor-result.json', 'advisor-synthesis.json']:
        receipt = read(evidence / name)
        for path, digest in {**receipt['reviewedSHA256'], **receipt.get('frontierReviewedSHA256', {})}.items():
            require(sha(root / path) == digest, 'Reviewed result/narrative changed: ' + path)
    synthesis = read(evidence / 'advisor-synthesis.json')
    require(synthesis['decision'] == 'approved-exact-snapshots' and len(synthesis['reviewedSHA256']) == 4,
            'Four exact final narratives missing')
    blog = root / progress['findingsLog']['path']
    require(progress['findingsLog']['lastUpdatedCycle'] == 90 and sha(blog) == progress['findingsLog']['sha256'],
            'Living log identity differs')

    report = read(public / 'report.json')
    supplement = read(evidence / 'signed-tradeoff.json')
    require(supplement['sourceReportSHA256'] == sha(public / 'report.json')
            and supplement['summarizerSourceSHA256'] == sha(root / 'generator/summarize_template_transport.py'),
            'Scalar supplement identity differs')
    for name, panel in report['panels'].items():
        a = panel['signedAccount']
        for centered, label in [(False, 'full'), (True, 'centered')]:
            energy = 'centeredEnergy' if centered else 'energy'
            error_key = 'centeredErrorEnergy' if centered else 'errorEnergy'
            target_key = 'centeredTargetEnergy' if centered else 'targetEnergy'
            ratio_key = 'centeredErrorOverCenteredTargetRMS' if centered else 'errorOverTargetRMS'
            for dynamic, fixed, left, right in [('dynamicOwnIncrement', 'fixedOwnIncrement', 'E1', 'Ef1'),
                                               ('dynamicResponse', 'fixedResponse', 'Q', 'Qf')]:
                require(panel[dynamic]['energies'][target_key] == panel[fixed]['energies'][target_key],
                        'Paired denominator differs')
                for metric, field in [(dynamic, left), (fixed, right)]:
                    e = panel[metric]['energies']
                    require(e[error_key] == a[field + '_' + energy]
                            and panel[metric][ratio_key] == math.sqrt(e[error_key] / e[target_key]),
                            'Actual increment/response scalar differs')
            scalar = supplement['panels'][name][label]
            parent = a['Ef0_' + energy] - a['E0_' + energy]
            response = a['Qf_' + energy] - a['Q_' + energy]
            own = a['Ef1_' + energy] - a['E1_' + energy]
            dynamic_cross = a['E1_' + energy] - a['E0_' + energy] - a['Q_' + energy]
            fixed_cross = a['Ef1_' + energy] - a['Ef0_' + energy] - a['Qf_' + energy]
            for key, value in [('parentErrorEnergyChange', parent), ('responseErrorEnergyChange', response),
                               ('ownErrorEnergyChange', own), ('dynamicTwiceParentErrorResponseInner', dynamic_cross),
                               ('fixedTwiceParentErrorResponseInner', fixed_cross),
                               ('signedCrossChange', fixed_cross - dynamic_cross)]:
                require(math.isclose(scalar[key], value, rel_tol=1e-14, abs_tol=1e-7), 'Scalar signed account differs')
            require(math.isclose(own, parent + response + fixed_cross - dynamic_cross, abs_tol=1e-7),
                    'Scalar expansion differs')
    joint = read(evidence / 'joint-outcomes.json')
    require(joint['sourceRowsSHA256'] == sha(public / 'observations.jsonl.gz'), 'Joint-row source differs')
    with gzip.open(public / 'observations.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    require(len(rows) == 1536, 'Dependent-row count differs')
    for name in ['all', 'text', 'image']:
        subset = rows if name == 'all' else [r for r in rows if r['stratum'] == name]
        for label, energy in [('full', 'errorEnergy'), ('centered', 'centeredErrorEnergy')]:
            counts = dict.fromkeys(joint['panels'][name][label], 0)
            for row in subset:
                s = row['statistics']
                own = s['fixedOwnIncrement_' + energy] - s['dynamicOwnIncrement_' + energy]
                response = s['fixedResponse_' + energy] - s['dynamicResponse_' + energy]
                key = ('oneOrBothTie' if own == 0 or response == 0 else
                       'bothImprove' if own < 0 and response < 0 else
                       'responseImprovesOwnWorsens' if own > 0 and response < 0 else
                       'responseWorsensOwnImproves' if own < 0 and response > 0 else 'bothWorsen')
                counts[key] += 1
            require(counts == joint['panels'][name][label], 'Joint row classification differs')
    parent = read(evidence / 'parent-association.json')
    require(parent['fileByteIdentical'] and parent['payloadByteIdentical']
            and parent['currentExitFileSHA256'] == parent['historicalExitFileSHA256']
            and parent['currentExitPayloadSHA256'] == parent['historicalExitPayloadSHA256']
            and report['panels']['all']['nonzeroParentAssociationRows'] == 0, 'Measured parent association differs')

    links = []
    for path in [root / 'docs/frozen-template-transport.md', plan, evidence / 'README.md', blog]:
        content = path.read_text()
        if path == blog:
            content = content.split('## Cycle 90:', 1)[1].split('## What would close the original goal', 1)[0]
        for target in re.findall(r'\]\(([^)]+)\)', content):
            if target.startswith(('http:', 'https:', '#')):
                continue
            resolved = (path.parent / target.split('#', 1)[0]).resolve()
            require(resolved.exists() or resolved == evidence / 'publication-qa.json', 'Broken link: ' + target)
            links.append(str(resolved))
    figdir = root / 'docs/results/template-transport-figure'
    figure = read(figdir / 'figure.json')
    require(sha(figdir / 'template-transport.png') == sha(figure['notebookCopy']) == figure['imageSHA256']
            and sha(root / 'generator/plot_template_transport.py') == figure['plotSourceSHA256']
            and sha(public / 'report.json') == figure['sourceReportSHA256']
            and sha(public / 'scheduler.json') == figure['schedulerSHA256']
            and read(evidence / 'render-qa.json')['imageSHA256'] == figure['imageSHA256'], 'Inspected figure differs')
    for action in ['run', 'verify-v2']:
        require(read(evidence / ('saturn-' + action + '.json'))['returncode'] == 0, 'Applied instrument differs')
        value = json.loads((evidence / ('saturn-' + action + '-stdout.txt')).read_text())
        require(Path(value['target']) == root / 'artifacts/template-transport-v1/result'
                and value['instrument_status'] == 'target-streams-only', 'Actual target-stream scope differs')
    instrument = read(evidence / 'saturn-instrument.json')
    require(read(evidence / 'saturn-verify.json')['returncode'] == 2
            and instrument['initialVerifyExitCode'] == 2 and instrument['actualVerifyExitCode'] == 0
            and 'unrecognized arguments' in (evidence / 'saturn-verify-stderr.txt').read_text(),
            'Preserved initial parser failure differs')
    require([instrument[k] for k in ['records', 'fires', 'formalClaims', 'qualifiedAddresses', 'expectedMissingNoFireLedgers']]
            == [1536, 0, 0, 0, 1], 'Instrument row scope differs')
    storage = read(evidence / 'storage.json')
    require([storage[k] for k in ['newArrayCount', 'newArrayFileBytes', 'compactCoreFiles', 'compactCoreBytes', 'newArchivedFiles', 'newDeletedFiles']]
            == [74, 1410856192, 18, 2084765, 0, 0], 'New retention scope differs')
    require(progress['reviews']['lastFullNewestPostSurveyCycle'] == 90
            and progress['reviews']['nextDueByCycle'] == 93
            and progress['storageMaintenance']['lastReviewCycle'] == 90
            and progress['storageMaintenance']['lastActualArchiveCycle'] == 78
            and progress['storageMaintenance']['nextDueByCycle'] == 93, 'Review/archive boundaries differ')
    survey = read(root / 'docs/results/research-review-cycle90/advisor-survey.json')
    historical = survey['reviewedWorkingSnapshots']['obsidian/blog/' + blog.name]
    require(len(survey['reviewedWorkingSnapshots']) == 55
            and historical == '45eec80c904a11e8d4f49f1f15cf8fc02891243d5dfe0125bf36a19c21bc2df9'
            and historical != sha(blog), 'Historical survey/current log distinction differs')
    storage_review = read(root / 'docs/results/storage-audit-cycle90/publication.json')
    require([storage_review[k] for k in ['totalRestorePointers', 'olderTerminalPointers', 'currentByteSnapshotPointers',
                                       'freshFullColdBytesVerified', 'remaining100MBCandidates', 'newArchivedFiles', 'newDeletedFiles']]
            == [93, 84, 9, 3806240603, 0, 0, 0], 'Full storage-review scope differs')
    result = dict(schema='tce-template-transport-publication-QA-v1', jobID=actual['job_id'],
                  authoritativeState=actual['state'], mechanics=mechanics, tests=372,
                  verifiedLocalLinks=len(links), reviewedSynthesisFiles=4, goalStatus='active',
                  nextCut='written-unexecuted', nextCutKind=cut['kind'],
                  fullResearchReviewCycle=90, fullStorageReviewCycle=90, nextPeriodicReviewCycle=93,
                  surveyedHistoricalBlogSHA256=historical, blogSHA256=sha(blog),
                  figureSHA256=figure['imageSHA256'], auditorSourceSHA256=sha(__file__),
                  scope='Actual custody/resources/frozen sources/contracts/intake/byte/scalar/figure/narrative/review/storage/goal bookkeeping. No independent real-array replay, old CUDA layout attestation, original full-attention sensitivity, image/generalization/zero-neural certification.')
    save(evidence / 'publication-qa.json', result)
    require(all(Path(path).exists() for path in links), 'Final linked member absent')
    return result


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
