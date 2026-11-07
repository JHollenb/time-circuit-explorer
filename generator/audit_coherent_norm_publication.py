"""Audit published source, bytes, typed joins and narrative; no model execution."""
import gzip
import hashlib
import json
from pathlib import Path
import re

from encoder_atoms import identity, read, save, sha
from image_dual_analysis import NAMES
from projected_complement import fields
from publish_coherent_norm import audit
from publish_diagonal_norm import audit as audit_diagonal
from publish_image_dual import audit as audit_dual


def require(condition, message):
    if not condition:
        raise ValueError(message)


def execute():
    from mrun.client.api import Api

    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/coherent-norm'
    e = root / 'docs/results/coherent-norm-evidence'
    paired = root / 'docs/results/paired-sketch-diagnostic'
    pe = root / 'docs/results/paired-sketch-evidence'
    dual = root / 'docs/results/image-dual'
    diagonal = root / 'docs/results/diagonal-norm'
    mechanics = audit(p)
    audit_dual(dual)
    audit_diagonal(diagonal)
    require(read(e / 'mechanics.json') == mechanics, 'Published mechanics changed')

    report, request, inputs, targets, actor, compiler = [read(p / n) for n in
        ['report.json', 'request.json', 'inputs.json', 'targets.json', 'actor.json', 'compiler.json']]
    progress = read(root / 'docs/zero-neural-progress.json')
    summary = read(e / 'scalar-summary.json')
    advice = read(e / 'advisor.json')
    require((progress['researchCycle'], progress['status'], progress['liveJobs']) == (60, 'active', []),
            'Goal cycle/status/live jobs differ')
    require(progress['lastCompletedResearchTurn']['commit'] == '73d9f706b368a695d19c98b4017715d2242bdb2f'
            and progress['lastCompletedResearchTurn']['cycle'] == 59
            and all(progress[n]['classification'] == 'progress' for n in ['thisGoalTurn', 'previousGoalTurn']),
            'Previous committed/publishing progress differs')
    primary = progress['coherentNormEvidence']
    require(primary['reportSHA256'] == sha(p / 'report.json')
            and primary['nextQ'] == report['scores']['nextQ']
            and primary['normDiagnostics'] == report['diagnostics']
            and primary['scalarSummarySHA256'] == sha(e / 'scalar-summary.json')
            and summary['reportSHA256'] == sha(p / 'report.json'), 'Main progress/scalars differ')
    require(sha(root / 'docs/plans/coherent-output-key-norm.md') == request['sourceHashes']['frozen-plan.md']
            == sha(p / 'frozen-plan.md'), 'Executed frozen plan changed')
    for key, parent in [('dualParentRecords', dual), ('diagonalParentRecords', diagonal)]:
        for n, descriptor in inputs[key].items():
            require(sha(parent / n) == descriptor['sha256'], 'Inherited record changed: ' + n)
    old_targets = read(diagonal / 'targets.json')['trajectories']
    require(all(targets['trajectories'][n] == old_targets[n] for n in ['PP', 'QQ', 'stale', 'retained']),
            'Historical trajectories differ')
    old_actor = read(diagonal / 'actor.json')
    old_fields = {**old_actor['source']['fields'], **old_actor['next_query']['fields']}
    require(targets['trajectories']['diagonal'] == {n: old_fields[n] for n in NAMES},
            'Diagonal control actual origin differs')

    pr, pi, pq, pv = [read(paired / n) for n in
        ['report.json', 'inputs.json', 'request.json', 'retained-array-verification.json']]
    require(identity({k: v for k, v in pq.items() if k != 'requestSHA256'}) == pq['requestSHA256']
            == pr['requestSHA256'], 'Paired request identity differs')
    for n in pq['ownedSources']:
        require(sha(root / 'generator' / n) == pq['sourceHashes'][n], 'Paired executed source changed: ' + n)
    for n, digest in read(paired / 'files.json').items():
        require(sha(paired / n) == digest, 'Paired publication member changed: ' + n)
    require(pv['verifierSourceSHA256'] == sha(root / 'generator/publish_paired_sketch.py')
            and pv['templateSourceSHA256'] == sha(root / 'generator/publish_sparse_text_kv.py'),
            'Paired byte verifier changed')
    for descriptor in [*fields(pi), *fields(pr)]:
        path = descriptor['path']
        require(pv['files'][path]['sha256'] == descriptor['sha256']
                and pv['npyPayloadWitnesses'][path]['payloadSHA256'] == descriptor['payloadSHA256']
                and list(pv['npyPayloadWitnesses'][path]['header']['shape']) == descriptor['shape'],
                'Paired full byte/payload/shape witness absent')
    dual_compiler, dual_reference = [read(dual / n) for n in ['compiler.json', 'reference.json']]
    expected = {'rawP': dual_compiler['compile']['fields']['rawP'],
        'pNorm': dual_compiler['compile']['fields']['pDenominator'],
        'rawQ': dual_reference['reference']['fields']['rawQ'],
        'qNorm': dual_reference['reference']['fields']['qDenominator'],
        'signs': compiler['compile']['fields']['signs'],
        'actualQProjection': actor['norm']['fields']['projection']}
    require(all(pi[n] == descriptor for n, descriptor in expected.items())
            and pi['parentReportSHA256'] == sha(p / 'report.json')
            and pi['sameInput'] == report['matchedControlPayloads']['actualModLNInput'],
            'Paired saved-input origin differs')
    provenance = read(paired / 'provenance.json')
    for stream in provenance['losslessStreams']:
        compressed = paired / stream['published']
        raw = gzip.decompress(compressed.read_bytes())
        require(sha(compressed) == stream['publishedSHA256']
                and hashlib.sha256(raw).hexdigest() == stream['sourceSHA256'], 'Lossless paired rows differ')
        rows = [json.loads(line) for line in raw.splitlines()]
        addresses = [(x['row'], x['head']) for x in rows]
        require(len(addresses) == 24576 and len(set(addresses)) == 24576
                and set(addresses) == {(row, head) for row in range(1024) for head in range(24)},
                'Paired unique address census differs')
    paired_progress = progress['pairedSketchEvidence']
    require(paired_progress['pooled'] == pr['pooled']
            and paired_progress['reportSHA256'] == sha(paired / 'report.json')
            and paired_progress['scalarSummarySHA256'] == sha(pe / 'scalar-summary.json')
            and read(pe / 'scalar-summary.json')['reportSHA256'] == sha(paired / 'report.json'),
            'Paired progress/scalars differ')

    terminal_jobs = []
    for name in ['coherent-norm-v1', 'paired-sketch-diagnostic-v1']:
        job = read(root / 'artifacts' / name / 'scheduler-job.json')
        live = Api().json('GET', '/api/jobs/' + job['job_id'])
        require(live['state'] == 'succeeded' and live['payload_custody'] == job['payload_custody'],
                'Current terminal scheduler custody differs')
        terminal_jobs.append(job['job_id'])
    require(advice['mainReportSHA256'] == sha(p / 'report.json')
            and advice['pairedReportSHA256'] == sha(paired / 'report.json')
            and advice['pairedSourceSHA256'] == sha(root / 'generator/paired_sketch_diagnostic.py')
            and advice['reviewedArticleSHA256'] == sha(root / 'docs/coherent-key-norm.md'),
            'Reviewed actual source/results/article differ')
    for n, digest in advice['reviewedSources'].items():
        require(sha(root / 'generator' / n) == digest == request['sourceHashes'][n],
                'Reviewed/current/executed source differs: ' + n)
    for phase, digest in advice['phaseSealSHA256'].items():
        require(sha(p / (phase + '-seal.json')) == digest, 'Reviewed phase seal differs')
    tests = read(e / 'tests.json')
    require(tests['sourceSHA256'] == sha(root / tests['source'])
            and (tests['passed'], tests['seconds'], tests['exitCode']) == (294, 17.14, 0),
            'Actual test receipt/source differs')
    for source, counts in [(e, (24924, 38)), (pe, (24576, 20))]:
        instrument = read(source / 'saturn-instrument.json')
        require((instrument['records'], instrument['predicates']) == counts
                and all(instrument[n] == 0 for n in ['runExit', 'verifyExit', 'formalReceipts', 'boundXrefAddresses']),
                'Executed instrument coverage/scope differs')

    plan = root / 'docs' / progress['nextCut']['plan']
    require(progress['nextCut']['status'] == advice['nextCut']['status'] == 'written-unexecuted'
            and progress['nextCut']['planSHA256'] == advice['nextCut']['sha256'] == sha(plan),
            'Unexecuted spatial plan binding differs')
    blog = root / progress['findingsLog']['path']
    require(progress['findingsLog']['lastUpdatedCycle'] == 60
            and progress['findingsLog']['sha256'] == sha(blog), 'Living blog binding differs')
    section = blog.read_text().split('## Cycle 60:', 1)[1].split('## What would close the original goal', 1)[0]
    links = 0
    for path, contents in [(blog, section), (root / 'docs/coherent-key-norm.md', None),
                           (plan, None), (e / 'README.md', None)]:
        for link in re.findall(r'\]\(([^)]+)\)', contents if contents is not None else path.read_text()):
            if not link.startswith(('http:', 'https:')):
                target = (path.parent / link.split('#', 1)[0]).resolve()
                require(target == (e / 'publication-qa.json').resolve() or target.exists(),
                        'Publication backlink absent: ' + link)
                links += 1
    figures = {}
    for name, plotter in [('coherent-norm', 'plot_coherent_norm.py'), ('paired-sketch', 'plot_paired_sketch.py')]:
        figure = root / 'docs/results' / (name + '-figure') / (name + '.png')
        fr = read(figure.parent / 'figure.json')
        report_path = p if name == 'coherent-norm' else paired
        require(sha(figure) == fr['imageSHA256']
                == sha(root.parent / 'obsidian/images' / ('zero-neural-cycle60-' + name + '.png'))
                and fr['plotSourceSHA256'] == sha(root / 'generator' / plotter)
                and fr['reportSHA256'] == sha(report_path / 'report.json'), 'Figure/source/data/notebook copy differs')
        figures[name] = sha(figure)

    cleanup_root = root / 'docs/results/storage-cleanup-cycle60'
    storage_root = root / 'docs/results/storage-audit-cycle60'
    for source in [cleanup_root, storage_root]:
        publication = read(source / 'publication.json')
        require(publication['archiverSourceSHA256'] == sha(root / 'generator/storage_transports.py'),
                'Executed archiver source changed')
        for n, digest in publication['files'].items():
            require(sha(source / n) == digest == sha(root / publication['source'] / n),
                    'Actual cleanup/audit publication differs')
    cleanup, cleanup_plan = [read(cleanup_root / n) for n in ['result.json', 'plan.json']]
    require(identity({k: v for k, v in cleanup_plan.items() if k != 'planSHA256'})
            == cleanup_plan['planSHA256'] == cleanup['planSHA256'], 'Actual cleanup plan identity differs')
    require((cleanup['files'], cleanup['archivedBytes']) == (4, 1508098618), 'Cleanup census differs')
    for row in cleanup['rows']:
        original = Path(row['path'])
        require(not original.exists() and Path(row['extractedRoot']).is_dir()
                and sha(row['custodyPath']) == row['custodySHA256']
                and read(str(original) + '.cold.json') == row
                and row['remoteVerification']['verified']
                and row['remoteVerification']['sha256'] == row['sha256'],
                'Actual transport restoration/custody joins differ')
    storage = read(storage_root / 'result.json')
    require((len(storage['pointerChecks']), storage['candidateFiles'], storage['candidateLogicalBytes'])
            == (64, 29, 9269736433), 'Post-cleanup metadata census differs')
    sm, review = progress['storageMaintenance'], progress['reviews']
    require((sm['lastActualArchiveCycle'], sm['lastAuditCycle'], sm['nextDueByCycle'],
             review['lastCycle'], review['nextDueByCycle'], review['latestActualResultCycle'])
            == (60, 60, 63, 59, 62, 60), 'Storage/full-post-review/result-review cadence differs')
    require(review['latestAdvisorSHA256'] == sha(e / 'advisor.json')
            and sm['archivedBytes'] == cleanup['archivedBytes'], 'Progress cleanup/advisor binding differs')

    result = {**mechanics, 'schema': 'tce-coherent-norm-publication-qa-v1',
        'reportSHA256': sha(p / 'report.json'), 'pairedReportSHA256': sha(paired / 'report.json'),
        'progressSHA256': sha(root / 'docs/zero-neural-progress.json'), 'blogSHA256': sha(blog),
        'figures': figures, 'advisorSHA256': sha(e / 'advisor.json'), 'nextPlanSHA256': sha(plan),
        'auditSourceSHA256': sha(__file__), 'testsPassed': tests['passed'], 'localLinksChecked': links,
        'pairedByteFiles': len(pv['files']), 'pairedNPYWitnesses': len(pv['npyPayloadWitnesses']),
        'pairedInputJoins': list(expected), 'pairedObservations': 24576, 'currentTerminalJobs': terminal_jobs,
        'cleanupFiles': cleanup['files'], 'cleanupBytes': cleanup['archivedBytes'],
        'nextStorageCycle': 63, 'lastFullNewestPostReviewCycle': 59, 'nextAdvisorCycle': 62,
        'mutableAdvisorPostObservations': [{**x, 'currentSHA256': sha((e / x['path']).resolve())}
            for x in advice['periodicNewestPostReview']['unchangedSnapshots']],
        'scope': 'Named publication mechanics, custody and narrative consistency. No effect-size gate or semantic/image/generalization/speedup/arbitrary-prompt zero-neural certification.'}
    save(e / 'publication-qa.json', result)
    return {k: result[k] for k in ['ownedSourcePins', 'typedActs', 'explicitOutputABIs',
        'ownInputNormHeadStateQueryJoins', 'observations', 'pairedInputJoins', 'pairedObservations',
        'localLinksChecked', 'cleanupBytes', 'nextStorageCycle', 'nextAdvisorCycle']}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
