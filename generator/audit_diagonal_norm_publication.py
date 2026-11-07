"""Source, custody, historical joins and narrative consistency; no model execution."""
import json
from pathlib import Path
import re
from encoder_atoms import read, save, sha, identity
from publish_diagonal_norm import audit
from publish_image_dual import audit as audit_parent
from projected_complement import fields


def execute():
    from mrun.client.api import Api
    root = Path(__file__).resolve().parent.parent;p = root / 'docs/results/diagonal-norm';e = root / 'docs/results/diagonal-norm-evidence'
    moment = root / 'docs/results/diagonal-norm-moments';me = root / 'docs/results/diagonal-norm-moments-evidence'
    mechanics = audit(p);parent = root / 'docs/results/image-dual';audit_parent(parent)
    report, progress, summary, advice, tests = [read(x) for x in [p / 'report.json', root / 'docs/zero-neural-progress.json',
        e / 'scalar-summary.json', e / 'advisor.json', e / 'tests.json']]
    if (progress['researchCycle'], progress['status'], progress['liveJobs']) != (59, 'active', []):
        raise ValueError('Active goal/completed jobs differ')
    if progress['lastCompletedResearchTurn']['commit'] != 'cdce08fce2276fe9725fbf5e6b2ab0891118af78' or any(progress[n]['classification'] != 'progress' for n in ['thisGoalTurn', 'previousGoalTurn']):
        raise ValueError('Previous committed/publishing progress differs')
    if progress['diagonalNormEvidence']['reportSHA256'] != sha(p / 'report.json') or progress['diagonalNormEvidence']['nextQ'] != report['scores']['nextQ'] or progress['diagonalNormEvidence']['normDiagnostics'] != report['diagnostics']:
        raise ValueError('Progress/primary result differs')
    if read(e / 'mechanics.json') != mechanics or progress['diagonalNormEvidence']['scalarSummarySHA256'] != sha(e / 'scalar-summary.json'):
        raise ValueError('Primary mechanics/scalar snapshot differs')
    request, inputs, targets = [read(p / n) for n in ['request.json', 'inputs.json', 'targets.json']]
    if sha(root / 'docs/plans/source-defined-diagonal-key-norm.md') != request['sourceHashes']['frozen-plan.md'] or sha(p / 'frozen-plan.md') != request['sourceHashes']['frozen-plan.md']:
        raise ValueError('Original executed norm plan changed')
    for n, x in inputs['dualParentRecords'].items():
        if sha(parent / n) != x['sha256']:
            raise ValueError('Historical dual parent differs: ' + n)
    old_targets = read(parent / 'targets.json')
    if any(targets['trajectories'][n] != old_targets[n] for n in ['PP', 'QQ', 'stale']):
        raise ValueError('Original PP/QQ/stale labels differ')
    from image_dual_analysis import NAMES
    parent_actor = read(parent / 'actor.json');f = {**parent_actor['source']['fields'], **parent_actor['next_query']['fields']}
    if targets['trajectories']['retained'] != {n: f[n] for n in NAMES}:
        raise ValueError('Retained-norm historical actor origin differs')
    mr, mi, mrequest, mv = [read(moment / n) for n in ['report.json', 'inputs.json', 'request.json', 'retained-array-verification.json']]
    if progress['diagonalMomentEvidence']['pooled'] != mr['pooled'] or progress['diagonalMomentEvidence']['scalarSummarySHA256'] != sha(me / 'scalar-summary.json'):
        raise ValueError('Post-hoc progress/scalar snapshot differs')
    if identity({k: v for k, v in mrequest.items() if k != 'requestSHA256'}) != mrequest['requestSHA256']:
        raise ValueError('Post-hoc request differs')
    for n in mrequest['ownedSources']:
        if sha(root / 'generator' / n) != mrequest['sourceHashes'][n]:
            raise ValueError('Frozen post-hoc source changed')
    for n, h in read(moment / 'files.json').items():
        if sha(moment / n) != h:
            raise ValueError('Post-hoc publication member changed')
    if mv['verifierSourceSHA256'] != sha(root / 'generator/publish_diagonal_moments.py') or mv['templateSourceSHA256'] != sha(root / 'generator/publish_sparse_text_kv.py'):
        raise ValueError('Post-hoc byte verifier changed')
    for x in [*fields(mi), *fields(mr)]:
        if mv['files'][x['path']]['sha256'] != x['sha256'] or mv['npyPayloadWitnesses'][x['path']]['payloadSHA256'] != x['payloadSHA256']:
            raise ValueError('Post-hoc complete byte/payload witness absent')
    actor, compiler = [read(p / n) for n in ['actor.json', 'compiler.json']]
    expected = {'x': actor['modln']['fields']['x'], 'diagonal': compiler['compile']['fields']['diagonal'],
        'rawQ': read(parent / 'reference.json')['reference']['fields']['rawQ'], 'actorEnergy': actor['norm']['fields']['energy']}
    if any(mi[n] != x for n, x in expected.items()) or mi['parentReportSHA256'] != sha(p / 'report.json'):
        raise ValueError('Post-hoc input origin differs')
    for name in ['diagonal-norm', 'diagonal-norm-moments']:
        job = read(root / 'artifacts' / (name + '-v1') / 'scheduler-job.json');live = Api().json('GET', '/api/jobs/' + job['job_id'])
        if live['state'] != 'succeeded' or live['payload_custody'] != job['payload_custody']:
            raise ValueError('Current terminal scheduler custody differs')
    if advice['mainReportSHA256'] != sha(p / 'report.json') or advice['momentReportSHA256'] != sha(moment / 'report.json') or advice['momentSourceSHA256'] != sha(root / 'generator/diagonal_norm_moments.py'):
        raise ValueError('Advisor actual source/result snapshot differs')
    if advice['reviewedArticleSHA256'] != sha(root / 'docs/diagonal-key-norm.md'):
        raise ValueError('Reviewed article differs')
    if tests['sourceSHA256'] != sha(root / tests['source']) or (tests['passed'], tests['seconds'], tests['exitCode']) != (292, 18.82, 0):
        raise ValueError('Actual primary test source/receipt differs')
    for source, counts in [(e, (24924, 29)), (me, (24, 18))]:
        inst = read(source / 'saturn-instrument.json')
        if (inst['records'], inst['predicates']) != counts or any(inst[n] != 0 for n in ['runExit', 'verifyExit', 'formalReceipts', 'boundXrefAddresses']):
            raise ValueError('Instrument coverage/scope differs')
    plan = root / 'docs' / progress['nextCut']['plan']
    if progress['nextCut']['status'] != 'written-unexecuted' or progress['nextCut']['totalNewMatrixMACs'] != 2705326080 or progress['nextCut']['planSHA256'] != sha(plan) or advice['nextCut']['sha256'] != sha(plan):
        raise ValueError('Unexecuted sketch/source/work differs')
    blog = root / progress['findingsLog']['path'];text = blog.read_text()
    if progress['findingsLog']['lastUpdatedCycle'] != 59 or progress['findingsLog']['sha256'] != sha(blog):
        raise ValueError('Living blog binding differs')
    section = text.split('## Cycle 59:', 1)[1].split('## What would close the original goal', 1)[0]
    links = 0
    for path, contents in [(blog, section), (root / 'docs/diagonal-key-norm.md', None), (plan, None), (e / 'README.md', None)]:
        for link in re.findall(r'\]\(([^)]+)\)', contents if contents is not None else path.read_text()):
            if not link.startswith(('http:', 'https:')):
                target = (path.parent / link.split('#', 1)[0]).resolve()
                if target != (e / 'publication-qa.json').resolve() and not target.exists():
                    raise ValueError('Publication backlink absent: ' + link)
                links += 1
    figure = root / 'docs/results/diagonal-norm-figure/diagonal-norm.png';fr = read(figure.parent / 'figure.json')
    if sha(figure) != sha(root.parent / 'obsidian/images/zero-neural-cycle59-diagonal-norm.png') or sha(figure) != fr['imageSHA256'] or fr['primaryReportSHA256'] != sha(p / 'report.json') or fr['momentsReportSHA256'] != sha(moment / 'report.json') or fr['plotSourceSHA256'] != sha(root / 'generator/plot_diagonal_norm.py'):
        raise ValueError('Figure/data/source/notebook copy differs')
    sm, review = progress['storageMaintenance'], progress['reviews']
    if (sm['lastActualArchiveCycle'], sm['lastAuditCycle'], sm['nextDueByCycle'], review['lastCycle'], review['nextDueByCycle']) != (57, 57, 60, 59, 62):
        raise ValueError('Actual storage/advisor cadence differs')
    if review['latestAdvisorSHA256'] != sha(e / 'advisor.json'):
        raise ValueError('Latest advisor receipt binding differs')
    result = {**mechanics, 'schema': 'tce-diagonal-norm-publication-qa-v1', 'reportSHA256': sha(p / 'report.json'),
        'momentReportSHA256': sha(moment / 'report.json'), 'progressSHA256': sha(root / 'docs/zero-neural-progress.json'),
        'blogSHA256': sha(blog), 'figureSHA256': sha(figure), 'advisorSHA256': sha(e / 'advisor.json'),
        'nextPlanSHA256': sha(plan), 'auditSourceSHA256': sha(__file__), 'testsPassed': 292, 'localLinksChecked': links,
        'postHocByteFiles': len(mv['files']), 'postHocNPYWitnesses': len(mv['npyPayloadWitnesses']),
        'postHocInputJoins': list(expected), 'currentTerminalJobs': ['job-188150561700', 'job-3f486b18e216'],
        'nextStorageCycle': 60, 'nextAdvisorCycle': 62,
        'mutableAdvisorPostObservations': [{**x, 'currentSHA256': sha((e / x['path']).resolve())} for x in advice['newestPostSnapshots']],
        'scope': 'Named publication mechanics and narrative consistency. No effect-size gate, semantic/image/generalization/speedup or arbitrary-prompt zero-neural certification.'}
    save(e / 'publication-qa.json', result)
    return {k: result[k] for k in ['ownedSourcePins', 'typedActs', 'explicitOutputABIs', 'ownInputNormHeadStateQueryJoins', 'observations', 'postHocByteFiles', 'postHocInputJoins', 'localLinksChecked', 'nextStorageCycle', 'nextAdvisorCycle']}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
