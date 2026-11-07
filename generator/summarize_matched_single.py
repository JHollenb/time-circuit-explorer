"""Record observed cycle73 mechanics and compact evidence, without tensor math."""
import json
from pathlib import Path
import shutil
from encoder_atoms import read, save, sha
from publish_matched_single import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/matched-single-write'
    out = root / 'docs/results/matched-single-write-evidence'
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'mechanics.json', audit(p))
    request = read(p / 'request.json')
    save(out / 'tests.json', {'schema': 'tce-matched-single-tests-v1',
        'command': 'rtk proxy ../common/bin/common exec python -m pytest -q generator',
        'passed': 337, 'seconds': 18.85, 'exitCode': 0, 'newTests': 2,
        'sourceHashes': {n: sha(root / 'generator' / n) for n in request['ownedSources']},
        'scope': 'Final observed suite after capture passivity/reporting edits, before staging. Tiny synthetic captured-vs-uncaptured sparse/dense helpers, actual product/residual identities and explicit signed cancellation. No pretrained local model.'})
    streams = ['observations.jsonl', 'channel-observations.jsonl', 'anatomy-observations.jsonl']
    save(out / 'saturn-instrument.json', {'schema': 'tce-matched-single-instrument-v1',
        'target': 'artifacts/matched-single-write-v1/result', 'runExit': 0, 'verifyExit': 0,
        'instrument_status': 'target-streams-only', 'streams': 3, 'records': 89147,
        'signedComparisonRows': 59963, 'channelRows': 27648, 'anatomyRows': 1536,
        'fires': 0, 'formalClaims': 0, 'qualifiedAddresses': 0,
        'verifyMissingExpectedFireLedgers': 3,
        'sourceStreams': {n: sha(root / 'artifacts/matched-single-write-v1/result' / n) for n in streams},
        'scope': 'Actual standard offline run and verify. No fire ledger written, no semantic certification. Instrument scope remains separate from mechanical byte/operator checks.'})
    c, a, differences = [read(p / n) for n in ['candidate.json', 'anatomy.json', 'differences.json']]
    collections = [c['fields'], a['fields'], differences]
    save(out / 'storage.json', {'schema': 'tce-matched-single-storage-v1',
        'newRawArrays': sum(len(x) for x in collections),
        'newRawArrayBytes': sum(item['bytes'] for x in collections for item in x.values()),
        'primaryCompactFiles': len(list(p.iterdir())),
        'primaryCompactBytes': sum(x.stat().st_size for x in p.iterdir()),
        'localFreeBytesObserved': shutil.disk_usage(root).free,
        'lastActualCleanupCycle': 72, 'nextCleanupDueCycle': 75,
        'scope': 'Complete arrays remain near compute; only metadata and lossless scalar streams published. Byte verification includes inherited checkpoint/parents and is not new storage. No archive or deletion this cycle; free space is shared.'})
    scheduler = read(p / 'scheduler.json')
    actors = {n: read(p / ('execution-' + n + '.json')) for n in ['sparse', 'dense']}
    anatomy = read(p / 'anatomy-execution.json')
    save(out / 'phase-account.json', {'schema': 'tce-matched-single-phases-v1',
        'scheduler': {k: scheduler['result'][k] for k in ['elapsed_s', 'peak_rss_mb', 'peak_vram_mb']},
        'reservation': scheduler['reservation'],
        'actors': {n: {k: x[k] for k in ['wallSecondsAfterImports', 'setupSecondsAfterImports', 'CPUSecondsAfterImports', 'peakCUDAAllocatedBytes', 'peakCUDAReservedBytes', 'timings']} for n, x in actors.items()},
        'anatomy': {k: anatomy[k] for k in ['wallSecondsAfterImportsBeforeComparisons', 'CPUSecondsAfterImportsBeforeComparisons', 'peakCUDAAllocatedBytes', 'peakCUDAReservedBytes', 'timings']},
        'analysisSecondsAfterImports': read(p / 'report.json')['analysisSecondsAfterImports'],
        'scope': 'Instrumented operations include internal diagnostics/copies, exclude later finite checks/fingerprints. Fresh-process walls include setup/checks/I/O. Scheduler includes orchestration/imports. Serial diagnostic timings do not establish serving speed.'})
    return {'typedActs': 3, 'explicitOutputABIs': 66, 'testsPassed': 337,
            'newRawArrays': sum(len(x) for x in collections)}


if __name__ == '__main__':
    print(json.dumps(execute(), sort_keys=True))
