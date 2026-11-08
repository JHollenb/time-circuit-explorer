"""Admission-only correction; execute the unchanged frozen cycle84 payload."""
import argparse
import json
import math
from pathlib import Path
from encoder_atoms import read, save, sha
from fixed_row_projection_study import verify


def submit(root):
    from mrun.client.submit import launch
    from response_study import command
    root = Path(root).resolve(); r = read(root/'request.json'); verify(root/'payload', r)
    if (root/'job.json').exists():
        raise ValueError('Already submitted; inspect same handle')
    basis = read(root/'payload/resource-basis.json')
    allowance = 512 << 20
    ram_mb = math.ceil((basis['workingArrayBudgetBytes']+allowance)/(512 << 20))*512
    if ram_mb != 2048:
        raise ValueError('Expected current source-derived RAM geometry changed')
    admission = {'schema': 'tce-fixed-row-projection-admission-correction-v1', 'requestSHA256': r['requestSHA256'],
        'frozenResourceBasisSHA256': sha(root/'payload/resource-basis.json'), 'launcherSHA256': sha(__file__),
        'initialClientRefusal': 'ValueError: cpu_threads and est_wall_s require a declared ram_mb', 'initialSchedulerJobCreated': False,
        'workingArrayBudgetBytes': basis['workingArrayBudgetBytes'], 'interpreterBLASAllocatorAllowanceBytes': allowance,
        'ramMB': ram_mb, 'vramMB': 0, 'cpuThreads': 4, 'diskGB': 2, 'expectedSeconds': 60, 'timeoutSeconds': 180,
        'scope': 'Admission-only declared exception after intent-first API refusal. No frozen science/plan/inputs/payload mutation, no scientific restart. Allowance is conservative planning, not measured RSS. Actual elapsed/RSS recorded by scheduler.'}
    save(root/'admission-correction.json', admission)
    cmd = [a.replace('THREADS=1', 'THREADS=4') for a in command()[:-5]]+['python', '-u', 'fixed_row_projection_study.py', 'execute', '.']
    cmd.insert(1, 'PYTHONDONTWRITEBYTECODE=1')
    job = launch(cmd, experiment='tce-fixed-row-projection-v1', task='analysis', payload=root/'payload', pin='beast', needs={'cpu': True},
        config={**{k: r[k] for k in ['schema', 'requestSHA256', 'device', 'backend', 'dtype', 'plannedWork']}, 'admissionCorrection': admission},
        ram_mb=ram_mb, vram_mb=0, disk_gb=2, cpu_threads=4, est_wall_s=60, timeout_s=180, detach=True,
        retry_on_kill=False, preflight=True, guarded=True,
        note='Admission-onlyv2, original frozen84request/payload unchanged. Saved-U diagnostic11sources23outputs;4FP64CPUcontractions1.208BMAC+explicit scalarGrams. No learnedmap/runtime/checkpoint/QR/fit/consumer/image.2048MiBRAM from1.146GB stagedarraybudget+512MiB runtimeallowance rounded512;CPU4,VRAM0,disk2GB,60expected180cap,noretry,guardedpreflight. Initial intent-firstclient refused before job creation. Historical formation paid.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root/'job.json', value); return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); args = parser.parse_args()
    print(json.dumps(submit(args.root)))
