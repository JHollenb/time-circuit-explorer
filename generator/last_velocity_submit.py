"""Automatic-sizing submission adapter for the immutable v1 scientific payload.

The original launcher was rejected locally before creating a job because it
combined cpu_threads/est_wall_s with automatic RAM. Omit declared-resource
arguments entirely; scheduler probe/history policy supplies the reservation.
"""
import argparse
from pathlib import Path
from encoder_atoms import read, save
from flux_clock_study import verify


def submit(root):
    from run import command
    from mrun.client.submit import launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():
        raise ValueError('Already submitted; inspect the existing job')
    cmd=command()[:-5]+['python','-u','last_velocity_study.py','execute','.']
    i=cmd.index('uv');cmd[i:i]=['OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=launch(cmd,experiment='tce-last-velocity-v1',task='vae-decode',payload=root/'payload',pin='beast',
        needs={'cuda':True},config={k:r[k] for k in ['schema','requestSHA256','specimen','component','backend','device','dtype','batch_size','seq_len','height','width','task_family']},
        timeout_s=300,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='VAE-only, automatic RAM/VRAM/CPU reservation:4serial B1/512 decodes of fixed own-prefix secant vs hold. No new encoder/denoiser calls or weights. All4images seal before targets;2postseal scheduler-only exact replays. Six historical prefix denoisers charged. Native CPU4 matches scheduler probe default; verify admission CPU4 before running.90sexpected300scap,noretries. Host-local prefix/VAE on Beast.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256'],'submissionAdapter':'last_velocity_submit.py'}
    save(root/'job.json',value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args()
    print(__import__('json').dumps(submit(a.run)))
