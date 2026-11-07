"""Submit frozen source-reader science after the pre-lease note-length rejection."""
import json
from pathlib import Path
import sys
from encoder_atoms import read,save,sha
from flux_clock_study import verify
from source_reader_study import WHEELS


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    note=('Source-state four-query reader, step2 joint1→2. Native Pquery/full hybrid QK→CPU FP64 allocation/deferredV→own MLP/nextQ. '
          '11Acts,13original tensors302MB,15.496BMAC. No archived operands/banks/duals/Pnorm before seal; no fullV/reference/suffix/image. '
          'Measured62:1868MB RSS/626MBdevice/41s; selected-map exception4096RAM/1024VRAM/4CPU/3GBdisk,120s expected600cap,no retries. '
          'Native prefix/context/maps paid. Frozen v1science; v2launch only shortens rejected note.')
    if len(note)>500:raise ValueError('Scheduler note exceeds 500 characters')
    save(root/'prelease-note-rejection.json',{'stage':'before lease','jobCreated':False,'HTTPStatus':422,
        'reason':'note must be at most 500 characters','frozenStudySHA256':sha(Path(__file__).with_name('source_reader_study.py')),
        'replacementLauncherSHA256':sha(__file__),'change':'Only scheduler note shortened; original scientific payload/request unchanged.'})
    cmd=command()[:-5]
    for n in WHEELS:cmd+=['--with','./'+n]
    cmd+=['python','-u','source_reader_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-source-reader-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={**{k:r[k] for k in ['schema','requestSHA256','model','backend','device','dtype','batch_size','seq_len','height','width','scheduling_contract','arithmeticSubplans']},
                     'submissionLauncherSHA256':sha(__file__)},
        ram_mb=4096,vram_mb=1024,cpu_threads=4,disk_gb=3,est_wall_s=120,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256'],'submissionLauncherSHA256':sha(__file__)}
    save(root/'job.json',value);return value


if __name__=='__main__':print(json.dumps(submit(sys.argv[1])))
