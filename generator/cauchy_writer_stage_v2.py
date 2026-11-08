"""Metadata-only staging adapter for the canonical explicit analysis family.

The first stage stopped in one_lease_scheduling_contract before request sealing
or submission. Keep that incomplete directory and all executed source intact.
This bounded adapter adds the required family declaration to the canonical
builder, then delegates all custody and payload staging to the reviewed source.
It does not alter any worker, numerical function, experiment arm or resource
reservation, and it does not bypass scheduler validation.
"""
import argparse
import json
import cauchy_writer_study as study
from mdb import job_support


def stage(root):
    original=job_support.one_lease_scheduling_contract
    owned=study.OWNED
    def classified_contract(**kwargs):
        extra=dict(kwargs.get('extra') or {})
        if 'task_family' in extra and extra['task_family']!='analysis':raise ValueError('Conflicting task family')
        extra['task_family']='analysis'
        return original(**{**kwargs,'extra':extra})
    try:
        # Only the stage-time metadata builder is adapted. The shipped original
        # worker/launcher retain their exact reviewed, tested source bytes.
        job_support.one_lease_scheduling_contract=classified_contract
        study.OWNED=owned+['cauchy_writer_stage_v2.py']
        return study.stage(root)
    finally:
        job_support.one_lease_scheduling_contract=original
        study.OWNED=owned


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args();print(json.dumps(stage(args.root)))
