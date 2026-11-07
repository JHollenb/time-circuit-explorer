"""Retrospective concentration from published row scalars; no tensor math."""
import gzip
import json
from pathlib import Path
from encoder_atoms import read,save,sha


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/rectifier-correction';rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()]
    total=sum(x['energyImprovementVsQuadratic'] for x in rows);exceptions=[rows[i] for i in [3,35]];gain=sum(x['energyImprovementVsQuadratic'] for x in exceptions)
    value={'schema':'tce-rectifier-concentration-v1','sourceSHA256':sha(p/'observations.jsonl.gz'),'summarizerSHA256':sha(__file__),'totalEnergyGainVsQuadratic':total,'historicalTwoRowsEnergyGainVsQuadratic':gain,'historicalTwoRowsShareOfNetEnergyGain':gain/total,'otherRowsNetEnergyGainVsQuadratic':total-gain,
        'historicalRows':[{k:x[k] for k in ['row','quadraticResidualEnergy','fullResidualEnergy','energyImprovementVsQuadratic','totalEnergy']} for x in exceptions],
        'worseVsZeroRows':[x['row'] for x in rows if x['energyImprovementVsZero']<0],
        'scope':'Published scalar-only retrospective concentration, all rows stay in every primary aggregate. Share of signed net energy gain can exceed100% when other rows networsen; not trimmed accuracy or independent causal component shares.'}
    save(root/'docs/results/rectifier-correction-evidence/concentration.json',value);return value


if __name__=='__main__':print(json.dumps(execute(),indent=2))
