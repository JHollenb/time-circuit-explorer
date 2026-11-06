"""Nominal projection accounting from the source-bound Qwen configuration."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from encoder_atoms import read, save, sha


def publish(intake, destination):
    intake = Path(intake); out = Path(destination); out.mkdir(parents=True, exist_ok=False)
    config = intake / 'compact/source-metadata/qwen_config.json'
    c = read(config); n = 512; h = c['hidden_size']; m = c['intermediate_size']
    q = c['num_attention_heads'] * c['head_dim']; kv = c['num_key_value_heads'] * c['head_dim']
    formation = n * h * (q + 2 * kv); output = n * q * h; mlp = 3 * n * h * m
    full = formation + output + mlp; remaining = output + mlp
    value = {'schema': 'tce-qwen-cached-qkv-nominal-budget-v1', 'sourceConfigSHA256': sha(config),
        'sourceVerificationSHA256': sha(intake / 'verification/verification.json'),
        'geometry': {'rows': n, 'hidden': h, 'mlpWidth': m, 'qWidth': q, 'kvWidth': kv},
        'nominalMACs': {'QKVFormation': formation, 'outputProjection': output, 'MLPProjections': mlp,
                       'fullLinear': full, 'cachedLinear': remaining},
        'avoidedFractionOfLinearMACs': formation / full,
        'scope': 'One edited block, nominal dense multiply-accumulates only. Not measured hardware operations or speedup. Source-positive construction, SDPA, norms, RoPE, elementwise work and later model stages excluded and still paid where applicable.'}
    save(out / 'budget.json', value)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(9, 3.7), constrained_layout=True)
    for i, (label, f) in enumerate([('Full edited block', formation), ('Transformed QKV', 0)]):
        start = 0
        for amount, title, color in [(f, 'Q/K/V formation', '#2563eb'), (output, 'Output projection', '#0d9488'), (mlp, 'MLP projections', '#d97706')]:
            ax.barh(i, amount / 1e9, left=start / 1e9, height=.48, color=color, label=title if i == 0 else None)
            start += amount
        ax.text(start / 1e9 + .5, i, f'{start / 1e9:.2f} B', va='center', fontsize=10)
    ax.set_yticks([0, 1], ['Full edited block', 'Transformed QKV']); ax.invert_yaxis()
    ax.set(xlabel='Nominal linear MACs per edited branch (billions)', xlim=(0, full / 1e9 + 7))
    fig.suptitle('The MLP dominates the remaining computation', fontweight='bold', fontsize=14)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=3, fontsize=9, frameon=False)
    ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    fig.savefig(out / 'projection-budget.png', dpi=160); plt.close(fig)
    save(out / 'figure.json', {'sourceBudgetSHA256': sha(out / 'budget.json'), 'figureSHA256': sha(out / 'projection-budget.png')})
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('intake'); p.add_argument('destination'); a = p.parse_args()
    print(json.dumps(publish(a.intake, a.destination)))
