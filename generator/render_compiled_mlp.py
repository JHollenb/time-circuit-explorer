"""Evaluate the published input-line write and its residual consumer."""
import argparse
import json
from pathlib import Path

from mlp_polynomial import read, sha, serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[1] / 'docs/results/compiled-mlp-write')
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1 or not 1 <= args.repeats <= 1000:
        parser.error('alpha must be in [0,1]; repeats must be in [1,1000]')
    bundle, output = args.bundle.resolve(), args.output.resolve()
    inventory = read(bundle / 'files.json')
    inputs = {}
    for key, name in [('coefficients', 'coefficients.npy'), ('consumer', 'consumer.npz')]:
        if sha(bundle / name) != inventory[name]:
            raise ValueError('Published serving input changed')
        inputs[key] = {'path': str(bundle / name), 'sha256': inventory[name]}
    output.mkdir(parents=True, exist_ok=False)
    spec = {**inputs, 'mode': 'primary', 'sourceRoots': [], 'alphas': [args.alpha],
            'repeats': args.repeats,
            'scope': 'Published supplied32-row MLP input line and fixed-parent residual update. No raw-prompt, arbitrary-state, full-block or image-generation capability.'}
    path = output / 'serve.json'
    path.write_text(json.dumps(spec, indent=2, allow_nan=False) + '\n')
    serve(path, output)
    receipt = read(output / 'prediction-seal.json')
    print(json.dumps({'output': str(output), 'scope': spec['scope'],
                      'arithmeticSeconds': receipt['rows'][0]['medianArithmeticSeconds'],
                      'nativeReferenceForThisDose': 'archived CPU BF16 midpoint' if args.alpha == .5 else 'archived endpoint' if args.alpha in [0, 1] else 'unmeasured'}))


if __name__ == '__main__':
    main()
