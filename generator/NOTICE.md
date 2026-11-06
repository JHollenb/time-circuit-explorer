The four files in `kernels/` and `retrieve_weights.py` are unmodified public
Saturn-pub / MDB sources, distributed under Apache-2.0. The complete upstream
license is retained in `UPSTREAM-LICENSE`.

Upstream: https://github.com/JHollenb/saturn-pub

The qualified source bytes come from the immutable payload of campaign
`job-4350531b1dcd`. `kernel-manifest.json` records exact hashes and the source
request. `prepare.py` extracts only parameter bindings/recipes and these source
files; it does not package any pretrained weights or observed activations.

The prediction-only driver, resource measurements, launcher, local server,
and website integration are separate code in this repository.
