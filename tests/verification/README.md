# Equivalence checks against the original implementations

Users of PyTextAD do not need to run these. They exist for maintainers: re-run them
whenever a detector's code changes, to prove it still reproduces the original code.

| Script | Original code it is compared with | What must match |
|---|---|---|
| date_ref_side.py + date_check.py | bit-ml/date (run in Python 3.8 + transformers 3.0.2) | weight layout, tokenisation and sliding windows, window scores, training loss, corrupted positions, 3 optimiser steps |
| cvdd_check.py | lukasruff/CVDD-PyTorch (CVDDTrainer + CVDDNet) | context vectors, attention weights, scores, alpha schedule after 1/10/15/23 epochs; variable-length documents with zero pad vectors |
| rsrae_ref_side.py + rsrae_check.py | dmzou/RSRAE (TensorFlow) | frozen BN statistics, forward pass, 20 training epochs (L21 and MSE) |
| fate_check.py | arav1ndajay/fate (src/main.py) | scores, full training loop with dropout, balanced batch sampler |

Each check was also run against deliberately broken copies of PyTextAD (wrong clipping,
wrong weight decay, wrong optimiser groups, no absolute value in FATE's top-k) and
failed as it should.

```bash
cd tests/verification
./setup_reference.sh     # once: clones the four repositories, builds a Python 3.8 env, tiny test models
./run_all.sh             # prints PASS/FAIL per check, exit code 1 on any failure
```

Set PYTEXTAD_REF to put the reference material elsewhere (default: ./reference).
Needs git, curl, uv and network access to GitHub and PyPI. CPU only: setup a few minutes, checks about 1 minute.
