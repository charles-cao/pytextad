# PyTextAD: Text Anomaly Detection in Python

[![PyPI](https://img.shields.io/pypi/v/pytextad.svg)](https://pypi.org/project/pytextad/)
[![Documentation](https://readthedocs.org/projects/pytextad/badge/?version=latest)](https://pytextad.readthedocs.io)
[![Tests](https://github.com/charles-cao/pytextad/actions/workflows/tests.yml/badge.svg)](https://github.com/charles-cao/pytextad/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)

**PyTextAD** is a Python library for detecting anomalies in text, at the document
and at the token level. Every detector follows the PyOD interface
(`fit`, `decision_function`, `predict`, `decision_scores_`, `labels_`), and every
re-implemented method is checked numerically against its original code.

## Installation

```bash
pip install pytextad
```

From source:

```bash
git clone https://github.com/charles-cao/pytextad.git
cd pytextad
pip install -e ".[test]"
```

Requires Python >= 3.9, PyTorch >= 1.13 and transformers >= 4.30. Install the
PyTorch build that matches your CUDA version first (https://pytorch.org).

## Quick start

```python
from pytextad import CVDD, DATE, FATE, RSRAE, TokenEmbedder, mean_pool

# frozen token embeddings from any Hugging Face encoder
emb = TokenEmbedder("bert-base-uncased")
H_train, _ = emb.transform(train_texts)
H_test, _ = emb.transform(test_texts)

scores = CVDD().fit(H_train).decision_function(H_test)                       # higher = more anomalous
scores = RSRAE().fit(mean_pool(H_train)).decision_function(mean_pool(H_test))
scores = DATE().fit(train_texts).decision_function(test_texts)               # raw text in, trains its own model
scores = FATE().fit(texts, y).decision_function(test_texts)                  # few-shot: y = 1 for labelled anomalies

word_scores = DATE().fit(train_texts).token_scores([t.split() for t in test_texts])
```

A runnable example on AG News: `python examples/quickstart.py`.

## Implemented methods

| Method | Year | Input | Token scores | Reference |
|---|---|---|---|---|
| CVDD | 2019 | frozen token embeddings | yes | Ruff et al., *Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text*, ACL 2019 |
| RSRAE | 2020 | document vectors | no | Lai et al., *Robust Subspace Recovery Layer for Unsupervised Anomaly Detection*, ICLR 2020 |
| DATE | 2021 | raw text | yes | Manolache et al., *DATE: Detecting Anomalies in Text via Self-Supervision of Transformers*, NAACL 2021 |
| FATE | 2023 | raw text (+ few labelled anomalies) | no | Das et al., *Few-shot Anomaly Detection in Text with Deviation Learning*, ICONIP 2023 |

Default hyperparameters are those of the official code. Each module's docstring
lists where the official code and the paper disagree and which one we follow.

## Faithfulness to the original implementations

`tests/verification/` runs each original implementation next to ours with the same
weights, inputs and random seeds and compares the results (DATE against the original
transformers 3.0.2 code, RSRAE against the original TensorFlow code). All checks
pass; see [tests/verification/README.md](tests/verification/README.md).

## Running the tests

```bash
pytest                               # fast API tests, a few seconds
PYTEXTAD_DEVICE=cuda pytest            # same, on GPU (PowerShell: $env:PYTEXTAD_DEVICE="cuda"; pytest)
```

## License

BSD 2-Clause. Third-party notices for the original implementations are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
