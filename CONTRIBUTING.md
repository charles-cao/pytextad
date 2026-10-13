# Contributing to PyTextAD

Contributions are welcome: bug reports, new detectors, datasets, documentation and fixes.
Please open an issue first for larger changes, so that we can agree on the approach.

## Development setup

```bash
git clone https://github.com/charles-cao/pytextad.git
cd pytextad
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build you need
pip install -e ".[test]"
```

## Running the tests

```bash
pytest                         # all tests, on CPU, with tiny randomly initialised models
pytest --cov                   # with a coverage report
PYTEXTAD_DEVICE=cuda pytest    # the same tests on a GPU
```

The tests build their own tiny BERT and GPT-2 models, so they need no downloads, except
`tests/test_datasets.py::test_real_hub_files`, which is skipped when the Hugging Face Hub
cannot be reached.

Continuous integration runs the tests on Linux, Windows and macOS with Python 3.9 and 3.12,
and once with the oldest dependency versions that `pyproject.toml` accepts (Python 3.9,
torch 1.13, transformers 4.30). PyTextAD's own code is fully covered by the tests; the
third-party code in `pytextad/models/embedding/_vendor` is excluded from the coverage count
and checked instead by the parity tests described below. New code should come with tests
that run every new line.

## Building the documentation

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Project layout

```
pytextad/
  models/
    base.py               BaseTextDetector: fit / decision_function / predict, threshold, labels
    cvdd.py, date.py, fate.py
                          text detectors (trained end to end)
    wrappers.py           DocumentDetector and TokenDetector
    embedding/            embedding detectors: one thin wrapper per detector
      base.py             BaseEmbeddingDetector: seeding, input checks, silencing
      _vendor/            the authors' code, unchanged except where stated in its header
  utils/embeddings.py     TokenEmbedder, SentenceEmbedder, word alignment helpers
  utils/rng.py            seeding helpers
  metrics.py              metrics, score aggregation, evaluate
  datasets.py             TextADDataset, load_dataset, load_local
tests/
  reference/              the original files, as published, used by the parity tests
  test_baselines.py       parity tests: PyTextAD gives exactly the scores of the original code
docs/                     Sphinx documentation; refs.bib holds every reference
scripts/                  how the Hugging Face dataset files were built and uploaded
```

## Adding an embedding detector

PyTextAD keeps every detector faithful to its authors' code. To add one:

1. **Vendor the original code.** Copy the original file, unchanged, to
   `pytextad/models/embedding/_vendor/<name>.py`. At the top, add a short header with the
   source (repository URL and commit), the licence, and either "Included unchanged below this
   header" or the list of changes and why they are needed (for example, a call removed from
   recent PyTorch). Put a second, untouched copy in `tests/reference/<name>_original.py`.
   Check that the licence allows redistribution and add the file to `THIRD_PARTY_NOTICES.md`.
2. **Write the wrapper.** Add `pytextad/models/embedding/<name>.py` with a subclass of
   `BaseEmbeddingDetector` that stores the hyper-parameters (same names and defaults as the
   original) and implements `_build(n_features)`, returning the unfitted original model.
   Override `_fit_model` only if the original is fitted differently. Write a NumPy-style
   docstring: a one-line summary with the citation (`` :cite:`key` ``), the parameters, and
   a `Notes` section saying where the code comes from.
3. **Export it** in `pytextad/models/embedding/__init__.py` and `pytextad/__init__.py`.
4. **Add a parity test.** Add a case to `CASES` in `tests/test_baselines.py`: the reference
   copy and the wrapper, fitted with the same seed on the same data, must give exactly the
   same scores (no tolerance). If the vendored file is the original byte for byte, also add
   it to `test_vendored_code_is_the_original` in `tests/test_embedding_detectors.py`.
5. **Document it.** Add the reference to `docs/refs.bib`, a row to the tables in
   `docs/algorithms.rst` and `README.md`, an entry in `docs/api.rst`, its provenance to
   `docs/faithfulness.rst`, and a line to `CHANGELOG.md`.

**Bugs in the original code.** If the original code has a bug, fix it by default and keep
the published behaviour behind an option, so that published numbers can still be
reproduced. Describe the bug and the option in the docstring and in
`docs/faithfulness.rst`. Examples: `DAGMM(detach_loss=True)`, `NormalizingFlow(legacy_score=True)`.

## Adding a text detector

Text detectors (like CVDD, DATE and FATE) subclass `BaseTextDetector`, implement `fit` and
`decision_function`, set `supports_token = True` and implement `token_scores` if they give
token scores, and finish `fit` with `self._process_decision_scores(training_scores)`.
Check them against the official code with a script in `tests/verification/` and describe
the check in `docs/faithfulness.rst`.

## Adding a dataset

Datasets are JSONL files on the Hugging Face Hub (`Charles-Cao/pytextad`), one document per
line with `tokens`, `labels` and `text`. `scripts/make_hub_files.py` builds them and
`scripts/upload_to_hub.py` uploads them. Register the file in `DATASETS` in
`pytextad/datasets.py` with its revision, SHA-256 and counts, so that `load_dataset` can
verify the download.

## Pull requests

* One topic per pull request, with tests and documentation for the change.
* Keep the common interface: `fit(X)`, `decision_function(X)` (higher means more anomalous),
  `predict(X)`, `decision_scores_`, `threshold_`, `labels_`.
* Run `pytest` and build the documentation before opening the pull request.
* Add a line to the `Unreleased` section of `CHANGELOG.md`.

## Releases

1. Set the version in `pytextad/version.py` and turn `Unreleased` in `CHANGELOG.md` into
   that version.
2. Push, and wait for the tests to pass.
3. Create a GitHub release with the tag `v<version>`; the `publish` workflow uploads it to PyPI.

## Code of conduct

Everyone taking part is expected to follow the [code of conduct](https://github.com/charles-cao/pytextad/blob/HEAD/CODE_OF_CONDUCT.md).
