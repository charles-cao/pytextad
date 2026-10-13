# Changelog

## 0.3.4
- `highlight(..., reference=...)`: the colour scale is set by the word scores of normal
  documents (white at their median, full colour at their 99th percentile), so that only
  words that stand out are coloured.
- Loading a model no longer prints the list of checkpoint weights the encoder does not use
  (for example BERT's pre-training heads); a warning is given only if weights are missing.
- Online demo (`space/`) runs on ZeroGPU hardware, the free option for new Spaces, using the
  CPU only.

## 0.3.3
- `highlight`: shows every word coloured by its anomaly score (in Jupyter, or saved as a web
  page), with the true anomalous words underlined.
- Tutorial notebook (`examples/tutorial.ipynb`) with an "Open in Colab" badge.
- `space/`: source of an online demo (Hugging Face Space) that colours the words of a typed
  text by anomaly score.
- `examples/benchmark.py`: `--train_ratio 0` (train and test on the whole dataset; it failed
  before) and `--retokenize`, as in `examples/token_level.py`.

## 0.3.2
- CVDD, DATE and FATE raise a clear "not fitted" error when used before `fit` (they raised
  an `AttributeError`).
- Tests: PyTextAD's own code is fully covered (third-party code in `_vendor` is checked by
  the parity tests instead); coverage report in CI.
- CI on Linux, Windows and macOS, and with the oldest supported dependency versions
  (Python 3.9, torch 1.13, transformers 4.30).
- `CONTRIBUTING.md` (also in the documentation), code of conduct, issue and pull request
  templates.
- How to use detectors from other libraries (PyOD) in the Quick Start and README.
- Up-to-date description of the `evaluate` parameters.
- Installation: how to upgrade and check the installed version; mirror example for `HF_ENDPOINT`.
- Logo in the README and the documentation; favicon; social preview image (`docs/_static`).
- `examples/benchmark.py`: several detectors (PyTextAD's or PyOD's, with parameters such as
  `"SIK(max_samples=8)"`) on several datasets in one run, with cached embeddings, a table per
  dataset and a summary, saved as CSV and Markdown.

## 0.3.1
- Documentation: document-level examples next to the token-level ones (README, home page,
  Quick Start); "Using your own embeddings" section, saying when `DocumentDetector` is needed
  (embedding detectors take document vectors directly); training on normal documents only in
  the home page example.
- CHANGELOG: everything shipped in 0.3.0 listed under 0.3.0.

## 0.3.0
- `TokenDetector` scores every sub-word and combines the scores of a word's sub-words into
  the word's score (`subword_aggregation="max" | "mean"`) when used with sub-word vectors
  (`TokenEmbedder` without `word_pooling`, now used in all examples). The vectors no longer
  depend on how the words are grouped (an annotated span given as one item). Pooling
  sub-word vectors first, as in TokenCore, remains available with `word_pooling="max"`.
- `TokenDetector` accepts the `(embeddings, ids)` pair returned by `TokenEmbedder.transform`;
  `align_labels` keeps the labels of the words that have a vector.
- `TokenEmbedder` warns about truncated words in sub-word mode too.
- Embedding detectors in `pytextad.models.embedding`: `TokenCore` (Cao et al., WWW 2026) and `SIK`
  (Cao et al., Findings of EMNLP 2025), the authors' code unchanged and checked against the
  originals; `ADERH` (Durani et al., NeurIPS 2025, official package) and `TCCM` (Li et al.,
  NeurIPS 2025, official code); `RSRAE` moved there (old import path still works).
- Embedding baselines from the SVEAD / TokenCore benchmark copies: DAGMM, GANomaly, DROCC, GOAD,
  ICL, MCM, SLAD, NormalizingFlow, DTECategorical, DTEInverseGamma, DTEGaussian,
  DTENonParametric, DDPM, DDAE, DRL; each gives exactly the scores of its benchmark copy.
- Corrected by default (old behaviour available): DAGMM's network was never trained
  (`detach_loss=True` reproduces); NormalizingFlow's score depended on the batch
  (`legacy_score=True` reproduces).
- Seeds: `evaluate`, `DocumentDetector`, `TokenDetector` and the embedding detectors seed the
  global random generators (Python, NumPy, PyTorch) before fitting, and score training data
  without disturbing them, so results equal those of a plain seeded benchmark script.
- `DocumentDetector` and `TokenDetector` keep float64 input vectors as float64 (they were
  rounded to float32, which changed scores slightly compared with running the detector directly).
- Documentation: references with citation labels linking to a References page; `CITATION.cff`.
- `pytextad.datasets`: `load_dataset` for six datasets with word-level anomaly labels -
  `sms_spam`, `restaurant_review`, `grammar_correction` (Cao et al., WWW 2026), `hate_speech`,
  `olid`, `restaurant_review2` (CA-PTD experiments) - downloaded from the Hugging Face Hub and
  verified by SHA-256 and document/word counts.
- `TextADDataset`: words, word labels, document labels and original texts; `subset`,
  `normal_indices` / `anomalous_indices`, `stats`, `retokenize` (one word segmentation for
  normal and anomalous documents), `to_jsonl`.
- `load_local` for JSONL, JSON and TokenCore `.npy` files; `TextADDataset.from_lists`.
- `examples/token_level.py`: embedding detectors on a dataset, token and document metrics.

## 0.2.0
- `SentenceEmbedder`: CLS, mean or last-token sentence vectors from any Hugging Face model.
- `TokenEmbedder`: word-level vectors (`word_pooling="max" | "mean" | "first"`), on-disk caching
  (`transform(..., cache=...)`), `cache_dir` for local model folders.
- Lists of words are now tokenised as running text and mapped back to words, so byte-level
  BPE tokenizers (GPT-2, RoBERTa, Qwen) see spaces between words; batches are right-padded so
  that decoder models give the same vectors with and without padding.
- `DocumentDetector` and `TokenDetector`: use any embedding detector on text.
- `pytextad.metrics`: document and token AUROC / AP / FPR95, score aggregation, multi-seed
  `evaluate` (reproduces the TokenCore benchmark numbers exactly).
- `supports_token` flag on every detector; `cache_dir` for DATE and FATE.

## 0.1.0 (unreleased)
- First release: CVDD, DATE, FATE, RSRAE with a common detector API.
- Frozen token-embedding extraction (`TokenEmbedder`).
- Numerical equivalence checks against the four original implementations (`tests/verification`).
