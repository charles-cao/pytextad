# Changelog

## Unreleased
- Vector detectors in `pytextad.models.vector`: `TokenCore` (Cao et al., WWW 2026) and `SIK`
  (Cao et al., Findings of EMNLP 2025), the authors' code unchanged and checked against the
  originals; `ADERH` (Durani et al., NeurIPS 2025, official package) and `TCCM` (Li et al.,
  NeurIPS 2025, official code); `RSRAE` moved there (old import path still works).
- Vector baselines from the SVEAD / TokenCore benchmark copies: DAGMM, GANomaly, DROCC, GOAD,
  ICL, MCM, SLAD, NormalizingFlow, DTECategorical, DTEInverseGamma, DTEGaussian,
  DTENonParametric, DDPM, DDAE, DRL; each gives exactly the scores of its benchmark copy.
- Corrected by default (old behaviour available): DAGMM's network was never trained
  (`detach_loss=True` reproduces); NormalizingFlow's score depended on the batch
  (`legacy_score=True` reproduces).
- Seeds: `evaluate`, `DocumentDetector`, `TokenDetector` and the vector detectors seed the
  global random generators (Python, NumPy, PyTorch) before fitting, and score training data
  without disturbing them, so results equal those of a plain seeded benchmark script.
- `DocumentDetector` and `TokenDetector` keep float64 input vectors as float64 (they were
  rounded to float32, which changed scores slightly compared with running the detector directly).
- Documentation: references with citation labels linking to a References page; `CITATION.cff`.

## 0.3.0
- `pytextad.datasets`: `load_dataset` for six datasets with word-level anomaly labels -
  `sms_spam`, `restaurant_review`, `grammar_correction` (Cao et al., WWW 2026), `hate_speech`,
  `olid`, `restaurant_review2` (CA-PTD experiments) - downloaded from the Hugging Face Hub and
  verified by SHA-256 and document/word counts.
- `TextADDataset`: words, word labels, document labels and original texts; `subset`,
  `normal_indices` / `anomalous_indices`, `stats`, `retokenize` (one word segmentation for
  normal and anomalous documents), `to_jsonl`.
- `load_local` for JSONL, JSON and TokenCore `.npy` files; `TextADDataset.from_lists`.
- `examples/token_level.py`: PyOD detectors on a dataset, token and document metrics.

## 0.2.0
- `SentenceEmbedder`: CLS, mean or last-token sentence vectors from any Hugging Face model.
- `TokenEmbedder`: word-level vectors (`word_pooling="max" | "mean" | "first"`), on-disk caching
  (`transform(..., cache=...)`), `cache_dir` for local model folders.
- Lists of words are now tokenised as running text and mapped back to words, so byte-level
  BPE tokenizers (GPT-2, RoBERTa, Qwen) see spaces between words; batches are right-padded so
  that decoder models give the same vectors with and without padding.
- `DocumentDetector` and `TokenDetector`: use any PyOD-style vector detector on text.
- `pytextad.metrics`: document and token AUROC / AP / FPR95, score aggregation, multi-seed
  `evaluate` (reproduces the TokenCore benchmark numbers exactly).
- `supports_token` flag on every detector; `cache_dir` for DATE and FATE.

## 0.1.0 (unreleased)
- First release: CVDD, DATE, FATE, RSRAE with a PyOD-style API.
- Frozen token-embedding extraction (`TokenEmbedder`).
- Numerical equivalence checks against the four original implementations (`tests/verification`).
