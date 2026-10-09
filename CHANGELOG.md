# Changelog

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
