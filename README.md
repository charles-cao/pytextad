# PyTextAD: Text Anomaly Detection in Python

[![PyPI](https://img.shields.io/pypi/v/pytextad?label=pypi)](https://pypi.org/project/pytextad/)
[![Documentation](https://img.shields.io/readthedocs/pytextad?label=docs)](https://pytextad.readthedocs.io)
[![Tests](https://github.com/charles-cao/pytextad/actions/workflows/tests.yml/badge.svg)](https://github.com/charles-cao/pytextad/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)

**PyTextAD** is a Python library for detecting anomalies in text, both whole anomalous
documents and the individual tokens that make them anomalous.

* **23 detectors** with one interface: `fit`, `decision_function`, `predict`.
* **Token-level detection**: any detector scores every token of a document, and token scores
  are aggregated into document scores.
* **Faithful implementations**: each detector is the authors' code, or is checked numerically
  against it.
* **Benchmark datasets** with a label for every word, and evaluation at the token and
  document levels.

Documentation: https://pytextad.readthedocs.io

## Citing PyTextAD

If you use PyTextAD, please cite it, together with the papers of the detectors you use
(listed under [References](#references)):

```bibtex
@software{cao2026pytextad,
  author = {Cao, Yang},
  title  = {{PyTextAD}: Text Anomaly Detection in Python},
  year   = {2026},
  url    = {https://github.com/charles-cao/pytextad}
}
```

If you use the built-in datasets, please also cite *Towards Token-Level Text Anomaly Detection*
(WWW 2026), listed under [References](#references).

## Installation

```bash
pip install pytextad
```

Requires Python 3.9 or later, PyTorch 1.13 or later and transformers 4.30 or later.
Install the PyTorch build that matches your CUDA version first (https://pytorch.org).

## Quick start

```python
from pytextad import SIK, TokenDetector, TokenEmbedder
from pytextad.datasets import load_dataset
from pytextad.metrics import evaluate, format_results

ds = load_dataset("restaurant_review")
normal, anomalous = ds.normal_indices, ds.anomalous_indices
train = ds.subset(normal[:500])                         # 500 normal reviews
test = ds.subset(list(normal[500:]) + list(anomalous))  # the other reviews

emb = TokenEmbedder("bert-base-uncased", word_pooling="max")      # one vector per word
X_train, _ = emb.transform(train.tokens)
X_test, kept = emb.transform(test.tokens)
y_test = [labels[k] for labels, k in zip(test.token_labels, kept)]

result = evaluate(TokenDetector(SIK()), X_train, X_test, token_labels=y_test)
print(format_results({"SIK": result}))     # token and document AUROC, AP, FPR95
```

## Implemented algorithms

**Text detectors** take text (or token embeddings) and are trained end to end.

| Abbr | Algorithm | Year | Token scores | Ref |
|---|---|:-:|:-:|:-:|
| CVDD | Context Vector Data Description | 2019 | yes | [1] |
| DATE | Detecting Anomalies in Text via Self-Supervision of Transformers | 2021 | yes | [2] |
| FATE | Few-shot Anomaly Detection in Text with Deviation Learning | 2023 | no | [3] |

**Embedding detectors** take one vector per document or per token. Wrapped in
`TokenDetector`, each of them scores every token.

| Abbr | Algorithm | Year | Ref |
|---|---|:-:|:-:|
| NormalizingFlow | Planar normalizing flow | 2015 | [4] |
| DAGMM | Deep Autoencoding Gaussian Mixture Model | 2018 | [5] |
| GANomaly | Adversarially trained encoder-decoder-encoder | 2018 | [6] |
| RSRAE | Robust Subspace Recovery AutoEncoder | 2020 | [7] |
| GOAD | Classification-based anomaly detection with random transformations | 2020 | [8] |
| DROCC | Distributionally Robust One-Class Classifier | 2020 | [9] |
| ICL | Internal Contrastive Learning | 2022 | [10] |
| SLAD | Scale Learning-based Anomaly Detection | 2023 | [11] |
| DTE | Diffusion Time Estimation (categorical, inverse-gamma, Gaussian, non-parametric) | 2024 | [12] |
| DDPM | Denoising diffusion model, reconstruction error | 2024 | [12] |
| MCM | Masked Cell Modeling | 2024 | [13] |
| DRL | Decomposed Representation Learning | 2025 | [14] |
| DDAE | Diffusion-Scheduled Denoising Autoencoder | 2025 | [15] |
| SIK | Simplified Isolation Kernel | 2025 | [16] |
| ADERH | Ensemble of Random Pairs of Hyperspheres | 2025 | [17] |
| TCCM | Time-Conditioned Contraction Matching | 2025 | [18] |
| TokenCore | Nearest-neighbour memory bank of token embeddings | 2026 | [19] |

**Wrappers** turn any embedding detector with `fit` and `decision_function`, including those of
PyOD [20], into a text detector: `DocumentDetector` (one embedding per document) and
`TokenDetector` (one embedding per token).

## Datasets

Six datasets with a 0/1 label for every word, downloaded once from the Hugging Face Hub.

```python
from pytextad.datasets import load_dataset
ds = load_dataset("restaurant_review")
ds.tokens, ds.token_labels, ds.labels      # words, word labels, document labels
```

| Dataset | Documents | Anomalous | Anomaly |
|---|--:|--:|---|
| `sms_spam` | 4,518 | 393 | injected gibberish |
| `restaurant_review` | 1,100 | 50 | negative sentiment |
| `grammar_correction` | 300 | 30 | grammatical errors |
| `hate_speech` | 4,302 | 140 | hateful or offensive words |
| `olid` | 650 | 30 | offensive words |
| `restaurant_review2` | 520 | 25 | negative sentiment |

## License

BSD 2-Clause, except for some third-party code under its own licence (CC BY-SA 4.0, and a
research-only licence for GOAD); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## References

[1] L. Ruff, Y. Zemlyanskiy, R. Vandermeulen, T. Schnake, M. Kloft. Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text. *ACL*, 2019.

[2] A. Manolache, F. Brad, E. Burceanu. DATE: Detecting Anomalies in Text via Self-Supervision of Transformers. *NAACL*, 2021.

[3] A. S. Das, A. Ajay, S. Saha, M. Bhuyan. Few-shot Anomaly Detection in Text with Deviation Learning. *ICONIP*, 2023.

[4] D. J. Rezende, S. Mohamed. Variational Inference with Normalizing Flows. *ICML*, 2015.

[5] B. Zong, Q. Song, M. R. Min, W. Cheng, C. Lumezanu, D. Cho, H. Chen. Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection. *ICLR*, 2018.

[6] S. Akcay, A. Atapour-Abarghouei, T. P. Breckon. GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training. *ACCV*, 2018.

[7] C.-H. Lai, D. Zou, G. Lerman. Robust Subspace Recovery Layer for Unsupervised Anomaly Detection. *ICLR*, 2020.

[8] L. Bergman, Y. Hoshen. Classification-Based Anomaly Detection for General Data. *ICLR*, 2020.

[9] S. Goyal, A. Raghunathan, M. Jain, H. V. Simhadri, P. Jain. DROCC: Deep Robust One-Class Classification. *ICML*, 2020.

[10] T. Shenkar, L. Wolf. Anomaly Detection for Tabular Data with Internal Contrastive Learning. *ICLR*, 2022.

[11] H. Xu, Y. Wang, J. Wei, S. Jian, Y. Li, N. Liu. Fascinating Supervisory Signals and Where to Find Them: Deep Anomaly Detection with Scale Learning. *ICML*, 2023.

[12] V. Livernoche, V. Jain, Y. Hezaveh, S. Ravanbakhsh. On Diffusion Modeling for Anomaly Detection. *ICLR*, 2024. [arXiv:2305.18593](https://arxiv.org/abs/2305.18593)

[13] J. Yin, Y. Qiao, Z. Zhou, X. Wang, J. Yang. MCM: Masked Cell Modeling for Anomaly Detection in Tabular Data. *ICLR*, 2024.

[14] H. Ye, H. Zhao, W. Fan, M. Zhou, D. Guo, Y. Chang. DRL: Decomposed Representation Learning for Tabular Anomaly Detection. *ICLR*, 2025.

[15] T. Sattarov, M. Schreyer, D. Borth. Diffusion-Scheduled Denoising Autoencoders for Anomaly Detection in Tabular Data. *KDD*, 2025. [arXiv:2508.00758](https://arxiv.org/abs/2508.00758)

[16] Y. Cao, S. Yang, Y. Yang, L. Qi, M. Liu. Text Anomaly Detection with Simplified Isolation Kernel. *Findings of EMNLP*, 2025. [doi:10.18653/v1/2025.findings-emnlp.680](https://doi.org/10.18653/v1/2025.findings-emnlp.680)

[17] W. Durani, C. Leiber, K. Durani, C. Plant, C. Böhm. Anomaly Detection by an Ensemble of Random Pairs of Hyperspheres. *NeurIPS*, 2025.

[18] Z. Li, Q. Huang, Y. Zhu, L. Yang, M. M. Amiri, N. van Stein, M. van Leeuwen. Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching. *NeurIPS*, 2025. [arXiv:2510.18328](https://arxiv.org/abs/2510.18328)

[19] Y. Cao, B. Yu, S. Yang, M. Liu, Y. Yang. Towards Token-Level Text Anomaly Detection. *The ACM Web Conference (WWW)*, 2026. [doi:10.1145/3774904.3792952](https://doi.org/10.1145/3774904.3792952)

[20] Y. Zhao, Z. Nasrullah, Z. Li. PyOD: A Python Toolbox for Scalable Outlier Detection. *JMLR*, 20(96):1-7, 2019.
