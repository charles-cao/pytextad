Algorithms
==========

All detectors share one interface: ``fit`` on training data, ``decision_function`` for
anomaly scores (higher means more anomalous), and ``predict`` for 0/1 labels. Each class is
documented in the :doc:`api`; the papers are listed under :doc:`references`.

Text detectors
--------------

Detectors that take text (or token embeddings) and are trained end to end.

.. list-table::
   :header-rows: 1
   :widths: 14 52 8 12 14

   * - Detector
     - Algorithm
     - Year
     - Token scores
     - Reference
   * - CVDD
     - Context Vector Data Description
     - 2019
     - yes
     - :cite:`ruff2019cvdd`
   * - DATE
     - Detecting Anomalies in Text via Self-Supervision of Transformers
     - 2021
     - yes
     - :cite:`manolache2021date`
   * - FATE
     - Few-shot Anomaly Detection in Text with Deviation Learning
     - 2023
     - no
     - :cite:`das2023fate`

Embedding detectors
-------------------

Detectors for embeddings: one vector per document, or one per token (any numeric vectors
work too, e.g. TF-IDF or tabular data). Wrapped in ``TokenDetector``, each of them scores
every token, and token scores are aggregated into document scores.

.. list-table::
   :header-rows: 1
   :widths: 18 58 8 16

   * - Detector
     - Algorithm
     - Year
     - Reference
   * - NormalizingFlow
     - Planar normalizing flow
     - 2015
     - :cite:`rezende2015planar`
   * - DAGMM
     - Deep Autoencoding Gaussian Mixture Model
     - 2018
     - :cite:`zong2018dagmm`
   * - GANomaly
     - Adversarially trained encoder-decoder-encoder
     - 2018
     - :cite:`akcay2018ganomaly`
   * - RSRAE
     - Robust Subspace Recovery AutoEncoder
     - 2020
     - :cite:`lai2020rsrae`
   * - GOAD
     - Classification-based anomaly detection with random transformations
     - 2020
     - :cite:`bergman2020goad`
   * - DROCC
     - Distributionally Robust One-Class Classifier
     - 2020
     - :cite:`goyal2020drocc`
   * - ICL
     - Internal Contrastive Learning
     - 2022
     - :cite:`shenkar2022icl`
   * - SLAD
     - Scale Learning-based Anomaly Detection
     - 2023
     - :cite:`xu2023slad`
   * - DTECategorical
     - Diffusion Time Estimation, categorical
     - 2024
     - :cite:`livernoche2024dte`
   * - DTEInverseGamma
     - Diffusion Time Estimation, inverse gamma
     - 2024
     - :cite:`livernoche2024dte`
   * - DTEGaussian
     - Diffusion Time Estimation, Gaussian
     - 2024
     - :cite:`livernoche2024dte`
   * - DTENonParametric
     - Diffusion Time Estimation, non-parametric
     - 2024
     - :cite:`livernoche2024dte`
   * - DDPM
     - Denoising diffusion model, reconstruction error
     - 2024
     - :cite:`livernoche2024dte`
   * - MCM
     - Masked Cell Modeling
     - 2024
     - :cite:`yin2024mcm`
   * - DRL
     - Decomposed Representation Learning
     - 2025
     - :cite:`ye2025drl`
   * - DDAE
     - Diffusion-Scheduled Denoising Autoencoder
     - 2025
     - :cite:`sattarov2025ddae`
   * - SIK
     - Simplified Isolation Kernel
     - 2025
     - :cite:`cao2025sik`
   * - ADERH
     - Ensemble of Random Pairs of Hyperspheres
     - 2025
     - :cite:`durani2025aderh`
   * - TCCM
     - Time-Conditioned Contraction Matching
     - 2025
     - :cite:`li2025tccm`
   * - TokenCore
     - Nearest-neighbour memory bank of token embeddings
     - 2026
     - :cite:`cao2026tokencore`

Wrappers
--------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Wrapper
     - Use
   * - DocumentDetector
     - one embedding per document; accepts texts or vectors
   * - TokenDetector
     - one embedding per token; gives token scores and document scores

Any object with ``fit(X)`` and ``decision_function(X)`` on a 2-D array can be wrapped,
including the detectors of PyOD :cite:`zhao2019pyod`.
