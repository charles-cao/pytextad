PyTextAD: Text Anomaly Detection in Python
==========================================

**PyTextAD** is a Python library for detecting anomalies in text, at the level of whole
documents and of individual tokens. All detectors share the interface of
`PyOD <https://pyod.readthedocs.io>`_ (``fit``, ``decision_function``, ``predict``), and
every re-implemented method is checked numerically against its original code.

.. code-block:: python

   from pyod.models.knn import KNN
   from pytextad import TokenDetector, TokenEmbedder
   from pytextad.datasets import load_dataset

   ds = load_dataset("restaurant_review")
   det = TokenDetector(KNN(), embedder=TokenEmbedder("bert-base-uncased", word_pooling="max"))
   det.fit(ds.tokens[:500])
   word_scores = det.token_scores(ds.tokens[500:])      # one score per word
   doc_scores = det.decision_function(ds.tokens[500:])  # one score per document

Implemented algorithms
----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 22 14 12

   * - Detector
     - Input
     - Token level
     - Reference
   * - TokenCore
     - token embeddings
     - via wrapper
     - :cite:`cao2026tokencore`
   * - SIK
     - vectors
     - via wrapper
     - :cite:`cao2025sik`
   * - ADERH
     - vectors
     - via wrapper
     - :cite:`durani2025aderh`
   * - TCCM
     - vectors
     - via wrapper
     - :cite:`li2025tccm`
   * - DAGMM
     - vectors
     - via wrapper
     - :cite:`zong2018dagmm`
   * - GANomaly
     - vectors
     - via wrapper
     - :cite:`akcay2018ganomaly`
   * - DROCC
     - vectors
     - via wrapper
     - :cite:`goyal2020drocc`
   * - GOAD
     - vectors
     - via wrapper
     - :cite:`bergman2020goad`
   * - ICL
     - vectors
     - via wrapper
     - :cite:`shenkar2022icl`
   * - MCM
     - vectors
     - via wrapper
     - :cite:`yin2024mcm`
   * - SLAD
     - vectors
     - via wrapper
     - :cite:`xu2023slad`
   * - NormalizingFlow
     - vectors
     - via wrapper
     - :cite:`rezende2015planar`
   * - DTECategorical, DTEInverseGamma, DTEGaussian, DTENonParametric, DDPM
     - vectors
     - via wrapper
     - :cite:`livernoche2024dte`
   * - DDAE
     - vectors
     - via wrapper
     - :cite:`sattarov2025ddae`
   * - DRL
     - vectors
     - via wrapper
     - :cite:`ye2025drl`
   * - CVDD
     - token embeddings
     - yes
     - :cite:`ruff2019cvdd`
   * - RSRAE
     - vectors
     - via wrapper
     - :cite:`lai2020rsrae`
   * - DATE
     - text
     - yes
     - :cite:`manolache2021date`
   * - FATE
     - text
     - no
     - :cite:`das2023fate`
   * - DocumentDetector
     - document embeddings
     - no
     - any PyOD detector :cite:`zhao2019pyod`
   * - TokenDetector
     - token embeddings
     - yes
     - any PyOD detector :cite:`zhao2019pyod`

"Via wrapper": a vector detector scores tokens when wrapped in ``TokenDetector``.

Get started with :doc:`install` and :doc:`quickstart`; the built-in benchmark data are
described in :doc:`datasets`.

.. toctree::
   :hidden:
   :caption: Getting Started

   install
   quickstart
   datasets
   examples

.. toctree::
   :hidden:
   :caption: API Documentation

   api

.. toctree::
   :hidden:
   :caption: Other

   faithfulness
   changelog
   citing
   references
   license
