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

==================  ===================  ============  ==========================
Detector            Input                Token level   Reference
==================  ===================  ============  ==========================
CVDD                token embeddings     yes           Ruff et al., ACL 2019
RSRAE               vectors              via wrapper   Lai et al., ICLR 2020
DATE                text                 yes           Manolache et al., NAACL 2021
FATE                text                 no            Das et al., ICONIP 2023
DocumentDetector    document embeddings  no            any PyOD detector
TokenDetector       token embeddings     yes           any PyOD detector
==================  ===================  ============  ==========================

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
   license
