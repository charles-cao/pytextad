PyTextAD: Text Anomaly Detection in Python
==========================================

**PyTextAD** is a Python library for detecting anomalies in text: whole anomalous documents,
and the individual tokens that make them anomalous.

* **23 detectors** with one interface (``fit``, ``decision_function``, ``predict``), from
  classic text detectors to recent embedding-based methods. See :doc:`algorithms`.
* **Token-level detection**: every detector can score each token of a document, and token
  scores are aggregated into document scores.
* **Faithful implementations**: each detector is its authors' code, or is checked
  numerically against it. See :doc:`faithfulness`.
* **Benchmark datasets** with a label for every word, and evaluation at the token and
  document levels. See :doc:`datasets`.

Installation
------------

.. code-block:: bash

   pip install pytextad

Example
-------

.. code-block:: python

   from pytextad import TokenCore, TokenDetector, TokenEmbedder
   from pytextad.datasets import load_dataset

   ds = load_dataset("restaurant_review")
   emb = TokenEmbedder("bert-base-uncased", word_pooling="max")   # one vector per word
   det = TokenDetector(TokenCore(), embedder=emb)
   det.fit(ds.tokens[:500])
   word_scores = det.token_scores(ds.tokens[500:])      # one score per word
   doc_scores = det.decision_function(ds.tokens[500:])  # one score per document

Continue with :doc:`quickstart`.

.. toctree::
   :hidden:
   :caption: Getting Started

   install
   quickstart
   algorithms
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
