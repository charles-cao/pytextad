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

Try it in the browser: `online demo <https://huggingface.co/spaces/Charles-Cao/pytextad>`_.

Installation
------------

.. code-block:: bash

   pip install pytextad

Example
-------

.. code-block:: python

   from pytextad.datasets import load_dataset

   ds = load_dataset("restaurant_review")
   train = ds.subset(ds.normal_indices[:500])            # 500 normal reviews
   test = ds.subset(list(ds.normal_indices[500:]) + list(ds.anomalous_indices))

Document level: one vector per document.

.. code-block:: python

   from pytextad import SIK, DocumentDetector, SentenceEmbedder

   det = DocumentDetector(SIK(), embedder=SentenceEmbedder("bert-base-uncased"))
   det.fit(train.texts)
   doc_scores = det.decision_function(test.texts)       # one score per document

Token level: one score per word, and word scores aggregated into document scores.

.. code-block:: python

   from pytextad import TokenCore, TokenDetector, TokenEmbedder

   det = TokenDetector(TokenCore(), embedder=TokenEmbedder("bert-base-uncased"))
   det.fit(train.tokens)
   word_scores = det.token_scores(test.tokens)          # one score per word
   doc_scores = det.decision_function(test.tokens)      # one score per document

Embeddings you already have can be passed instead of texts. See :doc:`quickstart`.

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
   contributing
   changelog
   citing
   references
   license
