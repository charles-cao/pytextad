API Reference
=============

This is the API documentation for PyTextAD.

Detectors
---------

The ``pytextad.models`` module includes the text anomaly detectors. All of them score
documents; CVDD and DATE also score every token (``token_scores``).

.. currentmodule:: pytextad.models

.. autosummary::
   :toctree: generated
   :nosignatures:

   cvdd.CVDD
   date.DATE
   fate.FATE
   rsrae.RSRAE

Wrappers
--------

Turn any vector detector with ``fit`` and ``decision_function`` (all of PyOD) into a
text detector.

.. autosummary::
   :toctree: generated
   :nosignatures:

   wrappers.DocumentDetector
   wrappers.TokenDetector

Base class
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   base.BaseTextDetector

Embeddings
----------

The ``pytextad.utils.embeddings`` module extracts frozen embeddings from Hugging Face
models.

.. currentmodule:: pytextad.utils.embeddings

.. autosummary::
   :toctree: generated
   :nosignatures:

   TokenEmbedder
   SentenceEmbedder
   mean_pool
   words_from_subwords
   encode_words

.. _api-datasets:

Datasets
--------

The ``pytextad.datasets`` module loads benchmark datasets with word-level labels.

.. currentmodule:: pytextad.datasets

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_dataset
   list_datasets
   dataset_info
   load_local
   TextADDataset

Metrics
-------

The ``pytextad.metrics`` module evaluates detectors at the document and token levels.

.. currentmodule:: pytextad.metrics

.. autosummary::
   :toctree: generated
   :nosignatures:

   evaluate
   format_results
   document_metrics
   token_metrics
   aggregate
   document_labels
   fpr_at_tpr
