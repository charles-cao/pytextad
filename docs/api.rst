API Reference
=============

This is the API documentation for PyTextAD.

Text detectors
--------------

Detectors that take text (or token embeddings) and are trained end to end. All of them
score documents; CVDD and DATE also score every token (``token_scores``).

.. currentmodule:: pytextad.models

.. autosummary::
   :toctree: generated
   :nosignatures:

   cvdd.CVDD
   date.DATE
   fate.FATE

Embedding detectors
-------------------

The ``pytextad.models.embedding`` module includes detectors for embeddings (one vector per
document or per token), taken from their authors' code. Use them on document embeddings with ``DocumentDetector`` or on token
embeddings with ``TokenDetector``.

.. autosummary::
   :toctree: generated
   :nosignatures:

   embedding.TokenCore
   embedding.SIK
   embedding.ADERH
   embedding.TCCM
   embedding.RSRAE
   embedding.DAGMM
   embedding.GANomaly
   embedding.DROCC
   embedding.GOAD
   embedding.ICL
   embedding.MCM
   embedding.SLAD
   embedding.NormalizingFlow
   embedding.DTECategorical
   embedding.DTEInverseGamma
   embedding.DTEGaussian
   embedding.DTENonParametric
   embedding.DDPM
   embedding.DDAE
   embedding.DRL

Wrappers
--------

Turn any embedding detector with ``fit`` and ``decision_function`` into a text detector.

.. autosummary::
   :toctree: generated
   :nosignatures:

   wrappers.DocumentDetector
   wrappers.TokenDetector

Base classes
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   base.BaseTextDetector
   embedding.BaseEmbeddingDetector

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
   align_labels
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
