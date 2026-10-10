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

Vector detectors
----------------

The ``pytextad.models.vector`` module includes detectors for vectors, taken from their
authors' code. Use them on document embeddings with ``DocumentDetector`` or on token
embeddings with ``TokenDetector``.

.. autosummary::
   :toctree: generated
   :nosignatures:

   vector.TokenCore
   vector.SIK
   vector.ADERH
   vector.TCCM
   vector.RSRAE
   vector.DAGMM
   vector.GANomaly
   vector.DROCC
   vector.GOAD
   vector.ICL
   vector.MCM
   vector.SLAD
   vector.NormalizingFlow
   vector.DTECategorical
   vector.DTEInverseGamma
   vector.DTEGaussian
   vector.DTENonParametric
   vector.DDPM
   vector.DDAE
   vector.DRL

Wrappers
--------

Turn any vector detector with ``fit`` and ``decision_function`` (all of PyOD) into a
text detector.

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
   vector.BaseVectorDetector

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
