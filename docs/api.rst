API reference
=============

Detectors
---------

Every detector gives document scores through ``decision_function``. Detectors with
``supports_token = True`` also give token scores through ``token_scores``.

.. autoclass:: pytextad.models.cvdd.CVDD
.. autoclass:: pytextad.models.date.DATE
.. autoclass:: pytextad.models.fate.FATE
.. autoclass:: pytextad.models.rsrae.RSRAE

Wrappers for vector detectors (PyOD, SIK, ...)
----------------------------------------------

.. autoclass:: pytextad.models.wrappers.DocumentDetector
.. autoclass:: pytextad.models.wrappers.TokenDetector

Base class
----------

.. autoclass:: pytextad.models.base.BaseTextDetector

Embeddings
----------

.. autoclass:: pytextad.utils.embeddings.TokenEmbedder
.. autoclass:: pytextad.utils.embeddings.SentenceEmbedder
.. autofunction:: pytextad.utils.embeddings.mean_pool
.. autofunction:: pytextad.utils.embeddings.words_from_subwords
.. autofunction:: pytextad.utils.embeddings.encode_words

Evaluation
----------

.. automodule:: pytextad.metrics
   :members:
   :no-inherited-members:
   :no-show-inheritance:

Module notes (faithfulness and deviations)
------------------------------------------

CVDD
~~~~

.. automodule:: pytextad.models.cvdd
   :no-index:
   :no-members:
   :no-inherited-members:
   :no-show-inheritance:

DATE
~~~~

.. automodule:: pytextad.models.date
   :no-index:
   :no-members:
   :no-inherited-members:
   :no-show-inheritance:

FATE
~~~~

.. automodule:: pytextad.models.fate
   :no-index:
   :no-members:
   :no-inherited-members:
   :no-show-inheritance:

RSRAE
~~~~~

.. automodule:: pytextad.models.rsrae
   :no-index:
   :no-members:
   :no-inherited-members:
   :no-show-inheritance:
