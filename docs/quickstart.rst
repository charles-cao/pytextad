Quick Start
===========

Document-level detection
------------------------

Any PyOD detector works on document embeddings through
:class:`~pytextad.models.wrappers.DocumentDetector`; end-to-end text detectors take the
texts directly.

.. code-block:: python

   from pyod.models.knn import KNN
   from pytextad import DATE, DocumentDetector, SentenceEmbedder

   det = DocumentDetector(KNN(), embedder=SentenceEmbedder("bert-base-uncased"))
   det.fit(train_texts)
   scores = det.decision_function(test_texts)   # higher = more anomalous
   labels = det.predict(test_texts)             # 0 / 1

   scores = DATE().fit(train_texts).decision_function(test_texts)

Token-level detection
---------------------

:class:`~pytextad.models.wrappers.TokenDetector` fits a vector detector on the words of
all training documents, scores every word, and aggregates word scores into document
scores.

.. code-block:: python

   from pytextad import TokenDetector, TokenEmbedder

   emb = TokenEmbedder("bert-base-uncased", word_pooling="max")
   det = TokenDetector(KNN(), embedder=emb, aggregation="max")
   det.fit(train_words)                       # documents as lists of words
   word_scores = det.token_scores(test_words)
   doc_scores = det.decision_function(test_words)

Evaluation
----------

:func:`~pytextad.metrics.evaluate` fits a detector once per seed and reports AUROC, AP and
FPR at 95 % TPR, at the document level and, for token-level detectors, at the token level.

.. code-block:: python

   from pytextad.metrics import evaluate, format_results

   X_train, _ = emb.transform(train_words, cache="train.npz")   # embed once, reuse
   X_test, _ = emb.transform(test_words, cache="test.npz")
   res = evaluate(TokenDetector(KNN()), X_train, X_test, token_labels=test_labels)
   print(format_results({"KNN": res}))

Input of each detector
----------------------

==================  ==================================================================
Detector            ``X`` in ``fit`` and ``decision_function``
==================  ==================================================================
CVDD                list of ``[n_tokens, dim]`` arrays
RSRAE               ``[n_documents, dim]`` array
DATE                list of strings or of word lists
FATE                list of strings, with ``y`` marking labelled anomalies
DocumentDetector    ``[n_documents, dim]`` array, or texts when given an embedder
TokenDetector       list of ``[n_tokens, dim]`` arrays, or texts when given an embedder
==================  ==================================================================
