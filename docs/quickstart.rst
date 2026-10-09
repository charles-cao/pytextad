Quick start
===========

Document-level detection
------------------------

.. code-block:: python

   from pytextad import CVDD, DATE, DocumentDetector, SentenceEmbedder, TokenEmbedder
   from pyod.models.knn import KNN

   # any PyOD detector on sentence embeddings
   det = DocumentDetector(KNN(), embedder=SentenceEmbedder("bert-base-uncased"))
   det.fit(train_texts)
   scores = det.decision_function(test_texts)        # higher = more anomalous
   labels = det.predict(test_texts)                  # 0 / 1, from the contamination rate

   # end-to-end text detectors
   scores = DATE().fit(train_texts).decision_function(test_texts)

Token-level detection
---------------------

.. code-block:: python

   from pytextad import TokenDetector, TokenEmbedder
   from pytextad.metrics import evaluate, format_results

   # documents as lists of words, token labels as one 0/1 array per document
   emb = TokenEmbedder("bert-base-cased", word_pooling="max", cache_dir="D:/models")
   X_train, _ = emb.transform(train_words, cache="train_emb.npz")   # one vector per word
   X_test, kept = emb.transform(test_words, cache="test_emb.npz")

   det = TokenDetector(KNN(), aggregation="max")     # fit on all training words pooled
   word_scores = det.fit(X_train).token_scores(X_test)

   res = evaluate(TokenDetector(KNN()), X_train, X_test, token_labels=test_labels, seeds=(0, 1, 2))
   print(format_results({"KNN": res}))

``evaluate`` reports, as mean and standard deviation over the seeds:

* ``document``: document metrics from ``decision_function``;
* ``document[max]``, ``document[mean]``: document metrics from aggregated token scores;
* ``token``: token metrics, all test tokens pooled.

Metrics are AUROC, AP and FPR at 95 % TPR, with anomalies as the positive class. Which
documents are used for training (normal only, contaminated, or the test set itself) is
decided by the caller.

Input types
-----------

==================  ======================================================================
Detector            ``X`` passed to ``fit`` / ``decision_function``
==================  ======================================================================
CVDD                list of ``[n_tokens, dim]`` arrays (e.g. from ``TokenEmbedder``)
RSRAE               ``[n_documents, dim]`` array (e.g. from ``SentenceEmbedder``)
DATE                list of strings, or list of word lists
FATE                list of strings; ``y`` marks the labelled anomalies (1)
DocumentDetector    ``[n_documents, dim]`` array, or texts if an embedder is given
TokenDetector       list of ``[n_tokens, dim]`` arrays, or texts if an embedder is given
==================  ======================================================================

``examples/quickstart.py`` runs the four end-to-end detectors on AG News.
