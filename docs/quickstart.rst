Quick start
===========

.. code-block:: python

   from pytextad import CVDD, DATE, FATE, RSRAE, TokenEmbedder, mean_pool

   emb = TokenEmbedder("bert-base-uncased")       # frozen encoder, last layer
   H_train, _ = emb.transform(train_texts)
   H_test, word_ids = emb.transform(test_texts)

   cvdd = CVDD().fit(H_train)
   scores = cvdd.decision_function(H_test)        # higher = more anomalous
   labels = cvdd.predict(H_test)                  # 0 / 1, using the contamination rate

   date = DATE().fit(train_texts)
   word_scores = date.token_scores([t.split() for t in test_texts])

``examples/quickstart.py`` runs all four detectors on AG News.

Input types
-----------

==========  =========================================================================
Detector    ``X`` passed to ``fit`` / ``decision_function``
==========  =========================================================================
CVDD        list of ``[n_tokens, dim]`` arrays (e.g. from ``TokenEmbedder``)
RSRAE       ``[n_documents, dim]`` array (e.g. ``mean_pool`` of token embeddings)
DATE        list of strings, or list of word lists
FATE        list of strings; ``y`` marks the labelled anomalies (1)
==========  =========================================================================
