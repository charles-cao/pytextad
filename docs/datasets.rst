Datasets
========

``pytextad.datasets`` gives benchmark datasets with a label for every word, so the same data
serve token-level and document-level evaluation. A document is anomalous if any of its words
is anomalous. How the documents are split into training and test data is up to you.

.. code-block:: python

   from pytextad.datasets import load_dataset, list_datasets

   list_datasets()        # ['grammar_correction', 'hate_speech', 'olid', 'restaurant_review', ...]
   ds = load_dataset("restaurant_review")
   ds                     # TextADDataset('restaurant_review': 1100 documents, 50 anomalous; ...)

   ds.tokens[0]           # list of words: pass these to detectors and embedders
   ds.token_labels[0]     # one 0/1 label per word
   ds.labels              # one 0/1 label per document
   ds.texts[0]            # original text

Running token-level detection
-----------------------------

A complete run on one dataset: split the documents, embed every word, fit a detector on
the training words, and evaluate at the token and document levels.

.. code-block:: python

   import numpy as np
   from pytextad import TokenCore, TokenDetector, TokenEmbedder, align_labels
   from pytextad.datasets import load_dataset
   from pytextad.metrics import evaluate, format_results

   ds = load_dataset("restaurant_review")

   # train on half of the normal documents, test on all the others
   # (the split of split_anomaly_data in the TokenCore code)
   np.random.seed(42)
   normal = ds.normal_indices
   train_idx = np.random.choice(normal, size=len(normal) // 2, replace=False)
   test_idx = np.setdiff1d(np.arange(len(ds)), train_idx)
   train, test = ds.subset(train_idx), ds.subset(test_idx)

   # one vector per sub-word, and the word each sub-word belongs to
   emb = TokenEmbedder("bert-base-uncased")
   X_train = emb.transform(train.tokens, cache="train.npz")
   X_test = emb.transform(test.tokens, cache="test.npz")
   # words beyond 512 sub-words have no vector: keep the labels of the embedded words
   y_test = align_labels(test.token_labels, X_test[1])

   res = evaluate(TokenDetector(TokenCore()), X_train, X_test, token_labels=y_test, seeds=(0, 1, 2))
   print(format_results({"TokenCore": res}))

The output has one line per level, each with AUROC, AP and FPR at 95 % TPR as mean and
standard deviation over the seeds:

* ``token``: all test words pooled together (a word's score is the maximum over its
  sub-words);
* ``document[max]`` and ``document[mean]``: each document scored by the maximum or the
  mean of its word scores;
* ``document``: the detector's own document score (here the same as ``document[max]``).

``examples/token_level.py`` runs this for any dataset and several detectors:

.. code-block:: bash

   python examples/token_level.py --dataset restaurant_review --model bert-base-uncased

Built-in datasets
-----------------

======================  =========  =========  ========  =================  ===========================
Name                    Documents  Anomalous  Words     Anomalous words    Anomaly
======================  =========  =========  ========  =================  ===========================
``sms_spam``            4,518      393        81,570    418                injected gibberish
``restaurant_review``   1,100      50         35,488    282                negative sentiment
``grammar_correction``  300        30         2,746     47                 grammatical errors
``hate_speech``         4,302      140        99,390    288                hateful or offensive words
``olid``                650        30         21,156    58                 offensive words
``restaurant_review2``  520        25         33,529    94                 negative sentiment
======================  =========  =========  ========  =================  ===========================

The first three are the datasets of *Towards Token-Level Text Anomaly Detection* (Cao et al.,
WWW 2026); all six are used in the CA-PTD experiments, where they are called ``spam``,
``grammar``, ``review1``, ``hate_speech``, ``olid`` and ``review2`` (these short names also
work in ``load_dataset``). ``hate_speech`` takes its tweets from Davidson et al. (ICWSM 2017)
and ``olid`` from Zampieri et al. (NAACL 2019); the word labels are new. They are hosted at `huggingface.co/datasets/Charles-Cao/pytextad
<https://huggingface.co/datasets/Charles-Cao/pytextad>`_. The first ``load_dataset`` call
downloads one JSONL file into the Hugging Face cache (or ``cache_dir``); later calls read it
offline. Every file is checked against a SHA-256 checksum and the counts above, so all users
evaluate on identical data. Set ``HF_ENDPOINT`` to use a mirror.

Word segmentation
-----------------

In these datasets (all but ``grammar_correction``) the normal documents were split into words by the regular expression
``\w+|[^\w\s]``, while the words of anomalous documents are the annotators' spans, which can
contain apostrophes or spaces (``"it's"``, ``"not fresh"``). The default keeps the original
words, as used in the paper. ``load_dataset(name, retokenize=True)`` (or
``ds.retokenize()``) splits every span by the same expression, each piece keeping the
span's label, so that normal and anomalous documents are segmented alike.

Your own data
-------------

.. code-block:: python

   from pytextad.datasets import TextADDataset, load_local

   ds = TextADDataset.from_lists(tokens, token_labels, texts=None)
   ds = load_local("my_data.jsonl")   # {"tokens": [...], "labels": [...], "text": "..."} per line
   ds = load_local("my_data.npy")     # TokenCore .npy format (pickled; trusted files only)
   ds.to_jsonl("my_data.jsonl")

See :ref:`api-datasets` for the full API.
