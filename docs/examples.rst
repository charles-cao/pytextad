Examples
========

A tutorial notebook walks through a dataset, token- and document-level detection, the
highlighted scores and a comparison of detectors; it runs in Google Colab without installing
anything: `Open in Colab <https://colab.research.google.com/github/charles-cao/pytextad/blob/main/examples/tutorial.ipynb>`_ (``examples/tutorial.ipynb``).

An `online demo <https://huggingface.co/spaces/Charles-Cao/pytextad>`_ colours the words of any text typed in the browser by
how anomalous they are, for restaurant reviews, SMS messages and English sentences.

The ``examples`` folder of the repository also has three scripts.

``examples/quickstart.py`` runs CVDD, DATE, FATE and RSRAE on a small AG News subset
(one class as normal data, the others as anomalies):

.. code-block:: bash

   python examples/quickstart.py

``examples/token_level.py`` runs embedding detectors on a built-in dataset and reports token-
and document-level AUROC, AP and FPR at 95 % TPR over three seeds:

.. code-block:: bash

   python examples/token_level.py --dataset restaurant_review --model bert-base-uncased

``--train_ratio 0.5`` (default) trains on half of the normal documents and tests on the
rest; ``--train_ratio 0`` trains and tests on the whole dataset.

Benchmark: several detectors on several datasets
------------------------------------------------

``examples/benchmark.py`` evaluates several detectors on several datasets in one run and
writes the results as tables:

.. code-block:: bash

   python examples/benchmark.py \
       --datasets restaurant_review grammar_correction sms_spam \
       --detectors TokenCore SIK ADERH pyod:IForest pyod:ECOD \
       --model bert-base-uncased

For every dataset, half of the normal documents are used for training and all other
documents for testing (``--train_ratio``, ``--split_seed``); ``--train_ratio 0`` trains and
tests on the whole dataset. Every document is embedded once,
and the embeddings are cached in ``--cache``, so that adding a detector later does not embed
the data again. Every detector is then evaluated over the seeds given by ``--seeds``.

Detectors are given by name: any embedding detector of PyTextAD (see :doc:`algorithms`), or a
PyOD detector as ``pyod:<Class>`` (needs ``pip install pyod``). Parameters go in brackets,
with the names of the detector's parameters (see the :doc:`api`), and the same detector can
be listed several times with different parameters; each specification is one row of the
tables. In a shell, put each one in quotes:

.. code-block:: bash

   python examples/benchmark.py --datasets restaurant_review \
       --detectors SIK "SIK(max_samples=8, n_estimators=100)" \
                   "TokenCore(n_neighbors=5, aggregation='mean')" "pyod:IForest(n_estimators=200)"

Only literal values (numbers, strings, ``True``/``False``/``None``) are accepted. A detector
that fails, for example because of a missing optional dependency or a misspelled parameter,
is reported and the others still run.
``--pool_vectors`` pools the sub-word vectors of each word before scoring, as in the
published TokenCore experiments.

The results are printed and saved to ``benchmark_results/results.csv`` (one row per dataset,
detector, level and metric, with mean, standard deviation and running time) and
``benchmark_results/results.md``, which has one table per dataset:

.. code-block:: text

   | Detector  | Token AUROC     | Token AP        | Doc AUROC (max) | Doc AUROC (mean) |
   |-----------|-----------------|-----------------|-----------------|------------------|
   | TokenCore | 0.xxxx ± 0.xxxx | 0.xxxx ± 0.xxxx | 0.xxxx ± 0.xxxx | 0.xxxx ± 0.xxxx  |
   | SIK       | ...             | ...             | ...             | ...              |

and a summary of token AUROC over all datasets. The functions of the script
(``run_benchmark``, ``to_markdown``) can also be imported and used with your own datasets.
