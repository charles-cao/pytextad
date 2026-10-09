Examples
========

The ``examples`` folder of the repository has two scripts.

``examples/quickstart.py`` runs CVDD, DATE, FATE and RSRAE on a small AG News subset
(one class as normal data, the others as anomalies):

.. code-block:: bash

   python examples/quickstart.py

``examples/token_level.py`` runs PyOD detectors on a built-in dataset and reports token-
and document-level AUROC, AP and FPR at 95 % TPR over three seeds:

.. code-block:: bash

   python examples/token_level.py --dataset restaurant_review --model bert-base-uncased

``--train_ratio 0.5`` (default) trains on half of the normal documents and tests on the
rest; ``--train_ratio 0`` trains and tests on the whole dataset.
