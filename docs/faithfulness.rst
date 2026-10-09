Faithfulness to the Original Code
=================================

Every re-implemented detector uses the default hyperparameters of its official code.
Where the official code and the paper disagree, the code is followed. Each detector's
notes below list such cases and every deliberate deviation.

Equivalence checks
------------------

``tests/verification`` runs each original implementation next to PyTextAD with the same
weights, inputs and random seeds, and compares the results. All checks pass. Users do
not need to run them.

=======  ==========================================  ================================================
Method   Compared with                               What must match
=======  ==========================================  ================================================
DATE     bit-ml/date (Python 3.8, transformers 3.0)  tokenisation, windows, scores, loss, 3 optimiser steps
CVDD     lukasruff/CVDD-PyTorch                      context vectors, attention, scores, alpha schedule
RSRAE    dmzou/RSRAE (TensorFlow)                    batch normalisation, forward pass, 20 epochs
FATE     arav1ndajay/fate                            scores, training loop, batch sampler
=======  ==========================================  ================================================

Each check also fails, as it should, on deliberately broken copies of PyTextAD.

Settings used in the papers
---------------------------

=======  ==============================  =================================================
Method   Setting                         Values
=======  ==============================  =================================================
DATE     AG News (default)               ``n_masks=50, mask_ratio=0.5, max_len=128``
DATE     20 Newsgroups                   ``n_masks=25, mask_ratio=0.25, max_len=498``
CVDD     results in the paper            mean over ``n_heads`` in {3, 5, 10}, ``lambda_p`` in {1, 10}
FATE     epochs                          4 AG News, 50 20 Newsgroups, 40 Reuters
FATE     labelled anomalies              10
RSRAE    protocol                        fit on the contaminated data, use ``decision_scores_``
=======  ==============================  =================================================

Notes per detector
------------------

CVDD
~~~~

.. automodule:: pytextad.models.cvdd
   :no-index:
   :no-members:

DATE
~~~~

.. automodule:: pytextad.models.date
   :no-index:
   :no-members:

FATE
~~~~

.. automodule:: pytextad.models.fate
   :no-index:
   :no-members:

RSRAE
~~~~~

.. automodule:: pytextad.models.rsrae
   :no-index:
   :no-members:
