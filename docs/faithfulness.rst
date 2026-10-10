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

TokenCore and SIK
~~~~~~~~~~~~~~~~~

The authors' code, unchanged (``pytextad/models/embedding/_vendor``); ``tests/test_embedding_detectors.py``
checks that the files are identical to the originals and that the scores are equal. SIK's
class gets scikit-learn's ``BaseEstimator`` as an extra base class, because its
``check_is_fitted`` calls fail on other objects from scikit-learn 1.8 on.

ADERH and TCCM
~~~~~~~~~~~~~~

ADERH is the official package (github.com/Walid10010/ADERH), unchanged; it gives the same
scores as the original code of the paper, which ``tests/test_embedding_detectors.py`` checks against that
code. The package documents three places where the original code differs from the paper's
description (how pair partners are chosen, the radius, and the alignment of radii with
centres); they are kept so that published results are reproduced.

TCCM is the official code (github.com/ZhongLIFR/TCCM-NIPS), unchanged except that the
training data are moved to the model's device; the official class trains on the CPU only.
``tests/test_embedding_detectors.py`` checks it against the copy used in the SVEAD benchmark.

Other embedding baselines
~~~~~~~~~~~~~~~~~~~~~~~~~

These are the copies used in the SVEAD and TokenCore benchmarks, which in turn come from the
repositories below. ``tests/test_baselines.py`` checks, for every one of them, that PyTextAD
gives exactly the same scores as the benchmark copy under the same seed.

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Detector
     - Source of the code
     - Changes in PyTextAD
   * - DTE (3 variants), DTE-NP, DDPM
     - official DTE code, github.com/vicliv/DTE (MIT)
     - none
   * - DDAE
     - official code, github.com/sattarov/AnoDDAE (MIT)
     - none (evaluation during training removed in the benchmark copy)
   * - DRL
     - rewrite by Yang Cao of github.com/HangtingYe/DRL
     - ``device`` honoured
   * - DROCC, GOAD, ICL, SLAD, NormalizingFlow
     - official code, adapted in the DTE repository, then in github.com/ZhongLIFR/TCCM-NIPS
     - none
   * - DAGMM, GANomaly
     - ADBench, then the TCCM repository
     - DAGMM: ``torch.linalg.cholesky``; optional loss detachment (below)
   * - MCM
     - official code, then the TCCM repository
     - none

Two published errors are corrected by default and can be reproduced:

* **DAGMM** (``detach_loss=True`` reproduces): the loss is returned as
  ``Variable(loss, requires_grad=True)``, a new tensor with no link to the network, so no
  gradient reaches it and the network keeps its random initial weights. This is in ADBench
  and in every copy above; we checked that the weights do not change during training.
* **NormalizingFlow** (``legacy_score=True`` reproduces): the original score adds a ``[n]``
  array to an ``[n, 1]`` array, which broadcasts to ``[n, n]``, and averages the rows, so a
  vector gets the batch's mean base log-density plus its own log-determinant. A far outlier
  scored alone gets 173.6 and 35.2 when scored with five normal vectors in our check. The
  corrected score is the vector's own negative log-likelihood. Training is not affected.

All embedding detectors seed the global random generators with ``random_state`` before the
model is built and fitted, and score the training data without changing the generators, so
a seed gives the same numbers as a benchmark script that seeds NumPy and PyTorch and then
calls ``fit`` and ``decision_function``.

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

.. automodule:: pytextad.models.embedding.rsrae
   :no-index:
   :no-members:
