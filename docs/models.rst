Implemented methods
===================

Document and token level
------------------------

All detectors give document scores. Token scores (``supports_token = True``):

=====================  ==============  ============  =========================================
Detector               Document level  Token level   Note
=====================  ==============  ============  =========================================
DATE                   yes             yes           token score = P(token replaced) (extension)
CVDD                   yes             yes           token-to-context distance (extension)
RSRAE                  yes             via wrapper   ``TokenDetector(RSRAE())``
FATE                   yes             no            end-to-end document model
DocumentDetector(any)  yes             no            vector detector on sentence embeddings
TokenDetector(any)     yes (aggreg.)   yes           vector detector on token embeddings
=====================  ==============  ============  =========================================

Default hyperparameters are those of the official code. Where the official code and
the paper disagree, the code is followed; the module docstrings (see :doc:`api`)
list every such case and every deliberate deviation.

Dataset-specific settings
-------------------------

=======  ==============================  ============================================
Method   Setting                         Values
=======  ==============================  ============================================
DATE     AG News (default)               ``n_masks=50, mask_ratio=0.5, max_len=128``
DATE     20 Newsgroups                   ``n_masks=25, mask_ratio=0.25, max_len=498``
CVDD     paper's Table 1                 average over ``n_heads`` in {3, 5, 10} and ``lambda_p`` in {1, 10}
CVDD     inputs                          GloVe 6B (Reuters) / fastText (20 Newsgroups), 300-d
FATE     epochs                          4 AG News, 50 20 Newsgroups, 40 Reuters, 80 if < 500 inliers
FATE     labelled anomalies              10
RSRAE    protocol in the paper           fit on the contaminated data and use ``decision_scores_``
=======  ==============================  ============================================
