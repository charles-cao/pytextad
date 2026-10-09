"""PyTextAD's evaluation reproduces the TokenCore benchmark code (benchmark_scoreagg.py)."""
import inspect

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from pytextad import TokenDetector
from pytextad.metrics import evaluate

pyod = pytest.importorskip("pyod")
from pyod.models.iforest import IForest  # noqa: E402
from pyod.models.knn import KNN  # noqa: E402


# ---- copied from benchmark_scoreagg.py (TokenCore), unchanged except comments removed ----
def evaluate_model_token_score_aggregation(clf, X_train, X_test, y_test, sentence_info):
    clf.fit(X_train)
    scores = clf.decision_function(X_test)
    token_roc_auc = roc_auc_score(y_test, scores)
    token_pr_auc = average_precision_score(y_test, scores)
    sentence_lengths = sentence_info["lengths"]
    sentence_labels = sentence_info["labels"]
    sentence_scores_mean, sentence_scores_max = [], []
    start = 0
    for length in sentence_lengths:
        end = start + length
        sent_scores = scores[start:end]
        sentence_scores_mean.append(np.mean(sent_scores))
        sentence_scores_max.append(np.max(sent_scores))
        start = end
    return (token_roc_auc, token_pr_auc,
            roc_auc_score(sentence_labels, sentence_scores_mean),
            average_precision_score(sentence_labels, sentence_scores_mean),
            roc_auc_score(sentence_labels, sentence_scores_max),
            average_precision_score(sentence_labels, sentence_scores_max))


def build_detector(detector_class, params, seed):
    params = dict(params)
    valid_args = inspect.signature(detector_class.__init__).parameters
    if "random_state" in valid_args and "random_state" not in params:
        params["random_state"] = seed
    elif "seed" in valid_args and "seed" not in params:
        params["seed"] = seed
    return detector_class(**params)
# -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("detector_class", [KNN, IForest])
def test_same_numbers_as_tokencore(detector_class):
    rng = np.random.RandomState(0)
    docs = [rng.randn(rng.randint(3, 15), 6) for _ in range(80)]
    labels = []
    for i, d in enumerate(docs):
        lab = np.zeros(len(d), dtype=int)
        if i % 4 == 0:
            d[0] += 3
            lab[0] = 1
        labels.append(lab)
    train, test, test_labels = docs[:40], docs[40:], labels[40:]
    info = {"lengths": [len(d) for d in test], "labels": [int(l.any()) for l in test_labels]}
    seeds = (0, 1, 2)

    ref = np.array([evaluate_model_token_score_aggregation(build_detector(detector_class, {}, s),
                                                           np.vstack(train), np.vstack(test),
                                                           np.concatenate(test_labels), info)
                    for s in seeds])
    res = evaluate(TokenDetector(detector_class()), train, test, token_labels=test_labels, seeds=seeds)
    ours = np.array([[r["auroc"], r["ap"]] for r in res["runs"]["token"]])
    ours = np.hstack([ours,
                      [[r["auroc"], r["ap"]] for r in res["runs"]["document[mean]"]],
                      [[r["auroc"], r["ap"]] for r in res["runs"]["document[max]"]]])
    np.testing.assert_allclose(ours, ref, atol=1e-12)
