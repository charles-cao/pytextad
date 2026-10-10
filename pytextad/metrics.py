"""Evaluation for document-level and token-level anomaly detection.

Conventions (the same as the TokenCore benchmark code):
  * the anomaly class is the positive class; higher scores mean more anomalous
  * a document is anomalous if any of its tokens is anomalous
  * token-level metrics pool all test tokens of all documents together
  * document scores of a token-level detector are its token scores aggregated per
    document ("max", "mean" or "topk")
"""

import copy
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .utils.rng import preserve_rng


# ----------------------------------------------------------------------------- basic metrics
def fpr_at_tpr(y_true, scores, tpr_level=0.95):
    """False-positive rate at the first threshold whose true-positive rate reaches ``tpr_level``."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(fpr[np.searchsorted(tpr, tpr_level, side="left")])


def document_metrics(y_true, scores):
    """AUROC, AP and FPR@95 for one score per document."""
    y_true, scores = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape:
        raise ValueError(f"{len(y_true)} labels but {len(scores)} scores")
    if np.isnan(scores).any():
        raise ValueError("document scores contain NaN")
    return {"auroc": float(roc_auc_score(y_true, scores)),
            "ap": float(average_precision_score(y_true, scores)),
            "fpr95": fpr_at_tpr(y_true, scores)}


def token_metrics(token_labels, token_scores):
    """AUROC, AP and FPR@95 over all tokens of all documents (pooled).

    Parameters
    ----------
    token_labels : list of array-like
        One 0/1 array per document.
    token_scores : list of array-like
        One score array per document, of the same lengths. Tokens with a NaN score (for
        example words cut off by truncation) are left out and counted in ``n_ignored``.

    Returns
    -------
    dict
        ``auroc``, ``ap``, ``fpr95`` and ``n_ignored``.
    """
    if len(token_labels) != len(token_scores):
        raise ValueError(f"{len(token_labels)} label lists but {len(token_scores)} score lists")
    for i, (l, s) in enumerate(zip(token_labels, token_scores)):
        if len(l) != len(s):
            raise ValueError(f"document {i}: {len(l)} token labels but {len(s)} token scores")
    y = np.concatenate([np.asarray(l, dtype=int) for l in token_labels])
    s = np.concatenate([np.asarray(x, dtype=float) for x in token_scores])
    ok = ~np.isnan(s)
    out = document_metrics(y[ok], s[ok])
    out["n_ignored"] = int((~ok).sum())
    return out


def document_labels(token_labels):
    """1 if any token of the document is anomalous."""
    return np.array([int(np.any(np.asarray(l) == 1)) for l in token_labels])


def aggregate(token_scores, how="max", k=0.1):
    """Document scores from token scores.

    Parameters
    ----------
    token_scores : list of array-like
        One score array per document; NaN scores are ignored.
    how : {"max", "mean", "topk"}, default="max"
        "topk" is the mean of the top ``k`` fraction of tokens (at least one).
    k : float, default=0.1
        Fraction used by "topk".

    Returns
    -------
    numpy.ndarray
        One score per document.
    """
    out = []
    for s in token_scores:
        s = np.asarray(s, dtype=float)
        s = s[~np.isnan(s)]
        if len(s) == 0:
            out.append(np.nan)
        elif how == "max":
            out.append(s.max())
        elif how == "mean":
            out.append(s.mean())
        elif how == "topk":
            m = max(1, int(np.ceil(k * len(s))))
            out.append(np.sort(s)[-m:].mean())
        else:
            raise ValueError("how must be 'max', 'mean' or 'topk'")
    return np.array(out)


# ----------------------------------------------------------------------------- full evaluation
def _seeded_copy(detector, seed):
    shared = {}
    for attr in ("embedder",):          # never deep-copy a loaded language model
        if getattr(detector, attr, None) is not None:
            shared[id(getattr(detector, attr))] = getattr(detector, attr)
    det = copy.deepcopy(detector, memo=shared)
    if hasattr(det, "random_state"):
        det.random_state = seed
    inner = getattr(det, "detector", None)
    if inner is not None and hasattr(inner, "random_state"):
        inner.random_state = seed
    return det


def _summary(runs):
    keys = runs[0].keys()
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in keys}


def evaluate(detector, X_train, X_test, doc_labels=None, token_labels=None, y_train=None,
             seeds=(0, 1, 2), aggregations=("max", "mean")):
    """Fit ``detector`` once per seed on ``X_train`` and evaluate it on ``X_test``.

    Which data are used for training (normal documents only, contaminated data, or the
    test set itself) is decided by the caller.

    Parameters
    ----------
    detector : BaseTextDetector
        Unfitted detector; a fresh copy is fitted for every seed, with ``random_state`` set
        on the detector and, for wrappers, on the wrapped vector detector. The global random
        generators are seeded with it before fitting.
    X_train, X_test
        Inputs accepted by the detector.
    doc_labels : array-like, optional
        One 0/1 label per test document; derived from ``token_labels`` if omitted.
    token_labels : list of arrays, optional
        One 0/1 array per test document, aligned with what ``token_scores`` returns
        (words for DATE and word-level embeddings, the embedded units otherwise).
    y_train : array-like, optional
        Passed to ``fit``; only semi-supervised detectors (FATE) use it.
    seeds : tuple of int
        Random seeds, one fit per seed.
    aggregations : tuple of str
        For token-level detectors, document scores are also computed by aggregating token
        scores in each of these ways ("max", "mean", "topk").

    Returns
    -------
    dict
        ``"document"``: (mean, std) of each document metric from ``decision_function``;
        ``"document[<agg>]"``: the same from aggregated token scores (token detectors);
        ``"token"``: token metrics (token detectors given ``token_labels``);
        ``"runs"``: the per-seed values.
    """
    if doc_labels is None:
        if token_labels is None:
            raise ValueError("give doc_labels or token_labels")
        doc_labels = document_labels(token_labels)
    token_capable = getattr(detector, "supports_token", False)
    if token_labels is not None and not token_capable:
        warnings.warn(f"{type(detector).__name__} has no token scores; token metrics skipped", stacklevel=2)
    runs = {"document": []}
    for seed in seeds:
        det = _seeded_copy(detector, seed)
        det.fit(X_train, y_train) if y_train is not None else det.fit(X_train)
        # both scorings start from the generator state right after fit, as in a plain
        # fit / decision_function script (matters for detectors whose scoring is random)
        with preserve_rng():
            runs["document"].append(document_metrics(doc_labels, det.decision_function(X_test)))
        if token_capable:
            ts = det.token_scores(X_test)
            for how in aggregations:
                runs.setdefault(f"document[{how}]", []).append(document_metrics(doc_labels, aggregate(ts, how)))
            if token_labels is not None:
                runs.setdefault("token", []).append(token_metrics(token_labels, ts))
    result = {k: _summary(v) for k, v in runs.items()}
    result["runs"] = runs
    return result


def format_results(results, digits=4):
    """Readable table of one or several ``evaluate`` outputs: {name: result}."""
    if "runs" in results:
        results = {"": results}
    lines = []
    for name, res in results.items():
        for level, metrics in res.items():
            if level == "runs":
                continue
            cells = [f"{m} {v[0]:.{digits}f}±{v[1]:.{digits}f}" for m, v in metrics.items() if m != "n_ignored"]
            lines.append(f"{name:<16} {level:<18} " + "   ".join(cells))
    return "\n".join(lines)
