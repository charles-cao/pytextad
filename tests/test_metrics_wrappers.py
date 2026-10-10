import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from pytextad import CVDD, RSRAE, DocumentDetector, TokenDetector, TokenEmbedder, SentenceEmbedder, align_labels
from pytextad.metrics import (aggregate, document_labels, document_metrics, evaluate, format_results,
                              fpr_at_tpr, token_metrics)

from conftest import DEV, WORDS, make_texts


class MeanDistance:
    """Minimal embedding detector: distance to the training mean."""
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X):
        self.mu_ = X.mean(0)
        return self

    def decision_function(self, X):
        return np.linalg.norm(X - self.mu_, axis=1)


# ----------------------------------------------------------------------------- metrics
def test_document_metrics_match_sklearn():
    rng = np.random.RandomState(0)
    y, s = rng.randint(0, 2, 300), rng.randn(300)
    m = document_metrics(y, s)
    assert m["auroc"] == pytest.approx(roc_auc_score(y, s))
    assert m["ap"] == pytest.approx(average_precision_score(y, s))


def test_fpr95_by_hand():
    # anomalies (1) scored 9..0; normals scored 4.5, 3.5, 2.5, 1.5, 0.5
    y = [1] * 10 + [0] * 5
    s = list(range(9, -1, -1)) + [4.5, 3.5, 2.5, 1.5, 0.5]
    # TPR reaches 95 % only when all 10 anomalies are flagged (threshold 0): all 5 normals pass
    assert fpr_at_tpr(y, s) == pytest.approx(1.0)
    assert fpr_at_tpr([1, 1, 0, 0], [4, 3, 2, 1]) == 0.0


def test_token_metrics_pool_tokens_and_skip_nan():
    labels = [np.array([0, 1, 0]), np.array([0, 0])]
    scores = [np.array([0.1, 0.9, 0.2]), np.array([0.3, np.nan])]
    m = token_metrics(labels, scores)
    assert m["auroc"] == pytest.approx(roc_auc_score([0, 1, 0, 0], [0.1, 0.9, 0.2, 0.3]))
    assert m["n_ignored"] == 1
    with pytest.raises(ValueError):
        token_metrics([np.array([0, 1])], [np.array([0.2])])


def test_aggregate_and_document_labels():
    s = [np.array([1.0, 3.0, 2.0, 0.0]), np.array([5.0])]
    np.testing.assert_allclose(aggregate(s, "max"), [3, 5])
    np.testing.assert_allclose(aggregate(s, "mean"), [1.5, 5])
    np.testing.assert_allclose(aggregate(s, "topk", k=0.5), [2.5, 5])
    np.testing.assert_array_equal(document_labels([[0, 0], [0, 1]]), [0, 1])


# ----------------------------------------------------------------------------- wrappers
def _token_data(seed=0):
    rng = np.random.RandomState(seed)
    X = [rng.randn(rng.randint(3, 12), 8) for _ in range(60)]
    labels = []
    for i, x in enumerate(X):
        lab = np.zeros(len(x), dtype=int)
        if i % 3 == 0:                                   # one shifted token in every third doc
            x[1] += 6
            lab[1] = 1
        labels.append(lab)
    return X, labels


def test_token_detector_scores_and_aggregation():
    X, labels = _token_data()
    det = TokenDetector(MeanDistance(), aggregation="mean").fit(X[:30])
    ts = det.token_scores(X[30:])
    assert [len(t) for t in ts] == [len(x) for x in X[30:]]
    np.testing.assert_allclose(det.decision_function(X[30:]), [t.mean() for t in ts])
    assert det.supports_token and len(det.decision_scores_) == 30


def test_document_detector_on_vectors_and_texts(tiny_bert):
    rng = np.random.RandomState(0)
    V = rng.randn(50, 8)
    det = DocumentDetector(MeanDistance()).fit(V)
    assert det.decision_function(V[:5]).shape == (5,) and not det.supports_token
    with pytest.raises(NotImplementedError):
        det.token_scores(V)
    path, _ = tiny_bert
    det = DocumentDetector(MeanDistance(), embedder=SentenceEmbedder(path, device=DEV))
    det.fit(make_texts(20, WORDS[:5], 0))
    assert det.decision_function(make_texts(4, WORDS, 1)).shape == (4,)


def test_token_detector_with_word_embedder_restores_word_positions(tiny_bert):
    path, _ = tiny_bert
    emb = TokenEmbedder(path, word_pooling="max", max_length=12, device=DEV)
    docs = [t.split() for t in make_texts(10, WORDS, 2)]
    det = TokenDetector(MeanDistance(), embedder=emb).fit(docs)
    with pytest.warns(UserWarning):
        ts = det.token_scores([["stock"] * 20])           # 10 words survive truncation
    assert len(ts[0]) == 20 and np.isnan(ts[0][10:]).all() and not np.isnan(ts[0][:10]).any()


def test_subword_scores_are_combined_per_word(tiny_bert):
    """Default: every sub-word is scored and scores are combined per given word. Grouping two
    words into one item ("bank price") changes neither the vectors nor the scores of the
    sub-words; the item's score is the max of the two words' scores."""
    path, _ = tiny_bert
    emb = TokenEmbedder(path, device=DEV)                      # sub-word vectors
    train = [t.split() for t in make_texts(20, WORDS, 3)]
    det = TokenDetector(MeanDistance(), embedder=emb).fit(train)
    split = [["stock", "bank", "price", "team"]]
    grouped = [["stock", "bank price", "team"]]
    s_split, s_grouped = det.token_scores(split)[0], det.token_scores(grouped)[0]
    assert len(s_split) == 4 and len(s_grouped) == 3
    assert s_grouped[1] == max(s_split[1], s_split[2])
    assert s_grouped[0] == s_split[0] and s_grouped[2] == s_split[3]
    assert np.array_equal(emb.transform(split)[0][0], emb.transform(grouped)[0][0])
    # "mean" combines by averaging
    det.subword_aggregation = "mean"
    assert np.isclose(det.token_scores(grouped)[0][1], np.mean(det.token_scores(split)[0][1:3]))


def test_precomputed_pair_matches_text_input(tiny_bert):
    path, _ = tiny_bert
    emb = TokenEmbedder(path, max_length=12, device=DEV)
    train = [t.split() for t in make_texts(20, WORDS, 4)]
    test = [t.split() for t in make_texts(6, WORDS, 5)] + [["stock"] * 20]   # last one truncated
    labels = [np.arange(len(d)) % 2 for d in test]
    a = TokenDetector(MeanDistance(), embedder=emb).fit(train)
    b = TokenDetector(MeanDistance()).fit(emb.transform(train))
    with pytest.warns(UserWarning):
        ta = a.token_scores(test)
    pair = emb.transform(test)
    tb = b.token_scores(pair)
    y = align_labels(labels, pair[1])
    for sa, sb, yy in zip(ta, tb, y):
        assert np.array_equal(sa[~np.isnan(sa)], sb) and len(yy) == len(sb)
    assert np.array_equal(a.decision_function(test[:6]), b.decision_function(emb.transform(test[:6])))


def test_rsrae_as_token_detector():
    X, _ = _token_data()
    det = TokenDetector(RSRAE(n_epochs=2, device=DEV)).fit(X[:30])
    assert det.decision_function(X[30:]).shape == (30,)


# ----------------------------------------------------------------------------- evaluate
def test_evaluate_token_and_document_levels():
    X, labels = _token_data()
    res = evaluate(TokenDetector(MeanDistance()), X[:30], X[30:], token_labels=labels[30:], seeds=(0, 1))
    assert set(res) == {"document", "document[max]", "document[mean]", "token", "runs"}
    assert res["token"]["auroc"][0] > 0.9                # the shifted tokens are easy
    assert res["document"] == res["document[max]"]       # decision_function uses "max"
    assert "token" in format_results({"MeanDistance": res})


def test_evaluate_document_only_detector_skips_tokens():
    X, labels = _token_data()
    V = np.stack([x.mean(0) for x in X])
    with pytest.warns(UserWarning, match="no token scores"):
        res = evaluate(DocumentDetector(MeanDistance()), V[:30], V[30:], token_labels=labels[30:], seeds=(0,))
    assert set(res) == {"document", "runs"}


def test_evaluate_reseeds_native_detectors():
    X, labels = _token_data()
    res = evaluate(CVDD(n_epochs=2, device=DEV), X[:30], X[30:], token_labels=labels[30:], seeds=(0, 1))
    assert len(res["runs"]["document"]) == 2 and "token" in res
