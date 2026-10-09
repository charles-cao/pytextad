"""Fast offline API tests (pytest). No downloads, no reference code, a few seconds on CPU.

The numerical equivalence with the original implementations lives in tests/verification/
and is only needed when the detector code changes.
"""
import os

import numpy as np
import pytest
import torch

from pytextad import CVDD, DATE, FATE, RSRAE

from conftest import DEV, WORDS, make_texts as _texts


def _check_api(det, Xtr, Xte):
    det.fit(Xtr)
    s = det.decision_function(Xte)
    assert s.shape == (len(Xte),) and np.isfinite(s).all()
    assert det.decision_scores_.shape == (len(Xtr),)
    assert det.labels_.mean() == pytest.approx(det.contamination, abs=0.05)
    assert set(np.unique(det.predict(Xte))) <= {0, 1}


def test_rsrae():
    rng = np.random.RandomState(0)
    Xtr = rng.randn(200, 16) * 0.1
    Xte = np.vstack([rng.randn(20, 16) * 0.1, rng.randn(20, 16)])
    det = RSRAE(n_epochs=5, device=DEV)
    _check_api(det, Xtr, Xte)
    assert det.activation_ == "tanh"


def test_cvdd_padding_invariance_and_tokens():
    rng = np.random.RandomState(0)
    X = [rng.randn(rng.randint(3, 20), 16).astype(np.float32) for _ in range(120)]
    det = CVDD(n_epochs=3, device=DEV)
    _check_api(det, X[:100], X[100:])
    batch = det.decision_function(X[:10])
    alone = np.array([det.decision_function([x])[0] for x in X[:10]])
    assert np.abs(batch - alone).max() < 1e-5            # padding must not change scores
    assert [len(t) for t in det.token_scores(X[:3])] == [len(x) for x in X[:3]]
    assert all(np.allclose(a.sum(1), 1) for a in det.attention(X[:3]))


def test_date(tiny_bert):
    _, tok = tiny_bert
    Xtr, Xte = _texts(60, WORDS[:5], 0), _texts(20, WORDS, 1)
    det = DATE(tokenizer=tok, max_len=16, n_masks=5, n_epochs=1, batch_size=8, device=DEV)
    _check_api(det, Xtr, Xte)
    wl = [t.split() for t in Xte[:3]]
    assert [len(s) for s in det.token_scores(wl)] == [len(w) for w in wl]


def test_fate_few_shot_and_unsupervised(tiny_bert):
    path, _ = tiny_bert
    X = _texts(40, WORDS[:5], 0) + _texts(4, WORDS[5:], 1)
    y = np.array([0] * 40 + [1] * 4)
    det = FATE(encoder=path, max_length=32, n_epochs=1, device=DEV)
    det.fit(X, y)
    assert det.few_shot_ and det.decision_function(X[:5]).shape == (5,)
    det2 = FATE(encoder=path, max_length=32, n_epochs=1, device=DEV).fit(X[:40])
    assert not det2.few_shot_
