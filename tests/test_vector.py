"""Vector detectors: identical to the authors' code, reproducible, and usable on tokens."""
import importlib.util
import os

import numpy as np
import pytest
import torch

from pytextad import SIK, TokenCore, TokenDetector
from pytextad.metrics import evaluate
from pytextad.models.vector.base import BaseVectorDetector

REF = os.path.join(os.path.dirname(__file__), "reference")
VENDOR = os.path.join(os.path.dirname(__file__), "..", "pytextad", "models", "vector", "_vendor")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REF, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _data(seed=0, n=300, d=8):
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n, d)
    X_test = np.vstack([rng.randn(n // 2, d), rng.randn(20, d) * 3 + 2])
    return X_train, X_test


@pytest.mark.parametrize("vendored, original", [("tokencore.py", "tokencore_original.py"),
                                                ("sik.py", "sik_original.py")])
def test_vendored_code_is_the_original(vendored, original):
    with open(os.path.join(REF, original), encoding="utf-8") as f:
        ref = f.read().splitlines()
    # the vendored file is the original preceded by a two-line provenance header
    with open(os.path.join(VENDOR, vendored), encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert lines[2:] == ref


@pytest.mark.parametrize("k, agg", [(1, "max"), (5, "mean"), (5, "median")])
def test_tokencore_matches_original(k, agg):
    X_train, X_test = _data()
    ref = _load("tokencore_original").TokenCore(n_neighbors=k, aggregation=agg).fit(X_train).decision_function(X_test)
    ours = TokenCore(n_neighbors=k, aggregation=agg).fit(X_train)
    np.testing.assert_array_equal(ours.decision_function(X_test), ref)
    assert ours.decision_scores_.shape == (len(X_train),) and ours.predict(X_test).shape == (len(X_test),)


def test_tokencore_is_pyod_1nn():
    pyod_knn = pytest.importorskip("pyod.models.knn")
    X_train, X_test = _data()
    knn = pyod_knn.KNN(n_neighbors=1, method="largest").fit(X_train)
    np.testing.assert_allclose(TokenCore().fit(X_train).decision_function(X_test),
                               knn.decision_function(X_test), atol=1e-12)


@pytest.mark.parametrize("novelty, sparse", [(True, False), (False, False), (True, True), (False, True)])
def test_sik_matches_original(novelty, sparse):
    from sklearn.base import BaseEstimator

    class Original(_load("sik_original").SIK, BaseEstimator):   # runnable on scikit-learn >= 1.8
        pass

    X_train, X_test = _data(1)
    for seed in (0, 7):
        ref = Original(novelty=novelty, sparse=sparse, random_state=seed)
        ref = ref.fit(X_train).decision_function(X_test)
        ours = SIK(novelty=novelty, sparse=sparse, random_state=seed).fit(X_train).decision_function(X_test)
        np.testing.assert_array_equal(ours, ref)


def test_sik_seed_matters():
    X_train, X_test = _data(2)
    a = SIK(random_state=0).fit(X_train).decision_function(X_test)
    b = SIK(random_state=0).fit(X_train).decision_function(X_test)
    c = SIK(random_state=1).fit(X_train).decision_function(X_test)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# ----------------------------------------------------------------------------- RNG conventions
class _NoisyOriginal:
    """Stand-in for published code that uses the global generators in __init__, fit and
    decision_function (like the diffusion baselines) and prints while training."""

    def __init__(self, n_features):
        self.w = torch.randn(n_features)

    def fit(self, X):
        print("epoch 1 loss 0.5")
        self.mu = X.mean(0) + np.random.randn(X.shape[1]) * 0.01
        return self

    def decision_function(self, X):
        noise = torch.randn(len(X)).numpy() * 0.01
        return np.linalg.norm(X - self.mu, axis=1) + float(self.w.sum()) * 1e-3 + noise


class _Noisy(BaseVectorDetector):
    def __init__(self, random_state=0, verbose=False):
        super().__init__(0.1, random_state, "cpu", verbose)

    def _build(self, n_features):
        return _NoisyOriginal(n_features)


def _benchmark_script(X_train, X_test, seed):
    """What the TokenCore / SVEAD benchmark scripts do."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    clf = _NoisyOriginal(X_train.shape[1])
    clf.fit(X_train)
    return clf.decision_function(X_test)


def test_same_numbers_as_a_benchmark_script(capsys):
    X_train, X_test = _data(3)
    for seed in (42, 123):
        ref = _benchmark_script(X_train, X_test, seed)
        capsys.readouterr()
        det = _Noisy(random_state=seed).fit(X_train)       # also scores the training data
        np.testing.assert_array_equal(det.decision_function(X_test), ref)
        assert capsys.readouterr().out == ""               # training log suppressed
    _Noisy(verbose=True).fit(X_train)
    assert "epoch 1" in capsys.readouterr().out


def test_token_detector_and_evaluate_reproduce_the_script():
    rng = np.random.RandomState(0)
    train = [rng.randn(rng.randint(3, 9), 6) for _ in range(30)]
    test = [rng.randn(rng.randint(3, 9), 6) for _ in range(20)]
    labels = [np.r_[1, np.zeros(len(t) - 1, int)] if i % 4 == 0 else np.zeros(len(t), int)
              for i, t in enumerate(test)]
    seeds = (42, 123, 456)
    res = evaluate(TokenDetector(_Noisy()), train, test, token_labels=labels, seeds=seeds)
    from sklearn.metrics import roc_auc_score
    for k, seed in enumerate(seeds):
        ref = _benchmark_script(np.vstack(train), np.vstack(test), seed)
        assert res["runs"]["token"][k]["auroc"] == roc_auc_score(np.concatenate(labels), ref)
    # SIK through the wrapper: the wrapper's seed reaches the partitions
    det = TokenDetector(SIK(random_state=None), random_state=5).fit(train)
    np.testing.assert_array_equal(np.concatenate(det.token_scores(test)),
                                  SIK(random_state=5).fit(np.vstack(train)).decision_function(np.vstack(test)))


def test_rsrae_old_import_path():
    from pytextad.models.rsrae import RSRAE as old, _RSRAENet  # noqa: F401
    from pytextad.models.vector.rsrae import RSRAE as new
    assert old is new


class _GlobalRNGDetector:
    """Like a deep PyOD model: has random_state but draws from the global generators."""

    def __init__(self):
        self.random_state = None

    def fit(self, X):
        self.inner = _NoisyOriginal(X.shape[1]).fit(X)
        return self

    def decision_function(self, X):
        return self.inner.decision_function(X)


def test_wrappers_seed_the_global_generators():
    from pytextad import DocumentDetector
    X_train, X_test = _data(4)
    for seed in (42, 123):
        ref = _benchmark_script(X_train, X_test, seed)
        det = DocumentDetector(_GlobalRNGDetector(), random_state=seed).fit(X_train)
        np.testing.assert_array_equal(det.decision_function(X_test), ref)


# ----------------------------------------------------------------------------- ADERH, TCCM
def test_vendored_aderh_is_the_official_file():
    import hashlib
    with open(os.path.join(VENDOR, "aderh.py"), "rb") as f:
        body = b"".join(f.readlines()[3:])           # drop the 3-line provenance header
    # sha256 of aderh/_aderh.py at github.com/Walid10010/ADERH commit 5942c45
    assert hashlib.sha256(body).hexdigest() == "b8e0529c2323f98f3b75de68ee623963d1dbf5fd7bdf832b49cfef27bc4d2c15"


def test_aderh_matches_the_code_used_in_the_paper():
    from pytextad import ADERH
    original = _load("aderh_original").ADERH          # the authors' original ADERH.py
    X_train, X_test = _data(5)
    for seed in (0, 42):
        ref = original(random_state=seed).fit(X_train).decision_function(X_test)
        ours = ADERH(random_state=seed).fit(X_train).decision_function(X_test)
        np.testing.assert_array_equal(ours, ref)


def _tccm_reference():
    """The TCCM file used in the SVEAD benchmark, run on the CPU (its only change to the
    official code is .to('cuda'))."""
    with open(os.path.join(REF, "tccm_original.py"), encoding="utf-8") as f:
        src = f.read().replace(".to('cuda')", "")
    mod = type(os)("tccm_original_cpu")
    exec(compile(src, "tccm_original.py", "exec"), mod.__dict__)
    return mod.TCCM


def test_tccm_matches_the_benchmark_script():
    from pytextad import TCCM
    original = _tccm_reference()
    X_train, X_test = _data(6)
    for seed in (42, 123):
        np.random.seed(seed)
        torch.manual_seed(seed)
        clf = original(n_features=X_train.shape[1], epochs=3)
        clf.fit(X_train)
        ref = clf.decision_function(X_test)
        ours = TCCM(epochs=3, random_state=seed, device="cpu").fit(X_train).decision_function(X_test)
        np.testing.assert_array_equal(ours, ref.astype(float))
