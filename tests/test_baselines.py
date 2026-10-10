"""Third-party baselines: same scores as the copies used in the benchmark scripts."""
import contextlib
import io
import os
import random

import numpy as np
import pytest
import torch

import pytextad
from pytextad.models.vector import (DAGMM, DDAE, DDPM, DRL, DROCC, DTECategorical, DTEGaussian,
                                    DTEInverseGamma, DTENonParametric, GANomaly, GOAD, ICL, MCM,
                                    NormalizingFlow, SLAD)

REF = os.path.join(os.path.dirname(__file__), "reference")
CPU = torch.device("cpu")


def _original(name, fixes=()):
    """Load tests/reference/<name>_original.py (the copies used in the benchmark scripts);
    ``fixes`` are (old, new) replacements needed to run them on current PyTorch."""
    with open(os.path.join(REF, name + "_original.py"), encoding="utf-8") as f:
        src = f.read()
    for old, new in fixes:
        assert old in src
        src = src.replace(old, new)
    mod = type(os)(name + "_original")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, name + "_original.py", "exec"), mod.__dict__)
    return mod


def _data(seed=0, n=300, d=8):
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n, d)
    X_test = np.vstack([rng.randn(60, d), rng.randn(12, d) * 3 + 2])
    return X_train, X_test


def _script(make, X_train, X_test, seed, fit=lambda c, X: c.fit(X)):
    """What the benchmark scripts do: seed, build, fit, score."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        clf = make()
        fit(clf, X_train)
        return np.asarray(clf.decision_function(X_test), dtype=float).reshape(-1)


CHOLESKY = [("torch.cholesky(a, False)", "torch.linalg.cholesky(a)")]
F32 = lambda c, X: c.fit(X.astype(np.float32))  # noqa: E731

CASES = {
    "dagmm": (lambda m, seed: m.DAGMM(num_epochs=2, batch_size=64, device=CPU), CHOLESKY, None,
              lambda: DAGMM(num_epochs=2, batch_size=64, detach_loss=True, device="cpu")),
    "ganomaly": (lambda m, seed: m.GANomaly(epochs=2, device=CPU), (), None,
                 lambda: GANomaly(epochs=2, device="cpu")),
    "drocc": (lambda m, seed: m.DROCC(device=CPU), (), None, lambda: DROCC(device="cpu")),
    "goad": (lambda m, seed: m.GOAD(n_rots=16, device=CPU), (), None, lambda: GOAD(n_rots=16, device="cpu")),
    "icl": (lambda m, seed: m.ICL(num_epochs=3, device=CPU), (), None, lambda: ICL(num_epochs=3, device="cpu")),
    "mcm": (lambda m, seed: m.MCM(8, epochs=2, device=CPU), (), None, lambda: MCM(epochs=2, device="cpu")),
    "slad": (lambda m, seed: m.SLAD(epochs=2, device=CPU, random_state=seed), (), None,
             lambda: SLAD(epochs=2, device="cpu")),
    "normalizing_flow": (lambda m, seed: m.NormalizingFlow(device=CPU), (), lambda c, X: c.fit(X.astype(np.float32), epochs=3),
                         lambda: NormalizingFlow(epochs=3, legacy_score=True, device="cpu")),
    "dte": (lambda m, seed: m.DTECategorical(epochs=2, device=CPU), (), F32,
            lambda: DTECategorical(epochs=2, device="cpu")),
    "dte_inverse_gamma": (lambda m, seed: m.DTEInverseGamma(epochs=2, device=CPU), (), F32,
                          lambda: DTEInverseGamma(epochs=2, device="cpu")),
    "dte_gaussian": (lambda m, seed: m.DTEGaussian(epochs=2, device=CPU), (), F32,
                     lambda: DTEGaussian(epochs=2, device="cpu")),
    "dte_nonparametric": (lambda m, seed: m.DTENonParametric(), (), None, lambda: DTENonParametric()),
    "ddpm": (lambda m, seed: m.DDPM(epochs=2, reconstruction_t=20, device=CPU), (), F32,
             lambda: DDPM(epochs=2, reconstruction_t=20, device="cpu")),
    "ddae": (lambda m, seed: m.DDAE(8, epochs=2), (), None, lambda: DDAE(epochs=2, device="cpu")),
    "drl": (lambda m, seed: m.DRL(8, epochs=2), (), None, lambda: DRL(epochs=2, device="cpu")),
}
FILE = {"dte_inverse_gamma": "dte", "dte_gaussian": "dte"}


@pytest.mark.parametrize("name", sorted(CASES))
def test_same_scores_as_the_benchmark_copy(name):
    needs = {"dte_nonparametric": "matplotlib", "icl": "pandas"}   # imported by the original code
    if name in needs:
        pytest.importorskip(needs[name])
    make_ref, fixes, fit, make_ours = CASES[name]
    mod = _original(FILE.get(name, name), fixes)
    X_train, X_test = _data(1)
    for seed in (42, 123):
        ref = _script(lambda: make_ref(mod, seed), X_train, X_test, seed, fit or (lambda c, X: c.fit(X)))
        det = make_ours()
        det.random_state = seed
        ours = det.fit(X_train).decision_function(X_test)
        np.testing.assert_allclose(ours, ref, rtol=0, atol=0, err_msg=name)


def test_dagmm_detached_loss_never_trains_and_the_fix_does():
    X_train, _ = _data(2, n=512)

    def weights(det):
        return torch.cat([p.detach().flatten() for p in det.model_.model_trainer.model.parameters()])

    torch.manual_seed(0)
    from pytextad.models.vector._vendor import dagmm as vendor
    torch.manual_seed(0)
    init = vendor.DAGMM_Model(8, 4, 1)
    init.apply(vendor.weights_init_normal)
    w0 = torch.cat([p.detach().flatten() for p in init.parameters()])
    legacy = DAGMM(num_epochs=2, batch_size=64, detach_loss=True, device="cpu", random_state=None)
    fixed = DAGMM(num_epochs=2, batch_size=64, device="cpu", random_state=None)
    torch.manual_seed(0)
    legacy.fit(X_train)
    torch.manual_seed(0)
    fixed.fit(X_train)
    assert torch.equal(weights(legacy), w0)               # untouched by training
    assert not torch.allclose(weights(fixed), w0)        # trained
    assert vendor.ComputeLoss.detach_loss is False       # switch restored


def test_normalizing_flow_score_is_the_negative_log_likelihood():
    X_train, X_test = _data(3)
    det = NormalizingFlow(epochs=3, device="cpu").fit(X_train)
    flow = det.model_.flow
    with torch.no_grad():
        z, log_det = flow(torch.tensor(X_test, dtype=torch.float32))
        nll = -(torch.distributions.Normal(0, 1).log_prob(z).sum(1) + log_det.reshape(-1)).numpy()
    np.testing.assert_allclose(det.decision_function(X_test), nll, rtol=1e-6)
    # a vector's score does not depend on the other vectors scored with it ...
    np.testing.assert_allclose(det.decision_function(X_test[-1:]), det.decision_function(X_test)[-1:], rtol=1e-6)
    # ... but it does in the original scoring
    legacy = NormalizingFlow(epochs=3, legacy_score=True, device="cpu").fit(X_train)
    assert not np.isclose(legacy.decision_function(X_test[-1:])[0], legacy.decision_function(X_test)[-1])
    # far vectors get higher scores with the corrected score
    assert det.decision_function(X_test[-12:]).mean() > det.decision_function(X_test[:60]).mean()


def test_import_prints_nothing():
    import importlib
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-c", "import pytextad"], capture_output=True, text=True)
    assert out.stdout == ""
