"""Global random-number-generator helpers.

Most published anomaly detectors draw their random numbers from the global NumPy and
PyTorch generators instead of a ``random_state`` argument. Seeding these generators before
a detector is built and fitted (as the TokenCore / SVEAD benchmark scripts do) is what
makes their results reproducible, so PyTextAD does the same.
"""
import contextlib
import random

import numpy as np
import torch


def seed_everything(seed):
    """Seed Python's ``random``, NumPy and PyTorch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def preserve_rng():
    """Run a block without changing the state of the global generators.

    Used when PyTextAD scores the training data after ``fit`` (for ``decision_scores_``):
    detectors whose scoring is random (diffusion models, for example) would otherwise see a
    different generator state when the test data are scored than in a plain
    ``fit(X_train); decision_function(X_test)`` script, and give different numbers.
    """
    py, npy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(npy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@contextlib.contextmanager
def silence(enabled=True):
    """Suppress ``print`` output of third-party code (training logs) unless ``enabled`` is False."""
    if not enabled:
        yield
        return
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        yield
