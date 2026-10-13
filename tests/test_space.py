"""The online demo (space/app.py) works end to end (tiny model, toy data)."""
import importlib.util
import os

import numpy as np
import pytest

from pytextad import datasets

from test_examples import _toy

APP = os.path.join(os.path.dirname(__file__), "..", "space", "app.py")


def test_demo_detect(tiny_bert, monkeypatch):
    pytest.importorskip("gradio")
    path, _ = tiny_bert
    monkeypatch.setenv("PYTEXTAD_DEMO_MODEL", path)
    monkeypatch.setenv("PYTEXTAD_DEMO_N_REF", "40")
    monkeypatch.setattr(datasets, "load_dataset", lambda name: _toy(0, n=160, n_anom=10))
    spec = importlib.util.spec_from_file_location("app", APP)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    html, summary = app.detect("stock market player score bank", "Restaurant reviews")
    assert html.count("<span") == 5 and summary.startswith("**More anomalous than")
    assert "player" in summary or "score" in summary         # unseen words rank highest
    assert app.detect("   ", "SMS messages") == ("", "Type or paste a text.")
    _, summary = app.detect("player , score !", "Restaurant reviews")
    assert summary.endswith("Most anomalous words: player, score.") or summary.endswith("score, player.")
    det, ref_docs, lo, hi = app.reference("Restaurant reviews")
    assert len(ref_docs) == 100 and np.isfinite(ref_docs).all() and lo < hi


def test_demo_on_zerogpu_hardware(tiny_bert, monkeypatch):
    """On ZeroGPU, a @spaces.GPU function exists at startup and the demo still runs on the CPU."""
    pytest.importorskip("gradio")
    import sys
    import types
    marked = []
    fake = types.ModuleType("spaces")
    fake.GPU = lambda f: marked.append(f.__name__) or f
    monkeypatch.setitem(sys.modules, "spaces", fake)
    path, _ = tiny_bert
    monkeypatch.setenv("PYTEXTAD_DEMO_MODEL", path)
    monkeypatch.setenv("PYTEXTAD_DEMO_N_REF", "40")
    monkeypatch.setattr(datasets, "load_dataset", lambda name: _toy(0, n=160, n_anom=10))
    spec = importlib.util.spec_from_file_location("app_zerogpu", APP)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    assert marked == ["_zerogpu_startup_check"]
    assert app.embedder().device == "cpu"
    html, _ = app.detect("stock player", "Restaurant reviews")
    assert html.count("<span") == 2
