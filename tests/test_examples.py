"""The example scripts run end to end (on toy data and a tiny local model)."""
import importlib.util
import os

import numpy as np
import pytest

from pytextad.datasets import TextADDataset

from conftest import DEV, WORDS

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXAMPLES, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy(seed, n=40, n_anom=8):
    rng = np.random.RandomState(seed)
    tokens, labels = [], []
    for i in range(n):
        t = list(rng.choice(WORDS[:5], rng.randint(4, 9)))
        lab = [0] * len(t)
        if i >= n - n_anom:                     # anomalies: words never seen in normal documents
            t[1], lab[1] = "player score", 1
        tokens.append(t)
        labels.append(lab)
    return TextADDataset.from_lists(tokens, labels, name=f"toy{seed}")


def test_benchmark_example(tiny_bert, tmp_path, monkeypatch):
    pytest.importorskip("pyod")
    bench = _load("benchmark")
    path, _ = tiny_bert
    toys = {"restaurant_review": _toy(0), "grammar_correction": _toy(1)}
    monkeypatch.setattr(bench, "load_dataset", lambda name, retokenize=False: toys[name])
    out = tmp_path / "out"
    bench.main(["--datasets", "restaurant_review", "grammar_correction",
                "--detectors", "TokenCore", "SIK", "pyod:ECOD", "NoSuchDetector",
                "TokenCore(n_neighbors=3, aggregation='mean')", "SIK(max_samples=4)", "SIK(no_such_param=1)",
                "--model", path, "--seeds", "0", "--device", DEV,
                "--cache", str(tmp_path / "cache"), "--out", str(out)])

    import csv
    rows = list(csv.DictReader(open(out / "results.csv", encoding="utf-8")))
    ok = [r for r in rows if not r["error"]]
    assert {r["detector"] for r in ok} == {"TokenCore", "SIK", "pyod:ECOD",
                                           "TokenCore(n_neighbors=3, aggregation='mean')", "SIK(max_samples=4)"}
    assert {r["dataset"] for r in ok} == {"restaurant_review", "grammar_correction"}
    assert {r["level"] for r in ok} == {"document", "document[max]", "document[mean]", "token"}
    failed = [r for r in rows if r["error"]]
    assert len(failed) == 4
    assert {r["detector"] for r in failed} == {"NoSuchDetector", "SIK(no_such_param=1)"}
    assert all("no_such_param" in r["error"] for r in failed if r["detector"].startswith("SIK"))
    md = (out / "results.md").read_text(encoding="utf-8")
    assert "### restaurant_review" in md and "| NoSuchDetector | failed" in md
    tokencore = [float(r["mean"]) for r in ok
                 if r["detector"] == "TokenCore" and r["level"] == "token" and r["metric"] == "auroc"]
    assert min(tokencore) > 0.9                  # the unseen words are easy to find
    assert len(os.listdir(tmp_path / "cache" / os.path.basename(os.path.normpath(path)))) == 4


def test_benchmark_split_is_the_paper_split():
    bench = _load("benchmark")
    ds = _toy(0)
    train, test = bench.split(ds, 0.5, 42)
    np.random.seed(42)                            # split_anomaly_data in the TokenCore code
    normal = np.where(ds.labels == 0)[0]
    idx = np.random.choice(normal, size=int(len(normal) * 0.5), replace=False)
    assert train.tokens == ds.subset(idx).tokens
    assert len(train) + len(test) == len(ds) and test.labels.sum() == ds.labels.sum()


def test_benchmark_whole_dataset_and_retokenize(tiny_bert, tmp_path, monkeypatch):
    bench = _load("benchmark")
    ds = _toy(0)
    train, test = bench.split(ds, 0, 42)
    assert train is ds and test is ds                     # contaminated setting
    path, _ = tiny_bert
    calls = []

    def load(name, retokenize=False):
        calls.append(retokenize)
        return _toy(0).retokenize() if retokenize else _toy(0)
    monkeypatch.setattr(bench, "load_dataset", load)
    bench.main(["--datasets", "restaurant_review", "--detectors", "TokenCore", "--model", path,
                "--seeds", "0", "--train_ratio", "0", "--retokenize", "--device", DEV,
                "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "out")])
    assert calls == [True]
    cached = os.listdir(tmp_path / "cache" / os.path.basename(os.path.normpath(path)))
    assert cached and all(f.startswith("restaurant_review_rt_r0.0") for f in cached)
    assert "TokenCore" in (tmp_path / "out" / "results.md").read_text(encoding="utf-8")


def test_benchmark_detector_specifications():
    bench = _load("benchmark")
    assert bench.parse_spec("SIK") == ("SIK", {})
    assert bench.parse_spec("pyod:IForest(n_estimators=200)") == ("pyod:IForest", {"n_estimators": 200})
    det = bench.make_detector("TokenCore(n_neighbors=5, aggregation='mean')")
    assert (det.n_neighbors, det.aggregation) == (5, "mean")
    assert bench.make_detector("SIK(device='cpu')", device="cuda").device == "cpu"   # explicit wins
    for bad in ["SIK(8)", "SIK(x=open('f'))", "SIK(", "__import__('os').system('ls')"]:
        with pytest.raises(ValueError, match="cannot read detector"):
            bench.parse_spec(bad)


def test_tutorial_notebook_runs(tiny_bert, monkeypatch):
    """Every code cell of examples/tutorial.ipynb runs (tiny model and toy data instead of
    BERT and the Hub dataset)."""
    pytest.importorskip("pyod")
    import json
    from pytextad import datasets
    path, _ = tiny_bert
    monkeypatch.setattr(datasets, "load_dataset", lambda name: _toy(0, n=60, n_anom=12))
    nb = json.load(open(os.path.join(EXAMPLES, "tutorial.ipynb"), encoding="utf-8"))
    ns = {}
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert cells[0].startswith("!pip install")
    for src in cells[1:]:
        # repr() keeps a Windows path (backslashes) a valid string literal
        exec(compile(src.replace('"bert-base-uncased"', repr(path)), "tutorial", "exec"), ns)
    assert ns["highlight"] is not None and len(ns["docs"]) == 5
    assert set(ns["results"]) == {"TokenCore", "SIK", "IForest", "ECOD"}