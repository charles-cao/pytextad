import json

import numpy as np
import pytest

from pytextad import TokenDetector, TokenEmbedder, datasets
from pytextad.datasets import TextADDataset, load_dataset, load_local
from pytextad.metrics import evaluate

from conftest import DEV

DOCS = [["the", "food", "was", "good", "."],
        ["it's", "not fresh", "at", "all", "!"],
        ["great", "staff"]]
LABELS = [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0]]
TEXTS = ["the food was good.", "it's not fresh at all!", "great staff"]


def _ds():
    return TextADDataset.from_lists(DOCS, LABELS, TEXTS, name="toy")


def test_views_and_subsets():
    ds = _ds()
    assert len(ds) == 3 and ds.labels.tolist() == [0, 1, 0]
    assert ds.normal_indices.tolist() == [0, 2] and ds.anomalous_indices.tolist() == [1]
    assert ds[1]["label"] == 1 and ds[1]["tokens"] == DOCS[1]
    for sub in (ds[[2, 0]], ds[np.array([False, False, True])], ds[2:]):
        assert sub.tokens[0] == DOCS[2] and sub.texts[0] == TEXTS[2]
    s = ds.stats()
    assert (s["n_tokens"], s["n_anomalous_tokens"], s["n_anomalous"]) == (12, 1, 1)


def test_validation():
    with pytest.raises(ValueError, match="2 words but 1 labels"):
        TextADDataset.from_lists([["a", "b"]], [[0]])
    with pytest.raises(ValueError, match="0 or 1"):
        TextADDataset.from_lists([["a"]], [[2]])


def test_retokenize_splits_spans_and_keeps_labels():
    r = _ds().retokenize()
    assert r.tokens[1] == ["it", "'", "s", "not", "fresh", "at", "all", "!"]
    assert r.token_labels[1].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    assert r.tokens[0] == DOCS[0] and r.tokens[2] == DOCS[2]        # already regex-split
    assert r.labels.tolist() == _ds().labels.tolist() and r.texts == TEXTS


@pytest.mark.parametrize("fmt", ["jsonl", "json", "npy"])
def test_local_formats_roundtrip(tmp_path, fmt):
    ds = _ds()
    p = tmp_path / f"toy.{fmt}"
    if fmt == "jsonl":
        ds.to_jsonl(p)
    elif fmt == "json":
        p.write_text(json.dumps([{"tokens": t, "labels": l, "text": x} for t, l, x in zip(DOCS, LABELS, TEXTS)]))
    else:   # the TokenCore format: sentences wrapped in one-element lists
        np.save(p, {"sentences": [[x] for x in TEXTS], "tokens": DOCS, "labels": LABELS}, allow_pickle=True)
    back = load_local(str(p))
    assert back.tokens == DOCS and back.texts == TEXTS and back.name == "toy"
    assert [l.tolist() for l in back.token_labels] == LABELS


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """Serve the toy dataset as the built-in "grammar_correction" from a local file."""
    f = tmp_path / "toy.jsonl"
    _ds().to_jsonl(f)
    meta = dict(datasets.DATASETS["grammar_correction"], sha256=datasets._sha256(f),
                n_documents=3, n_anomalous=1, n_tokens=12, n_anomalous_tokens=1)
    monkeypatch.setitem(datasets.DATASETS, "grammar_correction", meta)
    calls = []

    def fake_download(repo_id, filename, repo_type=None, revision=None, cache_dir=None):
        calls.append((repo_id, filename, repo_type, revision, cache_dir))
        return str(f)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return f, calls


def test_load_dataset_from_hub(fake_hub):
    f, calls = fake_hub
    ds = load_dataset("grammar", cache_dir="X")                      # alias
    assert ds.tokens == DOCS and ds.name == "grammar_correction"
    assert calls[0][2] == "dataset" and calls[0][4] == "X"
    assert load_dataset("grammar_correction", retokenize=True).info["retokenized"]


def test_load_dataset_rejects_changed_file(fake_hub):
    f, _ = fake_hub
    f.write_text(f.read_text().replace('"good"', '"bad"'))
    with pytest.raises(RuntimeError, match="SHA-256"):
        load_dataset("grammar_correction")


def test_unknown_dataset():
    with pytest.raises(ValueError, match="available"):
        load_dataset("nope")


def test_registry_is_consistent():
    assert datasets.list_datasets() == ["grammar_correction", "hate_speech", "olid", "restaurant_review",
                                        "restaurant_review2", "sms_spam"]
    assert all(v in datasets.DATASETS for v in datasets.ALIASES.values())
    for name in datasets.list_datasets():
        m = datasets.dataset_info(name)
        assert len(m["sha256"]) == 64 and m["filename"].endswith(".jsonl")
        assert 0 < m["n_anomalous"] < m["n_documents"] and 0 < m["n_anomalous_tokens"] < m["n_tokens"]
    assert len({datasets.DATASETS[n]["sha256"] for n in datasets.list_datasets()}) == 6


def test_dataset_feeds_word_level_pipeline(tiny_bert):
    """Word lists with multi-word spans ("bank price") get one vector and one score each,
    aligned with the labels, so ``evaluate`` can use ``token_labels`` directly."""
    path, _ = tiny_bert
    rng = np.random.RandomState(0)
    vocab = ["stock", "market", "bank", "price", "trade", "goal", "match", "team"]
    tokens, labels = [], []
    for i in range(30):
        t = list(rng.choice(vocab, rng.randint(4, 9)))
        lab = [0] * len(t)
        if i % 5 == 0:
            t.insert(2, "bank price")
            lab.insert(2, 1)
        tokens.append(t)
        labels.append(lab)
    ds = TextADDataset.from_lists(tokens, labels)
    emb = TokenEmbedder(path, word_pooling="max", device=DEV)
    E, ids = emb.transform(ds.tokens)
    assert [len(e) for e in E] == [len(t) for t in ds.tokens] and ids[0] == list(range(len(ds.tokens[0])))

    class Dist:
        def fit(self, X):
            self.mu_ = X.mean(0)
            return self

        def decision_function(self, X):
            return np.linalg.norm(X - self.mu_, axis=1)

    train = ds.subset(ds.normal_indices)
    res = evaluate(TokenDetector(Dist(), embedder=emb), train.tokens, ds.tokens,
                   token_labels=ds.token_labels, seeds=(0,))
    assert res["token"]["n_ignored"] == (0.0, 0.0)


def test_real_hub_files():
    """Downloads the real files when the Hub is reachable and they are uploaded."""
    from huggingface_hub import hf_hub_download
    for name in datasets.list_datasets():
        m = datasets.DATASETS[name]
        try:
            hf_hub_download(m["repo_id"], m["filename"], repo_type="dataset", revision=m["revision"])
        except Exception as e:                                       # offline or not uploaded yet
            pytest.skip(f"{name} not downloadable: {type(e).__name__}")
        ds = load_dataset(name)                                       # checks checksum and counts
        assert ds.stats()["n_documents"] == m["n_documents"]
