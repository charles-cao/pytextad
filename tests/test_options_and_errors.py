"""Input checks, error messages and less-used options of PyTextAD's own code."""
import pickle
import sys

import numpy as np
import pytest

from pytextad import (CVDD, DATE, FATE, ICL, RSRAE, DocumentDetector, DTENonParametric, SIK,
                      TokenDetector, TokenEmbedder, SentenceEmbedder, mean_pool, words_from_subwords)
from pytextad.datasets import TextADDataset, load_local
from pytextad.metrics import aggregate, document_metrics, evaluate, format_results, token_metrics
from pytextad.models.cvdd import _alpha_schedule
from pytextad.utils.embeddings import encode_words

from conftest import DEV, WORDS, make_texts


def _tokens(n=12, seed=0, dim=6):
    rng = np.random.RandomState(seed)
    return [rng.randn(rng.randint(3, 7), dim) for _ in range(n)]


# ----------------------------------------------------------------------------- datasets
def test_dataset_checks_and_repr(tmp_path):
    with pytest.raises(ValueError, match="2 documents, 2 label lists, 1 texts"):
        TextADDataset([["a"], ["b"]], [[0], [1]], ["a"])
    ds = TextADDataset.from_lists([["a", "b"], ["c"]], [[0, 1], [0]], name="toy")
    with pytest.raises(ValueError, match="wrong length"):
        ds.subset(np.array([True]))
    assert repr(ds) == "TextADDataset('toy': 2 documents, 1 anomalous; 3 words, 1 anomalous)"
    bad = tmp_path / "x.csv"
    bad.write_text("a")
    with pytest.raises(ValueError, match="unsupported file type"):
        load_local(str(bad))


def test_load_dataset_download_error_and_count_check(monkeypatch, tmp_path):
    import huggingface_hub
    from pytextad import datasets

    def offline(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(RuntimeError, match="(?s)could not download.*HF_ENDPOINT"):
        datasets.load_dataset("grammar_correction")

    f = tmp_path / "toy.jsonl"
    TextADDataset.from_lists([["a", "b"]], [[0, 1]]).to_jsonl(f)
    meta = dict(datasets.DATASETS["grammar_correction"], sha256=datasets._sha256(f),
                n_documents=1, n_anomalous=1, n_tokens=99, n_anomalous_tokens=1)
    monkeypatch.setitem(datasets.DATASETS, "grammar_correction", meta)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(f))
    with pytest.raises(RuntimeError, match="expected"):
        datasets.load_dataset("grammar_correction")


# ----------------------------------------------------------------------------- metrics
def test_metric_checks():
    with pytest.raises(ValueError, match="2 labels but 3 scores"):
        document_metrics([0, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="NaN"):
        document_metrics([0, 1], [0.1, np.nan])
    with pytest.raises(ValueError, match="1 label lists but 2 score lists"):
        token_metrics([[0, 1]], [[0.1, 0.2], [0.3]])
    with pytest.raises(ValueError, match="how must be"):
        aggregate([np.array([1.0])], "median")
    assert np.isnan(aggregate([np.array([np.nan, np.nan])], "max")[0])
    assert aggregate([np.array([1.0, 2.0, 3.0, 4.0])], "topk", k=0.5)[0] == 3.5


def test_evaluate_needs_labels_and_formats_one_result():
    X = _tokens(20)
    with pytest.raises(ValueError, match="give doc_labels or token_labels"):
        evaluate(TokenDetector(SIK()), X[:10], X[10:])
    labels = [np.r_[np.zeros(len(x) - 1, int), i % 2] for i, x in enumerate(X[10:])]
    res = evaluate(TokenDetector(SIK()), X[:10], X[10:], token_labels=labels, seeds=(0,))
    assert format_results(res).startswith(" " * 16 + " document")


# ----------------------------------------------------------------------------- base classes
def test_base_detector_checks(capsys):
    with pytest.raises(ValueError, match="contamination"):
        SIK(contamination=0.9)
    X = _tokens(30, dim=8)
    det = CVDD(n_epochs=1, random_state=None, verbose=True, device=DEV)
    for method in (det.decision_function, det.token_scores, det.attention):
        with pytest.raises(RuntimeError, match="not fitted"):
            method(X)
    labels = det.fit_predict(X)
    assert set(np.unique(labels)) <= {0, 1} and len(labels) == 30
    assert "[CVDD]" in capsys.readouterr().out


def test_embedding_detector_checks():
    with pytest.raises(ValueError, match="2-D array"):
        SIK().fit(np.zeros(5))
    with pytest.raises(RuntimeError, match="not fitted"):
        SIK().decision_function(np.zeros((2, 3)))


def test_optional_dependencies_are_reported(monkeypatch):
    X = np.random.RandomState(0).randn(20, 4)
    for missing, vendored, det in [("pandas", "icl", ICL(num_epochs=1)),
                                   ("matplotlib", "dte_nonparametric", DTENonParametric())]:
        with monkeypatch.context() as m:
            m.setitem(sys.modules, missing, None)
            m.delitem(sys.modules, f"pytextad.models.embedding._vendor.{vendored}", raising=False)
            with pytest.raises(ImportError, match=f"pip install {missing}"):
                det.fit(X)


# ----------------------------------------------------------------------------- CVDD
@pytest.mark.parametrize("name,n_alphas", [("soft", 5), ("linear", 5), ("logarithmic", 5), ("hard", 4)])
def test_cvdd_alpha_schedules(name, n_alphas):
    milestones, alphas = _alpha_schedule(name, 10)
    assert milestones == [2, 4, 6, 8, 10] and len(alphas) == n_alphas


def test_cvdd_input_checks():
    with pytest.raises(ValueError, match="unknown alpha_scheduler"):
        _alpha_schedule("cosine", 10)
    with pytest.raises(ValueError, match="non-empty"):
        CVDD(n_epochs=1, device=DEV).fit([np.zeros((0, 4))])
    with pytest.raises(ValueError, match="same dimension"):
        CVDD(n_epochs=1, device=DEV).fit([np.zeros((2, 4)), np.zeros((2, 5))])


# ----------------------------------------------------------------------------- DATE
def test_date_vocab_file_official_masks_and_attention_mask(tiny_bert, tmp_path):
    _, tok = tiny_bert
    vocab = tmp_path / "vocab.txt"                               # tokenizer given as a vocab file
    vocab.write_text("\n".join(sorted(tok.get_vocab(), key=tok.get_vocab().get)), encoding="utf-8")
    L, K = 16, 3
    rng = np.random.RandomState(0)
    masks = [{"label": k, "mask": rng.rand(L) < 0.5} for k in range(K)]
    f = tmp_path / "masks.pkl"
    f.write_bytes(pickle.dumps(masks))
    X = make_texts(20, WORDS, 0)
    det = DATE(tokenizer=str(vocab), max_len=L, n_masks=K, masks=str(f), n_epochs=1,
               batch_size=8, use_attention_mask=True, hidden=32, layers=1, heads=2, intermediate=32,
               emb=16, device=DEV)
    with pytest.raises(RuntimeError, match="not fitted"):
        det.decision_function(X)
    assert det.fit(X).decision_function(X[:4]).shape == (4,)

    path, _ = tiny_bert
    by_name = DATE(tokenizer=path)                                # model folder or Hub name
    by_name._load_tokenizer()
    assert by_name.tok_.is_fast
    with pytest.raises(ValueError, match="fast tokenizer"):
        DATE(tokenizer=object())._load_tokenizer()

    bad = tmp_path / "bad.pkl"
    bad.write_bytes(pickle.dumps([{"label": 0, "mask": np.zeros(L + 1, bool)}]))
    with pytest.raises(ValueError, match="mask length"):
        DATE(tokenizer=str(vocab), max_len=L, n_masks=1, masks=str(bad), n_epochs=1,
             device=DEV).fit(X)


# ----------------------------------------------------------------------------- FATE
def test_fate_mask_padding_and_label_check(tiny_bert):
    path, _ = tiny_bert
    X = make_texts(12, WORDS, 0)
    det = FATE(encoder=path, max_length=16, n_epochs=1, mask_padding=True, device=DEV)
    with pytest.raises(RuntimeError, match="not fitted"):
        det.decision_function(X)
    det.fit(X)
    assert det.decision_function(X[:3]).shape == (3,)
    with pytest.raises(ValueError, match="different lengths"):
        FATE(encoder=path, max_length=16, n_epochs=1, device=DEV).fit(X, [0, 1])


# ----------------------------------------------------------------------------- RSRAE
def test_rsrae_options(capsys):
    X = np.abs(np.random.RandomState(0).randn(40, 6))          # non-negative: relu
    small = dict(hidden_layer_sizes=(8, 8, 8), intrinsic_size=2, batch_size=16, device=DEV)
    for bn_mode in (None, "batch"):
        for norm in ("mse", "l1", "l21"):
            det = RSRAE(n_epochs=1, bn_mode=bn_mode, loss_norm_type=norm, **small).fit(X)
            assert det.activation_ == "relu"
    RSRAE(n_epochs=20, all_alt=False, verbose=True, **small).fit(X)
    assert "epoch 20/20" in capsys.readouterr().out
    with pytest.raises(ValueError, match="unknown bn_mode"):
        RSRAE(n_epochs=1, bn_mode="layer", **small).fit(X)
    with pytest.raises(ValueError, match="unknown norm"):
        RSRAE(n_epochs=1, loss_norm_type="l3", **small).fit(X)


# ----------------------------------------------------------------------------- wrappers
def test_wrapper_checks(tiny_bert):
    path, _ = tiny_bert
    with pytest.raises(ValueError, match="no embedder"):
        DocumentDetector(SIK()).fit(["a text"])
    with pytest.raises(ValueError, match="no embedder"):
        TokenDetector(SIK()).fit(["a text"])
    det = TokenDetector(SIK(), embedder=TokenEmbedder(path, device=DEV)).fit(make_texts(10, WORDS, 0))
    test = make_texts(3, WORDS, 1)
    scores = det.token_scores(test)                              # running text: one per sub-word
    assert [len(s) for s in scores] == [len(e) for e in det.embedder.transform(test)[0]]
    det.subword_aggregation = "median"
    with pytest.raises(ValueError, match="subword_aggregation"):
        det.token_scores([t.split() for t in test])


# ----------------------------------------------------------------------------- embedders
def test_embedder_checks_and_word_pooling(tiny_bert, monkeypatch):
    path, _ = tiny_bert
    with pytest.raises(ValueError, match="word_pooling must be"):
        TokenEmbedder(path, word_pooling="sum", device=DEV)
    with pytest.raises(ValueError, match="pooling must be"):
        SentenceEmbedder(path, pooling="max", device=DEV)
    with pytest.raises(ValueError, match="lists of words"):
        TokenEmbedder(path, word_pooling="max", device=DEV).transform(["stock market"])
    docs = [["stocks", "market"], ["banks"]]                     # "stocks" -> stock + ##s
    sub, ids = TokenEmbedder(path, device=DEV).transform(docs)
    for pooling, ref in [("mean", lambda v: v.mean(0)), ("first", lambda v: v[0]), ("max", lambda v: v.max(0))]:
        words, _ = TokenEmbedder(path, word_pooling=pooling, device=DEV).transform(docs)
        for e, s, i in zip(words, sub, ids):
            i = np.array(i)
            for w in range(len(e)):
                assert np.allclose(e[w], ref(s[i == w]), atol=1e-6)
    assert mean_pool(sub).shape == (2, sub[0].shape[1])

    class Slow:
        is_fast = False
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Slow())
    with pytest.raises(ValueError, match="fast tokenizer"):
        TokenEmbedder(path, device=DEV)


def test_encode_words_space_token_and_words_from_subwords(tiny_gpt):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tiny_gpt)
    ids, wids = encode_words(tok, ["stock", "zqxj"])               # unseen word: the space is its own token
    assert len(ids) == len(wids) and None not in wids and wids[0] == 0 and wids[-1] == 1
    s = words_from_subwords([1.0, 5.0, 2.0, 7.0], [0, None, 1, 1], n_words=3)     # default: max
    assert s[0] == 1.0 and s[1] == 7.0 and np.isnan(s[2])
    assert words_from_subwords([1.0, 5.0, 2.0, 7.0], [0, None, 1, 1], agg="mean")[1] == 4.5
    assert words_from_subwords([1.0, 5.0, 2.0, 7.0], [0, None, 1, 1], agg="first")[1] == 2.0


def test_unused_weights_are_silent_and_missing_weights_warned(tmp_path, recwarn):
    import json
    import warnings
    from transformers import BertConfig, BertForPreTraining, BertModel, BertTokenizerFast
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b"]))
    cfg = BertConfig(vocab_size=7, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32)
    full = tmp_path / "pretraining"                       # extra pre-training heads: no warning
    BertTokenizerFast(str(vocab)).save_pretrained(full)
    BertForPreTraining(cfg).save_pretrained(full)
    TokenEmbedder(str(full), device=DEV)
    assert not [w for w in recwarn if "randomly initialised" in str(w.message)]
    short = tmp_path / "short"                            # one layer saved, two expected
    BertTokenizerFast(str(vocab)).save_pretrained(short)
    BertModel(cfg).save_pretrained(short)
    conf = json.loads((short / "config.json").read_text())
    conf["num_hidden_layers"] = 2
    (short / "config.json").write_text(json.dumps(conf))
    with pytest.warns(UserWarning, match="randomly initialised"):
        TokenEmbedder(str(short), device=DEV)


def test_every_public_class_has_a_docstring():
    """A docstring built by string concatenation is not a docstring; set ``__doc__`` instead."""
    import pytextad
    missing = [n for n in dir(pytextad)
               if isinstance(getattr(pytextad, n), type) and not getattr(pytextad, n).__dict__.get("__doc__")]
    assert missing == []
