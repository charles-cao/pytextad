import numpy as np
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from pytextad import SentenceEmbedder, TokenEmbedder

from conftest import DEV, WORDS, make_texts


def _align1_reference(path, docs, model_type):
    """The word-embedding logic of the TokenCore script align1.py (one sentence at a time,
    max over sub-words, CLS or last-token sentence vector), kept here as the reference."""
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).eval()
    words_out, sent_out = [], []
    with torch.no_grad():
        for tokens in docs:
            enc = tok(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, padding=False)
            h = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            if model_type == "bert":
                sent_out.append(h[0, 0].numpy())
            else:
                last = enc["attention_mask"].sum(1) - 1
                sent_out.append(h[0, last[0]].numpy())
            groups = {}
            for pos, w in enumerate(enc.word_ids(0)):
                if w is not None:
                    groups.setdefault(w, []).append(h[0, pos].numpy())
            words_out.append(np.stack([np.max(np.stack(groups[w]), 0) for w in sorted(groups)]))
    return words_out, np.stack(sent_out)


def test_word_embeddings_match_align1(tiny_bert):
    path, _ = tiny_bert
    docs = [t.split() + ["scores"] for t in make_texts(6, WORDS, 3)]          # "scores" -> 2 sub-words
    ref_words, ref_sent = _align1_reference(path, docs, "bert")
    emb = TokenEmbedder(path, word_pooling="max", device=DEV, batch_size=4)
    words, ids = emb.transform(docs)
    for a, b, d, i in zip(words, ref_words, docs, ids):
        assert a.shape == (len(d), b.shape[1]) and list(i) == list(range(len(d)))
        np.testing.assert_allclose(a, b, atol=1e-5)
    sent = SentenceEmbedder(path, device=DEV).transform(docs)            # auto -> CLS for BERT
    np.testing.assert_allclose(sent, ref_sent, atol=1e-5)


def test_last_token_pooling_decoder_left_padding(tiny_gpt):
    docs = [t.split() for t in make_texts(5, WORDS, 1)]
    emb = SentenceEmbedder(tiny_gpt, device=DEV, batch_size=5)
    assert emb.pooling == "last"                                          # no CLS token
    batch = emb.transform(docs)
    alone = np.concatenate([emb.transform([d]) for d in docs])
    np.testing.assert_allclose(batch, alone, atol=1e-5)                   # padding side irrelevant
    # reference: last position of each sentence encoded alone, without padding
    s = SentenceEmbedder(tiny_gpt, device=DEV).transform([" ".join(d) for d in docs])
    tok = AutoTokenizer.from_pretrained(tiny_gpt)
    model = AutoModel.from_pretrained(tiny_gpt).eval()
    with torch.no_grad():
        ref_str = np.stack([model(**tok(" ".join(d), return_tensors="pt")).last_hidden_state[0, -1].numpy()
                            for d in docs])
    np.testing.assert_allclose(s, ref_str, atol=1e-5)


def test_word_lists_read_like_running_text(tiny_gpt, tiny_bert):
    """A list of words must give exactly the vectors of the same sentence given as a string
    (byte-level BPE tokenizers would otherwise glue the words together)."""
    for path in (tiny_gpt, tiny_bert[0]):
        emb = TokenEmbedder(path, device=DEV)
        docs = [t.split() for t in make_texts(4, WORDS, 5)]
        a, ia = emb.transform(docs)
        b, ib = emb.transform([" ".join(d) for d in docs])
        for x, y, wa, wb in zip(a, b, ia, ib):
            np.testing.assert_allclose(x, y, atol=1e-5)
            assert wa == wb


@pytest.mark.parametrize("pooling", ["cls", "mean", "last"])
def test_sentence_pooling_shapes(tiny_bert, pooling):
    path, _ = tiny_bert
    V = SentenceEmbedder(path, pooling=pooling, normalize=True, device=DEV).transform(make_texts(7, WORDS, 2))
    assert V.shape == (7, 32)
    np.testing.assert_allclose(np.linalg.norm(V, axis=1), 1, atol=1e-5)


def test_cache_roundtrip(tiny_bert, tmp_path):
    path, _ = tiny_bert
    docs = make_texts(5, WORDS, 4)
    f = str(tmp_path / "emb.npz")
    emb = TokenEmbedder(path, device=DEV)
    a, ia = emb.transform(docs, cache=f)
    b, ib = emb.transform(docs, cache=f)                                   # loaded from disk
    assert ia == ib and all(np.array_equal(x, y) for x, y in zip(a, b))
    with pytest.warns(UserWarning, match="recomputing"):
        c, _ = emb.transform(docs[:3], cache=f)                            # different texts
    assert len(c) == 3
    s = SentenceEmbedder(path, device=DEV)
    v1 = s.transform(docs, cache=str(tmp_path / "s.npz"))
    v2 = s.transform(docs, cache=str(tmp_path / "s.npz"))
    np.testing.assert_array_equal(v1, v2)


def test_truncated_words_are_reported(tiny_bert):
    path, _ = tiny_bert
    doc = ["stock"] * 30
    emb = TokenEmbedder(path, word_pooling="max", max_length=12, device=DEV)
    with pytest.warns(UserWarning, match="without an embedding"):
        E, ids = emb.transform([doc])
    assert E[0].shape[0] == 10 and ids[0] == list(range(10)) and emb.n_truncated_ == 1
