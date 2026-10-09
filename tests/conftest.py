"""Shared fixtures: tiny randomly initialised models built locally (no downloads)."""
import os

import numpy as np
import pytest
import torch

DEV = os.environ.get("PYTEXTAD_DEVICE", "cpu")   # PYTEXTAD_DEVICE=cuda pytest ... to test on GPU
WORDS = ["stock", "market", "bank", "price", "trade", "goal", "match", "team", "player", "score"]


def make_texts(n, words, seed):
    rng = np.random.RandomState(seed)
    return [" ".join(rng.choice(words, rng.randint(5, 30))) for _ in range(n)]


@pytest.fixture(scope="session")
def tiny_bert(tmp_path_factory):
    """WordPiece tokenizer + 1-layer BERT. Returns (path, tokenizer)."""
    from transformers import BertConfig, BertModel, BertTokenizerFast
    d = tmp_path_factory.mktemp("tinybert")
    vocab = {w: i for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS, "##s", "s"])}
    try:
        tok = BertTokenizerFast(vocab=vocab, do_lower_case=True)
    except TypeError:                                   # transformers 4.x
        f = d / "vocab.txt"
        f.write_text("\n".join(vocab))
        tok = BertTokenizerFast(vocab_file=str(f), do_lower_case=True)
    torch.manual_seed(0)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64)).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d), tok


@pytest.fixture(scope="session")
def tiny_gpt(tmp_path_factory):
    """Byte-level BPE tokenizer (no CLS, left padding, no pad token) + 1-layer GPT-2."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2Model, PreTrainedTokenizerFast
    d = tmp_path_factory.mktemp("tinygpt")
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(make_texts(300, WORDS, 0), vocab_size=300, special_tokens=["<|endoftext|>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, eos_token="<|endoftext|>",
                                  bos_token="<|endoftext|>", unk_token="<|endoftext|>",
                                  add_prefix_space=False, padding_side="left")
    tok.save_pretrained(d)
    torch.manual_seed(0)
    GPT2Model(GPT2Config(vocab_size=len(tok), n_embd=32, n_layer=1, n_head=2, n_positions=128,
                         bos_token_id=0, eos_token_id=0)).save_pretrained(d)
    return str(d)
