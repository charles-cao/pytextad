"""Convert the benchmark .npy files to the JSONL files hosted on the Hugging Face Hub and
check that nothing changes on the way.

    python scripts/make_hub_files.py <output folder> <data folder> [<data folder> ...]

Data folders: TokenCore/data (sms_spam, restaurant_review, grammar) and CA-PTD/data (spam,
grammar, hate_speech, olid, review1, review2). For every dataset: the JSONL read back equals
the .npy (words, labels, texts); when a dataset is found under several names or folders, all
copies must be identical; when TokenCore's Label Studio export and its converter
(tokens_labels/json_to_npy.py) are present, the .npy must equal the converter's output.
Prints the SHA-256 and counts for pytextad/datasets.py.
"""
import importlib.util
import os
import sys

import numpy as np

from pytextad.datasets import DATASETS, _sha256, load_local

# dataset -> (file names of the .npy in the data folders, Label Studio export or None)
FILES = {"sms_spam": (["sms_spam.npy", "spam.npy"], "sms_spam_labeled.json"),
         "restaurant_review": (["restaurant_review.npy", "review1.npy"], "restaurant_review_labeled.json"),
         "grammar_correction": (["grammar.npy"], "grammar_correction_labeled.json"),
         "hate_speech": (["hate_speech.npy"], None),
         "olid": (["olid.npy"], None),
         "restaurant_review2": (["review2.npy"], None)}


def same(a, b):
    return (a.tokens == b.tokens and a.texts == b.texts and
            all(np.array_equal(x, y) for x, y in zip(a.token_labels, b.token_labels)))


def converter(folder):
    path = os.path.join(folder, "..", "tokens_labels", "json_to_npy.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("json_to_npy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(out, folders):
    os.makedirs(out, exist_ok=True)
    assert set(FILES) == set(DATASETS), "FILES and DATASETS list different datasets"
    for name, (npys, ls_json) in FILES.items():
        copies = [os.path.join(d, f) for d in folders for f in npys if os.path.exists(os.path.join(d, f))]
        if not copies:
            print(f"{name:20s} not found, skipped")
            continue
        ds = load_local(copies[0], name)
        for c in copies[1:]:
            assert same(ds, load_local(c, name)), f"{name}: {copies[0]} and {c} differ"
        target = os.path.join(out, f"{name}.jsonl")
        ds.to_jsonl(target)
        assert same(ds, load_local(target, name)), f"{name}: JSONL differs from .npy"
        checks = [f"jsonl == npy ({len(copies)} identical cop{'y' if len(copies) == 1 else 'ies'})"]
        for d in folders:
            conv = converter(d)
            if ls_json and conv and os.path.exists(os.path.join(d, ls_json)):
                res = conv.label_studio_to_tokens_labels(os.path.join(d, ls_json))
                raw = np.load(copies[0], allow_pickle=True).item()
                assert [r["tokens"] for r in res] == raw["tokens"], f"{name}: tokens differ from Label Studio export"
                assert [r["labels"] for r in res] == raw["labels"], f"{name}: labels differ from Label Studio export"
                assert [[r["sentence"]] for r in res] == raw["sentences"], f"{name}: texts differ"
                checks.append("npy == Label Studio export")
        st = ds.stats()
        print(f"{name:20s} {ds!r}\n{'':20s} {', '.join(checks)}\n{'':20s} sha256 {_sha256(target)}")
        meta = DATASETS[name]
        for k in ("n_documents", "n_anomalous", "n_tokens", "n_anomalous_tokens"):
            if meta[k] != st[k]:
                print(f"{'':20s} !! {k}: registry {meta[k]}, data {st[k]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
