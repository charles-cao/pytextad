"""Quick smoke run of all four detectors on a small AG News subset.

    pip install pytextad          # or, from a clone:  pip install -e .
    python examples/quickstart.py

Local Hugging Face cache (models downloaded earlier with from_pretrained(..., cache_dir=...)):
    set HF_HUB_CACHE to that folder and HF_HUB_OFFLINE=1 before running.

Downloads AG News (CSV, ~30 MB) into ./data on first use, and the encoders from
Hugging Face (in mainland China, set HF_ENDPOINT=https://hf-mirror.com first).
Settings are shortened so that everything finishes in a few minutes on a GPU; the
AUROC values are NOT comparable with the papers.
"""
import argparse
import csv
import os
import random
import time
import urllib.request

import numpy as np
from sklearn.metrics import roc_auc_score

from pytextad import CVDD, DATE, FATE, RSRAE, TokenEmbedder, mean_pool

URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/{}.csv"
CLASSES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def load_ag(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    out = {}
    for split in ("train", "test"):
        path = os.path.join(data_dir, f"{split}.csv")
        if not os.path.exists(path):
            print(f"downloading AG News {split} ...")
            urllib.request.urlretrieve(URL.format(split), path)
        out[split] = [(int(r[0]), (r[1] + ". " + r[2]).replace("\\", " ")) for r in csv.reader(open(path, encoding="utf-8"))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--inlier", type=int, default=3, help="AG News class used as normal (3 = Business)")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=500, help="per side: n_test inliers + n_test anomalies")
    ap.add_argument("--bert", default="bert-base-uncased", help="frozen encoder for CVDD / RSRAE, tokenizer for DATE")
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2", help="encoder fine-tuned by FATE")
    ap.add_argument("--methods", default="rsrae,cvdd,date,fate")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = ap.parse_args()

    random.seed(0)
    ag = load_ag(args.data_dir)
    train = [t for c, t in ag["train"] if c == args.inlier]
    random.shuffle(train)
    train = train[:args.n_train]
    t_in = [t for c, t in ag["test"] if c == args.inlier][:args.n_test]
    t_out = [t for c, t in ag["test"] if c != args.inlier]
    random.shuffle(t_out)
    t_out = t_out[:args.n_test]
    test, y_test = t_in + t_out, np.r_[np.zeros(len(t_in)), np.ones(len(t_out))]
    print(f"normal class: {CLASSES[args.inlier]} | train {len(train)} | test {len(t_in)} normal + {len(t_out)} anomalous")

    methods = args.methods.split(",")
    results = {}

    def run(name, fn):
        t0 = time.time()
        scores = fn()
        results[name] = (roc_auc_score(y_test, scores), time.time() - t0)
        print(f"  {name:6s} AUROC {results[name][0]:.4f}  ({results[name][1]:.0f}s)")

    if "rsrae" in methods or "cvdd" in methods:
        print(f"encoding with frozen {args.bert} ...")
        emb = TokenEmbedder(args.bert, max_length=128, device=args.device)
        H_tr, _ = emb.transform(train)
        H_te, _ = emb.transform(test)
        if "rsrae" in methods:
            run("RSRAE", lambda: RSRAE(n_epochs=50, device=args.device).fit(mean_pool(H_tr)).decision_function(mean_pool(H_te)))
        if "cvdd" in methods:
            run("CVDD", lambda: CVDD(n_epochs=20, lr_milestones=(8,), device=args.device).fit(H_tr).decision_function(H_te))
    if "date" in methods:
        run("DATE", lambda: DATE(tokenizer=args.bert, n_epochs=2, device=args.device, verbose=True)
            .fit(train).decision_function(test))
    if "fate" in methods:
        anomalies = [t for c, t in ag["train"] if c != args.inlier]
        random.shuffle(anomalies)
        X, y = train + anomalies[:10], np.r_[np.zeros(len(train)), np.ones(10)]   # 10 labelled anomalies
        run("FATE", lambda: FATE(encoder=args.sbert, n_epochs=1, device=args.device, verbose=True)
            .fit(X, y).decision_function(test))

    print("\nall requested detectors ran:", ", ".join(results))


if __name__ == "__main__":
    main()
