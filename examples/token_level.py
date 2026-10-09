"""Token-level and document-level benchmark of PyOD detectors on a built-in dataset.

    pip install pytextad pyod
    python examples/token_level.py --dataset restaurant_review --model bert-base-uncased

Word vectors from a frozen encoder (max over sub-words); each detector is fitted on the
words of the training documents; every test word gets a score (token-level AUROC / AP) and
word scores are aggregated per document by max and mean (document-level AUROC / AP).
Seeds 0, 1, 2.

The split is that of ``split_anomaly_data`` in the TokenCore / CA-PTD code:
``--train_ratio 0.5`` trains on half of the normal documents (seed 42) and tests on all
other documents; ``--train_ratio 0`` trains and tests on the whole, contaminated dataset.
"""
import argparse

import numpy as np
from pyod.models.iforest import IForest
from pyod.models.knn import KNN

from pytextad import TokenDetector, TokenEmbedder
from pytextad.datasets import list_datasets, load_dataset
from pytextad.metrics import evaluate, format_results


def split(ds, train_ratio, seed):
    """Same indices as split_anomaly_data(X, y, z, train_ratio, random_state=seed)."""
    if not train_ratio:
        idx = np.arange(len(ds))
        return idx, idx
    normal = np.where(ds.labels == 0)[0]
    np.random.seed(seed)
    train_idx = np.random.choice(normal, size=int(len(normal) * train_ratio), replace=False)
    return train_idx, np.setdiff1d(np.arange(len(ds)), train_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="restaurant_review", choices=list_datasets())
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--cache_dir", default=None, help="Hugging Face cache folder for the encoder")
    ap.add_argument("--device", default=None)
    ap.add_argument("--retokenize", action="store_true")
    ap.add_argument("--train_ratio", type=float, default=0.5)
    ap.add_argument("--split_seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, retokenize=args.retokenize)
    print(ds)
    train_idx, test_idx = split(ds, args.train_ratio, args.split_seed)
    train, test = ds.subset(train_idx), ds.subset(test_idx)
    print(f"train: {len(train)} documents, test: {len(test)} documents ({test.labels.sum()} anomalous)")

    emb = TokenEmbedder(args.model, word_pooling="max", cache_dir=args.cache_dir, device=args.device)
    tag = (f"{args.dataset}{'_rt' if args.retokenize else ''}_r{args.train_ratio}_s{args.split_seed}_"
           f"{args.model.replace('/', '_').replace(':', '_').replace(chr(92), '_')}")
    X_train, _ = emb.transform(train.tokens, cache=f"{tag}_train.npz")
    X_test, ids = emb.transform(test.tokens, cache=f"{tag}_test.npz")
    # words cut off by max_length have no vector: keep the labels of the embedded words only
    y_test = [lab[i] for lab, i in zip(test.token_labels, ids)]

    results = {}
    for name, det in [("KNN", KNN()), ("IForest", IForest())]:
        results[name] = evaluate(TokenDetector(det), X_train, X_test, token_labels=y_test, seeds=(0, 1, 2))
    print(format_results(results))


if __name__ == "__main__":
    main()
