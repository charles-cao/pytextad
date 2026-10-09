"""Token-level benchmark on the TokenCore datasets with PyOD detectors.

    pip install pytextad pyod
    python examples/token_level.py --dataset restaurant_review --model bert-base-uncased

A setup like that of Cao et al. (WWW 2026): word vectors from a frozen encoder (max over
sub-words), detectors fitted on half of the normal documents and evaluated on all other
documents; token scores are aggregated per document by max and mean. Seeds 0, 1, 2.
Change the split below to match your own protocol.
"""
import argparse

import numpy as np
from pyod.models.iforest import IForest
from pyod.models.knn import KNN

from pytextad import TokenDetector, TokenEmbedder
from pytextad.datasets import list_datasets, load_dataset
from pytextad.metrics import evaluate, format_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="restaurant_review", choices=list_datasets())
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--cache_dir", default=None, help="Hugging Face cache folder for the encoder")
    ap.add_argument("--device", default=None)
    ap.add_argument("--retokenize", action="store_true")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, retokenize=args.retokenize)
    print(ds)
    rng = np.random.RandomState(0)
    normal = rng.permutation(ds.normal_indices)
    train_idx = normal[: len(normal) // 2]
    test = ds.subset(np.setdiff1d(np.arange(len(ds)), train_idx))
    train = ds.subset(train_idx)

    emb = TokenEmbedder(args.model, word_pooling="max", cache_dir=args.cache_dir, device=args.device)
    tag = f"{args.dataset}{'_rt' if args.retokenize else ''}_{args.model.replace('/', '_')}"
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
