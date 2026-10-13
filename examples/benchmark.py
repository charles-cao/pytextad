"""Benchmark several detectors on several datasets, at the token and document levels.

    python examples/benchmark.py
    python examples/benchmark.py --datasets restaurant_review grammar_correction \\
        --detectors TokenCore SIK ADERH pyod:IForest pyod:ECOD --model bert-base-uncased

For every dataset: the documents are split (half of the normal documents for training, all
other documents for testing), every document is embedded once with a frozen encoder (cached
on disk), and every detector is fitted and evaluated with ``evaluate`` over several seeds.
The results are printed as one table per dataset and a summary table, and saved to
``<out>/results.csv`` (one row per dataset, detector, level and metric) and
``<out>/results.md``.

Detectors are given by name: any embedding detector of PyTextAD (``TokenCore``, ``SIK``,
``ADERH``, ``DTEGaussian``, ...), or a PyOD detector as ``pyod:<Class>`` (``pyod:IForest``,
``pyod:ECOD``, ``pyod:LOF``; needs ``pip install pyod``). A detector that fails (for example
a missing optional dependency) is reported and skipped.
"""
import argparse
import csv
import importlib
import inspect
import os
import time

import numpy as np

import pytextad
from pytextad import TokenDetector, TokenEmbedder, align_labels
from pytextad.datasets import list_datasets, load_dataset
from pytextad.metrics import evaluate

DEFAULT_DATASETS = ["restaurant_review", "grammar_correction", "sms_spam"]
DEFAULT_DETECTORS = ["TokenCore", "SIK", "ADERH", "pyod:IForest", "pyod:ECOD"]
COLUMNS = [("token", "auroc"), ("token", "ap"), ("document[max]", "auroc"), ("document[mean]", "auroc")]


def make_detector(name, device=None):
    """A fresh detector from its name: a PyTextAD embedding detector or ``pyod:<Class>``."""
    if name.startswith("pyod:"):
        cls_name = name.split(":", 1)[1]
        cls = getattr(importlib.import_module(f"pyod.models.{cls_name.lower()}"), cls_name)
    else:
        cls = getattr(pytextad, name)
    kwargs = {}
    if device is not None and "device" in inspect.signature(cls).parameters:
        kwargs["device"] = device
    return cls(**kwargs)


def split(ds, train_ratio=0.5, seed=42):
    """Train on ``train_ratio`` of the normal documents, test on all the others."""
    rng = np.random.RandomState(seed)
    normal = ds.normal_indices
    train_idx = rng.choice(normal, size=int(len(normal) * train_ratio), replace=False)
    return ds.subset(train_idx), ds.subset(np.setdiff1d(np.arange(len(ds)), train_idx))


def run_benchmark(datasets, detectors, embedder, seeds=(0, 1, 2), train_ratio=0.5, split_seed=42,
                  device=None, cache=None, log=print):
    """Evaluate every detector on every dataset.

    Parameters
    ----------
    datasets : dict
        {name: TextADDataset}.
    detectors : list of str
        Detector names (see ``make_detector``).
    embedder : TokenEmbedder
        Embeds the words of every document (sub-word vectors by default).
    cache : str or None
        Folder for the embeddings, reused across runs.

    Returns
    -------
    list of dict
        One row per dataset, detector, level and metric, with ``mean`` and ``std`` over the
        seeds, the fitting time in seconds, or ``error`` if the detector failed.
    """
    rows = []
    for ds_name, ds in datasets.items():
        train, test = split(ds, train_ratio, split_seed)
        log(f"\n{ds_name}: {len(train)} training documents, {len(test)} test documents "
            f"({int(test.labels.sum())} anomalous)")
        paths = [None, None]
        if cache is not None:
            os.makedirs(cache, exist_ok=True)
            tag = f"{ds_name}_r{train_ratio}_s{split_seed}_{embedder.word_pooling or 'sub'}"
            paths = [os.path.join(cache, f"{tag}_{part}.npz") for part in ("train", "test")]
        X_train = embedder.transform(train.tokens, cache=paths[0])
        X_test = embedder.transform(test.tokens, cache=paths[1])
        y_test = align_labels(test.token_labels, X_test[1])
        for det_name in detectors:
            start = time.time()
            try:
                res = evaluate(TokenDetector(make_detector(det_name, device)), X_train, X_test,
                               token_labels=y_test, seeds=seeds)
            except Exception as e:                      # keep going with the other detectors
                log(f"  {det_name:<16} failed: {type(e).__name__}: {e}")
                rows.append({"dataset": ds_name, "detector": det_name, "error": f"{type(e).__name__}: {e}"})
                continue
            seconds = time.time() - start
            for level, metrics in res.items():
                if level == "runs":
                    continue
                for metric, (mean, std) in metrics.items():
                    if metric != "n_ignored":
                        rows.append({"dataset": ds_name, "detector": det_name, "level": level,
                                     "metric": metric, "mean": mean, "std": std, "seconds": seconds})
            tok = res["token"]["auroc"][0]
            log(f"  {det_name:<16} token AUROC {tok:.4f}   ({seconds:.0f} s)")
    return rows


def _value(rows, ds, det, level, metric):
    for r in rows:
        if (r["dataset"], r["detector"], r.get("level"), r.get("metric")) == (ds, det, level, metric):
            return r["mean"], r["std"]
    return None


def to_markdown(rows):
    """One table per dataset (mean ± std over the seeds) and a summary of token AUROC."""
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    detectors = list(dict.fromkeys(r["detector"] for r in rows))
    head = "| Detector | Token AUROC | Token AP | Doc AUROC (max) | Doc AUROC (mean) |\n|---|---|---|---|---|\n"
    out = []
    for ds in datasets:
        out.append(f"### {ds}\n\n{head}")
        for det in detectors:
            cells = []
            for level, metric in COLUMNS:
                v = _value(rows, ds, det, level, metric)
                cells.append(f"{v[0]:.4f} ± {v[1]:.4f}" if v else "failed")
            out[-1] += f"| {det} | " + " | ".join(cells) + " |\n"
    out.append("### Token AUROC, all datasets\n\n| Detector | " + " | ".join(datasets) + " | Mean |\n|---|"
               + "---|" * (len(datasets) + 1) + "\n")
    for det in detectors:
        vals = [_value(rows, ds, det, "token", "auroc") for ds in datasets]
        nums = [v[0] for v in vals if v]
        mean = f"{np.mean(nums):.4f}" if len(nums) == len(datasets) else "-"
        out[-1] += f"| {det} | " + " | ".join(f"{v[0]:.4f}" if v else "failed" for v in vals) + f" | {mean} |\n"
    return "\n".join(out)


def save(rows, out):
    os.makedirs(out, exist_ok=True)
    fields = ["dataset", "detector", "level", "metric", "mean", "std", "seconds", "error"]
    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    md = to_markdown(rows)
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    return md


def _model_tag(model):
    """Folder name for a model: the folder's own name for a local model, 'org_name' for a Hub model."""
    if os.path.isdir(model):
        return os.path.basename(os.path.normpath(model))
    return model.replace("/", "_").replace("\\", "_").replace(":", "_")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=list_datasets())
    ap.add_argument("--detectors", nargs="+", default=DEFAULT_DETECTORS)
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--pool_vectors", action="store_true",
                    help="max-pool sub-word vectors per word before scoring (published TokenCore)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--train_ratio", type=float, default=0.5)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model_cache_dir", default=None, help="Hugging Face cache folder for the encoder")
    ap.add_argument("--cache", default="benchmark_cache", help="folder for the embeddings")
    ap.add_argument("--out", default="benchmark_results")
    args = ap.parse_args(argv)

    embedder = TokenEmbedder(args.model, word_pooling="max" if args.pool_vectors else None,
                             device=args.device, cache_dir=args.model_cache_dir)
    datasets = {name: load_dataset(name) for name in args.datasets}
    rows = run_benchmark(datasets, args.detectors, embedder, seeds=tuple(args.seeds),
                         train_ratio=args.train_ratio, split_seed=args.split_seed,
                         device=args.device, cache=os.path.join(args.cache, _model_tag(args.model)))
    print("\n" + save(rows, args.out))
    print(f"\nSaved to {os.path.join(args.out, 'results.csv')} and results.md")


if __name__ == "__main__":
    main()
