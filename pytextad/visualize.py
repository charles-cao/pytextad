"""Show token scores on the text: every word coloured by how anomalous it is."""
import html

import numpy as np

__all__ = ["highlight", "Highlight"]


class Highlight:
    """HTML of highlighted documents; shown as coloured text in Jupyter.

    Attributes
    ----------
    html : str
        The HTML fragment (one ``<div>`` per document).
    """

    def __init__(self, html_fragment):
        self.html = html_fragment

    def _repr_html_(self):
        return self.html

    def __str__(self):
        return self.html

    def save(self, path):
        """Write a standalone HTML page that can be opened in a browser."""
        with open(path, "w", encoding="utf-8") as f:
            f.write('<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>PyTextAD</title></head>'
                    f'<body style="font-family:sans-serif;max-width:900px;margin:24px auto">{self.html}'
                    "</body></html>\n")


def highlight(tokens, scores, labels=None, vmin=None, vmax=None, color=(232, 89, 12), show_scores=True):
    """Colour every word by its anomaly score.

    The more anomalous a word, the stronger its background colour. Scores are scaled to
    [0, 1] with ``vmin`` and ``vmax``, by default the smallest and largest score over all
    the documents given, so that documents shown together are comparable.

    Parameters
    ----------
    tokens : list of str, or list of list of str
        One document (a list of words) or several.
    scores : array-like, or list of array-like
        One score per word, for example from ``TokenDetector.token_scores``. Words with a
        NaN score (cut off by truncation) are shown in grey.
    labels : array-like, or list of array-like, optional
        True 0/1 word labels; anomalous words are underlined, to compare the scores with
        the ground truth.
    vmin, vmax : float, optional
        Scores mapped to no colour and to full colour.
    color : tuple of int, default=(232, 89, 12)
        RGB colour of the highlight.
    show_scores : bool, default=True
        Show each word's score when the mouse is over it.

    Returns
    -------
    Highlight
        Displayed as coloured text in Jupyter; ``.html`` holds the HTML and ``.save(path)``
        writes a web page.

    Examples
    --------
    >>> det = TokenDetector(TokenCore(), embedder=TokenEmbedder("bert-base-uncased")).fit(train.tokens)
    >>> highlight(test.tokens[:5], det.token_scores(test.tokens[:5]), labels=test.token_labels[:5])
    """
    single = len(tokens) == 0 or isinstance(tokens[0], str)
    docs = [tokens] if single else list(tokens)
    score_docs = [scores] if single else list(scores)
    label_docs = [None] * len(docs) if labels is None else ([labels] if single else list(labels))
    if not len(docs) == len(score_docs) == len(label_docs):
        raise ValueError(f"{len(docs)} documents, {len(score_docs)} score lists, {len(label_docs)} label lists")
    score_docs = [np.asarray(s, dtype=float) for s in score_docs]
    for i, (t, s, lab) in enumerate(zip(docs, score_docs, label_docs)):
        if len(t) != len(s) or (lab is not None and len(lab) != len(t)):
            raise ValueError(f"document {i}: {len(t)} words, {len(s)} scores"
                             + ("" if lab is None else f", {len(lab)} labels"))
    finite = np.concatenate([s[np.isfinite(s)] for s in score_docs]) if score_docs else np.array([])
    lo = vmin if vmin is not None else (float(finite.min()) if finite.size else 0.0)
    hi = vmax if vmax is not None else (float(finite.max()) if finite.size else 1.0)
    r, g, b = color

    parts = []
    for words, s, lab in zip(docs, score_docs, label_docs):
        spans = []
        for j, (w, v) in enumerate(zip(words, s)):
            style = ["padding:1px 3px", "border-radius:3px", "line-height:1.9"]
            if np.isfinite(v):
                alpha = 0.0 if hi <= lo else float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
                style.append(f"background:rgba({r},{g},{b},{alpha:.2f})")
                title = f"{v:.4g}"
            else:
                style.append("color:#888")
                title = "no score"
            if lab is not None and int(lab[j]) == 1:
                style.append("text-decoration:underline;text-decoration-thickness:2px")
            tip = f' title="{title}"' if show_scores else ""
            spans.append(f'<span style="{";".join(style)}"{tip}>{html.escape(str(w))}</span>')
        parts.append('<div style="margin:6px 0">' + " ".join(spans) + "</div>")
    return Highlight("".join(parts))
