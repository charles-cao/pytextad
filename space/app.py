"""PyTextAD online demo (Hugging Face Space): which words of a text are anomalous?

For each domain, TokenCore learns what normal text looks like from normal documents of a
PyTextAD dataset. A text typed by the user is split into words; every word gets an anomaly
score, shown as a colour, and the text gets a score compared with held-out normal documents.

Everything runs on the CPU (one text at a time is fast), so the demo uses no GPU quota.
On ZeroGPU hardware, which requires a function marked with @spaces.GPU at startup, an
unused marked function is defined.

Environment variables (for local testing): PYTEXTAD_DEMO_MODEL (encoder, default
bert-base-uncased), PYTEXTAD_DEMO_N_REF (normal documents per domain, default 300).
"""
try:                                    # ZeroGPU hardware: must be imported before torch, and
    import spaces                       # needs one @spaces.GPU function at startup

    @spaces.GPU
    def _zerogpu_startup_check():
        """Not called: the demo runs on the CPU."""
except ImportError:                     # any other hardware, or local runs
    pass

import os
import re
from functools import lru_cache

import gradio as gr
import numpy as np

from pytextad import TokenCore, TokenDetector, TokenEmbedder, highlight
from pytextad import datasets

MODEL = os.environ.get("PYTEXTAD_DEMO_MODEL", "bert-base-uncased")
N_REF = int(os.environ.get("PYTEXTAD_DEMO_N_REF", "300"))
N_HELD_OUT = 100
WORD = re.compile(r"\w+|[^\w\s]")      # the word segmentation of the datasets' normal documents

DOMAINS = {
    "Restaurant reviews": ("restaurant_review", "normal restaurant reviews", [
        "The pasta was cold and the waiter was rude, we will never come back.",
        "Great food, friendly staff and a cozy atmosphere. Highly recommended!",
    ]),
    "SMS messages": ("sms_spam", "normal SMS messages", [
        "Hey, are we still meeting at 6 tonight? Let me know.",
        "Call me when you get home xqzv kplmw drtyy ok",
    ]),
    "English sentences": ("grammar_correction", "correct English sentences", [
        "She has finished her homwork yesterday.",
        "He has lived in this city for ten years.",
    ]),
}


@lru_cache(maxsize=None)
def embedder():
    return TokenEmbedder(MODEL, device="cpu")


@lru_cache(maxsize=None)
def reference(domain):
    """Detector fitted on normal documents of the domain, and the scores of held-out normal
    documents (TokenCore scores its own training words 0, so they cannot serve as reference).
    A dataset with fewer than N_REF + 100 normal documents keeps a third of them for the
    reference (grammar_correction has 270: 180 for fitting, 90 for the reference)."""
    ds = datasets.load_dataset(DOMAINS[domain][0])
    normal = np.random.RandomState(0).permutation(ds.normal_indices)
    n_held = min(N_HELD_OUT, max(len(normal) - N_REF, len(normal) // 3))
    n_fit = min(N_REF, len(normal) - n_held)
    fit_docs = [ds.tokens[i] for i in normal[:n_fit]]
    held_out = [ds.tokens[i] for i in normal[n_fit:n_fit + n_held]]
    det = TokenDetector(TokenCore(), embedder=embedder()).fit(fit_docs)
    word_scores = det.token_scores(held_out)
    doc_scores = np.array([np.nanmean(s) for s in word_scores])
    words = np.concatenate(word_scores)
    words = words[np.isfinite(words)]
    # colour scale: the 75% most typical normal words stay white, words beyond the
    # 99.5th percentile of normal words get the full colour
    return det, doc_scores, float(np.percentile(words, 75)), float(np.percentile(words, 99.5))


def detect(text, domain):
    """Coloured words (HTML) and a one-line summary for one text."""
    words = WORD.findall(text or "")
    if not words:
        return "", "Type or paste a text."
    det, ref_docs, lo, hi = reference(domain)
    scores = det.token_scores([words])[0]
    doc = float(np.nanmean(scores))
    pct = 100.0 * float(np.mean(ref_docs < doc))
    ranked = sorted(zip(np.nan_to_num(scores, nan=-np.inf), words), reverse=True)
    top = [w for _, w in ranked if any(c.isalnum() for c in w)][:3]          # skip punctuation
    summary = (f"**More anomalous than {pct:.0f}% of {DOMAINS[domain][1]}.** "
               f"Most anomalous words: {', '.join(top)}.")
    return highlight(words, scores, vmin=lo, vmax=hi).html, summary


def build():
    with gr.Blocks(title="PyTextAD demo") as demo:
        gr.Markdown(
            "# PyTextAD: which words are anomalous?\n"
            "Choose a domain, type a text and click **Detect**. Every word is coloured by how "
            "unusual it is compared with normal texts of that domain (stronger colour, more "
            "anomalous); the detector (TokenCore on BERT embeddings) learned only from normal "
            "texts, without any labelled anomaly. The first run of a domain takes about a "
            "minute while the reference texts are embedded.  \n"
            "[GitHub](https://github.com/charles-cao/pytextad) · "
            "[Documentation](https://pytextad.readthedocs.io) · `pip install pytextad`")
        domain = gr.Dropdown(list(DOMAINS), value=next(iter(DOMAINS)), label="Domain")
        text = gr.Textbox(lines=3, label="Text")
        button = gr.Button("Detect", variant="primary")
        summary = gr.Markdown()
        coloured = gr.HTML()
        gr.Examples([[t, d] for d, (_, _, ex) in DOMAINS.items() for t in ex], inputs=[text, domain])
        button.click(detect, inputs=[text, domain], outputs=[coloured, summary])
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch()
