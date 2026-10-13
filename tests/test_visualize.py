import numpy as np
import pytest

from pytextad import TokenCore, TokenDetector, TokenEmbedder, highlight
from pytextad.visualize import Highlight

from conftest import DEV, WORDS, make_texts


def test_single_document_colours_and_tooltips():
    h = highlight(["good", "food", "<b>"], [0.0, 1.0, 0.5])
    assert isinstance(h, Highlight) and h._repr_html_() == h.html == str(h)
    assert h.html.count("<div") == 1 and h.html.count("<span") == 3
    assert "rgba(232,89,12,0.00)" in h.html and "rgba(232,89,12,1.00)" in h.html and "rgba(232,89,12,0.50)" in h.html
    assert 'title="1"' in h.html and "&lt;b&gt;" in h.html          # words are escaped


def test_several_documents_share_one_scale_and_show_labels(tmp_path):
    h = highlight([["a", "b"], ["c"]], [[0.0, 2.0], [np.nan]], labels=[[0, 1], [0]], show_scores=False)
    assert h.html.count("<div") == 2
    assert "rgba(232,89,12,1.00)" in h.html and "color:#888" in h.html and "title=" not in h.html
    assert h.html.count("underline") == 1
    page = tmp_path / "h.html"
    h.save(page)
    assert page.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_fixed_scale_and_degenerate_cases():
    h = highlight(["x", "y"], [5.0, 15.0], vmin=0, vmax=10, color=(0, 0, 255))
    assert "rgba(0,0,255,0.50)" in h.html and "rgba(0,0,255,1.00)" in h.html       # clipped at vmax
    assert "rgba(232,89,12,0.00)" in highlight(["x", "y"], [3.0, 3.0]).html          # equal scores
    assert highlight([], []).html == '<div style="margin:6px 0"></div>'
    assert "color:#888" in highlight(["x"], [np.nan]).html                            # no finite score


def test_checks_lengths():
    with pytest.raises(ValueError, match="2 words, 1 scores"):
        highlight(["a", "b"], [1.0])
    with pytest.raises(ValueError, match="2 labels"):
        highlight(["a"], [1.0], labels=[0, 1])
    with pytest.raises(ValueError, match="2 documents, 1 score lists"):
        highlight([["a"], ["b"]], [[1.0]])


def test_with_a_token_detector(tiny_bert):
    path, _ = tiny_bert
    det = TokenDetector(TokenCore(), embedder=TokenEmbedder(path, device=DEV))
    det.fit([t.split() for t in make_texts(20, WORDS[:5], 0)])
    test = [["stock", "player", "score", "market"]]
    h = highlight(test, det.token_scores(test))
    assert h.html.count("<span") == 4


def test_reference_sets_the_colour_scale():
    ref = [np.array([1.0, 2.0, 3.0]), np.array([2.0, np.nan, 100.0])]     # median 2, 99th pct ~96
    h = highlight(["normal", "odd"], [2.0, 200.0], reference=ref)
    assert "rgba(232,89,12,0.00)" in h.html and "rgba(232,89,12,1.00)" in h.html
    lo, hi = np.percentile([1, 2, 3, 2, 100], [50, 99])
    mid = highlight(["w"], [(lo + hi) / 2], reference=[1, 2, 3, 2, 100])   # flat reference
    assert "rgba(232,89,12,0.50)" in mid.html
    assert "rgba(232,89,12,1.00)" in highlight(["w"], [5.0], reference=ref, vmax=5.0).html  # vmax wins
    with pytest.raises(ValueError, match="no finite score"):
        highlight(["w"], [1.0], reference=[np.nan])
