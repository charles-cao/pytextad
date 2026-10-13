---
title: PyTextAD
emoji: 🔍
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: bsd-2-clause
short_description: Which words of a text are anomalous?
---

# PyTextAD demo

Type a text and see which of its words are anomalous compared with normal texts of a domain
(restaurant reviews, SMS messages or English sentences). The detector is TokenCore on BERT
embeddings, fitted on normal documents of the PyTextAD datasets only.

* Library: https://github.com/charles-cao/pytextad (`pip install pytextad`)
* Documentation: https://pytextad.readthedocs.io
* Datasets: https://huggingface.co/datasets/Charles-Cao/pytextad
