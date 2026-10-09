---
license: cc-by-4.0
task_categories:
- token-classification
- text-classification
language:
- en
tags:
- anomaly-detection
- token-level
- arxiv:2601.13644
pretty_name: PyTextAD benchmark datasets
configs:
- config_name: sms_spam
  data_files: data/sms_spam.jsonl
- config_name: restaurant_review
  data_files: data/restaurant_review.jsonl
- config_name: grammar_correction
  data_files: data/grammar_correction.jsonl
- config_name: hate_speech
  data_files: data/hate_speech.jsonl
- config_name: olid
  data_files: data/olid.jsonl
- config_name: restaurant_review2
  data_files: data/restaurant_review2.jsonl
---

# PyTextAD benchmark datasets

Text anomaly detection datasets with a 0/1 anomaly label for every word, used by the
[PyTextAD](https://github.com/charles-cao/pytextad) library. A document is anomalous if any of
its words is anomalous, so each dataset serves both token-level and document-level evaluation.

```python
from pytextad.datasets import load_dataset     # pip install pytextad
ds = load_dataset("restaurant_review")
ds.tokens, ds.token_labels, ds.labels
```

| Dataset | Documents | Anomalous | Words | Anomalous words | Anomaly |
|---|---|---|---|---|---|
| `sms_spam` | 4,518 | 393 | 81,570 | 418 | injected gibberish |
| `restaurant_review` | 1,100 | 50 | 35,488 | 282 | negative sentiment |
| `grammar_correction` | 300 | 30 | 2,746 | 47 | grammatical errors |
| `hate_speech` | 4,302 | 140 | 99,390 | 288 | hateful or offensive words |
| `olid` | 650 | 30 | 21,156 | 58 | offensive words |
| `restaurant_review2` | 520 | 25 | 33,529 | 94 | negative sentiment |

Each line of a file is one document: `{"id", "text", "tokens", "labels"}`, where `labels`
has one 0/1 value per entry of `tokens`.

## Sources

- `sms_spam`: SMS Spam Collection, taken from [NLP-ADBench](https://huggingface.co/datasets/kendx/NLP-ADBench);
  meaningless character sequences were injected into some messages.
- `restaurant_review`: Google Maps reviews of a restaurant in the USA.
- `grammar_correction`: [Kaggle grammar-correction](https://www.kaggle.com/datasets/satishgunjal/grammar-correction).
- `hate_speech`: tweets from Davidson et al., *Automated Hate Speech Detection and the Problem
  of Offensive Language*, ICWSM 2017.
- `olid`: tweets from Zampieri et al., *Predicting the Type and Target of Offensive Posts in
  Social Media* (OLID), NAACL 2019.
- `restaurant_review2`: reviews of a second restaurant.

The word-level annotations are released under CC BY 4.0; the texts remain subject to the terms
of their original sources.

## Citation

The first three datasets were introduced in
[Towards Token-Level Text Anomaly Detection](https://huggingface.co/papers/2601.13644):

```bibtex
@inproceedings{cao2026tokenlevel,
  title     = {Towards Token-Level Text Anomaly Detection},
  author    = {Cao, Yang and Yu, Bicheng and Yang, Sikun and Liu, Ming and Yang, Yujiu},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  doi       = {10.1145/3774904.3792952}
}
```
