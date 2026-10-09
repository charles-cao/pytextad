#!/usr/bin/env bash
# Prepares everything the verification scripts need. Run once, from this directory.
#   REF  (default ./reference): clones of the original repositories, a Python 3.8 env, tiny test models
# Requires: git, curl, uv (pip install uv), a Python >= 3.10 env with pytextad's dependencies.
set -euo pipefail
REF=${PYTEXTAD_REF:-$(pwd)/reference}
mkdir -p "$REF/data" && cd "$REF"

clone() { [ -d "$2" ] || git clone -q --depth 1 "$1" "$2"; }
clone https://github.com/bit-ml/date.git            date
clone https://github.com/lukasruff/CVDD-PyTorch.git CVDD-PyTorch
clone https://github.com/dmzou/RSRAE.git            RSRAE
clone https://github.com/arav1ndajay/fate.git       fate

# data: AG News (csv) and the bert-base-uncased WordPiece vocabulary
[ -f data/train.csv ] || curl -sSfL -o data/train.csv https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv
[ -f data/test.csv ]  || curl -sSfL -o data/test.csv  https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv
[ -f data/vocab.txt ] || curl -sSfL -o data/vocab.txt https://raw.githubusercontent.com/microsoft/SDNet/master/bert_vocab_files/bert-base-uncased-vocab.txt

# Python 3.8 env that runs the ORIGINAL DATE code (transformers 3.0.2)
if [ ! -x py38/bin/python ]; then
  uv python install 3.8
  uv venv -q -p 3.8 py38
  VIRTUAL_ENV="$REF/py38" uv pip install -q --prerelease=allow --index-strategy unsafe-best-match \
      "torch==2.4.1" "transformers==3.0.2" "tokenizers==0.8.1rc1" "numpy<1.25" "scikit-learn==1.0.2" \
      tensorboardx matplotlib pandas tqdm regex sentencepiece requests filelock sacremoses packaging
  VIRTUAL_ENV="$REF/py38" uv pip install -q --no-deps -e "$REF/date/simpletransformers"
fi

# extra packages for the current env: official RSRAE (TensorFlow), CVDD (torchnlp, nltk, joypy), FATE (sentence-transformers)
python -m pip install -q tensorflow-cpu tf_keras pytorch-nlp nltk joypy sentence-transformers

# tiny randomly initialised encoders (no model download needed)
python - <<'PY'
import os
from transformers import BertConfig, BertModel, BertTokenizerFast
from sentence_transformers import SentenceTransformer, models
ref = os.environ.get("PYTEXTAD_REF", os.getcwd())
vocab = {w.rstrip("\n"): i for i, w in enumerate(open(f"{ref}/data/vocab.txt", encoding="utf-8"))}
try:
    tok = BertTokenizerFast(vocab=vocab, do_lower_case=True)
except TypeError:
    tok = BertTokenizerFast(vocab_file=f"{ref}/data/vocab.txt", do_lower_case=True)
import torch; torch.manual_seed(0)
BertModel(BertConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128)).save_pretrained(f"{ref}/tinybert")
tok.save_pretrained(f"{ref}/tinybert")
t = models.Transformer(f"{ref}/tinybert", max_seq_length=128)
SentenceTransformer(modules=[t, models.Pooling(t.get_word_embedding_dimension(), "mean")]).save(f"{ref}/tinysbert")
print("tiny models ready")
PY
echo "reference setup done in $REF"
