"""Runs under REF/py38 (Python 3.8 + transformers 3.0.2) with the ORIGINAL DATE code. Dumps reference outputs."""
import csv, os, pickle, random
import numpy as np, torch
torch.Tensor.cuda = lambda self, *a, **k: self                 # original hard-codes .cuda()
from transformers import ElectraConfig, ElectraTokenizer
from simpletransformers.custom_models.models import ElectraForLanguageModelingModel
from simpletransformers.language_modeling.language_modeling_model import LanguageModelingModel
from simpletransformers.language_modeling.language_modeling_utils import encode_sliding_window_custom, mask_tokens

REF = os.environ["PYTEXTAD_REF"]
MASKS = f"{REF}/date/experiments/pseudo_labels128_p50.pkl"
torch.manual_seed(0); random.seed(0); np.random.seed(0)
tok = ElectraTokenizer(vocab_file=f"{REF}/data/vocab.txt", do_lower_case=True)
masks = pickle.load(open(MASKS, "rb"))
extra = {"max_seq_length": 130, "random_generator": 1, "replace_tokens": 0}
gen_cfg = ElectraConfig(embedding_size=128, hidden_size=16, num_hidden_layers=1)              # train_ag.py
dis_cfg = ElectraConfig(hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5, embedding_size=128,
                        hidden_size=256, num_hidden_layers=4)
model = ElectraForLanguageModelingModel(dis_cfg, output_size=len(masks), extra_args=extra,
                                        generator_config=gen_cfg, discriminator_config=dis_cfg, random_generator=1)
model.eval()

texts = [(r[1] + " . " + r[2]).replace("\\", " ") for r in csv.reader(open(f"{REF}/data/test.csv"))][:40]
texts.append(" ".join(texts[:4]))                               # a long document -> several windows
windows = [w for t in texts for w in encode_sliding_window_custom((t, 130, 2, 0.8, False), tok)]
batch = torch.tensor(windows)

class Args: mlm = True; mlm_probability = 0.15; model_type = "electra"
lm = LanguageModelingModel.__new__(LanguageModelingModel)
lm.args, lm.tokenizer, lm.masks, lm.extra_args, lm.model, lm.device = Args(), tok, masks, extra, model, "cpu"
pl_rtd = np.asarray(LanguageModelingModel.test_anomaly(lm, batch)[3])   # PL_RTD per window

random.seed(1); torch.manual_seed(1)
inp, labels, clf = mask_tokens(batch[:16], tok, pickle.load(open(MASKS, "rb")), Args(), train=True)
torch.manual_seed(2)
out = model(inp, masked_lm_labels=labels)                     # the call made in the training loop
rmd = torch.nn.CrossEntropyLoss()(out[2], clf.squeeze(-1).long())
loss = 100 * rmd + 50 * out[4]

sd0 = {k: v.detach().numpy().copy() for k, v in model.discriminator_model.state_dict().items()}

# --- (3) a few optimisation steps with the ORIGINAL optimiser set-up
# (parameter groups copied from LanguageModelingModel.train_anomaly; eval mode so that dropout,
#  whose call order differs between transformers versions, does not enter the comparison)
from torch.optim import AdamW
LR, WD = 1e-3, 0.1
no_decay = ["bias", "LayerNorm.weight"]
groups = [{"params": [p for n, p in model.generator_model.named_parameters() if not any(nd in n for nd in no_decay)],
           "weight_decay": WD, "lr": LR},
          {"params": [p for n, p in model.generator_model.named_parameters() if any(nd in n for nd in no_decay)], "lr": LR}]
shared = ["electra.embeddings.word_embeddings.weight", "electra.embeddings.position_embeddings.weight",
          "electra.embeddings.token_type_embeddings.weight"]
disc = [(n, p) for n, p in model.discriminator_model.named_parameters() if n not in shared]
groups.append({"params": [p for n, p in disc if not any(nd in n for nd in no_decay)], "weight_decay": WD, "lr": LR})
groups.append({"params": [p for n, p in disc if any(nd in n for nd in no_decay)], "lr": LR})
opt = AdamW(groups, lr=LR, eps=1e-8, amsgrad=True)
steps = []
random.seed(3); torch.manual_seed(3)
for k in range(3):
    inp3, labels3, clf3 = mask_tokens(batch[16 * k % 32:16 * k % 32 + 16], tok, pickle.load(open(MASKS, "rb")), Args(), train=True)
    o3 = model(inp3, masked_lm_labels=labels3)
    l = 100 * torch.nn.CrossEntropyLoss()(o3[2], clf3.squeeze(-1).long()) + 50 * o3[4]
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    steps.append((o3[3].numpy(), o3[6].numpy(), clf3.squeeze(-1).numpy(), l.item()))
sd3 = {k: v.detach().numpy() for k, v in model.discriminator_model.state_dict().items()}

pickle.dump(dict(sd=sd0, sd3=sd3, steps=steps, lr=LR,
                 texts=texts, windows=windows, pl_rtd=pl_rtd, corrupted=out[3].numpy(),
                 rtd_labels=out[6].numpy(), rmd_labels=clf.squeeze(-1).numpy(), loss=loss.item()),
            open(f"{REF}/date_ref.pkl", "wb"))
print("reference DATE outputs written")
