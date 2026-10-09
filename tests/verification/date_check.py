"""Loads the original DATE discriminator weights into pytextad.DATE and compares every stage."""
import pickle, sys, numpy as np, torch
from common import REF, DATA, report
from pytextad.models.date import DATE, _DATENet

d = pickle.load(open(f"{REF}/date_ref.pkl", "rb"))
m = DATE(tokenizer=f"{DATA}/vocab.txt", device="cpu", masks=f"{REF}/date/experiments/pseudo_labels128_p50.pkl")
m._load_tokenizer(); m.vocab_size_ = len(m.tok_); m._build_masks()
net = _DATENet(m.vocab_size_, 50)
def rename(k):
    for a, b in [("electra.", "encoder."), ("discriminator_predictions.dense_prediction.", "rtd_out."),
                 ("discriminator_predictions.dense.", "rtd_dense."), ("rmd_predictions.fc1.", "rmd_fc1."),
                 ("rmd_predictions.fc2.", "rmd_fc2.")]:
        if k.startswith(a):
            return b + k[len(a):]
    return k
sd = {rename(k): torch.from_numpy(v) for k, v in d["sd"].items()}
missing, unexpected = net.load_state_dict(sd, strict=False)
missing = [k for k in missing if "position_ids" not in k]
ok = report("DATE weights map 1:1", not missing and not unexpected, f"missing={missing} unexpected={unexpected}")
m.net_ = net.eval()

ours = m._pack([w for ids, _ in m._encode(d["texts"]) for _, w in m._windows(ids)])[0].numpy()
ref = np.array(d["windows"])
ok &= report("DATE tokenisation + sliding windows", ours.shape == ref.shape and (ours == ref).all(), f"{len(ref)} windows")

win, _, _ = m._score(d["texts"])
diff = np.abs(np.concatenate([np.array(w) for w in win]) - d["pl_rtd"]).max()
ok &= report("DATE window scores (PL_RTD)", diff < 1e-5, f"max diff {diff:.2e}")

Xc = torch.from_numpy(d["corrupted"])
with torch.no_grad():
    loss = m._loss(Xc, Xc != m.tok_.pad_token_id, torch.from_numpy(d["rtd_labels"]).float(),
                   torch.from_numpy(d["rmd_labels"]).long()).item()
ok &= report("DATE training loss (RTD x50 + RMD x100)", abs(loss - d["loss"]) < 1e-3, f"orig {d['loss']:.5f} ours {loss:.5f}")

orig_in = torch.tensor(d["windows"][:16])
ours_pos = m.masks_[torch.from_numpy(d["rmd_labels"]).long()] & (orig_in != m.tok_.pad_token_id)
changed = torch.from_numpy(d["corrupted"]) != orig_in
same = bool(not (changed & ~ours_pos).any() and ((torch.from_numpy(d["rtd_labels"]) == 1) == changed).all())
ok &= report("DATE corrupted positions and RTD labels", same, "replaced positions follow the chosen masks")
# optimisation steps on the same corrupted batches with the same optimiser set-up
m.lr = d["lr"]
net.load_state_dict({rename(k): torch.from_numpy(v) for k, v in d["sd"].items()}, strict=False)
net.eval()
opt = m._make_optimizer()
loss_diff = 0.0
for Xc, rtd_y, rmd_y, l_orig in d["steps"]:
    Xc = torch.from_numpy(Xc)
    l = m._train_step(Xc, Xc != m.tok_.pad_token_id, torch.from_numpy(rtd_y).float(), torch.from_numpy(rmd_y).long(), opt)
    loss_diff = max(loss_diff, abs(l - l_orig))
ours_sd = net.state_dict()
# Adam rescales gradients, so a gradient that is exactly zero in theory (e.g. attention key
# biases, which softmax ignores) turns float noise into a visible update. Compare per tensor,
# mean |difference| / mean |update|, over tensors that really move (mean update > 1 % of lr).
ratio = max(np.abs(ours_sd[rename(k)].numpy() - v).mean() / np.abs(v - d["sd"][k]).mean()
            for k, v in d["sd3"].items() if "position_ids" not in k and np.abs(v - d["sd"][k]).mean() > 0.01 * d["lr"])
ok &= report("DATE 3 optimiser steps (AdamW groups, weight decay, clipping)", ratio < 1e-3 and loss_diff < 1e-5,
             f"loss diff per step {loss_diff:.1e}; worst tensor mean|diff|/mean|update| = {ratio:.1e}")
sys.exit(0 if ok else 1)
