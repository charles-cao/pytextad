"""Official TF RSRAE vs pytextad.RSRAE: identical initial weights and batch order."""
import pickle, sys, numpy as np, torch
from common import REF, report
from pytextad.models.rsrae import RSRAE, _RSRAENet
EPOCHS, NORM = int(sys.argv[1]), sys.argv[2]
d = pickle.load(open(f"{REF}/rsrae_ref_{EPOCHS}_{NORM}.pkl", "rb")); X, W0 = d["X"], d["W0"]
m = RSRAE(n_epochs=EPOCHS, loss_norm_type=NORM, device="cpu")
m.net_ = net = _RSRAENet(40, (32, 64, 128), 10, torch.tanh, True, "official")
put = lambda p, a: setattr(p, "data", torch.from_numpy(np.ascontiguousarray(a)).float())
for lin, n in zip(list(net.enc) + list(net.dec), ["encoder/fc1", "encoder/fc2", "encoder/fc3",
                                                   "decoder/revealed", "decoder/dfc3", "decoder/dfc2"]):
    put(lin.weight, W0[n + "/kernel:0"].T); put(lin.bias, W0[n + "/bias:0"])
put(net.out.weight, W0["decoder/dfc1/kernel:0"].T); put(net.out.bias, W0["decoder/dfc1/bias:0"])
put(net.A, W0["rsr/layer_rsr:0"])
bns = ["encoder/batch_normalization", "encoder/batch_normalization_1", "encoder/batch_normalization_2",
       "decoder/batch_normalization_3", "decoder/batch_normalization_4", "decoder/batch_normalization_5"]
for bn, n in zip(list(net.enc_bn) + list(net.dec_bn), bns):
    put(bn.gamma, W0[n + "/gamma:0"]); put(bn.beta, W0[n + "/beta:0"])
ok = report("RSRAE official BN never updates its statistics", d["bn_stats_untouched"], "moving mean/var unchanged by training")
d0 = np.abs(m.reconstruct(X) - d["out0"]).max()
ok &= report("RSRAE forward pass at initialisation", d0 < 1e-5, f"max diff {d0:.1e}")
m._train_loop(torch.from_numpy(X), d["idx"])
R = m.reconstruct(X)
cos = lambda R: (X * R).sum(1) / (np.linalg.norm(R, axis=1) + 1e-6) / (np.linalg.norm(X, axis=1) + 1e-6)
d1, d2 = np.abs(R - d["out"]).max(), np.abs(cos(R) - cos(d["out"])).max()
ok &= report(f"RSRAE after {EPOCHS} epochs ({NORM})", d1 < 1e-3 and d2 < 1e-3,
             f"reconstruction diff {d1:.1e}, score diff {d2:.1e} (float32 accumulation)")
sys.exit(0 if ok else 1)
