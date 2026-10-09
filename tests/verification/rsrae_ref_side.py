"""Runs the OFFICIAL TensorFlow RSRAE (in its own process: TF and PyTorch in one process can segfault)."""
import os, pickle, sys, numpy as np
os.environ["TF_USE_LEGACY_KERAS"] = "1"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
REF = os.environ["PYTEXTAD_REF"]; sys.path.insert(0, f"{REF}/RSRAE")
from RSRAE.model import CAE
EPOCHS, NORM, SEED = int(sys.argv[1]), sys.argv[2], 7
rng = np.random.RandomState(0)
X = np.clip(np.concatenate([rng.randn(300, 40) * 0.2, rng.randn(30, 40) * 0.6]), -0.99, 0.99).astype(np.float32)
kw = dict(input_shape=(40,), hidden_layer_sizes=(32, 64, 128), intrinsic_size=10, activation=tf.nn.tanh,
          norm_type="L21", loss_norm_type=NORM, if_rsr=True, enforce_proj=True, all_alt=True,
          learning_rate=0.00025, batch_show=None, normalize=True, bn=True, random_seed=SEED)   # experiments.py
def weights(c):
    with c.graph.as_default():
        return {v.name: c.sess.run(v) for v in tf.trainable_variables()}
a = CAE(epoch_size=0, **kw); a.fit(X); W0 = weights(a)
b = CAE(epoch_size=0, **kw); b.fit(X); assert all(np.array_equal(W0[k], v) for k, v in weights(b).items())
with a.graph.as_default():
    moving = [a.sess.run(v) for v in tf.global_variables() if "moving_" in v.name]
c = CAE(epoch_size=EPOCHS, **kw); c.fit(X)
with c.graph.as_default():
    moving_after = [c.sess.run(v) for v in tf.global_variables() if "moving_" in v.name]
np.random.seed(SEED); idx = np.arange(len(X)); np.random.shuffle(idx)          # the official shuffle
pickle.dump(dict(X=X, W0=W0, out0=a.get_output(X), out=c.get_output(X), idx=idx,
                 bn_stats_untouched=all(np.array_equal(m0, m1) for m0, m1 in zip(moving, moving_after))),
            open(f"{REF}/rsrae_ref_{EPOCHS}_{NORM}.pkl", "wb"))
print("reference RSRAE outputs written")
