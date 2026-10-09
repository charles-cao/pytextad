"""Official CVDD (CVDDTrainer + CVDDNet) vs pytextad.CVDD: same init, same batches, several epoch counts."""
import logging, sys, types, numpy as np, torch, torch.nn as nn
from common import REF, report
sys.path.insert(0, f"{REF}/CVDD-PyTorch/src")
six = types.ModuleType("torch._six"); six.int_classes = int; six.string_classes = str; sys.modules["torch._six"] = six
for name in ["pytorch_pretrained_bert", "pytorch_pretrained_bert.modeling", "pytorch_pretrained_bert.tokenization"]:
    mod = types.ModuleType(name); mod.BertModel = object; mod.BertTokenizer = object; sys.modules[name] = mod
import optim.cvdd_trainer as ot
from networks.cvdd_Net import CVDDNet
import pytextad.models.cvdd as tc
from sklearn.cluster import KMeans
logging.disable(logging.CRITICAL)

def run(epochs, milestones):
    torch.manual_seed(0); rng = np.random.RandomState(0)
    V, D, L, N, B = 300, 30, 12, 256, 64
    emb = (10 * rng.randn(V, D)).astype(np.float32)   # large norms so that gradient clipping (0.5) is active
    tokens = rng.randint(1, V, size=(N, L))
    X = [emb[t] for t in tokens]

    class Frozen(nn.Module):
        def __init__(s):
            super().__init__(); s.e = nn.Embedding.from_pretrained(torch.from_numpy(emb), freeze=True); s.embedding_size = D
        def forward(s, x): return s.e(x)

    class FakeDS:
        train_set = None; encoder = None
        def loaders(self, batch_size, num_workers=0):
            b = [(list(range(s, s + batch_size)), torch.from_numpy(tokens[s:s + batch_size]).t().contiguous(),
                  torch.zeros(batch_size), torch.empty(0)) for s in range(0, N, batch_size)]
            return b, b

    official = CVDDNet(Frozen(), attention_size=20, n_attention_heads=3)
    W1, W2 = official.self_attention.W1.weight.data.clone(), official.self_attention.W2.weight.data.clone()
    means = np.stack([x.mean(0) for x in X]); means /= np.linalg.norm(means, axis=1, keepdims=True)
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(means)
    centers = (km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)).astype(np.float32)
    ot.initialize_context_vectors = lambda net, loader, device: centers
    ot.get_top_words_per_context = lambda *a, **k: None
    ot.CVDDTrainer(lr=0.01, n_epochs=epochs, lr_milestones=milestones, batch_size=B, lambda_p=10.0,
                   alpha_scheduler="logarithmic", weight_decay=0.5e-6, device="cpu").train(FakeDS(), official)

    real = tc._CVDDNet
    def make(dim, a, r):
        n = real(dim, a, r); n.W1.weight.data = W1.clone(); n.W2.weight.data = W2.clone(); return n
    tc._CVDDNet = make
    ours = tc.CVDD(n_heads=3, attention_size=20, lambda_p=10.0, alpha_scheduler="logarithmic", n_epochs=epochs,
                   lr=0.01, lr_milestones=milestones, batch_size=B, device="cpu", random_state=0)
    ours._batches = lambda n, shuffle: (np.arange(s, min(s + B, n)) for s in range(0, n, B))
    ours.fit(X)
    tc._CVDDNet = real
    with torch.no_grad():
        d_off = official(torch.from_numpy(tokens).t())[0].mean(1).numpy()
    diffs = [(official.c.squeeze(0) - ours.net_.c).abs().max().item(),
             (official.self_attention.W1.weight - ours.net_.W1.weight).abs().max().item(),
             np.abs(d_off - ours.decision_scores_).max()]
    return max(diffs), official.alpha, ours.net_.alpha


def run_padded():
    """Variable-length documents. The official code pads with index 0 and does not mask; with static
    word vectors the pad vector is zero, which only rescales M, so cosine distances must be identical."""
    torch.manual_seed(0); rng = np.random.RandomState(0)
    V, D, N, B, EP = 300, 30, 256, 64, 23
    emb = (10 * rng.randn(V, D)).astype(np.float32); emb[0] = 0          # index 0 = padding, zero vector
    lens = rng.randint(3, 25, N); Lmax = lens.max()
    tokens = np.zeros((N, Lmax), dtype=int)
    for i, l in enumerate(lens): tokens[i, :l] = rng.randint(1, V, l)
    X = [emb[tokens[i, :l]] for i, l in enumerate(lens)]
    class Frozen(nn.Module):
        def __init__(s): super().__init__(); s.e = nn.Embedding.from_pretrained(torch.from_numpy(emb), freeze=True); s.embedding_size = D
        def forward(s, x): return s.e(x)
    class DS:
        train_set = None; encoder = None
        def loaders(self, batch_size, num_workers=0):
            b = []
            for s in range(0, N, batch_size):
                L = lens[s:s + batch_size].max()        # official pads each batch to its longest document
                b.append((None, torch.from_numpy(tokens[s:s + batch_size, :L]).t().contiguous(), torch.zeros(1), torch.empty(0)))
            return b, b
    off = CVDDNet(Frozen(), attention_size=20, n_attention_heads=3)
    W1, W2 = off.self_attention.W1.weight.data.clone(), off.self_attention.W2.weight.data.clone()
    # official k-means input: mean over ALL positions of the padded batch (pads included), then L2-normalised
    means = np.stack([x.mean(0) for x in X]); means /= np.linalg.norm(means, axis=1, keepdims=True)
    km = KMeans(3, n_init=10, random_state=0).fit(means)
    C = (km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)).astype(np.float32)
    ot.initialize_context_vectors = lambda *a: C; ot.get_top_words_per_context = lambda *a, **k: None
    ot.CVDDTrainer(lr=0.01, n_epochs=EP, lr_milestones=(10,), batch_size=B, lambda_p=10.0, alpha_scheduler="logarithmic",
                   weight_decay=0.5e-6, device="cpu").train(DS(), off)
    real = tc._CVDDNet
    def mk(d, a, r):
        n = real(d, a, r); n.W1.weight.data = W1.clone(); n.W2.weight.data = W2.clone(); return n
    tc._CVDDNet = mk
    ours = tc.CVDD(n_heads=3, attention_size=20, lambda_p=10.0, n_epochs=EP, lr=0.01, lr_milestones=(10,), batch_size=B, device="cpu")
    ours._batches = lambda n, shuffle: (np.arange(s, min(s + B, n)) for s in range(0, n, B))
    ours.fit(X)
    tc._CVDDNet = real
    with torch.no_grad():
        s_off = np.concatenate([off(torch.from_numpy(tokens[s:s + B, :lens[s:s + B].max()]).t())[0].mean(1).numpy() for s in range(0, N, B)])
    return max(np.abs(s_off - ours.decision_scores_).max(), (off.c.squeeze(0) - ours.net_.c).abs().max().item())

ok = True
for epochs, ms in [(1, ()), (10, ()), (15, (6,)), (23, (10,))]:
    diff, a_off, a_our = run(epochs, ms)
    ok &= report(f"CVDD {epochs} epochs, lr milestones {ms}", diff < 1e-5 and a_off == a_our,
                 f"max diff (c, W1, scores) {diff:.1e}, alpha {a_off:g}/{a_our:g}")
diff = run_padded()
ok &= report("CVDD variable lengths, zero pad vectors (official: no masking; ours: masking)", diff < 1e-5,
             f"max diff (scores, c) {diff:.1e}")
sys.exit(0 if ok else 1)
