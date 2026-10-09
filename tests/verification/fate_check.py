"""Official FATE (arav1ndajay/fate src/main.py) vs pytextad.FATE: forward, full training loop, batch sampler."""
import csv, sys, numpy as np, torch
from common import REF, DATA, report
torch.Tensor.cuda = lambda self, *a, **k: self                      # official DeviationLoss calls .cuda()
torch.utils.data.Sampler.__init__ = lambda self, *a, **k: None       # official sampler uses the old signature
ENC, LR, EPOCHS = f"{REF}/tinysbert", 1e-3, 2
sys.argv = ["main.py", "--learning_rate", str(LR)]
sys.path.insert(0, f"{REF}/fate/src")
import main as off                                                   # official module (parses args, defines classes)
from balanced_sampler import BalancedBatchSampler
from utils import CustomDataset
from pytextad.models.fate import FATE
from transformers import AutoTokenizer

texts = [(r[1] + " . " + r[2]).replace("\\", " ") for r in csv.reader(open(f"{DATA}/train.csv"))][:200]
enc = AutoTokenizer.from_pretrained(ENC)(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
rng = np.random.RandomState(0); y = np.array([0] * 190 + [1] * 10)
batches = []
for _ in range(6):
    idx = np.concatenate([rng.choice(190, 8), 190 + rng.choice(10, 8)])
    batches.append((enc["input_ids"][idx], enc["attention_mask"][idx], torch.tensor(y[idx], dtype=torch.float64)))

torch.manual_seed(0)
official = off.SBERTWithAttention(ENC)
ours = FATE(encoder=ENC, lr=LR, device="cpu"); ours._build()
ours.net_.W1.weight.data = official.W1.weight.data.clone(); ours.net_.W2.weight.data = official.W2.weight.data.clone()
def off_scores():
    official.eval()
    with torch.no_grad():
        return official({"input_ids": enc["input_ids"][:40], "attention_mask": enc["attention_mask"][:40]})[0].numpy()
d0 = np.abs(off_scores() - ours.decision_function(texts[:40])).max()
ok = report("FATE scores at initialisation", d0 < 1e-6, f"max diff {d0:.1e}")

fm = off.FATEModel.__new__(off.FATEModel)                           # official training loop, unchanged
fm.model, fm.train_dataloader, fm.num_epochs, fm.device = official, batches, EPOCHS, torch.device("cpu")
fm.evaluate_during_training_steps, fm.criterion = 10 ** 9, off.DeviationLoss()
W1_init = official.W1.weight.detach().clone()
torch.manual_seed(123); off.FATEModel.train_model(fm)
torch.manual_seed(123); opt = torch.optim.Adam(ours.net_.parameters(), lr=LR)
for _ in range(EPOCHS):
    ours._train_steps(batches, opt)
moved = (official.W1.weight - W1_init).abs().max().item()
pairs = list(zip([p for n, p in official.named_parameters() if n.startswith("sbert.")], ours.net_.encoder.parameters()))
pairs += [(official.W1.weight, ours.net_.W1.weight), (official.W2.weight, ours.net_.W2.weight)]
dp = max((a - b).abs().max().item() for a, b in pairs)
d1 = np.abs(off_scores() - ours.decision_function(texts[:40])).max()
ok &= report(f"FATE training ({EPOCHS} epochs x {len(batches)} steps, dropout on, reference redrawn per batch)",
             moved > 1e-4 and dp < 1e-6 and d1 < 1e-6, f"weights moved {moved:.1e}; param diff {dp:.1e}; score diff {d1:.1e}")

# batch sampler: official BalancedBatchSampler vs the indices fit() draws, same NumPy seed
yy = np.array([0] * 50 + [1] * 10)
ds = CustomDataset(torch.zeros(60, 2), torch.zeros(60, 2), torch.tensor(yy), np.where(yy == 0)[0], np.where(yy == 1)[0])
np.random.seed(0); s = BalancedBatchSampler(ds, 16); ref = [list(map(int, b)) for _ in range(3) for b in s]
got = []
m = FATE(encoder=ENC, n_epochs=3, device="cpu", random_state=None)
m._train_steps = lambda bs, opt: [got.append(list(map(int, ids[:, 0]))) for ids, am, yb in bs] and 0.0
m._build = lambda: (setattr(m, "tokenizer_", None), setattr(m, "net_", torch.nn.Linear(1, 1)))
m._tok = lambda T: {"input_ids": torch.arange(len(T)).unsqueeze(1).repeat(1, 2), "attention_mask": torch.ones(len(T), 2)}
m.decision_function = lambda X: np.zeros(len(X))
np.random.seed(0); m.fit([f"doc {i}" for i in range(60)], yy)
ok &= report("FATE balanced batch sampler", ref == got, f"{len(ref)} batches compared")
sys.exit(0 if ok else 1)
