# replicate_top_method.py
# BiGRU + CTC + tiny word N-gram LM (chars only)

import argparse, math, random, string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from unidecode import unidecode
from create_data_sample import load_data_by_day

# optional fast WER
try:
    import Levenshtein as Lev
except Exception:
    Lev = None

PHONEME_NON = "<blank>"
PHONEME_SPC = "<sp>"

def words_to_units(words):
    seq = []
    for w in words:
        for ch in w:
            if ch in string.ascii_lowercase:
                seq.append(ch)
        seq.append(PHONEME_SPC)
    if seq and seq[-1] == PHONEME_SPC:
        seq.pop()
    return seq

def norm_text(s: str) -> str:
    s = unidecode(str(s)).lower().strip()
    s = "".join(ch if (ch.isalnum() or ch in " '") else " " for ch in s)
    return " ".join(s.split())

class NgramLM:
    def __init__(self, n=3):
        self.n = n
        self.counts = [Counter() for _ in range(n)]
        self.context_counts = [Counter() for _ in range(n)]
        self.vocab = set()
    @staticmethod
    def _prep_line(s):
        s = unidecode(s).lower().strip().replace("’", "'")
        s = "".join(ch if (ch.isalnum() or ch in " '") else " " for ch in s)
        return " ".join(s.split())
    def fit(self, texts):
        for t in texts:
            t = self._prep_line(t)
            if not t: continue
            words = t.split()
            self.vocab.update(words)
            pad = ["<s>"]*(self.n-1) + words + ["</s>"]
            for i in range(len(words) + 1):
                for k in range(1, self.n+1):
                    if i + k > len(pad): continue
                    ng = tuple(pad[i:i+k])
                    ctx = tuple(pad[i:i+k-1]) if k > 1 else ()
                    self.counts[k-1][ng] += 1
                    self.context_counts[k-1][ctx] += 1
        self.vocab = sorted(self.vocab | {"<s>", "</s>"})
        self.V = len(self.vocab)
    def logprob(self, ctx_words, next_word):
        n = self.n
        ctx = tuple((["<s>"]*(n-1) + ctx_words)[-(n-1):]) if n > 1 else ()
        num = self.counts[n-1][ctx + (next_word,)] + 1.0
        den = self.context_counts[n-1][ctx] + self.V
        return math.log(num / den + 1e-12)

@dataclass
class Trial:
    feats: np.ndarray
    transcript: str
    day: str

class BrainTextTrials(Dataset):
    def __init__(self, split_dict, day_names):
        self.items = []
        feats = split_dict["neural_features"]
        txts  = split_dict.get("transcriptions") or [None]*len(feats)
        def dec(x): return (x.decode("utf-8","ignore") if isinstance(x,(bytes,bytearray)) else (str(x) if x is not None else ""))
        txts = [dec(t) for t in txts]
        for f, t, d in zip(feats, txts, day_names):
            self.items.append(Trial(f.astype(np.float16, copy=False), t, d))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate_trials(batch, token_map, max_len=None, upsample=1, min_ratio=1.1, max_repeat=8):
    tokens, tgt_lens, refs, days, feats_list = [], [], [], [], []
    for ex in batch:
        feats_list.append(ex.feats)
        words = norm_text(ex.transcript or "").split()
        ids = [token_map.get(p, token_map["<unk>"]) for p in words_to_units(words)]
        tokens.append(ids); tgt_lens.append(len(ids))
        refs.append(" ".join(words)); days.append(ex.day)
    repeats, base_lens = [], [f.shape[0] for f in feats_list]
    for T0, yL in zip(base_lens, tgt_lens):
        if yL == 0: r = max(1, int(upsample))
        else:
            need = math.ceil(min_ratio * yL)
            r = max(1, int(math.ceil(need / max(1, T0))))
            r = max(r, int(upsample))
        repeats.append(min(r, max_repeat))
    T_batch = max(T0*r for T0, r in zip(base_lens, repeats))
    if max_len is not None: T_batch = min(T_batch, max_len)
    F = feats_list[0].shape[1]
    X = np.zeros((len(batch), T_batch, F), dtype=np.float32)
    x_lens = np.zeros((len(batch),), dtype=np.int32)
    flat = []
    for i, (feats, ids, r) in enumerate(zip(feats_list, tokens, repeats)):
        t = np.repeat(feats, repeats=r, axis=0)
        if max_len is not None: t = t[:max_len]
        X[i, :t.shape[0]] = t
        x_lens[i] = t.shape[0]
        flat.extend(ids)
    X = torch.from_numpy(X)
    x_lens = torch.from_numpy(x_lens)
    y = torch.tensor(flat, dtype=torch.long) if flat else torch.zeros(1, dtype=torch.long)
    y_lens = torch.tensor(tgt_lens, dtype=torch.long)
    return X, x_lens, y, y_lens, days, refs

class DayAdaptLinear(nn.Module):
    def __init__(self, in_dim, day_to_idx):
        super().__init__()
        self.day_to_idx = day_to_idx
        n_days = len(day_to_idx)
        self.weight = nn.Parameter(torch.ones(n_days, in_dim))
        self.bias   = nn.Parameter(torch.zeros(n_days, in_dim))
    def forward(self, x, day_names):
        idx = torch.tensor([self.day_to_idx[d] for d in day_names], device=x.device)
        w = self.weight[idx].unsqueeze(1)
        b = self.bias[idx].unsqueeze(1)
        return x * w + b

class GRUAcousticModel(nn.Module):
    def __init__(self, feat_dim, hidden, num_layers, num_classes, day_to_idx, dropout=0.2):
        super().__init__()
        self.adapt = DayAdaptLinear(feat_dim, day_to_idx)
        self.input_proj = nn.Linear(feat_dim, hidden)
        self.in_ln = nn.LayerNorm(hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.out = nn.Linear(hidden*2, num_classes)
    def forward(self, x, x_lens, day_names):
        x = self.adapt(x, day_names)
        x = F.relu(self.input_proj(x))
        x = self.in_ln(x)
        pk = nn.utils.rnn.pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        po, _ = self.rnn(pk)
        y, _ = nn.utils.rnn.pad_packed_sequence(po, batch_first=True)
        return self.out(y).transpose(0, 1)

def ctc_beam_search(log_probs_TBC, x_len, id2ph, lm=None, alpha=0.5, beta=1.0, beam_size=50):
    blank_id = [i for i,s in id2ph.items() if s=="<blank>"][0]
    sp_id = next((i for i,s in id2ph.items() if s=="<sp>"), None)
    log_probs_TBC = log_probs_TBC.detach().cpu().float()
    T = int(x_len)
    if T <= 0: return ""
    beams = {(" ",): (0.0, float("-inf"), [])}
    for t in range(T):
        nb = defaultdict(lambda: (float("-inf"), float("-inf"), []))
        logp = log_probs_TBC[t, 0].numpy()
        C = logp.shape[0]
        for prefix, (pb, pnb, ids) in beams.items():
            bpb, bpnb, bpath = nb[prefix]
            bpb = np.logaddexp(bpb, pb + logp[blank_id])
            nb[prefix] = (bpb, bpnb, bpath)
            for c in range(C):
                if c == blank_id: continue
                p_c = float(logp[c])
                end = ids[-1] if ids else None
                if c == end:
                    bpb, bpnb, bpath = nb[prefix]
                    bpnb = np.logaddexp(bpnb, pnb + p_c)
                    nb[prefix] = (bpb, bpnb, bpath)
                else:
                    new_prefix = prefix
                    new_ids = ids + [c]
                    bonus = 0.0
                    if (sp_id is not None) and (c == sp_id) and (lm is not None):
                        words = " ".join(new_prefix).split()
                        if len(words) >= 1:
                            ctx, nxt = words[:-1], words[-1]
                            bonus = alpha * lm.logprob(ctx, nxt) + beta
                    val = np.logaddexp(pb, pnb) + p_c + bonus
                    sym = id2ph[c] if c != sp_id else " "
                    key = new_prefix + (sym,)
                    ppb, ppnb, ppath = nb[key]
                    nb[key] = (ppb, np.logaddexp(ppnb, val), new_ids)
        beams = dict(sorted(nb.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]), reverse=True)[:beam_size])
    best = max(beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]))
    hyp = "".join(best[0]).strip()
    return " ".join(hyp.split())

def wer(refs, hyps):
    if Lev is None:
        def ed(a,b):
            A,B=a.split(),b.split()
            dp=[[0]*(len(B)+1) for _ in range(len(A)+1)]
            for i in range(len(A)+1): dp[i][0]=i
            for j in range(len(B)+1): dp[0][j]=j
            for i in range(1,len(A)+1):
                for j in range(1,len(B)+1):
                    dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(A[i-1]!=B[j-1]))
            return dp[-1][-1]
        num = sum(ed(r,h) for r,h in zip(refs,hyps))
    else:
        num = sum(Lev.distance(r,h) for r,h in zip(refs,hyps))
    den = sum(len(r.split()) for r in refs) + 1e-8
    return num/den

def build_label_maps(train_texts):
    s = {PHONEME_NON, PHONEME_SPC, "<unk>"}
    for t in train_texts:
        t = unidecode(t).strip()
        if not t: continue
        s.update(words_to_units(t.split()))
    lst = sorted(s)
    ph2id = {p:i for i,p in enumerate(lst)}
    id2ph = {i:p for p,i in ph2id.items()}
    return ph2id, id2ph

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(Path(__file__).parent / "data" / "t15_copyTask_neuralData" / "hdf5_data_final"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beam", type=int, default=50)
    p.add_argument("--percent_days", type=int, default=100)
    p.add_argument("--upsample_factor", type=int, default=1)
    p.add_argument("--max_T", type=int, default=2000)
    p.add_argument("--dropout", type=float, default=0.2)
    args = p.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    root = Path(args.root)
    if not root.exists(): raise FileNotFoundError(f"Data root not found: {root}")

    splits = load_data_by_day(str(root), percent_of_days_to_read=args.percent_days)
    if splits["train"] is None: raise RuntimeError("no train split")

    texts, train_days = [], []
    for txt, day in zip(splits["train"]["transcriptions"], splits["train"]["session"]):
        t = txt.decode("utf-8","ignore") if isinstance(txt,(bytes,bytearray)) else str(txt)
        d = day.decode("utf-8","ignore") if isinstance(day,(bytes,bytearray)) else str(day)
        texts.append(norm_text(t)); train_days.append(d)
    day_to_idx = {d:i for i,d in enumerate(sorted(set(train_days)))}

    ph2id, id2ph = build_label_maps(texts)
    lm = NgramLM(n=3); lm.fit(texts)

    def _mk(split):
        if split is None: return None
        days = [(s.decode("utf-8","ignore") if isinstance(s,(bytes,bytearray)) else str(s)) for s in split["session"]]
        return BrainTextTrials(split, days)
    ds_train = _mk(splits["train"])
    ds_val   = _mk(splits["val"])
    ds_test  = _mk(splits["test"])

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=lambda b: collate_trials(b, ph2id, max_len=args.max_T,
                                                              upsample=args.upsample_factor, min_ratio=1.1, max_repeat=8))
    tr_loader = _loader(ds_train, True)
    va_loader = _loader(ds_val, False) if ds_val else None
    te_loader = _loader(ds_test, False) if ds_test else None

    assert PHONEME_NON in ph2id and "<unk>" in ph2id and ph2id[PHONEME_NON] != ph2id["<unk>"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUAcousticModel(feat_dim=ds_train[0].feats.shape[1], hidden=args.hidden,
                             num_layers=args.layers, num_classes=len(ph2id),
                             day_to_idx=day_to_idx, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ctc_loss = nn.CTCLoss(blank=ph2id[PHONEME_NON], reduction='none', zero_infinity=True)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss, n_items = 0.0, 0
        skipped_empty = skipped_short = skipped_inf = 0
        for X, Xlen, Y, Ylen, days, _ in tr_loader:
            X = X.to(device); Xlen = Xlen.to(device)
            Y = Y.to(device); Ylen = Ylen.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(X, Xlen, days)
                logp = F.log_softmax(out, dim=-1)
                Tcur = out.size(0)
                input_lengths = Xlen.clamp(max=Tcur)
                per = ctc_loss(logp, Y, input_lengths, Ylen)
            skipped_empty += int((Ylen == 0).sum().item())
            skipped_short += int((input_lengths < Ylen).sum().item())
            skipped_inf   += int((~torch.isfinite(per)).sum().item())
            mask = (Ylen > 0) & (input_lengths >= Ylen) & torch.isfinite(per)
            if not torch.any(mask): continue
            loss = per[mask].mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            bs = int(mask.sum().item())
            tot_loss += loss.item() * bs
            n_items  += bs
        print(f"Epoch {epoch}: loss={tot_loss / max(1,n_items):.4f} (valid={n_items}, empty={skipped_empty}, short={skipped_short}, nonfinite={skipped_inf})")

        if va_loader is not None:
            model.eval()
            with torch.no_grad():
                refs, hyps = [], []
                for i, (X, Xlen, Y, Ylen, days, refs_batch) in enumerate(va_loader):
                    X = X.to(device); Xlen = Xlen.to(device)
                    lp = F.log_softmax(model(X, Xlen, days), dim=-1)
                    for b in range(X.size(0)):
                        T = int(Xlen[b].item())
                        path = lp[:T, b].argmax(dim=-1).tolist()
                        last=None; toks=[]
                        for pid in path:
                            if pid == ph2id[PHONEME_NON] or pid == last: last=pid; continue
                            toks.append(pid); last=pid
                        hyp = " ".join(id2ph[i] for i in toks).replace("<sp>", " ")
                        hyp = " ".join(hyp.split())
                        refs.append(refs_batch[b]); hyps.append(hyp)
                    if i >= 2: break
                print(f"Val WER (subset): {wer(refs, hyps):.3f}")

    outdir = Path("artifacts"); outdir.mkdir(exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "ph2id": ph2id, "id2ph": id2ph, "day_to_idx": day_to_idx},
               outdir / "gru_ctc_checkpoint.pt")
    print(f"Saved → {outdir/'gru_ctc_checkpoint.pt'}")

    if te_loader is not None:
        model.eval()
        finals = []
        with torch.no_grad():
            for X, Xlen, Y, Ylen, days, _ in te_loader:
                X = X.to(device)
                lp = F.log_softmax(model(X, Xlen.to(device), days), dim=-1)
                for b in range(X.size(0)):
                    T = int(Xlen[b].item())
                    hyp = ctc_beam_search(lp[:, b:b+1, :], T, id2ph, lm=lm, alpha=args.alpha, beta=args.beta, beam_size=args.beam)
                    finals.append(hyp)
        print("Sample test decodes:")
        for i in range(min(5, len(finals))):
            print(f"{i+1:02d}: {finals[i]}")

if __name__ == "__main__":
    main()