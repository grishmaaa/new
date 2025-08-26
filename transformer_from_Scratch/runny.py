# train_reverse_task.py
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# CHANGE THIS to your actual filename if needed
from model import build_transformer

# ---------------------------
# 1) Toy vocab + utils
# ---------------------------
class Vocab:
    def __init__(self):
        # specials
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        # base symbols: digits + lowercase letters
        symbols = [chr(c) for c in range(ord('0'), ord('9') + 1)] + \
                  [chr(c) for c in range(ord('a'), ord('z') + 1)]
        self.offset = 3
        self.sym2id = {s: i + self.offset for i, s in enumerate(symbols)}
        self.id2sym = {i + self.offset: s for i, s in enumerate(symbols)}
        self.vocab_size = self.offset + len(symbols)

    def encode_seq(self, s):
        return [self.sym2id[c] for c in s]

    def decode_seq(self, ids):
        out = []
        for i in ids:
            if i in (self.PAD, self.BOS, self.EOS):
                continue
            out.append(self.id2sym.get(int(i), '?'))
        return ''.join(out)

# ---------------------------
# 2) Synthetic dataset: reverse task
#    input:  "abc12"
#    target: "21cba"
# ---------------------------
@dataclass
class GenCfg:
    min_len: int = 4
    max_len: int = 20
    num_train: int = 20000
    num_val: int = 1000

class ReverseDataset(Dataset):
    def __init__(self, vocab: Vocab, n_samples: int, cfg: GenCfg,
                 max_src_len: int, max_tgt_len: int, seed: int = 123):
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.samples = []
        rng = random.Random(seed)

        base_syms = list(vocab.sym2id.keys())
        for _ in range(n_samples):
            L = rng.randint(cfg.min_len, cfg.max_len)
            seq_chars = [rng.choice(base_syms) for _ in range(L)]
            src = vocab.encode_seq(seq_chars)
            tgt_rev = list(reversed(src))

            # Build model-ready sequences (pad to fixed lengths)
            src_pad = self.pad_to_len(src, max_src_len, vocab.PAD)

            tgt_in = [vocab.BOS] + tgt_rev
            tgt_out = tgt_rev + [vocab.EOS]
            tgt_in = self.pad_to_len(tgt_in, max_tgt_len, vocab.PAD)
            tgt_out = self.pad_to_len(tgt_out, max_tgt_len, vocab.PAD)

            self.samples.append((
                torch.tensor(src_pad, dtype=torch.long),
                torch.tensor(tgt_in, dtype=torch.long),
                torch.tensor(tgt_out, dtype=torch.long),
                L  # true (un-padded) source length
            ))

    @staticmethod
    def pad_to_len(seq, L, pad):
        if len(seq) > L:
            return seq[:L]
        return seq + [pad] * (L - len(seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # (src, tgt_in, tgt_out, src_len)

# ---------------------------
# 3) Mask helpers
# ---------------------------
def make_pad_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    seq: (B, T)
    returns: (B, 1, 1, T) with True for keep, False for mask
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(1)

def make_causal_mask(size: int, device) -> torch.Tensor:
    """
    returns: (1, 1, size, size) lower-triangular (True allowed, False masked)
    """
    return torch.tril(torch.ones((1, 1, size, size), dtype=torch.bool, device=device))

def combine_masks(pad_mask: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
    """
    pad_mask: (B, 1, 1, T)
    causal_mask: (1, 1, T, T)
    returns: (B, 1, T, T)
    """
    return pad_mask & causal_mask

# ---------------------------
# 4) Greedy decoding
# ---------------------------
@torch.no_grad()
def greedy_decode(model, src, src_len, max_tgt_len, pad_idx, bos_idx, eos_idx, device):
    """
    src: (1, S) single example
    """
    model.eval()
    src = src.to(device)
    src_mask = make_pad_mask(src, pad_idx)  # (1,1,1,S)
    memory = model.encode(src, src_mask)    # (1,S,D)

    ys = torch.full((1, 1), bos_idx, dtype=torch.long, device=device)  # (1,1)
    for _ in range(max_tgt_len - 1):
        tgt_mask_pad = make_pad_mask(ys, pad_idx)         # (1,1,1,t)
        tgt_mask_causal = make_causal_mask(ys.size(1), device)  # (1,1,t,t)
        tgt_mask = combine_masks(tgt_mask_pad, tgt_mask_causal) # (1,1,t,t)

        out = model.decode(memory, src_mask, ys, tgt_mask)       # (1,t,D)
        logits = model.project(out)                              # (1,t,V)
        next_token = logits[:, -1, :].argmax(-1)                 # (1,)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

        if int(next_token.item()) == eos_idx:
            break
    return ys.squeeze(0)  # (t,)

# ---------------------------
# 5) Training
# ---------------------------
def train():
    # Repro
    torch.manual_seed(42)

    # Configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab()
    gen_cfg = GenCfg(min_len=4, max_len=20, num_train=24000, num_val=1200)

    max_src_len = gen_cfg.max_len
    max_tgt_len = gen_cfg.max_len + 1  # +1 for EOS

    # Model
    d_model = 256
    N = 3
    h = 4
    d_ff = 1024
    dropout = 0.1

    model = build_transformer(
        src_vocab_size=vocab.vocab_size,
        tgt_vocab_size=vocab.vocab_size,
        src_seq_len=max_src_len,
        tgt_seq_len=max_tgt_len,
        d_model=d_model,
        N=N,
        h=h,
        dropout=dropout,
        d_ff=d_ff
    ).to(device)

    # Data
    train_ds = ReverseDataset(vocab, gen_cfg.num_train, gen_cfg, max_src_len, max_tgt_len, seed=13)
    val_ds   = ReverseDataset(vocab, gen_cfg.num_val, gen_cfg, max_src_len, max_tgt_len, seed=999)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)

    # Optim + loss
    optim = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD)

    # Simple training loop
    epochs = 8
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for src, tgt_in, tgt_out, _ in train_loader:
            src = src.to(device)       # (B,S)
            tgt_in = tgt_in.to(device) # (B,T)
            tgt_out = tgt_out.to(device) # (B,T)

            # Masks
            src_mask = make_pad_mask(src, vocab.PAD)                   # (B,1,1,S)
            tgt_mask_pad = make_pad_mask(tgt_in, vocab.PAD)            # (B,1,1,T)
            tgt_mask_causal = make_causal_mask(tgt_in.size(1), device) # (1,1,T,T)
            tgt_mask = combine_masks(tgt_mask_pad, tgt_mask_causal)    # (B,1,T,T)

            # Forward
            memory = model.encode(src, src_mask)                        # (B,S,D)
            out = model.decode(memory, src_mask, tgt_in, tgt_mask)      # (B,T,D)
            logits = model.project(out)                                 # (B,T,V)

            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += loss.item()

        # Validation (token-level accuracy, ignoring PAD)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for src, tgt_in, tgt_out, _ in val_loader:
                src = src.to(device)
                tgt_in = tgt_in.to(device)
                tgt_out = tgt_out.to(device)

                src_mask = make_pad_mask(src, vocab.PAD)
                tgt_mask = combine_masks(make_pad_mask(tgt_in, vocab.PAD),
                                         make_causal_mask(tgt_in.size(1), device))

                mem = model.encode(src, src_mask)
                out = model.decode(mem, src_mask, tgt_in, tgt_mask)
                logits = model.project(out)  # (B,T,V)
                preds = logits.argmax(-1)    # (B,T)

                mask = (tgt_out != vocab.PAD)
                correct += (preds.eq(tgt_out) & mask).sum().item()
                total += mask.sum().item()

        avg_loss = total_loss / len(train_loader)
        acc = correct / max(1, total)
        print(f"Epoch {epoch:02d} | train_loss={avg_loss:.4f} | val_token_acc={acc:.4f}")

    # ---------------------------
    # Qualitative check: decode a few
    # ---------------------------
    print("\nGreedy decode samples:")
    model.eval()
    for _ in range(5):
        # sample a random example from val set
        src, _, _, L = val_ds[random.randint(0, len(val_ds)-1)]
        src = src.unsqueeze(0)  # (1,S)

        out_ids = greedy_decode(model, src, L, max_tgt_len,
                                vocab.PAD, vocab.BOS, vocab.EOS, device)
        # strip BOS from decoded
        out_trim = out_ids.tolist()
        if out_trim and out_trim[0] == vocab.BOS:
            out_trim = out_trim[1:]
        # until EOS
        if vocab.EOS in out_trim:
            out_trim = out_trim[:out_trim.index(vocab.EOS)]

        print("SRC :", vocab.decode_seq(src[0].tolist()))
        print("PRED:", vocab.decode_seq(out_trim))
        print()

if __name__ == "__main__":
    train()
