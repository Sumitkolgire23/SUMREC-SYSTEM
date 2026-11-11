"""Data pipeline for SUMREC-System: Diginetica loading, chronological split,
and the controlled noise / transition interventions described in the paper."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MAX_LEN = 30  # maximum observed-prefix length fed to the model
N_, C_, S_ = 0, 1, 2  # state ids: noise, continuation, transition onset


# ----------------------------------------------------------------------------
# Loading and splitting
# ----------------------------------------------------------------------------
def load_diginetica(path, min_item_count=5, val_frac=0.1, test_frac=0.1):
    """Reads train-item-views.csv (';'-separated: sessionId, userId, itemId,
    timeframe, eventdate). Returns (train, val, test, n_items); each split is a
    list of (items[int array, 1..n_items], gaps[float array, seconds])."""
    df = pd.read_csv(path, sep=";", usecols=["sessionId", "itemId", "timeframe", "eventdate"])
    df = df.dropna()
    df["eventdate"] = pd.to_datetime(df["eventdate"])
    df = df.sort_values(["sessionId", "timeframe"], kind="stable")
    same = (df["sessionId"] == df["sessionId"].shift()) & (df["itemId"] == df["itemId"].shift())
    df = df[~same]  # drop consecutive repeats
    counts = df["itemId"].value_counts()
    df = df[df["itemId"].isin(counts[counts >= min_item_count].index)]

    sess = []
    for _, g in df.groupby("sessionId", sort=False):
        if len(g) < 2:
            continue
        t = g["timeframe"].to_numpy(dtype=float) / 1000.0
        gaps = np.clip(np.diff(t, prepend=t[0]), 0, None)
        end = g["eventdate"].iloc[0] + pd.Timedelta(seconds=float(t.max()))
        sess.append((g["itemId"].to_numpy(), gaps, end))

    sess.sort(key=lambda x: x[2])  # chronological by session end time
    n = len(sess)
    n_te, n_va = int(n * test_frac), int(n * val_frac)
    tr, va, te = sess[: n - n_va - n_te], sess[n - n_va - n_te: n - n_te], sess[n - n_te:]

    vocab = {it: i + 1 for i, it in enumerate(sorted(set(np.concatenate([s[0] for s in tr]).tolist())))}

    def remap(part):
        out = []
        for it, gp, _ in part:
            keep = np.array([x in vocab for x in it.tolist()], dtype=bool)  # drop unseen items
            if keep.sum() < 2:
                continue
            out.append((np.array([vocab[x] for x in it[keep].tolist()]), gp[keep]))
        return out

    return remap(tr), remap(va), remap(te), len(vocab)


def popularity_cdf(train, n_items):
    cnt = np.zeros(n_items + 1)
    for items, _ in train:
        np.add.at(cnt, items, 1)
    p = cnt[1:] / cnt[1:].sum()
    return np.cumsum(p)


# ----------------------------------------------------------------------------
# Interventions
# ----------------------------------------------------------------------------
def sample_noise(rng, n_items, cdf):
    """Uniform or popularity-weighted unrelated item (50/50)."""
    if rng.random() < 0.5:
        return int(rng.integers(1, n_items + 1))
    return int(min(np.searchsorted(cdf, rng.random()) + 1, n_items))


def corrupt_noise(items, gaps, rng, rate, how, n_items, cdf):
    """Insertion (after an event) or replacement of events with noise.
    Returns items, gaps, labels (0 = noise, -1 = unlabeled)."""
    L = len(items)
    hit = rng.random(L) < rate
    if not hit.any():
        hit[rng.integers(L)] = True
    oi, og, ol = [], [], []
    for k in range(L):
        if hit[k] and how == "replace":
            oi.append(sample_noise(rng, n_items, cdf)); og.append(gaps[k]); ol.append(N_)
        else:
            oi.append(int(items[k])); og.append(gaps[k]); ol.append(-1)
            if hit[k]:
                oi.append(sample_noise(rng, n_items, cdf)); og.append(gaps[k]); ol.append(N_)
    return oi, og, ol


def make_transition(items, gaps, rng, pool, max_seg=4):
    """Splice a contiguous segment from a *training* session after the last k
    events of the current prefix. First spliced event -> S, rest -> C. The
    target becomes the item that follows the segment in its source session.
    Segment length is 1..max_seg so the onset is sometimes the last event
    (persistence cue masked), matching real prediction time."""
    for _ in range(10):
        src_i, src_g = pool[rng.integers(len(pool))][:2]
        m = int(rng.integers(1, max_seg + 1))
        if len(src_i) < m + 1:
            continue
        s = int(rng.integers(0, len(src_i) - m))
        seg = src_i[s: s + m].tolist()
        if set(seg) & set(int(x) for x in items):
            continue
        target = int(src_i[s + m])
        k = int(rng.integers(1, len(items) + 1))
        new_i = [int(x) for x in items[-k:]] + seg
        seg_g = [float(gaps[-1])] + [float(x) for x in src_g[s + 1: s + m]]
        new_g = [float(x) for x in gaps[-k:]] + seg_g
        labels = [-1] * k + [S_] + [C_] * (m - 1)
        return new_i, new_g, labels, target
    return None


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class SessionDataset(Dataset):
    """mode: 'clean' | 'noise' | 'trans' | 'mixed' | 'train'.
    Non-train modes are deterministic per index (fixed seed) so every model
    sees identical corrupted samples."""

    def __init__(self, sessions, mode, pool, n_items, cdf, rate=None, how=None,
                 aug_prob=0.5, seed=0):
        self.sess, self.mode, self.pool = sessions, mode, pool
        self.n_items, self.cdf = n_items, cdf
        self.rate, self.how, self.aug_prob, self.seed = rate, how, aug_prob, seed
        self.ex = [(si, t) for si, (it, _) in enumerate(sessions) for t in range(1, len(it))]
        self.shared_rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.ex)

    def _rng(self, idx):
        return self.shared_rng if self.mode == "train" else np.random.default_rng([self.seed, idx])

    def __getitem__(self, idx):
        si, t = self.ex[idx]
        items, gaps = self.sess[si]
        lo = max(0, t - MAX_LEN)
        pi, pg, target = items[lo:t].tolist(), gaps[lo:t].tolist(), int(items[t])
        labels, kind = [-1] * len(pi), 0
        rng = self._rng(idx)
        mode = self.mode
        if mode == "train":
            if rng.random() >= self.aug_prob:
                mode = "clean"
            else:
                mode = "noise" if rng.random() < 0.5 else "trans"
        elif mode == "mixed":
            mode = "noise" if rng.random() < 0.5 else "trans"

        if mode == "noise":
            rate = self.rate if self.rate else float(rng.uniform(0.1, 0.4))
            how = self.how or ("insert" if rng.random() < 0.5 else "replace")
            pi, pg, labels = corrupt_noise(pi, pg, rng, rate, how, self.n_items, self.cdf)
            kind = 1
        elif mode == "trans":
            res = make_transition(pi, pg, rng, self.pool)
            if res is not None:
                pi, pg, labels, target = res
                kind = 2
        return pi[-MAX_LEN:], pg[-MAX_LEN:], labels[-MAX_LEN:], target, kind


def collate(batch):
    B, L = len(batch), max(len(b[0]) for b in batch)
    items = torch.zeros(B, L, dtype=torch.long)
    buckets = torch.zeros(B, L, dtype=torch.long)
    labels = torch.full((B, L), -1, dtype=torch.long)
    target = torch.zeros(B, dtype=torch.long)
    kind = torch.zeros(B, dtype=torch.long)
    for i, (pi, pg, lab, tg, k) in enumerate(batch):
        n = len(pi)
        items[i, :n] = torch.tensor(pi)
        b = np.minimum(np.log2(1.0 + np.asarray(pg, dtype=float)), 15).astype(np.int64)
        buckets[i, :n] = torch.from_numpy(b)
        labels[i, :n] = torch.tensor(lab)
        target[i], kind[i] = tg, k
    return items, buckets, labels, target, kind
