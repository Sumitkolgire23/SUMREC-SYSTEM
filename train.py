"""Train and evaluate one SUMREC variant for one seed.

  python train.py --data train-item-views.csv --variant full --seed 1
Results are written to results/<variant>_seed<seed>.json
"""
import argparse, copy, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import load_diginetica, popularity_cdf, SessionDataset, collate
from model import SUMREC, VARIANTS, state_losses

TEST_SEED, VAL_SEED = 12345, 777  # fixed intervention seeds, identical for every model


# ----------------------------------------------------------------------------- metrics
def rank_metrics(ranks):
    r = np.asarray(ranks)
    hit = r <= 20
    return {"Recall@20": float(hit.mean()),
            "MRR@20": float(np.where(hit, 1.0 / r, 0).mean()),
            "NDCG@20": float(np.where(hit, 1.0 / np.log2(r + 1), 0).mean()),
            "n": int(len(r))}


def calib_metrics(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    edges, ece = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return {"ECE": float(ece),
            "NLL": float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean()),
            "Brier": float(((probs - np.eye(3)[y]) ** 2).sum(1).mean())}


def class_metrics(pred, y):
    names, out, f1s = ["N", "C", "S"], {}, []
    for k, nm in enumerate(names):
        tp = float(((pred == k) & (y == k)).sum())
        p = tp / max(1, (pred == k).sum()); r = tp / max(1, (y == k).sum())
        f = 2 * p * r / max(1e-12, p + r)
        out[nm] = {"precision": p, "recall": r, "f1": f, "support": int((y == k).sum())}
        f1s.append(f)
    out["macro_f1"] = float(np.mean(f1s))
    out["accuracy"] = float((pred == y).mean())
    return out


def fit_temperature(logits, labels):
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / logT.exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp().item())


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def collect(model, ds, dev, bs=512):
    model.eval()
    ranks, lens, lg, lb = [], [], [], []
    for items, buckets, labels, target, _ in DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate):
        items, buckets, labels, target = items.to(dev), buckets.to(dev), labels.to(dev), target.to(dev)
        out = model(items, buckets)
        s = out["scores"]
        t = s.gather(1, (target - 1)[:, None])
        ranks.append(((s > t).sum(1) + 1).cpu())
        lens.append((items > 0).sum(1).cpu())
        if out["state_logits"] is not None:
            m = labels >= 0
            lg.append(out["state_logits"][m].cpu()); lb.append(labels[m].cpu())
    return {"ranks": torch.cat(ranks).numpy(), "lens": torch.cat(lens).numpy(),
            "logits": torch.cat(lg) if lg else None, "labels": torch.cat(lb) if lb else None}


def evaluate_all(model, te, tr, n_items, cdf, dev):
    mk = lambda mode, **kw: SessionDataset(te, mode, tr, n_items, cdf, seed=TEST_SEED, **kw)
    res = {}
    clean = collect(model, mk("clean"), dev)
    res["clean"] = rank_metrics(clean["ranks"])
    bl = {"len1": (1, 1), "len2-3": (2, 3), "len4-6": (4, 6), "len7+": (7, 10 ** 6)}  # E4
    res["clean_by_len"] = {k: rank_metrics(clean["ranks"][(clean["lens"] >= a) & (clean["lens"] <= b)])
                           for k, (a, b) in bl.items() if ((clean["lens"] >= a) & (clean["lens"] <= b)).any()}
    base = res["clean"]["Recall@20"]
    res["noise"] = {}
    for how in ["insert", "replace"]:  # E2
        for rate in [0.1, 0.2, 0.3, 0.4]:
            m = rank_metrics(collect(model, mk("noise", rate=rate, how=how), dev)["ranks"])
            m["rel_degradation_R20"] = 1 - m["Recall@20"] / max(base, 1e-12)
            res["noise"][f"{how}_{rate}"] = m
    res["transition_ranking"] = rank_metrics(collect(model, mk("trans"), dev)["ranks"])  # E3 (ranking)
    if model.variant != "no_state":  # E3 (state) and E5 (calibration)
        mx = collect(model, mk("mixed"), dev)
        y = mx["labels"].numpy()
        p_un = mx["logits"].softmax(-1).numpy()
        p_cal = (mx["logits"] / model.T.cpu()).softmax(-1).numpy()
        res["state"] = class_metrics(p_cal.argmax(1), y)
        res["calib_uncalibrated"] = calib_metrics(p_un, y)
        res["calib_temperature_scaled"] = calib_metrics(p_cal, y)
        res["temperature"] = float(model.T.item())
    # E8: single-request latency
    ds, lat = mk("clean"), []
    for i in range(min(200, len(ds))):
        items, buckets, *_ = collate([ds[i]])
        items, buckets = items.to(dev), buckets.to(dev)
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(): model(items, buckets)
        if dev.type == "cuda": torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    res["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}
    res["params"] = int(sum(p.numel() for p in model.parameters()))
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", default="full", choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--aug_prob", type=float, default=0.5)
    ap.add_argument("--lam_state", type=float, default=0.5)
    ap.add_argument("--lam_brier", type=float, default=0.1)
    ap.add_argument("--min_item_count", type=int, default=5)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te, n_items = load_diginetica(args.data, args.min_item_count)
    print(f"items={n_items} sessions train/val/test={len(tr)}/{len(va)}/{len(te)} device={dev}")
    cdf = popularity_cdf(tr, n_items)

    train_ds = SessionDataset(tr, "train", tr, n_items, cdf, aug_prob=args.aug_prob, seed=args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_clean = SessionDataset(va, "clean", tr, n_items, cdf, seed=VAL_SEED)

    model = SUMREC(n_items, args.d, args.heads, args.layers, args.dropout, args.variant).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_state, bad = -1.0, None, 0
    lam_s = 0.0 if args.variant == "no_state" else args.lam_state
    lam_b = 0.0 if args.variant == "no_state" else args.lam_brier

    for ep in range(args.epochs):
        model.train(); t0, tot = time.time(), 0.0
        for items, buckets, labels, target, _ in loader:
            items, buckets, labels, target = items.to(dev), buckets.to(dev), labels.to(dev), target.to(dev)
            out = model(items, buckets)
            l_rank = F.cross_entropy(out["scores"], target - 1)               # Eq. 15
            l_state, l_brier = state_losses(out["state_logits"], labels)
            loss = l_rank + lam_s * l_state + lam_b * l_brier                # Eq. 17 (no L_cons)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += loss.item()
        v = rank_metrics(collect(model, val_clean, dev)["ranks"])["Recall@20"]
        print(f"epoch {ep + 1} loss {tot / len(loader):.4f} val R@20 {v:.4f} ({time.time() - t0:.0f}s)")
        if v > best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)

    if args.variant != "no_state":  # temperature scaling on validation interventions only
        mx = collect(model, SessionDataset(va, "mixed", tr, n_items, cdf, seed=VAL_SEED), dev)
        model.T.fill_(fit_temperature(mx["logits"], mx["labels"]))
        print(f"fitted temperature T={model.T.item():.3f}")

    metrics = evaluate_all(model, te, tr, n_items, cdf, dev)
    metrics["best_val_R20"] = best
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.variant}_seed{args.seed}.json")
    json.dump({"args": vars(args), "metrics": metrics}, open(path, "w"), indent=2)
    print(json.dumps(metrics["clean"], indent=2)); print("saved", path)


if __name__ == "__main__":
    main()
