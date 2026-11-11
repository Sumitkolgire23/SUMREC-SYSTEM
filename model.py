"""SUMREC-System model (Eqs. 1-10, 14, 15) with ablation variants.

variant:
  full             - complete event-state routing (Eq. 8-10)
  no_state         - matched baseline: same encoder, attention pooling, ranker; no state head
  fixed_retention  - r_i = 1 - p(N), ignores uncertainty
  no_unc_path      - drops the uncertainty pathway z_U
  no_trans_path    - drops the transition pathway z_S

Both baseline and proposed model add a last-event term W_last * c_last, a
standard session-recommendation component; it is not in the paper text and
should be stated in the implementation details.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import MAX_LEN

VARIANTS = ["full", "no_state", "fixed_retention", "no_unc_path", "no_trans_path"]


class SUMREC(nn.Module):
    def __init__(self, n_items, d=128, heads=4, layers=2, dropout=0.2, variant="full"):
        super().__init__()
        assert variant in VARIANTS
        self.n_items, self.variant = n_items, variant
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, d)
        self.gap = nn.Embedding(16, d)  # log2 time-gap buckets
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        # evidence vector g_i = [h_i ; c_i ; dt ; a_i ; b_i ; has_prev ; has_next]  (Eq. 2)
        self.state_head = nn.Sequential(nn.Linear(2 * d + 5, d), nn.ReLU(), nn.Linear(d, 3))
        self.att1 = nn.Linear(d, d)
        self.att2 = nn.Linear(d, d, bias=False)
        self.att_v = nn.Linear(d, 1, bias=False)
        self.W_last = nn.Linear(d, d)
        self.W_C = nn.Linear(d, d, bias=False)
        self.W_S = nn.Linear(d, d, bias=False)
        self.W_U = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d)
        self.item_bias = nn.Parameter(torch.zeros(n_items))
        self.register_buffer("T", torch.ones(1))  # temperature, fitted on validation interventions

    def evidence(self, h0, c, items, buckets, valid):
        B, L = items.shape
        e = F.normalize(self.item(items), dim=-1)
        sim = e @ e.transpose(1, 2)  # B,L,L item-embedding cosine
        idx = torch.arange(L, device=items.device)
        prev = (idx[None, :] < idx[:, None])[None] & valid[:, None, :]   # j < i
        nxt = (idx[None, :] > idx[:, None])[None] & valid[:, None, :]    # j > i (already observed)
        has_prev, has_next = prev.any(-1), nxt.any(-1)
        a = (sim * prev).sum(-1) / prev.sum(-1).clamp(min=1)             # local compatibility
        b = sim.masked_fill(~nxt, -2.0).max(-1).values                   # persistence
        b = torch.where(has_next, b, torch.zeros_like(b))                # masked, never estimated
        dt = (buckets.float() / 15.0)
        feats = torch.stack([dt, a, b, has_prev.float(), has_next.float()], dim=-1)
        return torch.cat([h0, c, feats], dim=-1)

    def forward(self, items, buckets):
        B, L = items.shape
        dev = items.device
        valid = items > 0
        h0 = self.item(items) + self.pos(torch.arange(L, device=dev))[None] + self.gap(buckets)  # Eq. 1
        c = self.enc(self.drop(h0), src_key_padding_mask=~valid)
        lengths = valid.sum(1)
        c_last = c[torch.arange(B, device=dev), lengths - 1]
        att = self.att_v(torch.sigmoid(self.att1(c) + self.att2(c_last)[:, None])).squeeze(-1)
        alpha = att.masked_fill(~valid, -1e9).softmax(-1)

        zL = self.W_last(c_last)
        state_logits = None
        if self.variant == "no_state":
            zL = zL + self.W_C((alpha.unsqueeze(-1) * c).sum(1))
        else:
            state_logits = self.state_head(self.evidence(h0, c, items, buckets, valid))
            p = (state_logits / self.T).softmax(-1)                       # Eq. 6 (T=1 until fitted)
            u = -(p * torch.log(p + 1e-8)).sum(-1) / math.log(3)          # Eq. 5/6
            r = 1 - p[..., 0] if self.variant == "fixed_retention" else 1 - p[..., 0] * (1 - u)  # Eq. 8
            w = alpha * r * valid
            zC = ((w * p[..., 1]).unsqueeze(-1) * c).sum(1)               # Eq. 9
            zS = ((w * p[..., 2]).unsqueeze(-1) * c).sum(1)
            zU = ((w * u).unsqueeze(-1) * c).sum(1)                       # Eq. 10
            zL = zL + self.W_C(zC)
            if self.variant != "no_trans_path":
                zL = zL + self.W_S(zS)
            if self.variant != "no_unc_path":
                zL = zL + self.W_U(zU)

        scores = self.W_o(zL) @ self.item.weight[1:].T + self.item_bias    # Eq. 14
        return {"scores": scores, "state_logits": state_logits}


def state_losses(state_logits, labels):
    """Masked state loss (Eq. 4) and Brier loss (Eq. 7) on intervention-labeled events."""
    m = labels >= 0
    if state_logits is None or not m.any():
        z = torch.zeros((), device=labels.device)
        return z, z
    lg, y = state_logits[m], labels[m]
    l_state = F.cross_entropy(lg, y)
    l_brier = ((lg.softmax(-1) - F.one_hot(y, 3).float()) ** 2).sum(-1).mean()
    return l_state, l_brier
