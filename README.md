# SUMREC-System reference implementation

Implements the model, losses, interventions, calibration and evaluation from the manuscript (Diginetica).

## Setup
```
pip install torch numpy pandas
```
Download `train-item-views.csv` from the CIKM Cup 2016 Diginetica release and check its terms of use.
Confirm the file is `;`-separated with columns `sessionId;userId;itemId;timeframe;eventdate`.

## Run
```
# matched baseline (same encoder/ranker, no state head) and the full model, 5 seeds each
for s in 1 2 3 4 5; do
  python train.py --data train-item-views.csv --variant no_state --seed $s
  python train.py --data train-item-views.csv --variant full     --seed $s
done
# ablations (Section V.G): fixed_retention, no_unc_path, no_trans_path
python aggregate.py results      # mean ± std per metric
```
Start with `--epochs 1 --d 64` on one seed to confirm it runs end to end.

## What is implemented
Eq. 1-10, 14, 15 (encoder, evidence vector, three-state head, entropy, retention, three pathways, scoring),
masked state loss (Eq. 4), Brier loss (Eq. 7), temperature scaling on validation interventions (Eq. 6),
noise insertion/replacement (uniform + popularity-weighted), transition splicing, and experiments
E1 (ranking), E2 (noise robustness), E3 (transition ranking + state P/R/F1), E4 (length buckets),
E5 (ECE/NLL/Brier before and after scaling), E8 (parameters, latency).

## Not implemented (say so in the paper, or add it)
- Prototype memory (E6) and the consistency loss L_cons; set up as future work or implement before claiming them.
- Other baselines (GRU4Rec, NARM, STAMP, SASRec, SR-GNN, HyPro, DDSG, DICSI, MintSR, DG-DAN). Use their official code
  or RecBole under the same split and candidate protocol. `--variant no_state` is only the matched internal baseline.
- RetailRocket and Yoochoose loaders.

## Choices you must describe in the paper (they differ from or go beyond the manuscript text)
- Split: last 10% of sessions by end time = test, previous 10% = validation, rest = training; unseen items dropped from val/test.
- Item embedding cosine is used for local compatibility and persistence; Diginetica category clusters are not used
  for transition sampling (the source session just must share no items with the current prefix).
- A last-event term is added to both the baseline and the full model.
- Transition examples change the target to the item following the spliced segment (so the ranker is rewarded for following the new segment).
- Intervention seeds are fixed (test 12345, validation 777) so all models see identical corrupted samples.
- Reported ranking uses the full catalog (no sampled negatives).

## Caveat
The code was compile-checked and the data/intervention logic was tested on synthetic data, but the PyTorch model
has not been executed. Expect to fix small issues on first run. Report exactly what you run: seeds, hyperparameters, and negative results.
