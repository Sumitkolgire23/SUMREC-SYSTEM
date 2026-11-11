"""Mean ± std across seeds for every numeric metric, grouped by variant.
  python aggregate.py results
"""
import collections, glob, json, sys
import numpy as np


def flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, p + k + ".")
        elif isinstance(v, (int, float)):
            yield p + k, v


groups = collections.defaultdict(list)
for f in sorted(glob.glob((sys.argv[1] if len(sys.argv) > 1 else "results") + "/*.json")):
    r = json.load(open(f))
    groups[r["args"]["variant"]].append(dict(flat(r["metrics"])))

for var, runs in groups.items():
    print(f"\n== {var} ({len(runs)} seeds)")
    for k in sorted(runs[0]):
        vals = [x[k] for x in runs if k in x]
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        print(f"{k}: {np.mean(vals):.4f} ± {sd:.4f}")
