"""
Rebuild results/main.csv correctly, recovering CTGAN from the schema-drifted
tier2.csv that the original merge silently dropped.

The bug: tier2.csv was written across two runs. Early rows have 25 columns;
later rows gained `sampled_with_replacement` and `effective_pool_size` (27
columns) but the header was never rewritten. pandas rejects the file, and the
merge caught the exception and skipped it, dropping all CTGAN rows while the
completion report still said OK.
"""
import csv
import glob
import os

import pandas as pd

os.chdir(os.path.join(os.path.dirname(__file__), "..", "results"))

CANON_EXTRA = ["sampled_with_replacement", "effective_pool_size"]


def read_drifted(path):
    """Read a CSV whose data rows may carry extra trailing columns beyond the
    header. Pad the header, then drop rows that are still ragged."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    width = max(len(r) for r in rows)
    while len(header) < width:
        header.append(CANON_EXTRA[len(header) - len(rows[0])]
                      if len(header) - len(rows[0]) < len(CANON_EXTRA)
                      else f"extra_{len(header)}")
    # de-duplicate any repeated column labels so df[c] returns a Series
    seen_c = {}
    uniq = []
    for h in header:
        if h in seen_c:
            seen_c[h] += 1
            uniq.append(f"{h}.{seen_c[h]}")
        else:
            seen_c[h] = 0
            uniq.append(h)
    fixed = [r for r in rows[1:] if len(r) == width]
    dropped = len(rows) - 1 - len(fixed)
    df = pd.DataFrame(fixed, columns=uniq)
    for c in df.columns:                       # re-type numerics where possible
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().any():
            df[c] = conv.where(conv.notna(), df[c])
    return df, dropped


frames = []
sources = (glob.glob("tier1_shards_*/*.csv")
           + glob.glob("nsweep*_shards_*/*.csv")
           + glob.glob("tier1_creditg_n500*_shards_*/*.csv")
           + ["nsweep.csv", "tier1_creditg_n500.csv", "tier2.csv", "tier3.csv"])
seen = set()
for f in sources:
    if not os.path.exists(f) or f in seen:
        continue
    seen.add(f)
    try:
        df = pd.read_csv(f)
        note = ""
    except Exception:
        df, dropped = read_drifted(f)
        note = f"  (recovered via drift-parser, dropped {dropped} ragged rows)"
    if "synth" not in df.columns or "generation" not in df.columns:
        continue
    df["_src"] = os.path.basename(f)
    frames.append(df)
    print(f"{f:45s} rows={len(df):5d} synths={sorted(df.synth.unique())}{note}")

d = pd.concat(frames, ignore_index=True)
# dedup on the config key, keep last (append-only => later supersedes)
keys = ["dataset", "synth", "regime", "alpha", "seed", "generation", "n"]
before = len(d)
d = d.drop_duplicates(subset=keys, keep="last")
print(f"\nmerged {before} -> {len(d)} rows after dedup on {keys}")

# write atomically
tmp = "main.csv.tmp"
d.to_csv(tmp, index=False)
os.replace(tmp, "main.csv")

print("\n=== main.csv census ===")
print(d.groupby("synth").agg(rows=("generation", "size"),
                             alphas=("alpha", "nunique"),
                             regimes=("regime", "nunique"),
                             gmax=("generation", "max")).to_string())
if "sampled_with_replacement" in d.columns:
    swr = d["sampled_with_replacement"].fillna(False).astype(str).str.lower()
    n_true = (swr == "true").sum()
    print(f"\nsampled_with_replacement True rows: {n_true}")
