"""Confirm the style pass changed style only: no numbers, citations, or refs moved."""
import glob
import os
import re

os.chdir(os.path.join(os.path.dirname(__file__), "..", "paper"))

NUM = re.compile(r"\d+\.?\d*")
CITE = re.compile(r"\\cite\{([^}]*)\}")
REF = re.compile(r"\\(?:ref|eqref|label)\{([^}]*)\}")
EMDASH = re.compile(r"---")
# Hedge words that MUST survive -- these exist because the data forced them.
HEDGES = ["suggestive", "not necessarily", "should not be read", "conservative",
          "more slowly", "within noise", "caveat", "we do not claim",
          "optimistic bound", "rather than decisive"]

print(f"{'file':<22}{'em-dash':>8}{'nums':>7}{'cites':>7}{'refs':>7}")
tot_em = 0
allnums = {}
for f in sorted(glob.glob("sec_*.tex")):
    t = open(f, encoding="utf-8", errors="replace").read()
    em = len(EMDASH.findall(t))
    tot_em += em
    nums = NUM.findall(t)
    allnums[f] = sorted(nums)
    print(f"{f:<22}{em:>8}{len(nums):>7}"
          f"{len(CITE.findall(t)):>7}{len(REF.findall(t)):>7}")
print(f"\nTOTAL em-dashes remaining: {tot_em}")

print("\n=== hedges still present (must not have been strengthened) ===")
whole = "".join(open(f, encoding="utf-8", errors="replace").read()
                for f in glob.glob("sec_*.tex")).lower()
for h in HEDGES:
    print(f"  {'OK  ' if h in whole else 'GONE'} {h}")

print("\n=== key result numbers still present verbatim ===")
must = ["0.674", "0.389", "0.670", "0.893",     # Adult TSTR collapse + controls
        "0.15", "0.63", "0.47",                  # per-axis thresholds + spread
        "0.770", "0.715", "0.832", "0.784",      # fresh vs fixed support
        "0.959", "0.925", "1.043"]               # variance non-collapse
for m in must:
    hits = sum(1 for f in allnums for v in allnums[f] if v == m)
    print(f"  {'OK  ' if hits else 'MISSING'} {m}  ({hits} occurrence(s))")
