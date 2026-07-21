"""Rough IEEE two-column page estimate, since no TeX toolchain is available."""
import glob
import os
import re

os.chdir(os.path.join(os.path.dirname(__file__), "..", "paper"))

SECTIONS = ["sec_abstract", "sec_intro", "sec_related", "sec_problem",
            "sec_theory", "sec_experiments", "sec_results", "sec_discussion",
            "sec_conclusion"]

tot = 0
print(f"{'section':<22}{'words':>7}")
for s in SECTIONS:
    t = open(s + ".tex", encoding="utf-8").read()
    t = re.sub(r"(?m)(?<!\\)%.*$", "", t)                       # comments
    t = re.sub(r"\\begin\{(figure|table|tabular)\*?\}.*?\\end\{\1\*?\}",
               "", t, flags=re.S)                                # floats counted separately
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)                        # macros
    t = re.sub(r"[{}$\\&_^]", " ", t)
    w = len([x for x in t.split() if x.strip()])
    print(f"{s:<22}{w:>7}")
    tot += w
print(f"{'TEXT TOTAL':<22}{tot:>7}")

src = "".join(open(f, encoding="utf-8").read() for f in glob.glob("sec_*.tex"))
figstar = len(re.findall(r"\\begin\{figure\*\}", src))
fig = len(re.findall(r"\\begin\{figure\}", src))
tab = len(re.findall(r"\\begin\{table\}", src))
print(f"\nfloats: {fig} single-col figures, {figstar} full-width figures, {tab} tables")

text_pg = tot / 1050.0            # IEEE 10pt two-column, ~1050 words/page
float_pg = figstar * 0.34 + fig * 0.22 + tab * 0.20
refs_pg = 34 / 38.0               # ~38 IEEE refs per column-page
overhead = 0.4                    # title block + abstract framing
total = text_pg + float_pg + refs_pg + overhead
print(f"\ntext {text_pg:.1f} + floats {float_pg:.1f} + refs {refs_pg:.1f} "
      f"+ overhead {overhead:.1f}")
print(f"ESTIMATED TOTAL: {total:.1f} pages")
for limit in (6, 8, 10):
    over = total - limit
    verdict = "FITS" if over <= 0 else f"OVER by {over:.1f}pg (~{int(over*1050)} words)"
    print(f"  {limit}-page limit: {verdict}")
