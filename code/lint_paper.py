"""Static checks on the LaTeX sources, since no TeX toolchain is installed here.
Catches the errors that would otherwise only surface on Overleaf.
    python code/lint_paper.py
"""
import glob
import os
import re

PAPER = os.path.join(os.path.dirname(__file__), "..", "paper")
os.chdir(PAPER)

bib = open("refs.bib", encoding="utf-8", errors="replace").read()
bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))

tex_files = sorted(glob.glob("sec_*.tex")) + ["main.tex"]
cited, labels, refs, graphics, inputs = set(), set(), set(), set(), set()
for f in tex_files:
    t = open(f, encoding="utf-8", errors="replace").read()
    t = re.sub(r"(?m)(?<!\\)%.*$", "", t)                     # strip real comments, keep \%
    for m in re.findall(r"\\cite\{([^}]*)\}", t):
        cited |= {k.strip() for k in m.split(",") if k.strip()}
    labels |= set(re.findall(r"\\label\{([^}]*)\}", t))
    refs |= set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]*)\}", t))
    graphics |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", t))
    inputs |= set(re.findall(r"\\input\{([^}]*)\}", t))

fail = 0


def check(name, bad, detail=""):
    global fail
    if bad:
        fail += 1
        print(f"  FAIL {name}: {sorted(bad) if not isinstance(bad, str) else bad} {detail}")
    else:
        print(f"  ok   {name}")


print("== citations ==")
check("all \\cite keys exist in refs.bib", cited - bib_keys)
print(f"       ({len(cited)} cited of {len(bib_keys)} bib entries; "
      f"{len(bib_keys - cited)} uncited: {sorted(bib_keys - cited)})")

print("== cross-references ==")
check("all \\ref targets have a \\label", refs - labels)

print("== \\input files exist ==")
check("inputs present", {i for i in inputs if not os.path.exists(i + ".tex")})

print("== figures ==")
missing_figs = set()
for g in graphics:
    if not any(os.path.exists(g + e) or os.path.exists(g)
               for e in (".pdf", ".png", ".eps", "")):
        missing_figs.add(g)
check("referenced graphics exist on disk", missing_figs)
if not graphics:
    print("  WARN paper contains NO figures at all")

print("== environments / braces ==")
for f in tex_files:
    t = open(f, encoding="utf-8", errors="replace").read()
    t = re.sub(r"(?m)(?<!\\)%.*$", "", t)
    begins = re.findall(r"\\begin\{([^}]*)\}", t)
    ends = re.findall(r"\\end\{([^}]*)\}", t)
    if sorted(begins) != sorted(ends):
        fail += 1
        print(f"  FAIL {f}: begin/end mismatch {sorted(begins)} vs {sorted(ends)}")
    body = t.replace(r"\{", "").replace(r"\}", "")
    if body.count("{") != body.count("}"):
        fail += 1
        print(f"  FAIL {f}: unbalanced braces "
              f"{body.count('{')} open / {body.count('}')} close")
    if t.count("$") % 2:
        fail += 1
        print(f"  FAIL {f}: odd number of $ (unclosed math)")
print(f"  ({len(tex_files)} files checked)")

print(f"\n{'ALL CHECKS PASSED' if not fail else str(fail) + ' CHECK(S) FAILED'}")
