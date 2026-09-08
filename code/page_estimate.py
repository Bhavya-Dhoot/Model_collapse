"""IEEE two-column page estimate, calibrated against a real pdflatex run.

Ground truth (TeX Live 2025, IEEEtran conference, 10pt):
    "Lines per column: 56 (exact)"  ->  112 lines per page
    the 4402-word source measured 7 pages.

The earlier word-only model was structurally blind and under-predicted by a
full page: it costed a display equation as a handful of words when it really
occupies ~4 lines, and ignored section headings entirely (this paper has 28).
Everything is therefore counted in LINES here, not words.
"""
import glob
import os
import re

os.chdir(os.path.join(os.path.dirname(__file__), "..", "paper"))

LINES_PER_COL = 56
LINES_PER_PAGE = 2 * LINES_PER_COL
WORDS_PER_LINE = 9.5          # 10pt Times in a ~3.5in column

SECTIONS = ["sec_abstract", "sec_intro", "sec_related", "sec_problem",
            "sec_theory", "sec_experiments", "sec_results", "sec_discussion",
            "sec_conclusion"]

src = "".join(open(f, encoding="utf-8").read() for f in glob.glob("sec_*.tex"))
src = re.sub(r"(?m)(?<!\\)%.*$", "", src)

# ---- body text -------------------------------------------------------------
words = 0
for s in SECTIONS:
    t = open(s + ".tex", encoding="utf-8").read()
    t = re.sub(r"(?m)(?<!\\)%.*$", "", t)
    t = re.sub(r"\\begin\{(figure|table|tabular)\*?\}.*?\\end\{\1\*?\}", "",
               t, flags=re.S)
    t = re.sub(r"\\\[.*?\\\]|\\begin\{(gather|align|equation)\*?\}.*?"
               r"\\end\{\1\*?\}", "", t, flags=re.S)   # displays counted below
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}$\\&_^]", " ", t)
    words += len([x for x in t.split() if x.strip()])
text_lines = words / WORDS_PER_LINE

# ---- structural costs ------------------------------------------------------
n_display = (len(re.findall(r"\\\[", src))
             + len(re.findall(r"\\begin\{(?:gather|align|equation)\*?\}", src)))
n_sec = len(re.findall(r"\\section\{", src))
n_sub = len(re.findall(r"\\subsection\*?\{", src))
n_thm = len(re.findall(r"\\begin\{(?:proposition|proof|remark|definition|"
                       r"theorem|lemma|corollary)\}", src))
n_item = len(re.findall(r"\\item", src))
n_list = len(re.findall(r"\\begin\{(?:itemize|enumerate)\}", src))

display_lines = n_display * 4.0      # equation body + above/below skip
heading_lines = n_sec * 3.5 + n_sub * 2.5
thm_lines = n_thm * 1.5              # extra skip around theorem envs
list_lines = n_item * 0.6 + n_list * 1.5

# ---- floats (in column-inches -> lines) ------------------------------------
COL_IN = 9.7
figstar = len(re.findall(r"\\begin\{figure\*\}", src))
tikz = len(re.findall(r"\\begin\{tikzpicture\}", src))
fig = len(re.findall(r"\\begin\{figure\}", src)) - tikz
tab = len(re.findall(r"\\begin\{table\}", src))

# figure* spans both columns: height h costs h/COL_IN of a PAGE = 2h/COL_IN cols
figstar_lines = figstar * (2.7 + 0.55) / COL_IN * LINES_PER_PAGE
tikz_lines = tikz * (1.5 + 0.55) / COL_IN * LINES_PER_COL
fig_lines = fig * (2.7 + 0.55) / COL_IN * LINES_PER_COL
tab_lines = tab * (3.2 + 0.9) / COL_IN * LINES_PER_COL

# ---- front/back matter -----------------------------------------------------
title_lines = 26          # title block + authors + abstract framing + keywords
n_refs = len(re.findall(r"(?m)^@\w+\{", open("refs.bib", encoding="utf-8").read()))
ref_lines = n_refs * 2.3

total = (text_lines + display_lines + heading_lines + thm_lines + list_lines
         + figstar_lines + tikz_lines + fig_lines + tab_lines
         + title_lines + ref_lines)
pages = total / LINES_PER_PAGE

print(f"body text     {words:>5} words -> {text_lines:6.1f} lines")
print(f"displays      {n_display:>5}       -> {display_lines:6.1f}")
print(f"headings      {n_sec}sec {n_sub:>2}sub -> {heading_lines:6.1f}")
print(f"theorem envs  {n_thm:>5}       -> {thm_lines:6.1f}")
print(f"lists         {n_item:>5} items -> {list_lines:6.1f}")
print(f"figure*       {figstar:>5}       -> {figstar_lines:6.1f}")
print(f"tikz fig      {tikz:>5}       -> {tikz_lines:6.1f}")
print(f"figures       {fig:>5}       -> {fig_lines:6.1f}")
print(f"tables        {tab:>5}       -> {tab_lines:6.1f}")
print(f"title block               -> {title_lines:6.1f}")
print(f"references    {n_refs:>5}       -> {ref_lines:6.1f}")
print(f"{'':-<44}")
print(f"TOTAL {total:.0f} lines / {LINES_PER_PAGE} per page = {pages:.2f} PAGES")
over = pages - 6.0
if over > 0:
    print(f"  6-page limit: OVER by {over:.2f}pg "
          f"(~{int(over * LINES_PER_PAGE * WORDS_PER_LINE)} words of text)")
else:
    print(f"  6-page limit: FITS with {-over:.2f}pg spare")
