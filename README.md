# Gas–star segregation and the resilience of King-family clusters to gas expulsion

Manuscript, figures, analysis code and data tables for

> B. Shukirgaliyev, *Gas–star segregation and the resilience of King-family
> clusters to gas expulsion: a semi-analytic study*, submitted to
> Astronomy & Astrophysics.

Everything needed to rebuild every figure and number in the paper is here.

## What is in the paper

For a King-model cluster whose stars form with a centrally peaked star
formation efficiency, the residual gas ends up more extended than the stars.
How damaging that gas is when it is expelled is set by a single dimensionless
parameter, the gas harmfulness `eta`, and `eta` is minimised near `W0 ~ 8-9`
because the King structural ratio `r_t/r_h` peaks there. The predicted
survivability is therefore **non-monotonic in concentration** — that ordering,
rather than any absolute threshold, is the falsifiable result.

Two static estimators are computed and kept carefully distinct:

| symbol | what it is |
|---|---|
| `SFE_vir` | global SFE at which the post-expulsion virial ratio reaches an adopted `eSFE_surv = 1/3` |
| `SFE_DF`  | global SFE at which the frozen distribution function retains a bound fraction `F_b = 0.02` |
| `SFE_crit`| the dynamical threshold of a live cluster — **not computed here**; the companion N-body campaign measures it |

## Layout

```
main.tex, main.pdf     manuscript (A&A class files included so it compiles as-is)
references.bib         bibliography (NASA ADS records)
figures/               the nine figures included by main.tex
code/                  analysis pipeline, one directory per stage
docs/figure_data_map.csv   which script and which input produce each figure
```

`docs/figure_data_map.csv` is the fastest way in: for every figure it names the
producing script, its inputs, whether AGAMA is required, and roughly what it
costs to recompute.

## Reproducing the figures

Requires Python 3 with NumPy, SciPy and Matplotlib; the Monte Carlo stages also
need [AGAMA](https://github.com/GalacticDynamics-Oxford/Agama).

Most figures redraw from committed CSVs in seconds, without repeating the
expensive distribution-function work:

```bash
cd code/08_rebuild_paper_figures && python3 rebuild_paper_figures.py
cd code/04_adams_df_boundfraction && python3 plot_df_figures.py
```

`code/REPRODUCTION_GUIDE.md` gives the full stage order and the cost of each,
and `code/build.sh` runs the whole pipeline end to end. The expensive parts —
the frozen-DF sweep (hours) and the N = 1e5 Monte Carlo grid (~3 h) — are only
needed if the distribution-function machinery itself changes; their outputs are
committed.

## Compiling the manuscript

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`aa.cls`, `aa.bst`, `lineno.sty` and `linenoaa.sty` are the Astronomy &
Astrophysics author package, included here so the source builds without extra
downloads; they are distributed by A&A and remain under their own terms.

## Citing

Please cite the paper. If you use the code or data tables directly, cite
the archived release as well:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22646889.svg)](https://doi.org/10.5281/zenodo.22646889)

    doi:10.5281/zenodo.22646889
