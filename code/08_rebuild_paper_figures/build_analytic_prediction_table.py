#!/usr/bin/env python3
"""
build_analytic_prediction_table.py
==================================
Produces `analytic_prediction_table.csv`, the input
`rebuild_paper_figures.fig12()` reads.

At every (W0, SFE) point the Monte Carlo run visited, evaluate the continuum
frozen-DF bound fraction `fbound_at_sfe` and column-merge it with the MC mean
and standard deviation at that point, split by particle number N.

Inputs (searched in '.', '../*/', '..'):
  mc_bound_fractions*.csv  -- per-realization MC output of
      mc_bound_fraction_agama.py. Columns:
      W0, SFE_target, SFE_actual, seed, F_bound, n_iter, N_part
      Run the MC once with N_PART = 10_000 and once with N_PART = 100_000
      (either as two files or appended into one); rows are grouped by the
      N_part column, so both end up in the table.

Output columns (exactly what fig12 expects):
  W0, SFE, F_analytic, F_MC_N10k_mean, F_MC_N10k_std,
      F_MC_N100k_mean, F_MC_N100k_std
MC columns are NaN for any (W0, SFE) not covered at that N.

Needs numpy/scipy + the analysis modules (no Agama). Runtime: ~1-2 min
(one fbound_at_sfe evaluation per grid point).
"""

import glob
import os
import sys
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

sys.path.append('../01_king_sfe_analysis')
sys.path.append('../03_multi_profile_analysis')
sys.path.append('../04_adams_df_boundfraction')
from king_sfe_analysis import king_model
from adams_df_boundfraction import fbound_at_sfe

OUTDIR = './'
N_BINS = {10000: 'N10k', 100000: 'N100k'}   # particle counts -> column tag
SFE_ROUND = 4                                # match MC SFE_target to this precision


def _find_all(pattern):
    hits = []
    for pat in (pattern, os.path.join('..', '*', pattern),
                os.path.join('..', pattern)):
        hits += sorted(glob.glob(pat))
    # de-duplicate by realpath, keep order
    seen, out = set(), []
    for h in hits:
        rp = os.path.realpath(h)
        if rp not in seen:
            seen.add(rp); out.append(h)
    return out


def load_mc():
    files = _find_all('mc_bound_fractions*.csv')
    if not files:
        sys.exit("no mc_bound_fractions*.csv found (run mc_bound_fraction_agama.py "
                 "at N_PART = 10_000 and 100_000 first)")
    print("reading MC data from:")
    rows = []
    for f in files:
        d = np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2)
        print(f"  {f}  ({d.shape[0]} rows)")
        rows.append(d)
    return np.vstack(rows)   # W0, SFE_target, SFE_actual, seed, F_bound, n_iter, N_part


def mc_stats(d):
    """{(W0, SFE): {N_part: (mean, std)}} over realizations."""
    out = {}
    W0 = d[:, 0]; sfe = np.round(d[:, 1], SFE_ROUND)
    Fb = d[:, 4]; Npart = np.round(d[:, 6]).astype(int)
    for key in {(w, s) for w, s in zip(W0, sfe)}:
        w, s = key
        m = (W0 == w) & (sfe == s)
        per_n = {}
        for n in np.unique(Npart[m]):
            v = Fb[m & (Npart == n)]
            per_n[int(n)] = (float(v.mean()),
                             float(v.std(ddof=1)) if v.size > 1 else 0.0)
        out[key] = per_n
    return out


def main():
    d = load_mc()
    stats = mc_stats(d)

    models = {}
    def model(w):
        if w not in models:
            models[w] = king_model(w)
        return models[w]

    rows = []
    n_invalid = 0
    for (w, s) in sorted(stats):
        # return_diag: adams_df_boundfraction's ERR_TOL reconstruction check
        # (see that module's doc); F_analytic is unchanged either way, but a
        # 'valid'=0 row should not be quoted without checking DF_err there.
        Fa, diag = fbound_at_sfe(model(w), float(s), convention='total', return_diag=True)
        rec = [w, s, Fa]
        for n in (10000, 100000):
            mean, std = stats[(w, s)].get(n, (np.nan, np.nan))
            rec += [mean, std]
        rec += [diag['err'], float(diag['valid'])]
        rows.append(rec)
        have = "+".join(N_BINS[n] for n in (10000, 100000)
                        if n in stats[(w, s)]) or "no MC"
        flag = '' if diag['valid'] else f"  ** DF reconstruction check FAILED (err={diag['err']:.2e}) **"
        n_invalid += (not diag['valid'])
        print(f"W0={w:5.1f} SFE={s:.3f}  F_analytic={Fa:.4f}  [{have}]{flag}")
    if n_invalid:
        print(f"\n  {n_invalid}/{len(rows)} F_analytic points above failed the "
              "DF reconstruction check -- see DF_err/DF_valid in the CSV.")

    hdr = ('W0, SFE, F_analytic, F_MC_N10k_mean, F_MC_N10k_std, '
           'F_MC_N100k_mean, F_MC_N100k_std, DF_err, DF_valid')
    np.savetxt(OUTDIR + 'analytic_prediction_table.csv', np.array(rows),
               fmt='%.6f', delimiter=', ', header=hdr, comments='')
    print("\nWrote analytic_prediction_table.csv -- rebuild_paper_figures.py "
          "fig12() picks it up automatically.")


if __name__ == '__main__':
    main()
