#!/usr/bin/env python3
"""
df_mc_thresholds_dense.py
=========================
For manuscript Fig. 10 (file fig11): critical SFE at every Delta W0 = 0.5,
from BOTH the continuum frozen-DF calculation and Agama Monte Carlo.
Run on your own machine (needs agama + the four analysis scripts in cwd).

Outputs: dense_thresholds.csv with columns
  W0, SFE_crit_continuum, SFE_crit_MC, MC_err_bisection, MC_err_stat,
  DF_worst_err, DF_valid
MC_err_bisection is the final bisection bracket's half-width -- a NUMERICAL
resolution tolerance, not a statistical uncertainty (this used
to be the sole "MC_err" column and was plotted as an error bar, which
overstates precision -- it shrinks with more N_BISECT regardless of how
noisy the underlying N_REAL-realization survival statistic actually is).
MC_err_stat is the real one: a Wilson score interval on the binomial
survival counts at the bisection's final bracket, propagated to SFE via the
local slope of the survival-probability curve -- reuses samples already
drawn, no extra Agama calls. Quote MC_err_stat, not MC_err_bisection, as
the Monte Carlo uncertainty; MC_err_bisection is kept for anyone checking
numerical convergence of the bisection itself.
DF_worst_err/DF_valid are adams_df_boundfraction's ERR_TOL reconstruction-
check diagnostic for SFE_crit_continuum (see that module for what it
means): DF_valid=0 means the Eddington-inversion self-consistency check
failed for this W0 and SFE_crit_continuum should not be quoted without
looking at DF_worst_err. No local N_EPS override here (
forced to 600, coarser than the module default, for wall-clock reasons
across a grid this dense): with the analytic inversion
that inversion is a convergent quadrature in N_EPS, so under-resolving it
here would silently reintroduce the accuracy loss that rewrite fixed --
this script now uses adams_df_boundfraction's own default (2000) like
everything else, at correspondingly longer runtime. Rerunning at a schema
older than this will produce a column-count mismatch in
dense_thresholds.csv on restart; delete/rename any pre-existing file from
before this diagnostic was added.
Runtime: the continuum part dominates; expect several hours for the full
W0 = 0.5..20 grid. Edit W0_GRID to split the work across sessions --
results are appended, so the script is restartable.

MC threshold definition: the SFE at which the survival probability over
N_REAL realizations of N_PART particles crosses 50 per cent, found by
bisection (each realization votes survive/collapse via F_b > F_MIN).
"""
import os
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
import sys 
sys.path.append('../04_adams_df_boundfraction')
import adams_df_boundfraction as A
sys.path.append('../01_king_sfe_analysis')
from king_sfe_analysis import king_model
sys.path.append('../05_mc_bound_fraction_agama')
from mc_bound_fraction_agama import (build_model, tsf_for_sfe,
                                     bound_fraction_particles)
import agama

W0_GRID = np.arange(0.5, 20.01, 0.5)
N_PART  = 10_000
N_REAL  = 9              # realizations per MC evaluation (odd: majority vote)
F_MIN   = 0.02
N_BISECT = 7
OUT = 'dense_thresholds.csv'


def mc_survival_prob(W0, sfe):
    """Raw survival COUNT out of N_REAL (not the fraction) -- callers need
    the count for the binomial statistics in mc_threshold below."""
    tsf = tsf_for_sfe(W0, sfe)
    pot_total, king_pot, df, _ = build_model(W0, tsf)
    gm = agama.GalaxyModel(potential=pot_total, df=df)
    n_surv = 0
    for i in range(N_REAL):
        agama.setRandomSeed(1000 + i)
        xv, _ = gm.sample(N_PART)
        F, _ = bound_fraction_particles(xv[:, :3], xv[:, 3:])
        n_surv += (F > F_MIN)
    return n_surv


def wilson_interval(k, n, z=1.0):
    """Wilson score confidence interval for a binomial proportion k/n.
    z=1.0 gives the ~68% (1-sigma-equivalent) interval, matching how MC
    spread is quoted elsewhere in this pipeline (a std, not a 95% CI) --
    NOT the normal approximation (unreliable at n=N_REAL=9 and p near 0/1,
    both of which happen here). Returns (phat, lo, hi)."""
    if n <= 0:
        return 0.5, 0.0, 1.0
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return phat, center - half, center + half


def mc_threshold(W0, lo=0.003, hi=0.6):
    """Returns (SFE_crit_MC, err_bisection, err_stat).

    err_bisection = (b-a)/2, the width of the final bracket -- a NUMERICAL
    resolution tolerance on where the survival probability crosses 50%, not
    a statistical uncertainty (do not export it as
    'MC_err' and plotted as an error bar, which is what it is not).

    err_stat is the actual statistical uncertainty: a Wilson score interval
    on the binomial survival counts at the two bracket endpoints (N_REAL
    realizations each), propagated to SFE via the local slope of the
    survival-probability curve across the (by then very narrow) bracket. No
    extra Agama sampling -- reuses the counts the bisection already computed.
    """
    # coarse bracket on survival probability crossing 0.5
    scan = np.geomspace(lo, hi, 8)
    counts = [mc_survival_prob(W0, s) for s in scan]
    ps = [c / N_REAL for c in counts]
    br = None
    for i in range(1, len(scan)):
        if ps[i - 1] < 0.5 <= ps[i]:
            br = (scan[i - 1], scan[i], counts[i - 1], counts[i])
    if br is None:
        return np.nan, np.nan, np.nan
    a, b, ka, kb = br
    for _ in range(N_BISECT):
        mid = np.sqrt(a * b)
        kmid = mc_survival_prob(W0, mid)
        if kmid / N_REAL < 0.5:
            a, ka = mid, kmid
        else:
            b, kb = mid, kmid
    sfe_c = np.sqrt(a * b)
    err_bisection = (b - a) / 2.0

    pa, lo_a, hi_a = wilson_interval(ka, N_REAL)
    pb, lo_b, hi_b = wilson_interval(kb, N_REAL)
    slope = (pb - pa) / (b - a) if b > a else 0.0
    if abs(slope) > 1e-12:
        p_halfwidth = 0.25 * ((hi_a - lo_a) + (hi_b - lo_b))
        err_stat = p_halfwidth / abs(slope)
    else:
        err_stat = np.nan   # ka==kb: no local slope info to propagate through
    return sfe_c, err_bisection, err_stat


done = set()
if os.path.exists(OUT):
    prev = np.loadtxt(OUT, delimiter=',', skiprows=1, ndmin=2)
    done = set(np.round(prev[:, 0], 2))
    print(f'restarting: {len(done)} W0 values already done')
else:
    with open(OUT, 'w') as f:
        f.write('W0, SFE_crit_continuum, SFE_crit_MC, MC_err_bisection, '
                'MC_err_stat, DF_worst_err, DF_valid\n')

for w in W0_GRID:
    if round(w, 2) in done:
        continue
    m = king_model(w, n_grid=2500)
    sc_cont, _, _, diag = A.threshold_by_bisection(m)
    sc_mc, err_bisection, err_stat = mc_threshold(w)
    with open(OUT, 'a') as f:
        f.write(f"{w:.2f}, {sc_cont:.6f}, {sc_mc:.6f}, {err_bisection:.6f}, "
                f"{err_stat:.6f}, {diag['worst_err']:.6f}, {int(diag['valid'])}\n")
    flag = '' if diag['valid'] else f"  ** DF reconstruction check FAILED (err={diag['worst_err']:.2e}) **"
    print(f'W0={w:5.2f}: continuum={sc_cont:.4f}  MC={sc_mc:.4f}'
          f' +-{err_stat:.4f} (stat) [+-{err_bisection:.4f} bisection]{flag}',
          flush=True)

print('done ->', OUT)

