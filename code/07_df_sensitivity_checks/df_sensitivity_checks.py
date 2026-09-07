#!/usr/bin/env python3
"""
df_sensitivity_checks.py
========================
Reproduces the two robustness checks quoted in Sect. 5 of the paper, both
each probing how sensitive the frozen-DF estimate is to a modelling choice.

(1) THRESHOLD CRITERION.  The DF-based critical SFE is defined by
    F_b < F_MIN with F_MIN = 0.02, which is a convention.  This recomputes
    the King thresholds for F_MIN = 0.01, 0.02 and 0.05 so that the
    sensitivity to that choice can be quoted rather than assumed.
    Result (paper values): the threshold is unchanged at W0 = 3 and moves
    by 25, 19 and 99 per cent at W0 = 6, 9 and 12 -- large in relative
    terms, but <= 0.014 in absolute SFE, against a curve spanning a factor
    ~20.  The non-monotonicity is unaffected.

(2) POSITIVITY OF THE DISTRIBUTION FUNCTION.  Eddington inversion of a
    density in a potential it does not generate is not guaranteed to give
    a non-negative f(eps); if it did not, the model would be unphysical.
    This scans (W0, SFE) and reports the most negative value found,
    normalized to the peak of f.
    Result (paper value): non-negative everywhere, worst normalized
    minimum +1e-15.

Outputs: df_threshold_sensitivity.csv, and both tables printed.
Runtime: a few minutes for (1) at adams_df_boundfraction's own N_EPS (2000
with the analytic inversion -- no local override here
any more; that inversion is now a convergent quadrature in N_EPS, so there
is no accuracy/speed excuse for silently running a robustness check at a
coarser resolution than the production DF numbers use); (2) is fast.

Needs only numpy/scipy plus the analysis modules (no Agama).
"""

import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
import sys
sys.path.append('../04_adams_df_boundfraction')
sys.path.append('../01_king_sfe_analysis')
import adams_df_boundfraction as A
from king_sfe_analysis import king_model, gas_density, tsf_for_sfe
from adams_df_boundfraction import relative_potential, eddington_f_analytic

# ----------------------------------------------------------------------
W0_LIST   = [3.0, 6.0, 9.0, 12.0]        # simulated grid of Paper I
F_MIN_SET = [0.01, 0.02, 0.05]
N_GRID    = 2500                         # radial grid of the King model
SFE_SCAN  = [0.02, 0.05, 0.10, 0.17, 0.30, 0.50]     # for the positivity scan
W0_SCAN   = [0.5, 3.0, 6.0, 9.0, 12.0, 16.0, 20.0]
OUT       = 'df_threshold_sensitivity.csv'


def threshold_sensitivity():
    """DF critical SFE for several bound-fraction criteria.

    Each threshold now carries adams_df_boundfraction's ERR_TOL
    reconstruction-check diagnostic (ANY_INVALID column); see that module's
    ERR_TOL doc for what it means. Nothing about the F_min sensitivity
    calculation itself changes."""
    print("(1) sensitivity of the DF threshold to the bound-fraction level")
    head = "".join(f"F_b>{f:<6.2f}" for f in F_MIN_SET)
    print(f"  {'W0':>5}  {head}{'spread/mid':>12}  any_invalid")
    rows = []
    for w in W0_LIST:
        m = king_model(w, n_grid=N_GRID)
        vals, valids = [], []
        for fmin in F_MIN_SET:
            A.F_MIN = fmin                     # used by threshold_by_bisection
            sc, _, _, diag = A.threshold_by_bisection(m, f_min=fmin)
            vals.append(sc)
            valids.append(diag['valid'])
        spread = 100.0 * (max(vals) - min(vals)) / vals[len(vals) // 2]
        any_invalid = not all(valids)
        rows.append([w] + vals + [float(any_invalid)])
        print(f"  {w:5.1f}  " + "".join(f"{v:<11.4f}" for v in vals)
              + f"{spread:11.1f}%  {'** YES **' if any_invalid else 'no'}", flush=True)
    A.F_MIN = 0.02                             # restore the default
    rows = np.array(rows)
    np.savetxt(OUT, rows, fmt='%.6f', delimiter=', ',
               header='W0, ' + ', '.join(f'SFE_crit_Fb{f:g}' for f in F_MIN_SET)
               + ', any_invalid', comments='')
    print(f"  -> {OUT}")
    return rows


def df_positivity():
    """Most negative value of the inverted DF over the (W0, SFE) grid.

    NOTE: eddington_f_analytic() unconditionally clips
    f=max(f,0) before returning, so f.min()/f.max() on the returned array is
    non-negative BY CONSTRUCTION and was not actually testing anything before
    return_diag existed -- any true negativity in the raw inversion was
    silently discarded. This reads the pre-clip diagnostic via
    return_diag=True, so 'worst normalized minimum' below is the real
    number, not a tautology."""
    print("\n(2) positivity of the inverted distribution function")
    print(f"  {'W0':>5}  {'min f_raw / max f':>18}  {'negative bins':>14}  {'clipped?':>9}")
    worst = np.inf
    any_clipped = False
    for w in W0_SCAN:
        m = king_model(float(w))
        rel_min, neg, clipped_here = 1.0, 0.0, False
        for sfe in SFE_SCAN:
            try:
                tsf = tsf_for_sfe(m, sfe)
            except Exception:
                continue                        # SFE unreachable for this model
            rho_g = gas_density(m['rho'], tsf)
            psi, _ = relative_potential(m['x'], m['rho'] + rho_g)
            eps, f, diag = eddington_f_analytic(m, rho_g, psi[-1], return_diag=True)
            rel_min = min(rel_min, diag['worst_rel_negativity'])
            neg = max(neg, diag['neg_frac'])
            clipped_here = clipped_here or diag['clipped_any']
        worst = min(worst, rel_min)
        any_clipped = any_clipped or clipped_here
        print(f"  {w:5.1f}  {rel_min:+18.3e}  {neg:14.3f}  {'YES' if clipped_here else 'no':>9}",
              flush=True)
    print(f"\n  worst normalized minimum over the grid (pre-clip): {worst:+.3e}")
    if worst < 0:
        print("  WARNING: the RAW inversion went negative somewhere in this grid "
              "(clipped to 0 before use by eddington_f) -- the model is not "
              "realizable with an isotropic DF at those parameters, or the "
              "inversion is not resolved there. See the per-W0 'clipped?' column.")
    elif any_clipped:
        print("  Note: 'clipped?' is YES somewhere above even though the worst "
              "normalized minimum stayed >=0 -- check the affected W0/SFE by eye.")
    return worst


if __name__ == '__main__':
    threshold_sensitivity()
    df_positivity()
