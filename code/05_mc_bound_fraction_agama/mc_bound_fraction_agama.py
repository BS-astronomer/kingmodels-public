#!/usr/bin/env python3
"""
mc_bound_fraction_agama.py
==========================
Monte Carlo verification of the semi-analytic DF-based bound fraction,
WITHOUT any N-body time integration. Designed to run standalone on your
own machine (requires: agama, numpy, scipy, matplotlib).

What it does
------------
For each (W0, SFE, seed):
 1. Builds the embedded King cluster exactly as gen_king_sc.py does:
    King stellar density + closed-form PP13 gas profile ->
    multipole gas potential -> QuasiSpherical DF in the total potential ->
    sample N equal-mass particles with full velocities (Agama).
 2. Applies instantaneous gas expulsion: gas potential vanishes, particles
    keep (r, v).
 3. Iterated escape criterion: particle i is bound iff
    v_i^2/2 < Psi_b(r_i), with Psi_b computed from the BOUND particles
    only (spherical estimator); iterate until the bound set is stable.
 4. Records the bound fraction F_b. Repeating over seeds gives the
    finite-N sampling scatter -- i.e. the stochastic uncertainty of the
    actual Paper I initial conditions.

Outputs
-------
 - mc_bound_fractions.csv : one row per (W0, SFE_target, SFE_actual, seed, F_b)
 - mc_vs_semianalytic.pdf/png : comparison plot; the semi-analytic (continuum
   frozen-DF) curves are computed live from adams_df_boundfraction /
   king_sfe_analysis, and the virial threshold is overlaid from
   mechanism_summary.csv when present (see _find_csv).

Runtime: with the default grid (4 W0 x 6 SFE x 10 seeds, N=10^4) expect
roughly 10-30 minutes on a laptop; sampling dominates, the criterion
itself is O(N log N) per iteration.

Units: G = M_star = r0 = 1 throughout (dimensionless; make sure Agama is
used WITHOUT physical units, i.e. do not call agama.setUnits).
"""

import numpy as np
if not hasattr(np, 'trapezoid'):          # numpy < 2.0 compatibility
    np.trapezoid = np.trapz
import agama
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# ----------------------------------------------------------------------
# CONFIGURATION -- edit freely
# ----------------------------------------------------------------------
N_PART   = 10_000
N_SEEDS  = 50            # realizations per (W0, SFE); 20 is enough away from
                         # threshold, 50 pays off in the marginal cells where
                         # the outcome is bimodal
W0_LIST  = [3.0, 6.0, 9.0, 12.0]
# SFE values per W0: chosen to bracket the semi-analytic DF thresholds
# (0.196, 0.147, 0.026, 0.016 for W0=3,6,9,12); edit as you like.
SFE_LIST = {
    3.0:  [0.25, 0.30, 0.305, 0.31, 0.35, 0.4],
    6.0:  [0.16, 0.17, 0.18, 0.20, 0.22, 0.30, 0.4],
    9.0:  [0.03, 0.04, 0.045, 0.05, 0.06, 0.10, 0.17, 0.20, 0.30],
    12.0: [0.01, 0.015, 0.03, 0.06, 0.12, 0.20, 0.30, 0.4],
}
SEEDS    = list(np.random.default_rng(42).integers(1_000_000, 99_999_999,
                                                 size=N_SEEDS))
EPS_FF   = 0.01   # only t_sf*eps_ff enters; committed CSVs are eps_ff-invariant
RMIN, RMAX = 1e-3, 1e4   # RMAX is now only a probe cap; grids are adaptive


def king_rt(king_pot, r_probe_max=1e6):
    """Truncation radius: outermost radius with nonzero King density."""
    rp = np.logspace(-3, np.log10(r_probe_max), 400)
    xyz = np.zeros((rp.size, 3)); xyz[:, 0] = rp
    pos = np.where(king_pot.density(xyz) > 0)[0]
    lo, hi = rp[pos[-1]], rp[pos[-1] + 1]
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if king_pot.density([[mid, 0, 0]])[0] > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
def _find_csv(name, verbose=True):
    """Find a CSV written by a sibling script.

    Searches, in order: the working directory, sibling directories (../*/),
    the parent, and one level further out (../../*/) -- enough for the
    numbered-folder reproduction layout. Prints where it looked when it fails,
    because a silently missing file is how curves quietly vanish from figures.
    """
    import glob, os
    pats = (name, os.path.join('..', '*', name), os.path.join('..', name),
            os.path.join('..', '..', '*', name))
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            if verbose:
                print(f"  found {name} at {hits[0]}")
            return hits[0]
    if verbose:
        print(f"  {name} NOT FOUND (looked in: {', '.join(pats)})")
    return None


OUTCSV   = 'mc_bound_fractions.csv'
SHOW_VIRIAL = True    # mark the virial threshold in each panel (see below)
ERRBAR   = 'std'      # 'std' = spread of outcomes over realizations (default,
                      # the physically meaningful quantity near threshold,
                      # where the outcome is bimodal); 'sem' = uncertainty of
                      # the mean, std/sqrt(n_real)

# ----------------------------------------------------------------------
# Gas profile: closed-form inversion of the PP13 local-SFE relation
# (identical to gen_king_sc.py / king_sfe_analysis.py, including the
# large-alpha asymptotic branch)
# ----------------------------------------------------------------------
def gas_density(rho_s, tsf, eps_ff=EPS_FF):
    k = np.sqrt(8.0 / (3.0 * np.pi)) * eps_ff * tsf
    k4 = k**4
    rho_s = np.asarray(rho_s, dtype=float)
    rs = np.maximum(rho_s, 1e-140)
    alpha = k4 * rs**2
    K0 = (alpha**3 + 36 * alpha**2 + 216 * alpha
          + 24 * alpha * np.sqrt(3.0 * (alpha + 27.0)))**(1.0 / 3.0)
    k4K0 = k4 * K0
    K1 = np.sqrt((alpha**2 + alpha * (K0 + 24.0) + K0 * (K0 + 12.0)) / (12.0 * k4K0))
    K2 = (alpha - K0 + 24.0) * (K0 - alpha) / (3.0 * k4K0)
    rho_g = 1.0 / k**2 - rs / 2.0 - np.sqrt(np.maximum(K2 + 8.0 / (k**6 * K1), 0.0)) / 2.0 + K1
    big = alpha > 1e8
    if np.any(big):
        rb = rs[big]
        rho_g[big] = 1.0 / k**2 - 2.0 / (k**3 * np.sqrt(rb)) + 3.0 / (k**4 * rb)
    # Small-alpha branch: for alpha < 1e-30 the exact expression is destroyed by
    # cancellation (all terms ~1/k^2 while the answer is ~1e-17 of that) and can
    # return exactly 0, which makes the local SFE rho_s/(rho_s+rho_g) spike to 1
    # at the truncation radius. The asymptotic limit is rho_g -> (rho_s/(2k))^(2/3)
    # (verified: exact/asymptotic = 1 - 1.7e-5 at alpha = 1e-28).
    tiny = alpha < 1e-30
    if np.any(tiny):
        rho_g[tiny] = (rs[tiny] / (2.0 * k))**(2.0 / 3.0)
    rho_g = np.maximum(rho_g, 0.0)
    rho_g[rho_s <= 0.0] = 0.0
    return rho_g


# ----------------------------------------------------------------------
# Model construction (mirrors gen_king_sc.py)
# ----------------------------------------------------------------------
def build_model(W0, tsf):
    """Return (pot_total, king_pot, df, sfe_actual) for given (W0, tsf)."""
    king_pot = agama.Potential(type='king', W0=W0, scaleRadius=1.0, mass=1.0)
    rmax_ad = 1.1 * king_rt(king_pot)        # adaptive: 1.1 * truncation radius
    npts = 3000
    r = np.logspace(np.log10(RMIN), np.log10(rmax_ad), npts)
    xyz = np.zeros((npts, 3)); xyz[:, 0] = r
    rho_s = king_pot.density(xyz)
    rho_g = gas_density(rho_s, tsf)

    gi = interp1d(r, rho_g, kind='linear', bounds_error=False,
                  fill_value=(rho_g[0], 0.0))
    def gasdens(xyz):
        xyz = np.asarray(xyz)
        rr = np.linalg.norm(xyz, axis=1) if xyz.ndim == 2 else np.linalg.norm(xyz)
        return gi(rr)

    # FIX: fit the multipole only over the region where gas actually exists.
    # With a fixed rmax=1e4 the fit spans decades of exactly-zero density
    # beyond the King truncation; Agama's internal grid placement then
    # shifts non-smoothly with tsf and can inject small potential
    # artifacts -- enough to flip marginal configurations (seen as
    # non-monotonic F_b vs SFE near threshold at W0=3).
    pot_gas = agama.Potential(type='multipole', symmetry='spherical',
                              density=gasdens, rmin=RMIN, rmax=rmax_ad,
                              gridSizeR=60)
    pot_total = agama.Potential(pot_gas, king_pot)
    df = agama.DistributionFunction(type='QuasiSpherical',
                                    potential=pot_total, density=king_pot)
    # actual global SFE from the gas mass integral
    Mg = np.trapezoid(4.0 * np.pi * rho_g * r**2, r)
    return pot_total, king_pot, df, 1.0 / (1.0 + Mg)


def tsf_for_sfe(W0, sfe_target):
    """Invert SFE(tsf) for this W0 (self-contained; no lookup files)."""
    king_pot = agama.Potential(type='king', W0=W0, scaleRadius=1.0, mass=1.0)
    rmax_ad = 1.1 * king_rt(king_pot)
    npts = 3000
    r = np.logspace(np.log10(RMIN), np.log10(rmax_ad), npts)
    xyz = np.zeros((npts, 3)); xyz[:, 0] = r
    rho_s = king_pot.density(xyz)
    def sfe_of_logt(lt):
        rho_g = gas_density(rho_s, 10.0**lt)
        Mg = np.trapezoid(4.0 * np.pi * rho_g * r**2, r)
        return 1.0 / (1.0 + Mg) - sfe_target
    return 10.0**brentq(sfe_of_logt, -2.0, 7.5, xtol=1e-6)


# ----------------------------------------------------------------------
# Iterated escape criterion on particles
# ----------------------------------------------------------------------
def spherical_potential_at_particles(r_sorted, m):
    """Psi(r_i) = M(<r_i)/r_i + sum_{r_j>r_i} m_j/r_j  (G=1), for particles
    sorted by radius; excludes the self-term."""
    n = len(r_sorted)
    M_inside = np.concatenate([[0.0], np.cumsum(np.full(n, m))])[:-1]  # mass strictly inside
    inv_r = m / r_sorted
    suffix = np.concatenate([np.cumsum(inv_r[::-1])[::-1][1:], [0.0]])  # sum over j>i
    return M_inside / r_sorted + suffix


def bound_fraction_particles(pos, vel, n_iter=200):
    """Iterated escape criterion; returns F_b and number of iterations."""
    r = np.linalg.norm(pos, axis=1)
    v2 = np.einsum('ij,ij->i', vel, vel)
    order = np.argsort(r)
    r_s, v2_s = r[order], v2[order]
    n = len(r_s)
    m = 1.0 / n
    bound = np.ones(n, dtype=bool)
    for it in range(n_iter):
        rb = r_s[bound]
        if len(rb) == 0:
            return 0.0, it
        # potential of the bound subset, evaluated at ALL particle radii:
        # M_b(<r_i)/r_i + sum_{bound j: r_j>r_i} m/r_j
        Mb_below = np.searchsorted(rb, r_s, side='left') * m
        inv_rb_suffix = np.concatenate([np.cumsum((m / rb)[::-1])[::-1], [0.0]])
        idx = np.searchsorted(rb, r_s, side='right')
        psi_b = Mb_below / r_s + inv_rb_suffix[idx]
        new_bound = 0.5 * v2_s < psi_b
        if np.array_equal(new_bound, bound):
            return new_bound.sum() / n, it
        bound = new_bound
    return bound.sum() / n, n_iter


# ----------------------------------------------------------------------
def main():
    results = []
    for W0 in W0_LIST:
        for sfe_t in SFE_LIST[W0]:
            tsf = tsf_for_sfe(W0, sfe_t)
            pot_total, king_pot, df, sfe_act = build_model(W0, tsf)
            gm = agama.GalaxyModel(potential=pot_total, df=df)
            for seed in SEEDS:
                agama.setRandomSeed(int(seed))
                np.random.seed(seed)
                xv, _ = gm.sample(N_PART)
                F, nit = bound_fraction_particles(xv[:, :3], xv[:, 3:])
                results.append([W0, sfe_t, sfe_act, seed, F, nit, N_PART])
                print(f"W0={W0:5.1f} SFE={sfe_act:.4f} seed={seed}: "
                      f"F_b={F:.4f} ({nit} iters)", flush=True)
    res = np.array(results)
    np.savetxt(OUTCSV, res, fmt='%.6f', delimiter=', ',
               header='W0, SFE_target, SFE_actual, seed, F_bound, n_iter, N_part',
               comments='')
    print(f"\nSaved {OUTCSV}")

    # ---------------- plot ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif',
                         'mathtext.fontset': 'dejavuserif',
                         'legend.frameon': False, 'savefig.bbox': 'tight',
                         'savefig.dpi': 300})
    # The continuum frozen-DF curve is COMPUTED HERE rather than read from a
    # CSV: a stale file from an earlier run is impossible to spot in a figure,
    # and the calculation is cheap next to the sampling. It needs only numpy /
    # scipy (no Agama), via the same routine used for the paper's Fig. 10.
    try:
        import sys
        sys.path.append('../04_adams_df_boundfraction')
        import adams_df_boundfraction as _adf
        sys.path.append('../01_king_sfe_analysis')
        from king_sfe_analysis import king_model as _king
        _have_cont = True
    except Exception as e:
        print(f"  continuum curve unavailable ({e}); plotting MC only. Put "
              f"adams_df_boundfraction.py, king_sfe_analysis.py and "
              f"multi_profile_analysis.py on sys.path to enable it.")
        _have_cont = False

    def continuum_curve(W0, sfe_pts):
        """F_b(SFE) from the continuum frozen-DF calculation for this W0."""
        m = _king(float(W0))
        grid = np.unique(np.concatenate([np.asarray(sfe_pts, dtype=float),
                                         np.arange(0.02, 0.401, 0.02)]))
        return grid, np.array([_adf.fbound_at_sfe(m, float(s)) for s in grid])

    # Three F_b(SFE) curves per panel: the continuum frozen-DF calculation,
    # this Monte Carlo run, and (optionally) a second MC run at a different
    # particle number, so the finite-N trend is visible in the same frame.
    def mc_curve(d, W0, npart=None):
        """Mean bound fraction and error bar per SFE for one MC dataset.

        UNCERTAINTIES.  Each row of the CSV is one independent realization
        (one random sampling of N particles from the same DF), so at a given
        (W0, SFE) we have n_real values of F_b.  Two different quantities can
        be quoted, and they answer different questions:

          ERRBAR='std'  sample standard deviation over realizations,
                        s = sqrt( sum (F_i - <F>)^2 / (n-1) ).
                        This is the SPREAD OF PHYSICAL OUTCOMES: how much a
                        single cluster of N stars differs from another with
                        the same initial conditions.  Near threshold the
                        distribution is bimodal (some realizations keep a
                        remnant, others dissolve entirely), so s is large and
                        <F> sits between the two modes -- that is the result,
                        not an error to be reduced.

          ERRBAR='sem'  standard error of the mean, s/sqrt(n_real).
                        This is the PRECISION OF THE ENSEMBLE AVERAGE, i.e.
                        how well <F> is determined; it shrinks as more seeds
                        are run and is the right bar when comparing <F> with
                        the continuum curve.

        The default is 'std' because the physics of interest here is the
        realization-to-realization scatter itself.  Note ddof=1 (unbiased);
        with n_real=1 the bar is undefined and set to zero.
        """
        sub = d[np.abs(d[:, 0] - W0) < 0.01]
        if npart is not None and sub.shape[1] > 6:
            sub = sub[sub[:, 6] == npart]
        sfes = np.unique(np.round(sub[:, 2], 6))
        mean = np.array([sub[np.round(sub[:, 2], 6) == s, 4].mean() for s in sfes])
        n = np.array([np.sum(np.round(sub[:, 2], 6) == s) for s in sfes])
        std = np.array([sub[np.round(sub[:, 2], 6) == s, 4].std(ddof=1)
                        if np.sum(np.round(sub[:, 2], 6) == s) > 1 else 0.0
                        for s in sfes])
        err = std if ERRBAR == 'std' else std / np.sqrt(np.maximum(n, 1))
        return sfes, mean, err

    # The virial criterion predicts a THRESHOLD, not a bound fraction: it says
    # only whether the cluster is globally sub- or super-virial after
    # expulsion, so there is no F_b(SFE) curve to draw for it. It is therefore
    # shown as a vertical marker, while the frozen-DF continuum -- which does
    # predict F_b at every SFE -- is drawn as a curve.
    eta_p = _find_csv('mechanism_summary.csv') if SHOW_VIRIAL else None
    eta = np.loadtxt(eta_p, delimiter=',', skiprows=1, ndmin=2) if eta_p else None


    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8), sharey=True)
    for ax, W0 in zip(axes.flat, W0_LIST):
        if _have_cont:
            sg, fg = continuum_curve(W0, np.unique(res[np.abs(res[:, 0] - W0)
                                                       < 0.01, 2]))
            ax.plot(sg, fg, color='0.2', lw=1.6, label='frozen DF, continuum')
        if eta is not None:
            ax.axvline(np.interp(W0, eta[:, 0], eta[:, 1]), color='0.55',
                       ls='-.', lw=1.2,
                       label=r'virial threshold ($e$SFE$_{\rm c}=1/3$)')
        s, m, e = mc_curve(res, W0, N_PART)
        ax.errorbar(s, m, yerr=e, fmt='o-', color='#b2182b', ms=4, lw=1.1,
                    capsize=2, label=fr'MC, $N={N_PART:g}$, {len(SEEDS)} real. '
                                     fr'(mean $\pm$ {ERRBAR})')
        ax.set(title=fr'$W_0={W0:g}$', xlabel='global SFE', xlim=(0, 0.4),
               ylim=(-0.03, 1.03))
        if ax in axes[:, 0]:
            ax.set_ylabel(r'bound fraction $F_{\rm b}$')
    axes[0, 0].legend(fontsize=7.5, loc='upper left')
    fig.tight_layout()
    fig.savefig('mc_vs_semianalytic.pdf')
    fig.savefig('mc_vs_semianalytic.png')
    print("Saved mc_vs_semianalytic.pdf/png")


if __name__ == '__main__':
    main()