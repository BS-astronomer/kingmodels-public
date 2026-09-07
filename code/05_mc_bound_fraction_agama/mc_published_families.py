#!/usr/bin/env python3
"""
mc_published_families.py
========================
Monte Carlo (frozen-DF) critical SFE for the PUBLISHED profile families --
Plummer and Dehnen(gamma = 0, 1, 2) -- so that Fig. 11b can show the same
three estimates (virial / continuum DF / Monte Carlo) for them as for the
King family.

It is the direct analogue of mc_bound_fraction_agama.py: same PP13 gas
inversion, same Agama machinery (multipole gas potential + QuasiSpherical DF
in the total potential), same particle-wise iterated escape criterion. Only
the stellar profile and the SFE convention differ.

CONVENTIONS -- these matter, and are chosen to match the semi-analytic code
--------------------------------------------------------------------------
* Stellar models: scaleRadius = 1, mass = 1, exactly as in
  multi_profile_analysis.plummer_model() / dehnen_model().
* SFE is measured WITHIN 10 SCALE RADII (the 'ap10' convention), because the
  Dehnen gas mass diverges (rho_g ~ rho_s^(2/3) ~ r^(-8/3) => M_g ~ r^(1/3))
  and because that is the convention under which the published thresholds and
  our continuum DF thresholds are quoted.  Note this is NOT the SFE_J
  convention of the IC generators (SFE inside R_J = 20 r_h); the two differ,
  see the SFE_J -> SFE_10 conversion discussed in the paper.
* The gas is truncated at RMAX_GAS. For an infinite-extent profile some outer
  cut is unavoidable; RMAX_GAS is therefore a real (small) systematic, and the
  script prints the enclosed gas mass at the cut so its size is visible.

Output: mc_thresholds_published.csv with columns
    family_index, SFE_crit_MC, err
with family_index 0..3 = Plummer, Dehnen g=0, g=1, g=2 -- the order and format
that adams_df_boundfraction.py expects when drawing Fig. 11b.

Runtime: roughly 20-60 min for the default settings on a laptop.
"""

import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
import agama
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import sys 
sys.path.append('../04_adams_df_boundfraction')
from mc_bound_fraction_agama import gas_density, bound_fraction_particles

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
N_PART   = 10_000        # particles per realization
N_REAL   = 9             # realizations per SFE (odd -> unambiguous majority)
F_MIN    = 0.02          # bound-fraction level defining survival
N_BISECT = 7
EPS_FF   = 0.01   # only t_sf*eps_ff enters; committed CSVs are eps_ff-invariant
RMIN     = 1e-3
RMAX_GAS = 1e3           # outer gas cut (see docstring); 1e2 for Plummer is
                         # what the IC generator uses -- change with care
APERTURE = 10.0          # SFE measured within 10 scale radii
OUT      = 'mc_thresholds_published.csv'

# ----------------------------------------------------------------------
# PUBLISHED N-BODY RESULTS (bound mass fraction after violent relaxation),
# transcribed from the author's Fb-SFE comparison script.  SFE is quoted in
# the 10-scale-radii convention -- SFE_P for Plummer, SFE_D for Dehnen --
# which is the convention this script uses throughout (see docstring).
#   columns: SFE_10, F_b, F_b error (0 where none was quoted)
# ----------------------------------------------------------------------
NBODY = {
    # Plummer with kick (Shukirgaliyev et al., newer set): x = SFE within 10 a
    'Plummer': np.array([
        [0.13, 0.0150584, 0.0209260],
        [0.15, 0.0540905, 0.0514172],
        [0.17, 0.1993974, 0.0327109],
        [0.20, 0.3339593, 0.0283606],
        [0.25, 0.5376716, 0.0241054]]),
    # Plummer, S17-2019 set (no kick): SFE_10 from the same tabulation
    'Plummer (S17)': np.array([
        [0.10, 0.0039845, 0.0],
        [0.13, 0.0177022, 0.0],
        [0.15, 0.0574038, 0.0360855],
        [0.17, 0.2304037, 0.0308382],
        [0.20, 0.3558248, 0.0431699],
        [0.25, 0.5007739, 0.0272825],
        [0.30, 0.6407412, 0.0],
        [0.35, 0.7152976, 0.0]]),
    # Dehnen gamma=0: SFE_10 = SFE_D column of the same tabulation
    'Dehnen g=0': np.array([
        [0.0920, 0.0135272, 0.0109633],
        [0.1498, 0.1421041, 0.0300854],
        [0.2049, 0.2866261, 0.0113079],
        [0.2824, 0.4290741, 0.0146770],
        [0.3983, 0.5629996, 0.0192096],
        [0.4988, 0.6446777, 0.0098058],
        [0.6569, 0.7231894, 0.0087164]]),
}

FAMILIES = [('Plummer', 'plummer', None),
            ('Dehnen g=0', 'dehnen', 0.0),
            ('Dehnen g=1', 'dehnen', 1.0),
            ('Dehnen g=2', 'dehnen', 2.0)]


# ----------------------------------------------------------------------
def stellar_potential(kind, gamma):
    if kind == 'plummer':
        return agama.Potential(type='plummer', scaleRadius=1.0, mass=1.0)
    # agama's Dehnen: gamma = 2 is allowed but the cusp is singular; nudge it
    g = 1.99999 if abs(gamma - 2.0) < 1e-9 else gamma
    return agama.Potential(type='dehnen', gamma=g, scaleRadius=1.0, mass=1.0)


def profile_arrays(pot, n=3000):
    r = np.logspace(np.log10(RMIN), np.log10(RMAX_GAS), n)
    xyz = np.zeros((n, 3)); xyz[:, 0] = r
    return r, pot.density(xyz)


def sfe_ap10(r, rho_s, rho_g):
    """SFE within 10 scale radii (both components integrated to the aperture)."""
    Ms = 4 * np.pi * np.trapezoid((rho_s * r**2)[r <= APERTURE], r[r <= APERTURE])
    Mg = 4 * np.pi * np.trapezoid((rho_g * r**2)[r <= APERTURE], r[r <= APERTURE])
    return Ms / (Ms + Mg)


def tsf_for_sfe(pot, target):
    r, rho_s = profile_arrays(pot)

    def f(lt):
        return sfe_ap10(r, rho_s, gas_density(rho_s, 10.0**lt, EPS_FF)) - target
    lo, hi = -3.0, 6.0
    for _ in range(6):                       # widen if not bracketed
        if f(lo) * f(hi) <= 0:
            break
        lo -= 2.0; hi += 2.0
    return 10.0**brentq(f, lo, hi, xtol=1e-8)


def build_model(kind, gamma, tsf, verbose=False):
    pot_s = stellar_potential(kind, gamma)
    r, rho_s = profile_arrays(pot_s)
    rho_g = gas_density(rho_s, tsf, EPS_FF)

    gi = interp1d(r, rho_g, kind='linear', bounds_error=False,
                  fill_value=(rho_g[0], 0.0))

    def gasdens(xyz):
        xyz = np.asarray(xyz)
        rr = np.linalg.norm(xyz, axis=1) if xyz.ndim == 2 else np.linalg.norm(xyz)
        return gi(rr)

    # Multipole fitted over exactly the region where gas is defined (the same
    # lesson as the King case: a fit spanning decades of zero density injects
    # DF-construction jitter that flips marginal configurations).
    pot_g = agama.Potential(type='multipole', symmetry='spherical',
                            density=gasdens, rmin=RMIN, rmax=RMAX_GAS,
                            gridSizeR=60)
    pot_tot = agama.Potential(pot_g, pot_s)
    df = agama.DistributionFunction(type='QuasiSpherical',
                                    potential=pot_tot, density=pot_s)
    if verbose:
        Mg_tot = 4 * np.pi * np.trapezoid(rho_g * r**2, r)
        Mg_ap = 4 * np.pi * np.trapezoid((rho_g * r**2)[r <= APERTURE],
                                         r[r <= APERTURE])
        print(f"      gas mass: {Mg_ap:.4f} within {APERTURE:g}a, "
              f"{Mg_tot:.4f} out to RMAX_GAS={RMAX_GAS:g} "
              f"(outer cut holds {100*(1-Mg_ap/Mg_tot):.0f}% of it)")
    return agama.GalaxyModel(potential=pot_tot, df=df)


def survival_probability(kind, gamma, sfe, seed0=1000):
    tsf = tsf_for_sfe(stellar_potential(kind, gamma), sfe)
    gm = build_model(kind, gamma, tsf)
    n_surv = 0
    for i in range(N_REAL):
        agama.setRandomSeed(seed0 + i)
        xv, _ = gm.sample(N_PART)
        F, _ = bound_fraction_particles(xv[:, :3], xv[:, 3:])
        n_surv += (F > F_MIN)
    return n_surv / N_REAL


def mc_threshold(kind, gamma, lo=0.005, hi=0.60, n_scan=8, max_extend=3):
    """Threshold where the survival probability crosses 1/2.

    The scan window is extended DOWNWARD when the lowest SFE already survives
    in the majority of realizations -- which happens for the cuspy Dehnen
    models, whose thresholds can lie below the default floor.  If the window
    cannot be bracketed even then, the lowest scanned SFE is returned as an
    UPPER LIMIT (flagged by err < 0) rather than NaN, so the point still
    appears in Fig. 11b instead of silently vanishing.
    """
    for _ in range(max_extend + 1):
        scan = np.geomspace(lo, hi, n_scan)
        ps = [survival_probability(kind, gamma, s) for s in scan]
        a = b = None
        for i in range(1, n_scan):
            if ps[i - 1] < 0.5 <= ps[i]:
                a, b = scan[i - 1], scan[i]
        if a is not None:
            break
        if ps[0] >= 0.5:                  # already surviving at the floor
            print(f"      survival >= 1/2 already at SFE={lo:.4f}; "
                  f"extending the scan downward")
            lo /= 10.0
            continue
        break                              # never survives: no threshold below hi
    if a is None:
        if ps[0] >= 0.5:
            print(f"      no crossing found; reporting SFE < {scan[0]:.4f} "
                  f"as an upper limit")
            return float(scan[0]), -1.0
        return np.nan, np.nan
    for _ in range(N_BISECT):
        mid = np.sqrt(a * b)
        if survival_probability(kind, gamma, mid) < 0.5:
            a = mid
        else:
            b = mid
    return np.sqrt(a * b), (b - a) / 2.0


def fbound_curve_mc(kind, gamma, sfe_values, seed0=1000):
    """Mean and spread of F_b over realizations at each requested SFE."""
    out = []
    for s in sfe_values:
        tsf = tsf_for_sfe(stellar_potential(kind, gamma), float(s))
        gm = build_model(kind, gamma, tsf)
        vals = []
        for i in range(N_REAL):
            agama.setRandomSeed(seed0 + i)
            xv, _ = gm.sample(N_PART)
            F, _ = bound_fraction_particles(xv[:, :3], xv[:, 3:])
            vals.append(F)
        vals = np.array(vals)
        out.append([s, vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0])
        print(f"      SFE={s:.4f}: F_b(MC) = {out[-1][1]:.4f} +- {out[-1][2]:.4f}",
              flush=True)
    return np.array(out)


def compare_with_nbody():
    """MC (and continuum, if importable) vs the published N-body bound
    fractions at the SAME SFE values -- the direct measurement of the
    frozen-DF bias for profiles where simulations exist."""
    try:
        import adams_df_boundfraction as _adf
        from multi_profile_analysis import plummer_model, dehnen_model
        cont = {'Plummer': plummer_model(), 'Dehnen g=0': dehnen_model(0.0)}
    except Exception as e:
        print(f"(continuum unavailable: {e})")
        cont = {}
    rows, store = [], {}
    for label, kind, gamma in [('Plummer', 'plummer', None),
                               ('Dehnen g=0', 'dehnen', 0.0)]:
        nb = NBODY[label]
        print(f"\n{label}: MC vs published N-body")
        mc = fbound_curve_mc(kind, gamma, nb[:, 0])
        store[label] = mc
        print(f"  {'SFE_10':>8}{'F_b(MC)':>10}{'F_b(N-body)':>13}"
              f"{'MC - Nbody':>12}{'F_b(cont)':>11}")
        for (s, fm, em), (_, fn, en) in zip(mc, nb):
            fc = (_adf.fbound_at_sfe(cont[label], float(s), convention='ap10')
                  if label in cont else np.nan)
            print(f"  {s:8.4f}{fm:10.4f}{fn:13.4f}{fm - fn:+12.4f}{fc:11.4f}")
            rows.append([s, fm, em, fn, en, fc])
    if rows:
        np.savetxt('mc_vs_nbody_published.csv', np.array(rows), fmt='%.6f',
                   delimiter=', ',
                   header='SFE_10, Fb_MC, Fb_MC_std, Fb_Nbody, Fb_Nbody_err, '
                          'Fb_continuum', comments='')
        print("\nWrote mc_vs_nbody_published.csv")
    _plot_comparison(store, cont)


def _plot_comparison(store, cont):
    """F_b(SFE) for the published families: continuum, Monte Carlo and the
    published N-body measurements on the same axes, so the bias of the
    frozen-DF estimate is read directly rather than inferred."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif',
                         'mathtext.fontset': 'dejavuserif',
                         'axes.linewidth': 0.8, 'xtick.direction': 'in',
                         'ytick.direction': 'in', 'xtick.top': True,
                         'ytick.right': True, 'legend.frameon': False,
                         'savefig.bbox': 'tight', 'savefig.dpi': 300})
    try:
        import adams_df_boundfraction as _adf
    except Exception:
        _adf = None

    labels = ['Plummer', 'Dehnen g=0']
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, label in zip(axes, labels):
        # continuum curve on a dense grid
        if _adf is not None and label in cont:
            grid = np.linspace(0.02, 0.70, 35)
            fc = [_adf.fbound_at_sfe(cont[label], float(s), convention='ap10')
                  for s in grid]
            ax.plot(grid, fc, color='0.2', lw=1.6, label='frozen DF, continuum')
        # Monte Carlo
        if label in store:
            mc = store[label]
            ax.errorbar(mc[:, 0], mc[:, 1], yerr=mc[:, 2], fmt='o-',
                        color='#b2182b', ms=4, lw=1.1, capsize=2,
                        label=fr'frozen DF, MC ($N={N_PART:g}$)')
        # published N-body
        nb = NBODY[label]
        ax.errorbar(nb[:, 0], nb[:, 1], yerr=nb[:, 2], fmt='s', color='#2166ac',
                    ms=5, mfc='none', capsize=2, lw=1.0,
                    label=r'$N$-body (published)')
        if label == 'Plummer' and 'Plummer (S17)' in NBODY:
            nb2 = NBODY['Plummer (S17)']
            ax.errorbar(nb2[:, 0], nb2[:, 1], yerr=nb2[:, 2], fmt='x',
                        color='#2166ac', ms=5, capsize=2, lw=1.0, alpha=0.7,
                        label=r'$N$-body (S17 set)')
        ax.set(title=label.replace('g=0', r'$\gamma=0$'), xlabel='global SFE'
               r' (within $10\,a$)', xlim=(0, 0.7), ylim=(-0.03, 1.03))
    axes[0].set_ylabel(r'bound fraction $F_{\rm b}$')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig('fbound_published_families.pdf')
    fig.savefig('fbound_published_families.png')
    plt.close(fig)
    print("Wrote fbound_published_families.pdf/png")


if __name__ == '__main__':
    import sys
    if '--compare' in sys.argv:      # F_b curves at the N-body SFE values
        compare_with_nbody()
        sys.exit(0)
    rows = []
    for idx, (label, kind, gamma) in enumerate(FAMILIES):
        print(f"[{idx}] {label}")
        # one verbose build so the gas-truncation systematic is visible
        build_model(kind, gamma, tsf_for_sfe(stellar_potential(kind, gamma), 0.15),
                    verbose=True)
        sc, err = mc_threshold(kind, gamma)
        rows.append([idx, sc, err])
        print(f"      SFE_crit(MC, N={N_PART}) = {sc:.4f} +- {err:.4f}", flush=True)
    np.savetxt(OUT, np.array(rows), fmt='%.6f', delimiter=', ',
               header='family_index, SFE_crit_MC, err', comments='')
    print(f"\nWrote {OUT} -- adams_df_boundfraction.py will pick it up "
          f"automatically for Fig. 11b.")