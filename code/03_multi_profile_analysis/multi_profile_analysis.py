#!/usr/bin/env python3
"""
multi_profile_analysis.py
=========================
Three extensions to king_sfe_analysis.py:

(1) VALIDATION: Plummer and Dehnen(gamma) profiles under the published
    conventions (Shukirgaliyev et al. 2017: total SFE; 2021: SFE within 10
    scale radii). Computes the global eSFE at the published survival
    thresholds and compares with the values quoted from those papers.

(2) UNIVERSALITY TEST: cumulative eSFE(<r) profiles at each family's
    measured threshold, plotted against enclosed stellar mass fraction.
    Tests whether survival is governed by a LOCAL criterion (some inner
    region keeping eSFE above a universal value) rather than the global
    virial ratio, which the published thresholds show is NOT universal
    (0.32 for Plummer vs 0.06-0.16 for Dehnen).

(3) t_ff TRUNCATION: gas is removed where its local free-fall time exceeds
    t_SF (no time to participate in star formation). In model units G=1:
        t_ff = sqrt(3 pi / (32 rho))  =>  rho_SF = 3 pi / (32 t_SF^2).
    Since the central gas plateau is rho_plateau = 1/k^2 = 3 pi/(8 eps_ff^2
    t_SF^2), the truncation happens at a FIXED density contrast
        rho_SF / rho_plateau = eps_ff^2 / 4,
    independent of t_SF and of all scales (only eps_ff enters). This gives
    Dehnen models a finite, physically defined gas mass (their total gas
    mass otherwise diverges as r^{1/3}), and shifts the King critical
    curve, which is recomputed here with and without truncation.

Units: G = M_star = a = 1 (a = scale radius: King r0 / Plummer a / Dehnen a).
"""

import numpy as np
if not hasattr(np, 'trapezoid'):          # numpy < 2.0 compatibility
    np.trapezoid = np.trapz
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from king_sfe_analysis import king_model, gas_density, EPS_FF

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
})
OUTDIR = './'

# Published survival thresholds (gamma>=1 values are the
# LOWEST SIMULATED SFE at which clusters still survived -> upper limits).
THRESHOLDS = [
    # label,        kind,      param, convention, SFE_threshold, quoted eSFE, censored
    ('Plummer',     'plummer', None,  'ap10',     0.15,  0.32, False),  # 2017 SFE was aperture-limited (cf. 2021 Sec.2)
    (r'Dehnen $\gamma=0$', 'dehnen', 0.0, 'ap10', 0.090, 0.16, False),
    # gamma>=1 SFE_10 values CONVERTED from the published SFE_J = 0.01
    # (S21: lambda=0.05 -> R_J = 20 r_h) via the profile model; see paper Sect. 3.
    (r'Dehnen $\gamma=1$', 'dehnen', 1.0, 'ap10', 0.0241, 0.06, True),
    (r'Dehnen $\gamma=2$', 'dehnen', 2.0, 'ap10', 0.0148, 0.08, True),
]
TSF_GRID = np.logspace(-1.5, 4.5, 160)
ESFE_MARK = 1.0 / 3.0  # open squares mark where each curve reaches this eSFE


# ----------------------------------------------------------------------
# Analytic profiles (M_tot = 1, a = 1)
# ----------------------------------------------------------------------
def plummer_model(rmin=1e-5, rmax=1e4, n=6000):
    x = np.logspace(np.log10(rmin), np.log10(rmax), n)
    rho = 3.0 / (4.0 * np.pi) * (1.0 + x**2)**(-2.5)
    M = x**3 * (1.0 + x**2)**(-1.5)
    rh = brentq(lambda r: r**3 * (1 + r**2)**(-1.5) - 0.5, 0.1, 10)
    # analytic d(rho)/dr, d^2(rho)/dr^2 (sympy-verified); see king_model's
    # rho1/rho2 docstring for why adams_df_boundfraction.eddington_f_analytic
    # wants these instead of differencing the rho array.
    rho1 = -15.0 / (4.0 * np.pi) * x * (1.0 + x**2)**(-3.5)
    rho2 = 15.0 / (4.0 * np.pi) * (1.0 + x**2)**(-4.5) * (6.0 * x**2 - 1.0)
    return dict(x=x, rho=rho, M=M, rh=rh, rt=np.inf, label='Plummer',
                rho1=rho1, rho2=rho2)


def dehnen_model(gamma, rmin=1e-6, rmax=1e4, n=6000):
    if gamma >= 2.5:
        raise ValueError("stellar self-energy diverges for gamma >= 2.5")
    x = np.logspace(np.log10(rmin), np.log10(rmax), n)
    rho = (3.0 - gamma) / (4.0 * np.pi) * x**(-gamma) * (1.0 + x)**(gamma - 4.0)
    M = (x / (1.0 + x))**(3.0 - gamma)
    rh = 1.0 / (2.0**(1.0 / (3.0 - gamma)) - 1.0)
    # rho = rho * g(x), g = -gamma/x + (gamma-4)/(1+x); rho' = rho*g,
    # rho'' = rho*(g^2 + g'), g' = gamma/x^2 - (gamma-4)/(1+x)^2 (sympy-verified)
    g = -gamma / x + (gamma - 4.0) / (1.0 + x)
    gp = gamma / x**2 - (gamma - 4.0) / (1.0 + x)**2
    rho1 = rho * g
    rho2 = rho * (g**2 + gp)
    return dict(x=x, rho=rho, M=M, rh=rh, rt=np.inf,
                label=fr'Dehnen $\gamma={gamma:g}$', rho1=rho1, rho2=rho2)


# ----------------------------------------------------------------------
# Generic analysis with optional t_ff truncation
# ----------------------------------------------------------------------
def rho_sf(tsf):
    """Density at which t_ff(rho) = t_SF, in G=1 units."""
    return 3.0 * np.pi / (32.0 * tsf**2)


def analyze_generic(model, tsf, truncate=False):
    x, rho_s, M_s = model['x'], model['rho'], model['M']
    rho_g = gas_density(rho_s, tsf)
    r_trunc = np.inf
    if truncate:
        thr = rho_sf(tsf)
        below = np.where(rho_g < thr)[0]
        # outer truncation only: first index beyond which gas stays sub-critical
        if len(below) > 0:
            i0 = below[0]
            rho_g = rho_g.copy()
            rho_g[i0:] = 0.0
            r_trunc = x[i0]
    M_g = 4.0 * np.pi * cumulative_trapezoid(rho_g * x**2, x, initial=0.0)

    dWss = 4.0 * np.pi * rho_s * M_s * x
    dWsg = 4.0 * np.pi * rho_s * M_g * x
    cWss = cumulative_trapezoid(dWss, x, initial=0.0)
    cWsg = cumulative_trapezoid(dWsg, x, initial=0.0)
    esfe_cum = np.where(cWss + cWsg > 0, cWss / np.maximum(cWss + cWsg, 1e-300), 1.0)
    esfe = cWss[-1] / (cWss[-1] + cWsg[-1])

    Ms10 = np.interp(10.0, x, M_s)
    Mg10 = np.interp(10.0, x, M_g)
    return dict(tsf=tsf, rho_g=rho_g, M_g=M_g, r_trunc=r_trunc,
                valid=(M_g[-1] > 0.0),
                sfe_total=1.0 / (1.0 + M_g[-1]),
                sfe_ap10=Ms10 / (Ms10 + Mg10),
                esfe=esfe, esfe_cum=esfe_cum,
                cWss=cWss, cWsg=cWsg)


def sfe_of(res, convention):
    return res['sfe_total'] if convention == 'total' else res['sfe_ap10']


def _bracket_and_solve(fun, target, lo=-2.0, hi=7.5, n=130, max_expand=5):
    """Root of fun(log10 tsf) = target on the physical (rising) branch.
    fun may return NaN for invalid configurations (e.g. fully truncated gas);
    under t_ff truncation eSFE(tsf) can be U-shaped, so we take the LAST
    increasing crossing, which continuously matches the untruncated case.

    The default window is widened automatically when the target lies outside
    it. This is needed because the t_sf required for a given SFE scales with
    the model: in r_0 units it grows as (r_t/r_0)^(3/2), so concentrated King
    models (x_t ~ 3e4 at W0=20) need t_sf well above 10^7.5 to reach high SFE,
    while diffuse ones (x_t ~ 1.3 at W0=0.5) need t_sf below 10^-2 to reach
    SFE ~ 0.004."""
    for attempt in range(max_expand + 1):
        lg = np.linspace(lo, hi, n)
        vals = np.array([fun(l) for l in lg])
        ok = np.isfinite(vals)
        bracket = None
        for i in range(1, n):
            if ok[i - 1] and ok[i] and (vals[i - 1] < target <= vals[i]):
                bracket = (lg[i - 1], lg[i])
        if bracket is not None:
            return brentq(lambda l: fun(l) - target, *bracket, xtol=1e-6)
        if attempt == max_expand:
            break
        finite = vals[ok]
        # widen towards whichever side the target is missing from (or both)
        if finite.size == 0 or target < finite.min():
            lo -= 2.0
        if finite.size == 0 or target > finite.max():
            hi += 2.0
    raise RuntimeError(
        f"target {target} not bracketed for log10(t_sf) in [{lo}, {hi}] "
        f"after {max_expand} expansions (reachable range: "
        f"{np.nanmin(vals):.4g}..{np.nanmax(vals):.4g})")


def tsf_for_threshold(model, sfe_target, convention, truncate=False, **_):
    def f(l):
        r = analyze_generic(model, 10.0**l, truncate)
        return sfe_of(r, convention) if r['valid'] else np.nan
    return 10.0**_bracket_and_solve(f, sfe_target)


def critical_sfe_generic(model, esfe_crit, convention, truncate=False, **_):
    def f(l):
        r = analyze_generic(model, 10.0**l, truncate)
        return r['esfe'] if r['valid'] else np.nan
    l_crit = _bracket_and_solve(f, esfe_crit)
    return sfe_of(analyze_generic(model, 10.0**l_crit, truncate), convention)


# ----------------------------------------------------------------------
# Part 1: validation against published thresholds
# ----------------------------------------------------------------------
def part1_validation():
    print("=" * 72)
    print("PART 1: eSFE at published survival thresholds (no truncation)")
    print("=" * 72)
    print(f"{'model':<22}{'convention':<11}{'SFE_thr':>8}{'eSFE calc':>11}"
          f"{'eSFE quoted':>13}{'note':>10}")
    results = []
    for label, kind, par, conv, sfe_thr, esfe_quoted, cens in THRESHOLDS:
        model = plummer_model() if kind == 'plummer' else dehnen_model(par)
        tsf = tsf_for_threshold(model, sfe_thr, conv)
        res = analyze_generic(model, tsf)
        note = 'UPPER LIM' if cens else ''
        print(f"{label.replace('$',''):<22}{conv:<11}{sfe_thr:>8.3f}"
              f"{res['esfe']:>11.4f}{esfe_quoted:>13.2f}{note:>10}")
        results.append((label, model, tsf, res, sfe_thr, esfe_quoted, cens))
    return results



# ----------------------------------------------------------------------
# Part 1b: eSFE(SFE) across profile families -- the non-universality picture
# ----------------------------------------------------------------------
def part1b_cross_profile(threshold_results):
    """eSFE vs global SFE for Plummer, Dehnen(0,1,2) and King(3,6,9), with the
    measured/predicted survival thresholds marked.

    CONVENTIONS (unavoidable, stated explicitly): the Dehnen and Plummer gas
    masses diverge, so their SFE is measured within 10 scale radii, exactly as
    in the papers that report their thresholds. King models are truncated, so
    their SFE is the true global one. For W0=9 the 10a aperture lies INSIDE
    r_t, so the two conventions differ there; the King curves are therefore
    shown in their own (global) convention and this is noted in the caption.
    """
    print("\n" + "=" * 72)
    print("PART 1b: eSFE(SFE) across profile families")
    print("=" * 72)

    # (label, model, convention, colour, linestyle, thr_sfe, censored, mark)
    # King models keep the green/blue/red of Figs 1-2 for W0 = 3/6/9 and are
    # drawn SOLID; the published families use a separate palette and are DASHED,
    # so colour never carries two meanings across the paper. Only the published
    # families get threshold markers -- the King crossings at eSFE = 1/3 are
    # predictions by construction, not measurements, so marking them would
    # imply a measured threshold that does not exist.
    entries = []
    pal_pub = {'Plummer': '#000000', 'D0': '#e08214', 'D1': '#807dba', 'D2': '#8c510a'}
    pal_king = {3.0: '#1b7837', 6.0: '#2166ac', 9.0: '#b2182b'}
    entries.append(('Plummer', plummer_model(), 'ap10', pal_pub['Plummer'],
                    '--', 0.150, False, True))
    for g, key, thr, cens in [(0.0, 'D0', 0.090, False), (1.0, 'D1', 0.0241, True),
                              (2.0, 'D2', 0.0148, True)]:
        entries.append((fr'Dehnen $\gamma={g:g}$', dehnen_model(g), 'ap10',
                        pal_pub[key], '--', thr, cens, True))
    for w in (3.0, 6.0, 9.0):
        m = king_model(w)
        sc = critical_sfe_generic(m, 1.0 / 3.0, 'total')
        entries.append((fr'King $W_0={w:g}$', m, 'total', pal_king[w],
                        '-', sc, False, False))

    # two stacked panels sharing the global-SFE axis: eSFE (top) and the
    # boost factor eSFE/SFE (bottom, = 1/(SFE + eta(1-SFE)), -> 1/eta as
    # SFE -> 0, so the intercept reads off the gas harmfulness directly)
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(4.6, 5.6), sharex=True,
                                  gridspec_kw={'height_ratios': [1.0, 0.62]})
    ax.plot([0, 1], [0, 1], color='0.6', ls=':', lw=1.2, zorder=0)
    ax.axhline(1.0 / 3.0, color='0.35', ls='--', lw=1.0, zorder=0)
    ax.text(0.40, 1.0 / 3.0 + 0.02, r'$e{\rm SFE}=1/3$', fontsize=7.5, color='0.35')

    sfe_at_mark = {}
    print(f"{'model':<20}{'eSFE at SFE=0.10':>18}{'0.17':>8}{'0.30':>8}"
          f"{'  thr SFE':>10}{'eSFE(thr)':>11}")
    for label, model, conv, col, ls, thr, cens, mark in entries:
        # sweep in the model's own time units; for King convert from clump units
        tgrid = (TSF_GRID if np.isinf(model['rt'])
                 else np.logspace(-3.5, 3.5, 220) * model['rt']**1.5)
        s, e = [], []
        for t in tgrid:
            r = analyze_generic(model, t)
            if r['valid']:
                s.append(sfe_of(r, conv)); e.append(r['esfe'])
        s, e = np.array(s), np.array(e)
        o = np.argsort(s); s, e = s[o], e[o]
        keep = (s > 0.002) & (s < 0.997)
        ax.plot(s[keep], e[keep], color=col, lw=1.6, ls=ls, label=label)
        with np.errstate(divide='ignore', invalid='ignore'):
            boost = np.where(s[keep] > 0, e[keep] / s[keep], np.nan)
        axb.plot(s[keep], boost, color=col, lw=1.6, ls=ls)
        e_thr = float(np.interp(thr, s, e))
        if mark:                      # measured thresholds only
            ax.plot(thr, e_thr, 'o' if not cens else 'v', color=col, ms=6,
                    mec='0.2', mew=0.7, zorder=5)
        # open square: where each curve reaches eSFE = ESFE_MARK
        s_mark = float(np.interp(ESFE_MARK, e, s))
        sfe_at_mark[label] = s_mark
        vals = [float(np.interp(q, s, e)) for q in (0.10, 0.17, 0.30)]
        print(f"{label.replace('$','').replace(chr(92)+'gamma','g'):<20}"
              + ''.join(f"{v:>18.3f}" if i == 0 else f"{v:>8.3f}"
                        for i, v in enumerate(vals))
              + f"{thr:>10.4f}{e_thr:>11.3f}" + ("  (upper lim)" if cens else ""))

    ax.set(xlim=(0, 0.5), ylim=(0, 1), ylabel=r'$e$SFE')
    ax.legend(fontsize=7.5, loc='lower right', ncol=2)
    ax.text(0.02, 0.92, '(a)', transform=ax.transAxes, fontsize=10)
    axb.axhline(1.0, color='0.6', ls=':', lw=1.2)
    axb.text(0.30, 1.10, 'identical profiles', fontsize=7.5, color='0.45')
    axb.set(xlim=(0, 0.5), ylim=(0.8, 6.4),
            xlabel='SFE',
            ylabel=r'boost $e$SFE$\,/\,$SFE')
    axb.text(0.02, 0.90, '(b)', transform=axb.transAxes, fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)      # panels flush: shared x-axis
    fig.savefig(OUTDIR + 'fig8b_esfe_cross_profile.pdf')
    fig.savefig(OUTDIR + 'fig8b_esfe_cross_profile.png')
    plt.close(fig)
    print("Fig 8b done  (circles: measured thresholds; triangles: upper limits)")

    # ---- companion view: the BOOST FACTOR eSFE/SFE ----------------------
    # Since eSFE = SFE/(SFE + eta (1-SFE)), the ratio is
    #     eSFE/SFE = 1/(SFE + eta (1-SFE)),
    # i.e. a boost factor that is exactly 1 for gas tracing stars (identical
    # profiles) and tends to 1/eta as SFE -> 0. The y-intercept therefore reads
    # off the gas-harmfulness parameter of Sect. 4 directly, and the curves
    # separate in the low-SFE regime that matters for survival -- which the
    # eSFE-vs-SFE view compresses into the top-left corner.
    def _boost_panel(logx):
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        rows = []
        for label, model, conv, col, ls, thr, cens, mark in entries:
            tgrid = (TSF_GRID if np.isinf(model['rt'])
                     else np.logspace(-3.5, 3.5, 220) * model['rt']**1.5)
            s, e = [], []
            for t in tgrid:
                r = analyze_generic(model, t)
                if r['valid']:
                    s.append(sfe_of(r, conv)); e.append(r['esfe'])
            s, e = np.array(s), np.array(e)
            o = np.argsort(s); s, e = s[o], e[o]
            keep = (s > 0.003) & (s < 0.997)
            ax.plot(s[keep], (e / s)[keep], color=col, lw=1.6, ls=ls, label=label)
            if mark:
                ax.plot(thr, float(np.interp(thr, s, e)) / thr,
                        'o' if not cens else 'v', color=col, ms=6, mec='0.2',
                        mew=0.7, zorder=5)
            s_mark = float(np.interp(ESFE_MARK, e, s))
            ax.plot(s_mark, ESFE_MARK / s_mark, 's', mfc='none', mec=col,
                    mew=1.3, ms=6, zorder=6)
            rows.append((label, s, e, s_mark))
        if logx:
            ax.set(xscale='log', xlim=(3e-3, 1))
        else:
            ax.set(xlim=(0, 1))
        ax.set(ylim=(1.0, 6.5), xlim=(0.8e-2, 1), xlabel='global SFE',
               ylabel=r'boost factor $e$SFE$\,/\,$SFE')
        ax.legend(fontsize=7.5, loc='upper right', ncol=2)
        name = 'fig8c_boost_factor' if logx else 'fig8d_boost_factor_linear'
        fig.savefig(OUTDIR + name + '.pdf')
        fig.savefig(OUTDIR + name + '.png')
        plt.close(fig)
        return rows

    rows = _boost_panel(logx=True)
    _boost_panel(logx=False)
    print(f"\n{'model':<20}{'boost at SFE=0.05':>18}{'0.10':>8}{'0.17':>8}"
          f"{'0.30':>8}{'1/eta':>8}{f'  SFE at eSFE={ESFE_MARK:.3f}':>20}")
    for label, s, e, s_mark in rows:
        vals = [float(np.interp(q, s, e)) / q for q in (0.05, 0.10, 0.17, 0.30)]
        keep = (s > 0.003) & (s < 0.997)
        print(f"{label.replace('$','').replace(chr(92)+'gamma','g'):<20}"
              + ''.join(f"{v:>18.2f}" if i == 0 else f"{v:>8.2f}"
                        for i, v in enumerate(vals))
              + f"{float((e / s)[keep][0]):>8.2f}{s_mark:>20.4f}")
    print("Figs 8c (log) and 8d (linear) done")


# ----------------------------------------------------------------------
# Part 2: cumulative eSFE universality test
# ----------------------------------------------------------------------
def part2_universality(threshold_results=None):
    """Cumulative eSFE(<r) vs enclosed stellar mass fraction, each published
    family evaluated at its measured survival threshold and the King models
    at their predicted threshold (eSFE_c = 1/3).

    If survival were set by a universal criterion -- cluster-wide, or some
    fixed inner mass fraction held above a threshold -- these curves would
    share a crossing. They do not (Sect. 3.5 / Phase B): the mass fraction
    above eSFE(<r) = 1/3 runs from ~0.005 to ~0.9 across the four published
    models.
    """
    print("\n" + "=" * 72)
    print("PART 2: cumulative eSFE(<r) at each family's threshold")
    print("=" * 72)

    pal_pub = {'Plummer': '#000000', 0.0: '#e08214', 1.0: '#807dba', 2.0: '#8c510a'}
    pal_king = {3.0: '#1b7837', 8.8: '#e7298a'}
    # (label, model, convention, colour, linestyle, target, is_esfe_target, censored)
    families = [('Plummer', plummer_model(), 'ap10', pal_pub['Plummer'], '-',
                 0.150, False, False)]
    for g, thr, cens in [(0.0, 0.090, False), (1.0, 0.0241, True),
                         (2.0, 0.0148, True)]:
        families.append((fr'Dehnen $\gamma={g:g}$', dehnen_model(g), 'ap10',
                         pal_pub[g], '--' if cens else '-', thr, False, cens))
    for w in (3.0, 8.8):
        families.append((fr'King $W_0={w:g}$', king_model(w), 'total',
                         pal_king[w], ':', 1.0 / 3.0, True, False))

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    print(f"{'model':<18}{'global SFE':>11}{'eSFE(glob)':>11}"
          f"{'f(eSFE<r>1/3)':>15}")
    for label, model, conv, col, ls, target, is_esfe, cens in families:
        if is_esfe:
            def f_obj(l):
                r = analyze_generic(model, 10.0**l)
                return r['esfe'] if r['valid'] else np.nan
        else:
            def f_obj(l, _c=conv):
                r = analyze_generic(model, 10.0**l)
                return sfe_of(r, _c) if r['valid'] else np.nan
        l_c = _bracket_and_solve(f_obj, target)
        res = analyze_generic(model, 10.0**l_c)
        f_enc = model['M'] / model['M'][-1]
        ok = f_enc > 1e-4
        sfe_glob = sfe_of(res, conv)
        sfe_tag = fr'$\mathrm{{SFE}}_{{10}}={sfe_glob:.3f}$' if conv == 'ap10' \
            else fr'$\mathrm{{SFE}}={sfe_glob:.3f}$'
        lab = f'{label}, {sfe_tag}' + (' (u.l.)' if cens else '')
        ax.plot(f_enc[ok], res['esfe_cum'][ok], color=col, lw=1.7, ls=ls,
                label=lab)
        above = np.where(res['esfe_cum'][ok] >= 1.0 / 3.0)[0]
        f_core = float(f_enc[ok][above[-1]]) if len(above) else 0.0
        print(f"{label.replace('$','').replace(chr(92)+'gamma','g'):<18}"
              f"{sfe_glob:>11.4f}{res['esfe']:>11.3f}{f_core:>15.3f}")

    ax.axhline(1.0 / 3.0, color='0.5', ls=':', lw=1.2)
    ax.text(0.02, 1.0 / 3.0 - 0.05, r'$e{\rm SFE}(<r)=1/3$', fontsize=8,
            color='0.35')
    ax.set(xlabel=r'enclosed stellar mass fraction $M_\star(<r)/M_\star$',
           ylabel=r'cumulative $e{\rm SFE}(<r)$', xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=7.0, loc='upper right')
    fig.savefig(OUTDIR + 'fig8_cumulative_esfe.pdf')
    fig.savefig(OUTDIR + 'fig8_cumulative_esfe.png')
    plt.close(fig)
    print("Fig 8 done (each family at its threshold)")


# ----------------------------------------------------------------------
# Part 3: t_ff truncation
# ----------------------------------------------------------------------
def _crit_trunc_at_eps(model, eps, target=1.0 / 3.0):
    """Virial SFE threshold under the t_ff truncation at an arbitrary
    efficiency per free-fall time. The untruncated problem is exactly
    independent of eps_ff (the eps_ff-t_sf degeneracy), so this is the only
    place the choice can matter."""
    def esfe_and_sfe(tsf):
        x, rho_s, M_s = model['x'], model['rho'], model['M']
        rho_g = gas_density(rho_s, tsf, eps_ff=eps)
        below = np.where(rho_g < rho_sf(tsf))[0]
        if len(below):
            rho_g = rho_g.copy(); rho_g[below[0]:] = 0.0
        M_g = 4.0 * np.pi * cumulative_trapezoid(rho_g * x**2, x, initial=0.0)
        if M_g[-1] <= 0:
            return np.nan, np.nan
        Wss = np.trapezoid(rho_s * M_s * x, x)
        Wsg = np.trapezoid(rho_s * M_g * x, x)
        return Wss / (Wss + Wsg), 1.0 / (1.0 + M_g[-1])

    grid = np.logspace(-1.5, 4.5, 200)
    vals = [esfe_and_sfe(t) for t in grid]
    e = np.array([v[0] for v in vals]); f = np.array([v[1] for v in vals])
    ok = np.isfinite(e)
    if ok.sum() < 3:
        return np.nan
    e, f = e[ok], f[ok]
    idx = [i for i in range(1, len(e)) if e[i - 1] < target <= e[i]]
    if not idx:
        return np.nan
    i = idx[-1]
    w = (target - e[i - 1]) / max(e[i] - e[i - 1], 1e-12)
    return float(f[i - 1] + w * (f[i] - f[i - 1]))


def part3_truncation():
    print("\n" + "=" * 72)
    print(f"PART 3: t_ff truncation (rho_SF/rho_plateau = eps_ff^2/4 "
          f"= {EPS_FF**2 / 4:.2e})")
    print("=" * 72)

    # (a) effect on the King critical curve
    W0_grid = np.unique(np.concatenate([np.arange(0.5, 20.001, 1.0),
                                        np.arange(7.0, 11.01, 0.4)]))
    rows = []
    c_vals = []          # c = log10(r_t/r_c) per W0, for the secondary top axis
    for w in W0_grid:
        m = king_model(w)
        sc0 = critical_sfe_generic(m, 1.0 / 3.0, 'total', truncate=False)
        sc1 = critical_sfe_generic(m, 1.0 / 3.0, 'total', truncate=True)
        tsf1 = tsf_for_threshold(m, sc1, 'total', truncate=True)
        r1 = analyze_generic(m, tsf1, truncate=True)
        rows.append([w, sc0, sc1, r1['r_trunc'] / m['rt']])
        c_vals.append(m['c'])
        print(f"  W0={w:5.2f}  SFE_crit: no-trunc={sc0:.4f}  trunc={sc1:.4f}  "
              f"r_SF/r_t={r1['r_trunc'] / m['rt']:.3f}")
    rows = np.array(rows)
    c_vals = np.array(c_vals)

    # (b) Dehnen: two flavors of the t_ff criterion.
    # REMOVAL flavor (gas beyond r_SF excluded from the potential): the
    # reachable SFE_10 has a MINIMUM -- the published low-SFE thresholds
    # cannot be realized. APERTURE flavor (full gas kept in the dynamics;
    # r_SF only defines where the clump/SFE is measured): always defined.
    print("\n  Dehnen under the t_ff criterion:")
    tsf_scan = np.logspace(-1.5, 4.0, 90)
    for label, kind, par, conv, sfe_thr, esfe_q, cens in THRESHOLDS[1:]:
        model = dehnen_model(par)
        # removal flavor: minimum reachable SFE_10
        s10 = []
        for t in tsf_scan:
            r = analyze_generic(model, t, truncate=True)
            if r['valid']:
                s10.append(r['sfe_ap10'])
        s10_min = min(s10)
        # aperture flavor at the published threshold (untruncated dynamics)
        tsf = tsf_for_threshold(model, sfe_thr, conv, truncate=False)
        r = analyze_generic(model, tsf, truncate=False)
        thr_rho = rho_sf(tsf)
        above = np.where(r['rho_g'] >= thr_rho)[0]
        r_SF = model['x'][above[-1]] if len(above) else np.nan
        Ms_a = np.interp(r_SF, model['x'], model['M'])
        Mg_a = np.interp(r_SF, model['x'], r['M_g'])
        sfe_rsf = Ms_a / (Ms_a + Mg_a)
        print(f"    gamma={par:g}: removal flavor min SFE_10 = {s10_min:.4f} "
              f"(published thr {sfe_thr:.3f} UNREACHABLE)" if s10_min > sfe_thr
              else f"    gamma={par:g}: removal flavor min SFE_10 = {s10_min:.4f}")
        print(f"              aperture flavor at thr: r_SF={r_SF:7.2f} a, "
              f"SFE(<r_SF)={sfe_rsf:.4f}  [vs SFE_10={sfe_thr:.3f}], eSFE={r['esfe']:.4f}")

    # Single panel: critical SFE on the left axis (black; solid = no
    # truncation, dashed = t_ff truncation), r_SF/r_t on the right axis (blue,
    # with matching blue axis label and ticks). The solid "no truncation"
    # curve is the headline critical curve; the grey verticals mark the
    # Paper I grid and the top axis carries c = log10(r_t/r_c) -- both
    # folded in from the former standalone critical-curve figure.
    fig, (ax, axe) = plt.subplots(2, 1, figsize=(5.0, 5.1), sharex=True,
                                  gridspec_kw={'height_ratios': [1.0, 0.60]})
    for wsc in (3.0, 6.0, 9.0):
        ax.axvline(wsc, color='0.85', lw=0.8, zorder=0)
    ax.text(0.015, 0.05, '(a)', transform=ax.transAxes, fontsize=10)
    l1, = ax.plot(rows[:, 0], rows[:, 1], color='k', lw=1.8,
                  label='no truncation')
    l2, = ax.plot(rows[:, 0], rows[:, 2], color='k', lw=1.8, ls='--',
                  label=r'$t_{\rm ff}$ truncation')
    ax.set(ylabel='virial SFE threshold', xlim=(0, 20.5))

    secax = ax.secondary_xaxis('top')
    cw = interp1d(rows[:, 0], c_vals, bounds_error=False,
                  fill_value='extrapolate')
    tick_w0 = [2, 5, 9, 13, 17]
    secax.set_xticks(tick_w0)
    secax.set_xticklabels([f'{float(cw(t)):.1f}' for t in tick_w0])
    secax.set_xlabel(r'$c=\log_{10}(r_{\rm t}/r_{\rm c})$', fontsize=9)

    axr = ax.twinx()
    BLUE = '#2166ac'
    l3, = axr.plot(rows[:, 0], rows[:, 3], color=BLUE, lw=1.6, ls='-.',
                   label=r'$r_{\rm SF}/r_{\rm t}$')
    axr.set_ylabel(r'$r_{\rm SF}/r_{\rm t}$ at threshold', color=BLUE)
    axr.tick_params(axis='y', colors=BLUE)
    axr.spines['right'].set_color(BLUE)
    axr.set_ylim(0, 1.05)
    axr.minorticks_on()

    ax.legend(handles=[l1, l2, l3], fontsize=8.5, loc='center right')

    # ---- panel (b): sensitivity to eps_ff (was the Appendix A figure) ----
    W0_e = np.arange(1.0, 20.01, 1.0)
    axe.plot(rows[:, 0], rows[:, 1], color='0.45', lw=1.4, ls=':',
             label='no truncation')
    for eps, col in ((0.01, '#1b7837'), (0.05, '#e08214'), (0.10, '#b2182b')):
        cs = [_crit_trunc_at_eps(king_model(w), eps) for w in W0_e]
        axe.plot(W0_e, cs, color=col, lw=1.6,
                 label=fr'$\epsilon_{{\rm ff}}={eps:g}$')
        print(f"    eps_ff={eps:.2f}: min truncated threshold "
              f"{np.nanmin(cs):.4f} at W0={W0_e[int(np.nanargmin(cs))]:.0f}")
    axe.set(xlabel=r'King concentration $W_0$',
            ylabel='virial SFE threshold', xlim=(0, 20.5))
    axe.text(0.015, 0.05, '(b)', transform=axe.transAxes, fontsize=10)
    axe.legend(fontsize=8, loc='upper right', ncol=2)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)      # panels flush: shared x-axis
    fig.savefig(OUTDIR + 'fig9_truncation.pdf')
    fig.savefig(OUTDIR + 'fig9_truncation.png')
    plt.close(fig)

    hdr = 'W0, SFE_crit_notrunc, SFE_crit_trunc, rSF_over_rt'
    np.savetxt(OUTDIR + 'truncation_summary.csv', rows, fmt='%.6f',
               delimiter=', ', header=hdr, comments='')
    print("Fig 9 + table done")


if __name__ == '__main__':
    thr = part1_validation()
    part1b_cross_profile(thr)
    part2_universality(thr)
    part3_truncation()
    print("\nAll three parts complete.")