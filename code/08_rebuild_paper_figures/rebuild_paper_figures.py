#!/usr/bin/env python3
"""
rebuild_paper_figures.py
========================
Regenerates every presentation-layer figure of the manuscript from the
analysis modules and their CSV outputs. Run AFTER the analysis scripts
(see REPRODUCTION_GUIDE.md for the order). Figures that need data files
you have not produced yet are skipped with a message.

Inputs expected in the current directory (all optional; a missing one just
skips its figure):
  mechanism_summary.csv          (king_sfe_mechanism.py)
  truncation_summary.csv         (multi_profile_analysis.py)
  df_threshold_summary.csv       (adams_df_boundfraction.py)
  analytic_prediction_table.csv  (build_analytic_prediction_table.py)
  envelope_slopes.csv            (build_envelope_slopes.py)
  dense_thresholds.csv           (df_mc_thresholds_dense.py)

Outputs: the three manuscript figures this stage owns -- fig1_structure,
fig3_sfe_vs_tsf_rh_units and fig12_validation (PDF+PNG). The others are
produced by their own analysis stages; see docs/figure_data_map.csv.
"""
import os
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import sys
sys.path.append('../01_king_sfe_analysis')
from king_sfe_analysis import (king_model, gas_density, analyze, sweep,
                               tsf_for_sfe, critical_sfe)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300})

SFE0 = 0.17
W0_FAMILY = np.array([0.5, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
NORM = colors.Normalize(vmin=W0_FAMILY.min(), vmax=W0_FAMILY.max())
CMAP = cm.viridis
CK = {3.0: '#1b7837', 6.0: '#2166ac', 9.0: '#b2182b'}


def save(fig, name):
    fig.savefig(name + '.pdf'); fig.savefig(name + '.png'); plt.close(fig)
    print(name, 'done')


def maybe(path):
    """Load a CSV written by this or a sibling pipeline stage.

    Inputs are searched in the current directory first, then in the sibling
    stage directories, because several of them are produced by another stage
    (e.g. mc_vs_nbody_published.csv comes from 05_mc_bound_fraction_agama).
    Returns None, with a message, if the file is nowhere to be found -- the
    dependent figure is then skipped rather than crashing the whole rebuild.
    """
    import glob
    for cand in [path, *sorted(glob.glob(os.path.join('..', '*', path))),
                 os.path.join('..', path)]:
        if os.path.exists(cand):
            return np.loadtxt(cand, delimiter=',', skiprows=1, ndmin=2)
    print(f'  ({path} not found -- skipping dependent figure)')
    return None


# ---------------- Fig 1 (ms Fig 1): density profiles, rh units --------
# Half-mass-radius units (G = M_star = r_h = 1): these are the units of
# the Paper I N-body campaign, which equates half-mass radii (fixed
# lambda = r_h/r_J) across W_0 rather than tidal radii. The truncation
# radius r_t/r_h now varies with W_0 and is drawn per model.
def fig1():
    """Merged structure figure (ms Fig. 1): density profiles (top) and
    local-SFE profiles (bottom), sharing the r/rh axis."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4.5, 6.2), sharex=True,
                                  gridspec_kw={'height_ratios': [1.0, 0.85]})
    for w in [3.0, 6.0, 9.0]:
        m = king_model(w)
        r = analyze(m, tsf_for_sfe(m, SFE0))
        xh = m['rh']; xr = m['x'] / xh
        ax.plot(xr, m['rho'] * xh**3, color=CK[w], lw=1.6, label=fr'$W_0={w:.0f}$')
        ax.plot(xr, r['rho_g'] * xh**3, color=CK[w], lw=1.6, ls='--')
        ax.axvline(1.0 / xh, color=CK[w], ls=':', lw=1.0, alpha=0.75)
        ax.axvline(m['rt'] / xh, color=CK[w], ls=(0, (1, 1)), lw=1.0, alpha=0.6)
    ax.axhline(3 / (8 * np.pi), color='0.6', ls=':', lw=1)
    ax.text(1.1e-2, 3 / (8 * np.pi) * 1.4,
            r'$\langle\rho_\star\rangle_{r_{\rm h}}$', fontsize=8, color='0.4')
    ax.plot([], [], color='0.3', lw=1.6, label='stars')
    ax.plot([], [], color='0.3', lw=1.6, ls='--', label='gas')
    ax.plot([], [], color='0.3', lw=1.0, ls=':', label=r'$r_{\rm c}$')
    ax.plot([], [], color='0.3', lw=1.0, ls=(0, (1, 1)), label=r'$r_{\rm t}$')
    ax.set(xscale='log', yscale='log', xlim=(1e-2, 12), ylim=(1e-5, 1e3),
           ylabel=r'$\rho\;\;[M_\star\,r_{\rm h}^{-3}]$')
    ax.set_title(fr'global SFE $= {SFE0}$', fontsize=10)
    ax.legend(ncol=2, fontsize=8.0, loc='upper right')
    ax.text(0.02, 0.05, '(a)', transform=ax.transAxes, fontsize=10)

    # ---- lower panel: local SFE for the whole W0 family ----
    for w in W0_FAMILY:
        m = king_model(w)
        r = analyze(m, tsf_for_sfe(m, SFE0))
        eps = np.where(m['rho'] > 0,
                       m['rho'] / np.maximum(m['rho'] + r['rho_g'], 1e-300), np.nan)
        ax2.plot(m['x'] / m['rh'], eps, color=CMAP(NORM(w)), lw=1.3)
    ax2.axhline(SFE0, color='0.4', ls=':', lw=1)
    ax2.text(1.3e-2, SFE0 * 1.08, 'global SFE', fontsize=8, color='0.35')
    ax2.set(ylim=(0, 1.02), xlabel=r'$r/r_{\rm h}$',
            ylabel=r'local SFE $\;\rho_\star/(\rho_\star+\rho_{\rm gas})$')
    ax2.text(0.02, 0.90, '(b)', transform=ax2.transAxes, fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)      # panels flush: shared x-axis
    # colorbar spans BOTH panels so the two axes keep the same width
    fig.colorbar(cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=[ax, ax2],
                 label=r'$W_0$', pad=0.02, fraction=0.06)
    save(fig, 'fig1_structure')


# ---------------- Fig 2 (ms Fig 2): local SFE vs r/rh -----------------
# ---------------- Fig 3 (ms Fig 3): SFE vs tsf in rh units -----------
def fig3():
    TSF = np.logspace(-2, 8, 220)
    fig, ax = plt.subplots(figsize=(4.6, 3.7))
    t_at_ref = []                       # t_sf to reach the fixed global SFE
    for w in W0_FAMILY:
        m = king_model(w)
        sfe, _ = sweep(m, TSF)
        trh = TSF / m['rh']**1.5
        ax.plot(trh, sfe, color=CMAP(NORM(w)), lw=1.3, alpha=0.9)
        t_at_ref.append(float(np.interp(SFE0, sfe, trh)))
    fig.colorbar(cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=ax, label=r'$W_0$')
    ax.set(xscale='log', xlabel=r'$t_{\rm sf}\;\;[G=M_\star=r_{\rm h}=1]$',
           ylabel='global SFE', ylim=(0, 1), xlim=(3e-2, 1e3))
    # inset: t_sf needed for the fixed global SFE, vs W_0 -- the near-universal
    # (factor ~2.6) spread the main panel's near-collapse hides
    axin = ax.inset_axes([0.13, 0.56, 0.40, 0.38])
    axin.plot(W0_FAMILY, t_at_ref, 'o-', color='0.25', lw=1.2, ms=3)
    axin.set_xlabel(r'$W_0$', fontsize=7, labelpad=1)
    axin.set_ylabel(fr'$t_{{\rm sf}}(\mathrm{{SFE}}={SFE0})$', fontsize=7, labelpad=1)
    axin.tick_params(labelsize=6, length=2)
    axin.set_xlim(0, 20.5)
    save(fig, 'fig3_sfe_vs_tsf_rh_units')


# ---------------- Figs 4-8: produced by the analysis modules -----------
# fig4_esfe_vs_sfe, fig5_critical_sfe_vs_W0: king_sfe_analysis.py main()
# fig6_mechanism, fig7_energy_profiles:     king_sfe_mechanism.py
# fig8_cumulative_esfe:                     multi_profile_analysis.py

# ---------------- Fig 9 (ms Fig 12): truncation, twinx ------------------
# ---------------- Fig 11 (ms Fig 10): thresholds vs W0 ------------------
# ---------------- Fig 12 (ms Fig 11): MC validation panels --------------
def _mc_points(w):
    """Mean and standard deviation of F_b over realizations at each SFE, for
    one W0, from the per-realization N=1e5 run. Returns (sfe, mean, sd) or
    None if that run is unavailable."""
    d = maybe('mc_bound_fractions_n1e5.csv')
    if d is None:
        return None
    sub = d[np.abs(d[:, 0] - w) < 0.01]
    if not len(sub):
        return None
    sfe = np.unique(np.round(sub[:, 2], 6))
    mean = np.array([sub[np.round(sub[:, 2], 6) == x, 4].mean() for x in sfe])
    sd = np.array([sub[np.round(sub[:, 2], 6) == x, 4].std() for x in sfe])
    return sfe, mean, sd


def fig12():
    tab = maybe('analytic_prediction_table.csv')
    if tab is None:
        return
    # Dense continuum from build_fig12_continuum.py; the prediction table
    # samples only the SFE values the Monte Carlo covers, so its own
    # continuum column is too coarse to draw.
    cont = maybe('fig12_continuum.csv')
    pub = maybe('published_continuum.csv')          # built once, see script
    pubmc = maybe('mc_vs_nbody_published.csv')      # from stage 05
    # panels share BOTH axes: with the SFE convention fixed once in the text
    # (total for King, SFE_10 for the published families) the abscissa is
    # simply SFE in both, so they can sit flush.
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(4.6, 5.6),
                                  sharex=True, sharey=True)

    # ---- (a) King family: all four W0 on one axes ----
    for w in [3.0, 6.0, 9.0, 12.0]:
        t = tab[np.abs(tab[:, 0] - w) < 0.01]
        t = t[np.argsort(t[:, 1])]
        col = CK.get(w, CMAP(NORM(w)))
        if cont is not None:
            c = cont[np.abs(cont[:, 0] - w) < 0.01]
            c = c[np.argsort(c[:, 1])]
            ax.plot(c[:, 1], c[:, 2], color=col, lw=1.6, zorder=1,
                    label=fr'$W_0={w:g}$')
        else:
            ax.plot(t[:, 1], t[:, 2], color=col, lw=1.6, zorder=1,
                    label=fr'$W_0={w:g}$')
        # particle points: prefer the per-realization N=1e5 run (150 seeds,
        # dense in SFE); fall back to the prediction table's aggregate columns
        mc = _mc_points(w)
        if mc is not None:
            sfe_m, mean_m, sd_m = mc
            ax.errorbar(sfe_m, mean_m, yerr=sd_m, fmt='o', color=col, ms=3.0,
                        capsize=1.5, lw=0.8, mec='0.25', mew=0.4, zorder=3)
        else:
            ok2 = np.isfinite(t[:, 5])
            ax.errorbar(t[ok2, 1], t[ok2, 5], yerr=t[ok2, 6], fmt='o',
                        color=col, ms=3.5, capsize=2, lw=1, mec='0.25',
                        mew=0.5, zorder=3)
    ax.plot([], [], color='0.3', lw=1.6, label='continuum')
    ax.plot([], [], 'o', color='0.3', ms=3.0, label=r'MC, $N=10^5$')
    ax.set(xlim=(0, 0.72), ylim=(-0.03, 1.03),
           ylabel=r'bound fraction $F_{\rm b}$')
    ax.legend(fontsize=7.5, ncol=2, loc='lower right')
    ax.text(0.02, 0.92, '(a) King family', transform=ax.transAxes, fontsize=9)

    # ---- (b) published families: continuum + MC + published N-body ----
    PC = {0: ('#000000', 'Plummer'), 1: ('#e08214', r'Dehnen $\gamma=0$')}
    if pub is not None:
        for code, (col, lab) in PC.items():
            c = pub[np.abs(pub[:, 0] - code) < 0.01]
            c = c[np.argsort(c[:, 1])]
            axp.plot(c[:, 1], c[:, 2], color=col, lw=1.6, zorder=1, label=lab)
    if pubmc is not None:
        # rows 0-4 are Plummer, 5-11 Dehnen g=0 (SFE_10 restarts at the split)
        split = int(np.argmax(np.diff(pubmc[:, 0]) < 0) + 1)
        for code, sl in ((0, slice(0, split)), (1, slice(split, None))):
            col = PC[code][0]; d = pubmc[sl]
            ok = d[:, 1] > 0
            axp.errorbar(d[ok, 0], d[ok, 1], yerr=d[ok, 2], fmt='o', color=col,
                         ms=3.5, capsize=2, lw=1, mec='0.25', mew=0.5, zorder=3)
            axp.errorbar(d[:, 0], d[:, 3], yerr=d[:, 4], fmt='s', color=col,
                         ms=4.5, mfc='none', capsize=2, lw=1, zorder=4)
    axp.plot([], [], 'o', color='0.3', ms=3.5, label='frozen-DF MC')
    axp.plot([], [], 's', color='0.3', ms=4.5, mfc='none',
             label=r'$N$-body (published)')
    axp.set(xlim=(0, 0.72), xlabel='SFE',
            ylabel=r'bound fraction $F_{\rm b}$')
    axp.legend(fontsize=7.5, loc='lower right')
    axp.text(0.02, 0.92, '(b) published families', transform=axp.transAxes,
             fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)      # panels flush
    save(fig, 'fig12_validation')


# ---------------- Fig A1: eps_ff sensitivity via truncation -------------
# ---------------- Fig B1: King-density catastrophic cancellation ------
# ---------------- Fig 13: envelope slopes ------------------------------
if __name__ == '__main__':
    fig1(); fig3(); fig12()
    # fig13() (envelope slopes) retired from the manuscript -- kept as a
    # function for reference but no longer rebuilt.
    print('\nAll presentation figures rebuilt (figs 4-8 come from the '
          'analysis modules themselves; see guide).')
