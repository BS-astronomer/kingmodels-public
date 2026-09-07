#!/usr/bin/env python3
"""
king_sfe_mechanism.py
=====================
Mechanism analysis for the non-monotonic critical SFE found by
king_sfe_analysis.py.

Reduction
---------
Write the star-gas cross energy as
    W_sg = eta * (M_g/M_star) * |W_ss|,
so eta = 1 for gas tracing the stars exactly. Then
    eSFE = SFE / (SFE + eta*(1-SFE)),
and the survival condition eSFE = 1/3 gives EXACTLY
    SFE_crit = eta / (eta + 2).
All W0 dependence therefore lives in the single dimensionless
"gas harmfulness" parameter eta (evaluated at the critical configuration).

Hypothesis tested here: eta is controlled by the radial segregation of gas
from stars. The gas avoids the stellar center (local SFE ~ 1 there) and
extends to the stellar truncation radius r_t, so the segregation should be
governed by the King structural ratio r_t/r_h -- which is known to be
non-monotonic in W0. We test whether argmax(r_t/r_h) coincides with
argmin(SFE_crit).

Outputs: fig6_mechanism.pdf/png, fig7_energy_profiles.pdf/png,
         mechanism_summary.csv
"""

import numpy as np
if not hasattr(np, 'trapezoid'):          # numpy < 2.0 compatibility
    np.trapezoid = np.trapz
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from king_sfe_analysis import (king_model, gas_density, analyze, sweep,
                               tsf_for_sfe, critical_sfe, TSF_GRID)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
})

OUTDIR = './'
# fine near the extremum, coarser elsewhere
W0_GRID = np.unique(np.concatenate([np.arange(0.5, 20.001, 0.5),
                                    np.arange(6.6, 11.4, 0.2)]))


def diagnostics_at_critical(model, esfe_crit=1.0 / 3.0):
    """eta, gas/star half-mass ratio, halo energy fractions at SFE_crit."""
    sc = critical_sfe(model, esfe_crit)
    tsf = tsf_for_sfe(model, sc)
    r = analyze(model, tsf)
    x, rho_s, M_s = model['x'], model['rho'], model['M']
    rho_g, M_g = r['rho_g'], r['M_g']
    Mg_tot = M_g[-1]

    dWss = 4.0 * np.pi * rho_s * M_s * x          # dW/dr integrands (G=1)
    dWsg = 4.0 * np.pi * rho_s * M_g * x
    W_ss = np.trapezoid(dWss, x)
    W_sg = np.trapezoid(dWsg, x)
    eta = W_sg / (Mg_tot * W_ss)                  # M_star = 1

    rh_gas = brentq(lambda rr: np.interp(rr, x, M_g) - 0.5 * Mg_tot,
                    x[1], model['rt'])

    # fraction of each energy integral originating outside the stellar rh
    cum_ss = cumulative_trapezoid(dWss, x, initial=0.0)
    cum_sg = cumulative_trapezoid(dWsg, x, initial=0.0)
    f_halo_ss = 1.0 - np.interp(model['rh'], x, cum_ss) / W_ss
    f_halo_sg = 1.0 - np.interp(model['rh'], x, cum_sg) / W_sg

    return dict(sfe_crit=sc, tsf=tsf, eta=eta,
                rh_gas=rh_gas, rh_ratio=rh_gas / model['rh'],
                rt_rh=model['rt'] / model['rh'],
                f_halo_ss=f_halo_ss, f_halo_sg=f_halo_sg,
                x=x, dWss=dWss, dWsg=dWsg, rho_g=rho_g, M_g=M_g)


def main():
    print("Sweeping W0 for mechanism diagnostics...")
    rows, models = [], {}
    for w in W0_GRID:
        m = king_model(w)
        d = diagnostics_at_critical(m)
        models[w] = (m, d)
        rows.append([w, d['sfe_crit'], d['eta'], d['eta'] / (d['eta'] + 2.0),
                     d['rt_rh'], d['rh_ratio'], d['f_halo_ss'], d['f_halo_sg']])
        print(f"  W0={w:5.2f}  SFE_crit={d['sfe_crit']:.4f}  eta={d['eta']:.4f}  "
              f"eta/(eta+2)={d['eta']/(d['eta']+2):.4f}  rt/rh={d['rt_rh']:.3f}  "
              f"rh_gas/rh_star={d['rh_ratio']:.3f}")
    rows = np.array(rows)

    # closure check: SFE_crit must equal eta/(eta+2) identically
    closure = np.max(np.abs(rows[:, 1] - rows[:, 3]))
    print(f"\nClosure check |SFE_crit - eta/(eta+2)|_max = {closure:.2e}")

    # locate extrema
    i_min = np.argmin(rows[:, 1])
    i_eta = np.argmin(rows[:, 2])
    i_rtrh = np.argmax(rows[:, 4])
    i_seg = np.argmax(rows[:, 5])
    print(f"argmin SFE_crit : W0 = {rows[i_min, 0]:.2f}")
    print(f"argmin eta      : W0 = {rows[i_eta, 0]:.2f}")
    print(f"argmax rt/rh    : W0 = {rows[i_rtrh, 0]:.2f}")
    print(f"argmax rh_g/rh_s: W0 = {rows[i_seg, 0]:.2f}")

    # ------------------- Fig 6: the mechanism chain ----------------------
    fig, axes = plt.subplots(3, 1, figsize=(4.6, 8.2), sharex=True)
    w = rows[:, 0]

    ax = axes[0]
    ax.plot(w, rows[:, 4], color='#2166ac', lw=1.8, label=r'$r_{\rm t}/r_{\rm h}$ (stars)')
    ax.plot(w, rows[:, 5] * 1.0, color='#b2182b', lw=1.8, ls='--',
            label=r'$r_{{\rm h,gas}}/r_{{\rm h,\star}}$')
    ax.axvline(rows[i_min, 0], color='0.75', lw=0.9, zorder=0)
    ax.set_ylabel('structural ratios')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 12)
    ax.text(0.02, 0.9, '(a) geometry: gas-star segregation', transform=ax.transAxes, fontsize=9)

    ax = axes[1]
    ax.plot(w, rows[:, 2], color='#1b7837', lw=1.8)
    ax.axvline(rows[i_min, 0], color='0.75', lw=0.9, zorder=0)
    ax.set_ylabel(r'gas harmfulness $\eta$')
    ax.text(0.02, 0.9, '(b) energy: '
            r'$\eta = W_{\rm sg}\,M_\star/(M_{\rm gas}\,|W_{\rm ss}|)$',
            transform=ax.transAxes, fontsize=9)
    ax.set_ylim(0, 1)

    ax = axes[2]
    ax.plot(w, rows[:, 1], color='k', lw=1.8, label=r'${\rm SFE}_{\rm crit}$')
    ax.plot(w, rows[:, 3], color='#e08214', lw=1.4, ls='--',
            label=r'$\eta/(\eta+2)$')
    ax.axvline(rows[i_min, 0], color='0.75', lw=0.9, zorder=0)
    ax.set_xlabel(r'King concentration $W_0$')
    ax.set_ylabel('virial SFE threshold')
    ax.legend(fontsize=9, loc='upper right')
    ax.text(0.02, 0.9, '(c) closure', transform=ax.transAxes, fontsize=9)
    ax.set_ylim(0.1, 0.275)

    fig.subplots_adjust(hspace=0.0)      # panels flush: shared x-axis
    fig.savefig(OUTDIR + 'fig6_mechanism.pdf')
    fig.savefig(OUTDIR + 'fig6_mechanism.png')
    plt.close(fig)
    print("Fig 6 done")

    # ------------------- Fig 7: where the cross energy lives -------------
    # stacked 2x1: this is a single-column appendix figure (Fig. C.1)
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 6.0), sharex=True)
    ck = {3.0: '#1b7837', 9.0: '#2166ac', 20.0: '#b2182b'}
    for w0 in [3.0, 9.0, 20.0]:
        m, d = models[w0]
        xr = d['x'] / m['rh']
        # differential energy per dlnr, normalized to unit area
        dss = d['dWss'] * d['x']
        dsg = d['dWsg'] * d['x']
        dss /= np.trapezoid(dss, np.log(d['x']))
        dsg /= np.trapezoid(dsg, np.log(d['x']))
        axes[0].plot(xr, dsg, color=ck[w0], lw=1.6, label=fr'$W_0={w0:.0f}$')
        axes[0].plot(xr, dss, color=ck[w0], lw=1.2, ls=':')
        # enclosed gas-to-star mass ratio, normalized by global ratio
        Mg_tot = d['M_g'][-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (d['M_g'] / Mg_tot) / np.maximum(m['M'], 1e-12)
        axes[1].plot(xr, ratio, color=ck[w0], lw=1.6)
    axes[0].plot([], [], color='0.3', lw=1.6, label=r'$dW_{\rm sg}/d\ln r$')
    axes[0].plot([], [], color='0.3', lw=1.2, ls=':', label=r'$dW_{\rm ss}/d\ln r$')
    axes[0].set(xscale='log', xlim=(1e-2, 20),
                ylabel=r'normalized $dW/d\ln r$')
    axes[0].legend(fontsize=7.5, ncol=2, loc='upper left')
    axes[1].axhline(1.0, color='0.5', ls=':', lw=1)
    axes[1].set(xscale='log', yscale='log', xlim=(1e-2, 20), ylim=(1e-3, 3),
                xlabel=r'$r/r_{\rm h}$',
                ylabel=r'$[M_{\rm g}(r)/M_{\rm g}]\,/\,[M_\star(r)/M_\star]$')
    axes[1].text(0.03, 0.92, 'gas lags stars',
                 transform=axes[1].transAxes, fontsize=8, color='0.35')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08)
    fig.savefig(OUTDIR + 'fig7_energy_profiles.pdf')
    fig.savefig(OUTDIR + 'fig7_energy_profiles.png')
    plt.close(fig)
    print("Fig 7 done")

    hdr = ('W0, SFE_crit, eta, eta_over_eta_plus_2, rt_over_rh, '
           'rh_gas_over_rh_star, f_halo_Wss, f_halo_Wsg')
    np.savetxt(OUTDIR + 'mechanism_summary.csv', rows, fmt='%.6f',
               delimiter=', ', header=hdr, comments='')
    print("Table saved. Done.")


if __name__ == '__main__':
    main()

