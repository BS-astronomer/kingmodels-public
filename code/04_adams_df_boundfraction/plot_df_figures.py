#!/usr/bin/env python3
"""
plot_df_figures.py
==================
Re-draws the three distribution-function figures -- fig10 (F_bound curves),
fig11 / fig11b (critical-SFE thresholds vs W0) -- from the CSV data that
adams_df_boundfraction.py writes on its (hours-long) run.

Nothing here touches king_model, fbound_at_sfe, threshold_by_bisection or
any DF machinery: it is pure numpy + matplotlib on pre-computed numbers.
Use it whenever you want a cosmetic change (colours, labels, limits,
legend) without paying for the DF sweep again. Re-run
adams_df_boundfraction.py only if the underlying numbers must change.

Inputs (written by adams_df_boundfraction.py, in its output directory):
  fig10_fbound_curves.csv   kind, W0, SFE, F_b
  fig11_plotdata.csv        series, x, virial, df_cont, df_mc, df_mc_err,
                            nbody, censored, valid

Usage:
  python plot_df_figures.py            # redraw all three into ./
"""
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300})

F_MIN = 0.02
PAL_PUB = {1: ('#000000', 'Plummer'), 2: ('#e08214', r'Dehnen $\gamma=0$'),
           3: ('#807dba', r'Dehnen $\gamma=1$'), 4: ('#8c510a', r'Dehnen $\gamma=2$')}
PUB_ORD = {-1.0: ('#000000', 'P'), -2.0: ('#e08214', 'D0'),
           -3.0: ('#807dba', 'D1'), -4.0: ('#8c510a', 'D2')}


def plot_fig10(csv='fig10_fbound_curves.csv', out='fig10_fbound_curves'):
    d = np.loadtxt(csv, delimiter=',', skiprows=1, ndmin=2)
    king = d[d[:, 0] == 0]
    w0s = np.unique(king[:, 1])
    norm = colors.Normalize(vmin=w0s.min(), vmax=w0s.max())
    cmap = cm.viridis
    fig, ax = plt.subplots(figsize=(4.6, 3.7))
    for w in w0s:
        c = king[king[:, 1] == w]
        c = c[np.argsort(c[:, 2])]
        ax.plot(c[:, 2], c[:, 3], color=cmap(norm(w)), lw=1.4)
    for kind, (col, lab) in PAL_PUB.items():
        c = d[d[:, 0] == kind]
        if not c.size:
            continue
        c = c[np.argsort(c[:, 2])]
        ax.plot(c[:, 2], c[:, 3], color=col, lw=1.3, ls='--', label=lab)
    ax.legend(fontsize=7, loc='lower right')
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$W_0$')
    ax.axhline(F_MIN, color='0.6', ls=':', lw=1)
    ax.set(xlabel='SFE',
           ylabel=r'bound fraction $F_{\rm b}$ (DF estimate)',
           xlim=(0, 0.62), ylim=(0, 1.02))
    fig.savefig(out + '.pdf'); fig.savefig(out + '.png'); plt.close(fig)
    print(f'{out} redrawn from {csv}')


def plot_fig11(csv='fig11_plotdata.csv', with_mc=True,
               out='fig11b_thresholds_with_mc'):
    d = np.loadtxt(csv, delimiter=',', skiprows=1, ndmin=2)
    vir = d[d[:, 0] == 0]
    dfc = d[d[:, 0] == 1]
    pts = d[d[:, 0] == 2]
    pub = d[d[:, 0] == 3]
    RED, MS, MEW = '#b2182b', 5, 1.1
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    if vir.size:
        o = np.argsort(vir[:, 1])
        ax.plot(vir[o, 1], vir[o, 2], color=RED, lw=1.3, ls='--')
    o = np.argsort(dfc[:, 1])
    ax.plot(dfc[o, 1], dfc[o, 3], '-', color=RED, lw=1.4)
    for r in pts:
        w, v, dc, mv, me, valid = r[1], r[2], r[3], r[4], r[5], r[8]
        ax.plot(w, v, 'x', color=RED, ms=MS + 2, mew=MEW, zorder=7)
        ax.plot(w, dc, 'o', color=RED, ms=MS, mfc='none', mew=MEW, zorder=7)
        if valid == 0.0:
            ax.plot(w, dc, '*', color='k', ms=MS + 3, mew=0.8, zorder=9)
        if with_mc and np.isfinite(mv):
            ax.errorbar(w, mv, yerr=me if np.isfinite(me) else None, fmt='+',
                        color=RED, ms=MS + 4, mew=MEW + 0.4, capsize=2, lw=1.0,
                        zorder=8)
    for r in pub:
        xo, v, dc, mv, me, nb, cens, valid = r[1:9]
        col = PUB_ORD[xo][0]
        ax.plot(xo, v, 'x', color=col, ms=MS + 2, mew=MEW, zorder=7)
        ax.plot(xo, dc, 'o', color=col, ms=MS, mfc='none', mew=MEW, zorder=7)
        if valid == 0.0:
            ax.plot(xo, dc, '*', color='k', ms=MS + 3, mew=0.8, zorder=9)
        if with_mc:
            if np.isfinite(mv):
                ax.errorbar(xo, mv, yerr=me if (np.isfinite(me) and me >= 0)
                            else None, fmt='+', color=col, ms=MS + 4,
                            mew=MEW + 0.4, capsize=2, lw=1.0, zorder=8)
                if np.isfinite(me) and me < 0:
                    ax.annotate('', xy=(xo, mv * 0.55), xytext=(xo, mv),
                                arrowprops=dict(arrowstyle='->', color=col,
                                                lw=MEW))
            ax.plot(xo, nb, 'v', color=col, ms=MS, mew=MEW,
                    mfc='none' if cens else col, mec=col, zorder=8)
        ax.axvline(xo, color=col, lw=0.6, ls=':', alpha=0.35, zorder=0)
    ax.plot([], [], 'x', color='0.35', ms=MS + 2, mew=MEW, ls='none',
            label=r'virial ($e{\rm SFE}_{\rm c}=1/3$)')
    ax.plot([], [], 'o', color='0.35', ms=MS, mfc='none', mew=MEW, ls='none',
            label='frozen DF, continuum')
    if with_mc:
        ax.plot([], [], '+', color='0.35', ms=MS + 4, mew=MEW + 0.4, ls='none',
                label='frozen DF, Monte Carlo')
        ax.plot([], [], 'v', color='0.35', ms=MS, mfc='none', mew=MEW,
                ls='none', label=r'$N$-body (open = upper lim.)')
    if np.any(pts[:, 8] == 0.0) or np.any(pub[:, 8] == 0.0):
        ax.plot([], [], '*', color='k', ms=MS + 3, mew=0.8, ls='none',
                label='DF reconstr. check failed')
    ax.axvline(0.0, color='0.75', lw=0.8, zorder=0)
    # The ordinate mixes conventions by necessity: total SFE for the finite
    # King models, SFE_10 for the infinite-extent published families. Say so
    # on the axis itself rather than only in the caption, and shade the
    # published strip so the two regions are visually distinct.
    ax.axvspan(-4.8, 0.0, color='0.93', zorder=-1)
    ax.set(xlabel=r'King concentration $W_0$', ylabel='SFE threshold',
           xlim=(-4.8, 20.5), ylim=(0, 0.36))
    labs = [PUB_ORD[xo][1] for xo in sorted(PUB_ORD)]
    xp = sorted(PUB_ORD)
    w_ticks = [0, 5, 10, 15, 20]
    ax.set_xticks(list(xp) + w_ticks)
    ax.set_xticklabels(labs + [str(t) for t in w_ticks])
    for t in ax.get_xticklabels()[:len(labs)]:
        t.set_fontsize(8)
    ax.legend(fontsize=8, loc='upper right')
    fig.savefig(out + '.pdf'); fig.savefig(out + '.png'); plt.close(fig)
    print(f'{out} redrawn from {csv} (with_mc={with_mc})')


if __name__ == '__main__':
    plot_fig10()
    plot_fig11('fig11_plotdata.csv', with_mc=False,
               out='fig11_df_vs_virial_threshold')
    plot_fig11('fig11_plotdata.csv', with_mc=True,
               out='fig11b_thresholds_with_mc')
