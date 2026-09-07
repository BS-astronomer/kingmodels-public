# REPRODUCTION GUIDE
## King-model clusters with a centrally peaked SFE — full analysis chain

This document explains every script in the project, in the order you would
run them to reproduce the entire paper from scratch, including the physics
each one implements, the numerical traps we hit and fixed, and how the
pieces feed each other.

---

## 0. Environment and conventions

**Dependencies.** Python ≥ 3.9, numpy (1.x or 2.x — every script carries a
two-line shim `np.trapezoid = np.trapz` for old numpy), scipy, matplotlib.
Only the `mc_*` scripts additionally need **Agama**. Everything else is
pure numpy/scipy and runs anywhere.

**Units.** All computations use G = M_star = r0 = 1, where r0 is the King
core radius (Agama's `scaleRadius`). Conversions to clump units
(G = M_star = r_t = 1), used in presentation figures 2–4:
r̂ = r/x_t, t̂ = t/x_t^(3/2), ρ̂ = ρ·x_t³, with x_t = r_t/r0.
**Never feed r_t-unit t_sf values into the IC generator** — its lookup
tables are r0-based.

**Two exact degeneracies worth internalizing** (they mean you never need
to re-run for different unit choices or ε_ff):
1. Rescaling ρ→cρ, k→c^(-1/2)k leaves the gas relation invariant → all
   dimensionless results are unit-independent.
2. ε_ff and t_SF enter only through k = sqrt(8/3π)·ε_ff·t_SF → every
   untruncated result is exactly ε_ff-independent. ε_ff matters only via
   the free-fall truncation, where ρ_SF/ρ_plateau = ε_ff²/4.

---

## 1. `gen_king_sc_fixed.py` — initial-condition generator (for Paper I)

**Purpose.** Generates N-body ICs: King stellar cluster in virial
equilibrium within the total (stars+gas) potential, for PeTar and
phi-GRAPE/GPU.

**Flow.** (1) Build King potential in Agama; (2) compute the residual gas
density on a radial grid via the closed-form inversion (Sect. 2 below);
(3) build the gas potential (multipole) and the total potential;
(4) construct the QuasiSpherical DF of the King density in the total
potential; (5) sample N phase-space points; (6) assign masses (Kroupa IMF
internally, or mcluster for binaries — mcluster supplies ONLY masses and
internal binary orbits; positions/velocities always come from the DF);
(7) rescale to physical units and write `.ini` / `-petar.init`.

**Fixes you should know about (vs. the original):**
- Exact-N bookkeeping: N_bin = round(N_tot·fb/(1+fb)), N_sys = N_tot−N_bin
  ⇒ total star count = N_tot exactly for any fb (constant mass/density
  across the fb grid).
- Size precedence: --lam > --rh > --rc > default rh=1 pc (rc was silently
  ignored before; defaults are now None so "explicitly set" is detectable).
- Gas is zeroed exactly where the true King density vanishes: the 1e-10
  clamp on ρ_star used to leak a spurious ~M_star-worth of fake gas halo
  (ρ_g ∝ ρ_s^(2/3) amplifies the clamp floor).
- mcluster parsing handles scientific notation; exit codes and star counts
  are checked; header records the actual row count.
- Note: the PeTar file stays in the CLUSTER frame (galactic orbit handled
  by PeTar's external potential); only the `.ini` gets the X_sc/V_sc shift.

---

## 2. `king_sfe_analysis.py` — core semi-analytic module (Figs 4, 5 of ms)

**This is the foundation; every other analysis script imports it.**

**`king_model(W0, n_grid)`** solves the King structure ODE
(d/dx)(x²dW/dx) = −9x²ρ̂(W)/ρ̂(W0) from W(0)=W0 outward to the truncation
W=0, then normalizes to M=1, r0=1. Returns a dict with the radial grid,
ρ(r), M(<r), r_t, r_h, c.
*Numerical trap:* the textbook density
ρ̂(W)=e^W erf(√W) − √(4W/π)(1+2W/3) is a difference of two √W-scale terms
whose true value is O(W^(5/2)) — catastrophic cancellation for W ≲ 1e-8.
Below W=1e-3 we use the sympy-verified series
(8/15√π)W^(5/2)(1 + 2W/7 + 4W²/63).

**`gas_density(rho_s, tsf, eps_ff)`** inverts the Parmentier & Pfalzner
(2013) local star-formation relation in closed form (the K0/K1/K2
quartic solution). Limits: ρ_g → 1/k² (plateau) at high ρ_s (gas-poor
centre, local SFE→1); ρ_g ∝ ρ_s^(2/3) at low ρ_s (gas-dominated
envelope); ρ_g → 0 where ρ_s → 0 (gas truncates WITH the stars at r_t —
no explicit cut needed).
*Numerical trap:* for α = k⁴ρ² ≳ 1e16 the exact expression loses all
precision in (K0−α); above α=1e8 we switch to the asymptotic
ρ_g = 1/k² − 2/(k³√ρ) + 3/(k⁴ρ) (continuous to 2e-4 at the switch).

**`analyze(model, tsf)`** computes the energy integrals
|W_ss| = 4πG∫ρ_s M_s r dr and W_sg = 4πG∫ρ_s M_g r dr, and from them
the post-expulsion effective SFE, eSFE = |W_ss|/(|W_ss|+W_sg)
(= 1/(2Q), Q the post-expulsion virial ratio). Also returns the global
SFE = 1/(1+M_gas).

**`verify_rho_hat_series()`** self-tests the small-W series: symbolically
re-derives the expansion with sympy when available (checking the 8/(15√π)
prefactor and the 2/7, 4/63 coefficients), and always runs a sympy-free
numerical check — the two branches must agree to <1e-7 on 1e-3 < W < 1e-2,
the window where both are simultaneously accurate, and rho_hat/W^(5/2) must
approach 8/(15√π). Injecting a typo (2/7 -> 2/9) fails it by 6e-4, so the
coefficients are guarded rather than merely commented. `verify_all()` runs
this plus the identical-profile checks; nothing runs at import time, so
sympy is never required on compute nodes.

**`verify(model)`** enforces the identical-profile check: gas ∝ stars
must give eSFE = SFE to machine precision — run it whenever you touch the
energy code.

**Sweeps and inversions:** `sweep` maps a t_sf grid to (SFE, eSFE);
`tsf_for_sfe` and `critical_sfe` invert (both auto-extend the t_sf range —
concentrated models have tiny densities in r0 units and need t_sf up to
1e7+).

Running `python king_sfe_analysis.py` regenerates figs 1–5 (files) and
`results_summary.csv`. The virial critical curve — the non-monotonic
headline with the minimum SFE_crit = 0.106 at W0 = 8.8 — converges to
1e-7 in resolution (Appendix B).

---

## 3. `king_sfe_mechanism.py` — why the minimum exists (ms Figs 7, 8)

Reduces the survival condition exactly: writing
W_sg = η·(M_g/M_s)·|W_ss| defines the "gas harmfulness" η (η=1 for gas
tracing stars), and eSFE = SFE/(SFE+η(1−SFE)) gives
**SFE_crit = η·eSFE_c/(1−eSFE_c+η·eSFE_c)** — all W0 dependence lives in
η. The script computes, at each W0's critical configuration: η, the
gas/star half-mass ratio (segregation), r_t/r_h, and the halo fractions
of both energy integrals; then verifies the closure SFE_crit = η/(η+2)
to 1e-8 and plots the causal chain (Fig 6 file) and the radial energy
distributions (Fig 7 file).

**The mechanism in one sentence:** by the shell theorem only enclosed gas
binds a star; segregation of gas from stars minimizes η; segregation
peaks where the King structural ratio r_t/r_h peaks (W0≈8, a pure King
property) — beyond that the stellar half-mass radius races into the
gas-rich halo and η rises again. Extrema: r_t/r_h max at 8.0 →
segregation max at 8.2 → η and SFE_crit min at 8.8 (offsets from
energy-weighting vs mass-weighting).

---

## 4. `multi_profile_analysis.py` — Plummer/Dehnen validation, truncation
(ms Figs 1, 12)

**Part 1 (validation).** Adds analytic Plummer and Dehnen(γ) profiles and
reproduces the published survival thresholds: eSFE at threshold = 0.317
(Plummer, quoted 0.32), 0.154 (γ=0, 0.16), 0.062 (γ=1, 0.06), 0.073
(γ=2, 0.08). Two conventions are essential: (i) the SFE of
infinite-extent models is measured within 10 scale radii — and the 2017
Plummer threshold only reproduces under this aperture, confirming it was
aperture-limited; (ii) the γ≥1 thresholds are the lowest SIMULATED SFEs
(censored upper limits).

**Part 2 (universality test).** Cumulative eSFE(<r) profiles at each
threshold do NOT collapse at any enclosed-mass fraction ⇒ no static
energy criterion is universal; the critical global eSFE genuinely
decreases with central concentration (motivates the DF method, Sect. 5).

**Part 3 (t_ff truncation).** Gas with t_ff(ρ) > t_SF hasn't collapsed;
threshold density ρ_SF = 3π/(32G·t_SF²), i.e. a FIXED contrast
ε_ff²/4 below the central plateau — scale-free. Two flavors:
*removal* (gas excluded from the potential — forbids the published
low-SFE Dehnen configs, so disfavored) and *aperture* (r_SF defines
where SFE is measured — compresses the threshold spread across profiles
from a factor 11 to ~2). For King models removal raises the critical
curve by 0.03–0.06 and moves the minimum to W0≈9.8; the
non-monotonicity survives.

*Numerical notes:* root-finding is done in t_sf space (monotone), taking
the LAST increasing crossing — under truncation eSFE(t_sf) is U-shaped
(full truncation at short t_sf ⇒ a minimum consistent SFE exists), and
fully-truncated configs are flagged invalid.

---

## 5. `adams_df_boundfraction.py` — bound fraction from the DF (ms Figs 9, 10)

Implements the Adams (2000) / Boily & Kroupa (2003a) approach on our
derived gas profiles.

**`eddington_f`**: numerical Eddington inversion of ρ_star(Ψ_tot).
Uses plain np.gradient + np.interp (a PCHIP-spline version behaved
non-portably across numpy/scipy builds); log-spaced energy grid
(resolves halo energies of infinite models); verified against the
analytic Plummer DF to 2e-3. For truncated models f(ε)=0 below
Ψ_t = Ψ_tot(r_t) automatically.

**`bound_fraction`**: frozen-DF iterated escape criterion. A star is
bound iff v²/2 < Ψ_b(r), Ψ_b from the bound stars only; iterate at fixed
positions. Two stabilizers, both physical: a pointwise correction factor
ρ_s/ρ_reconstructed cancels inversion error systematically (without it,
outer-halo errors feed a runaway of spurious bound mass in Dehnen
models), and ρ_b ≤ ρ_s (the bound subset cannot exceed the total).

**Evaluation is SFE-parametrized** (your improvement): `fbound_at_sfe`
inverts SFE→t_sf per model (cheap) so different W0 are compared at
identical SFE points; `threshold_by_bisection` finds SFE_crit
(F_b < 0.02) with ~14 expensive evaluations.

**Cheap replot (no DF sweep).** A full run of this script is hours. It
now writes everything the three figures plot to CSV —
`fig10_fbound_curves.csv` (every F_b(SFE) point, King + published) and
`fig11_plotdata.csv` (virial/DF-continuum curves, the W0 = 3/6/9/12
markers, and the P/D0/D1/D2 strip). `plot_df_figures.py` redraws
fig10 / fig11 / fig11b from those two files (plus the committed MC CSVs)
with **zero DF computation** — use it for any cosmetic change (colours,
labels, limits). Re-run `adams_df_boundfraction.py` only when the
numbers must change.

**Known limitations, quantified:** the method neglects violent
relaxation/recapture (conservative for diffuse profiles — kills Plummer
at its N-body threshold); near threshold the iteration is nearly
runaway, giving the thresholds a 15–25% numerical sensitivity AND a
one-sided continuum-vs-particle offset (see Sect. 6) — which is why the
MC values are definitive there, while the continuum agrees with N=1e5
MC to <2% on the developed branch.

---

## 6. `mc_bound_fraction_agama.py` — Monte Carlo verification (ms Fig 11)

Samples N equal-mass particles from the same Agama machinery as the IC
generator (QuasiSpherical DF in the total potential), then applies the
iterated escape criterion particle-wise with the exact spherical
estimator Ψ_b(r_i) = M_b(<r_i)/r_i + Σ_{bound j: r_j>r_i} m/r_j
(O(N log N) via sorting; no softening). Config at top: W0_LIST, SFE_LIST,
SEEDS, N_PART.

**Critical fix (learned the hard way):** the multipole gas potential must
be fitted only over the region where gas exists
(rmax = 2×gas truncation radius, gridSizeR=60). A fixed rmax=1e4 spans
decades of exactly-zero density; the resulting DF-construction jitter
(~2% in the velocity tail — invisible in the potential itself!) is enough
to flip marginal configurations, producing non-monotonic F_b(SFE).
`mc_diagnose_w3.py` demonstrates this: compares both settings and prints
t_sf smoothness, multipole-vs-direct potential, single-pass unbound
fraction (the smoking gun), and F_b.

**Findings to expect:** per-mille agreement with the continuum on the
developed branch, N-independent means with √N-shrinking errors; near
threshold the transition sharpens with N (at W0=3, SFE=0.30: 26% of
N=1e4 realizations survive, none at N=1e5 ⇒ the continuum is the N→∞
limit and shot noise rescues sub-threshold clusters at finite N).
Note: Agama's OpenMP sampling is not bit-reproducible across runs even at
fixed seed — treat every run as an independent realization; quote
distributions, not per-seed values.

---

## 6b. `mc_published_families.py` — Monte Carlo for Plummer/Dehnen

Same machinery as `mc_bound_fraction_agama.py`, applied to the published
profile families so Fig. 11b can show virial / continuum-DF / Monte Carlo for
them too. Writes `mc_thresholds_published.csv` (family_index 0-3 = Plummer,
Dehnen gamma = 0, 1, 2), which `adams_df_boundfraction.py` picks up
automatically.

Two conventions to keep straight: SFE is measured within 10 SCALE RADII
(matching the continuum thresholds and the published values), NOT within
R_J = 20 r_h as in the IC generators; and the gas of these infinite-extent
profiles must be cut at RMAX_GAS, a real systematic that the script reports
(it prints how much of the gas mass lies beyond the aperture).

## 6c. `df_sensitivity_checks.py` — robustness of the frozen-DF estimate

Two robustness checks quoted in Sect. 5:
(1) how much the DF threshold moves if the bound-fraction criterion F_b < 0.02
is changed to 0.01 or 0.05 — unchanged at W0 = 3, moving by 25/19/99 per cent
at W0 = 6/9/12, i.e. <= 0.014 in absolute SFE against a curve spanning a factor
~20; and (2) whether the Eddington inversion returns a non-negative f(eps)
over the whole (W0, SFE) grid — it does, worst normalized minimum +1e-15.
Writes `df_threshold_sensitivity.csv`. No Agama needed.

## 7. `df_mc_thresholds_dense.py` — dense threshold curves (for ms Fig 10)

Continuum bisection + MC survival-probability bisection (threshold =
SFE where P(survive) crosses 50% over N_REAL realizations) at every
ΔW0 = 0.5. Restartable (appends to dense_thresholds.csv). Hours of
runtime; edit W0_GRID to split sessions.

**`analytic_prediction_table.csv`** (used by Fig 11 / file fig12) and
**`envelope_slopes.csv`** (file fig13) are produced by
`build_analytic_prediction_table.py` and `build_envelope_slopes.py`
respectively (both in `08_rebuild_paper_figures/`). The first evaluates
`fbound_at_sfe` at exactly the MC SFE points and column-merges the MC
mean/std split by N (run `mc_bound_fraction_agama.py` at N_PART = 10_000
and 100_000 first); the second fits rho ~ r^-p between the 10% and 90%
enclosed-mass radii for rho_gas and rho_star+rho_gas at SFE =
0.10/0.15/0.20. Neither needs Agama.

---

## 8. `rebuild_paper_figures.py` — all presentation figures

Regenerates ms Figs 2, 3, 4 (r_t-unit versions), 12 (twinx truncation),
10 (thresholds incl. your dense scan if present), 11 (MC validation),
A1 (ε_ff via truncation), B1 (numerics). Files it expects are listed in
its docstring; missing ones are skipped gracefully. Figs 5–9 of the
manuscript come from the analysis modules' own `main()`s.

---

## 9. Full reproduction checklist (fresh machine)

```
# analysis core (no agama needed)
python king_sfe_analysis.py          # figs 1-5 files, results_summary.csv
python king_sfe_mechanism.py         # figs 6-7, mechanism_summary.csv
python multi_profile_analysis.py     # figs 8-9, truncation_summary.csv
python adams_df_boundfraction.py     # figs 10-11, df_threshold_summary.csv

# Monte Carlo (needs agama; your machine)
python mc_bound_fraction_agama.py    # mc_bound_fractions.csv (+figure)
#   ... run with N_PART=10_000 and 100_000; 50 seeds
python df_mc_thresholds_dense.py     # dense_thresholds.csv (long)

# build the fig inputs, then the figures (no agama needed)
python build_analytic_prediction_table.py   # analytic_prediction_table.csv (fig12 MC points)
python build_fig12_continuum.py             # fig12_continuum.csv           (fig12 dense curve; ~15 min)
python build_envelope_slopes.py             # envelope_slopes.csv           (unused since fig13 was dropped)
python rebuild_paper_figures.py             # all presentation figures
```

**Cosmetic-only rebuilds** (no expensive recomputation): the DF-figure
data is cached in CSV, so
`python plot_df_figures.py` redraws fig10 / fig11 / fig11b, and
`python rebuild_paper_figures.py` redraws fig12 from `fig12_continuum.csv`.
Everything else (`king_sfe_analysis.py`, `king_sfe_mechanism.py`,
`multi_profile_analysis.py` figs) recomputes in seconds and needs no
cache.

Sanity anchors to confirm at each stage: identical-profile eSFE = SFE to
1e-12; King concentrations c = 0.672/1.255/2.119 for W0 = 3/6/9; virial
minimum 0.10624 at W0=8.8; η closure to 1e-8; Plummer DF inversion 2e-3;
Plummer/Dehnen threshold eSFEs 0.317/0.154/0.060/0.083 (matches main.tex
Table tab:validation; the 0.062/0.073 previously here was a stale
transcription, same one caught in README.md -- see docs/feedback-2026-09.md);
MC vs continuum < 2% on the developed branch.

## 10. Gotchas index (one line each)

- King density near edge: series below W=1e-3 (cancellation).
- Gas inversion at α>1e8: asymptotic branch (cancellation in K0−α).
- Clamp floors amplify: ρ_g ∝ ρ_s^(2/3) turns a 1e-10 floor into ~M_star
  of fake gas — always zero gas where true ρ_s = 0.
- Dehnen gas mass diverges (r^(1/3)); use the 10a aperture or the t_ff
  truncation; energy integrals converge regardless.
- Root-find in t_sf (monotone), never by interpolating the inverted
  eSFE(SFE) relation; take the last increasing crossing under truncation.
- Iterated escape criterion is nearly runaway at threshold: expect
  bimodality, N-dependence, and quote survival probabilities there.
- Multipole fits: never span decades of zero density; match rmax to the
  gas extent.
- Agama OpenMP sampling: not seed-reproducible run-to-run.
- numpy <2: np.trapezoid shim (already in every script).
- r_t-unit t_sf values must never enter the r0-based IC pipeline.
- eps_ff single-source rule: t_SF and eps_ff are exactly degenerate (only
  k = sqrt(8/3pi)*eps_ff*t_SF enters), so a t_SF value is meaningless
  without its eps_ff convention. Tables made at eps_1 convert to eps_2 via
  t_SF * eps_1/eps_2. The generator now inverts SFE->t_SF internally with
  its own --eff (default 0.05, the paper's fiducial) and self-checks the
  built model's actual SFE against the target, aborting on >0.005 mismatch
  -- legacy target_sfe_vs_tsf tables (e.g. from get_sfe_tsf.py at
  eps_ff=0.01) are no longer read and cannot corrupt the ICs.

### Numerical checks

`code/04_adams_df_boundfraction/df_positivity_check.py` sweeps the King
family (and the published families) and records the PRE-CLIP diagnostic of
`eddington_f_analytic`: `neg_frac`, `worst_rel_negativity`. Result reported in
Appendix B: the DF is non-negative at every interior energy; only the final
grid node (eps = Psi_tot(0), where the analytic boundary term ~ q^-1/2 is
evaluated) can go negative, and it carries exactly zero weight because every
phase-space integral multiplies f by sqrt(Psi_tot - eps), which vanishes
there. Output: `df_positivity.csv`. Runtime ~1 h (each model needs a
threshold bisection); not needed to rebuild any figure.

`code/08_rebuild_paper_figures/build_published_continuum.py` -> `published_continuum.csv`
(dense frozen-DF curve for Plummer and Dehnen gamma=0; feeds Fig. 7b). ~10 min.

### Monte Carlo at N = 1e5 (the paper's single particle dataset)

`code/05_mc_bound_fraction_agama/mc_bound_fraction_n1e5.py` -> `mc_bound_fractions_n1e5.csv`
  150 realizations per (W0, SFE) at N = 1e5, on a refined SFE grid across each
  transition. Needs Agama. ~2.5 h. Replaces the earlier N = 1e4 run.

`mc_thresholds_n1e5.py` -> `mc_thresholds_n1e5.csv`
  Thresholds with a BOOTSTRAP sampling uncertainty (4000 resamples over
  realizations), not an SFE-grid half-width. Seconds, reads the CSV above.

`mc_refine_threshold.py`
  Inserts finely spaced SFE points across an already-located transition and
  appends them to the input CSV. Needed because the transition narrows with N:
  at N = 1e5 the W0 = 3 transition completes inside one 0.005 grid step, so
  without refinement the bootstrap error would be smaller than the interval the
  crossing is actually known to lie in. Needs Agama. ~40 min.

`mc_finiteN_scaling.py [W0 SFE]` -> `mc_finiteN_scaling.csv`
  Survival probability vs N at fixed SFE, six values of N over two decades, 150
  realizations each with binomial errors. Run for W0 = 3 (SFE 0.30) and W0 = 9
  (SFE 0.045); pick an SFE inside the stochastic zone or the scan is
  uninformative. Needs Agama. ~20 min per configuration.

`mc_threshold_convergence.py [N] [ref_csv]` -> `mc_bound_fractions_n<N>.csv`
  Re-measures the threshold at a higher N on the same grid. Result: tripling N
  shifts the threshold by <1e-4 (within the bootstrap error, opposite signs at
  W0 = 3 and 6) while the transition width falls as N^-1/2, so the quoted
  thresholds are converged. Needs Agama; ~4.7 s per realization, so restrict it
  to the two concentrations rather than the whole grid.
