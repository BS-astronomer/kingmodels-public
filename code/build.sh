#!/usr/bin/env bash
# build.sh -- single-command driver for the King-SFE analysis pipeline.
# Runs every stage in REPRODUCTION_GUIDE.md
# Sect. 9's order, from ITS OWN directory (the scripts use sibling-module
# imports and write to './'), and FAILS LOUD (set -e, plus an explicit
# artifact check after each stage) instead of silently continuing past a
# missing declared output -- unlike rebuild_paper_figures.py's own
# maybe()/skip behaviour, which is deliberate there (presentation figures
# with genuinely optional inputs) and left as is.
#
# Usage:
#   ./build.sh                 # everything that needs no Agama (default)
#   ./build.sh --with-agama    # also stages 05/06 -- 05 is ~10-60 min,
#                               # 06 is HOURS even after the R01 analytic-
#                               # inversion fix (it now runs a real N_EPS,
#                               # not the old N_EPS=600 shortcut); the
#                               # committed mc_bound_fractions*.csv already
#                               # cover what stage 05's default grid needs,
#                               # so --with-agama is for regenerating those
#                               # from scratch or extending the dense scan.
#
# Requires activating an environment with the packages in requirements.txt
# first (agama only if --with-agama; see that file's notes on installing it).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD"
WITH_AGAMA=0
[ "${1:-}" = "--with-agama" ] && WITH_AGAMA=1

MANIFEST="$ROOT/build_manifest.txt"
: > "$MANIFEST"
STAGE_START=$(date +%s)

run_stage () {
    local dir="$1" script="$2"; shift 2
    local want=("$@")             # expected output files, relative to $dir
    echo "=== $dir/$script ==="
    ( cd "$ROOT/$dir" && python "$script" )
    local missing=0
    for f in "${want[@]}"; do
        if [ ! -e "$ROOT/$dir/$f" ]; then
            echo "  MISSING declared output: $dir/$f" >&2
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        echo "build.sh: $dir/$script did not produce all declared outputs -- stopping." >&2
        exit 1
    fi
    for f in "${want[@]}"; do
        printf '%-55s %s\n' "$dir/$f" "$(date -r "$ROOT/$dir/$f" '+%Y-%m-%d %H:%M:%S')" >> "$MANIFEST"
    done
}

run_stage 01_king_sfe_analysis        king_sfe_analysis.py        results_summary.csv
run_stage 02_king_sfe_mechanism       king_sfe_mechanism.py       mechanism_summary.csv
run_stage 03_multi_profile_analysis   multi_profile_analysis.py   truncation_summary.csv
run_stage 04_adams_df_boundfraction   adams_df_boundfraction.py   df_threshold_summary.csv df_threshold_published_families.csv
run_stage 07_df_sensitivity_checks    df_sensitivity_checks.py    df_threshold_sensitivity.csv

if [ "$WITH_AGAMA" -eq 1 ]; then
    run_stage 05_mc_bound_fraction_agama  mc_bound_fraction_agama.py  mc_bound_fractions.csv
    run_stage 05_mc_bound_fraction_agama  mc_published_families.py   mc_thresholds_published.csv
    run_stage 06_df_mc_thresholds_dense   df_mc_thresholds_dense.py  dense_thresholds.csv
else
    echo "=== skipping 05/06 (Agama, long) -- rerun with --with-agama, or rely on the committed"
    echo "    05_mc_bound_fraction_agama/mc_bound_fractions*.csv used below ==="
fi

# --- stage 08: presentation figures. rebuild_paper_figures.py + the two
# build_*.py producers read only from cwd (per their own docstrings), so
# stage every input they declare, from wherever this run actually produced
# it -- fail loud (not "skip") if a REQUIRED one (needed by build_*.py
# itself, as opposed to rebuild_paper_figures.py's own genuinely-optional
# ms-figure inputs) is missing.
D8="$ROOT/08_rebuild_paper_figures"
for pair in \
    "01_king_sfe_analysis/results_summary.csv" \
    "02_king_sfe_mechanism/mechanism_summary.csv" \
    "03_multi_profile_analysis/truncation_summary.csv" \
    "04_adams_df_boundfraction/df_threshold_summary.csv" \
    "04_adams_df_boundfraction/df_threshold_published_families.csv" \
    "05_mc_bound_fraction_agama/mc_bound_fractions.csv" \
    "05_mc_bound_fraction_agama/mc_bound_fractions_N100k.csv" \
    "06_df_mc_thresholds_dense/dense_thresholds.csv" ; do
    src="$ROOT/$pair"
    [ -f "$src" ] && cp -f "$src" "$D8/" || echo "  (optional: $pair not found, skipping)"
done

run_stage 08_rebuild_paper_figures build_analytic_prediction_table.py analytic_prediction_table.csv
run_stage 08_rebuild_paper_figures build_envelope_slopes.py           envelope_slopes.csv
cp -f "$D8/analytic_prediction_table.csv" "$D8/envelope_slopes.csv" "$D8/" 2>/dev/null || true
run_stage 08_rebuild_paper_figures rebuild_paper_figures.py           fig1_density_profiles_rh.pdf

ELAPSED=$(( $(date +%s) - STAGE_START ))
echo
echo "build.sh complete in ${ELAPSED}s. Manifest: $MANIFEST"
cat "$MANIFEST"
