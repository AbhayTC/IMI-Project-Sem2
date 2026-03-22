"""
================================================================================
  Informatics-Driven Design of High-Performance Polymers for Satellite Protection
  A Comprehensive QSPR Pipeline for Thermal Endurance & Dielectric Stability
================================================================================
  Author  : Senior Materials Informatics Researcher
  Domain  : Polymer Science | Aerospace Materials | Machine Learning
================================================================================

TASK 7 ▶ Soft Outlier Physics Validity Check

  Examines the top and bottom 5% of samples by Tg and Dk and verifies that
  extreme values are physically consistent with the underlying morphological
  and processing features.

  Physical rules checked (Tg-axis):
  ───────────────────────────────────
  LOW Tg  (< mean − σ)  MUST have:
    ✓ FreeVolumeFraction   > polymer mean  (loose packing → soft / mobile)
    ✓ ChainRigidityIndex   < polymer mean  (flexible chains → low Tg)
    ✓ ThermalDiffusivity   > polymer mean  (fast heat dissipation → quench proxy)

  HIGH Tg (> mean + σ)  MUST have:
    ✓ FreeVolumeFraction   < polymer mean  (dense packing → stiff)
    ✓ DegreeOfCrystallinity > polymer mean (ordered phase → raises Tg)
    ✓ ThermalDiffusivity   < polymer mean  (slow cooling → annealing proxy)

  Physical rules checked (Dk-axis):
  ────────────────────────────────────
  LOW Dk  (< mean − σ)  MUST have:
    ✓ DielectricPolarizability < polymer mean  (fewer polarisable units)
    ✓ FreeVolumeFraction       > polymer mean  (air gaps lower permittivity)

  HIGH Dk (> mean + σ)  MUST have:
    ✓ DielectricPolarizability > polymer mean  (more polarisable units)
    ✓ DegreeOfCrystallinity    > polymer mean  (denser crystal lattice)

  NOTE: CoolingRate is not a direct feature. ThermalDiffusivity is used as a
  processing proxy: higher ThermalDiffusivity → faster heat dissipation →
  faster effective quenching rate during processing.

  Outputs: physics_check_report.pkl  (dict with pass/fail counts per rule)
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"

POLYMERS = ["Polyimide", "PEEK", "PTFE"]
POLY_COLOR = {
    "Polyimide": "\033[94m",
    "PEEK"     : "\033[92m",
    "PTFE"     : "\033[93m",
}

# ──────────────────────────────────────────────────────────────────────────────
#  PHYSICS RULES DEFINITION
# ──────────────────────────────────────────────────────────────────────────────

# Each rule: (feature, direction, description)
#   direction = "above_mean" → sample feature must be above polymer mean
#               "below_mean" → sample feature must be below polymer mean
TG_LOW_RULES = [
    ("FreeVolumeFraction",    "above_mean",
     "Loose packing → high FVF expected for mobile, low-Tg chains"),
    ("ChainRigidityIndex",    "below_mean",
     "Flexible chains → low rigidity expected for low Tg"),
    ("ThermalDiffusivity",    "above_mean",
     "Fast heat dissipation → quench proxy → amorphous / low Tg"),
]

TG_HIGH_RULES = [
    ("FreeVolumeFraction",    "below_mean",
     "Dense packing → low FVF expected for stiff, high-Tg chains"),
    ("DegreeOfCrystallinity", "above_mean",
     "Ordered crystal phase → raises Tg via restricted segmental motion"),
    ("ThermalDiffusivity",    "below_mean",
     "Slow cooling → annealing proxy → crystalline / high Tg"),
]

DK_LOW_RULES = [
    ("DielectricPolarizability", "below_mean",
     "Fewer polarisable units → lower dielectric constant"),
    ("FreeVolumeFraction",       "above_mean",
     "Air gaps in free volume dilute permittivity → lower Dk"),
]

DK_HIGH_RULES = [
    ("DielectricPolarizability", "above_mean",
     "More polarisable units → higher dielectric constant"),
    ("DegreeOfCrystallinity",    "above_mean",
     "Denser crystal lattice → higher Dk via increased polarisation"),
]

# ──────────────────────────────────────────────────────────────────────────────
#  CORE CHECK FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _check_rule(sample_val: float, poly_mean: float,
                direction: str) -> bool:
    """Return True if the sample satisfies the physical rule."""
    if direction == "above_mean":
        return sample_val > poly_mean
    elif direction == "below_mean":
        return sample_val < poly_mean
    raise ValueError(f"Unknown direction: {direction}")


def check_target_rules(df: pd.DataFrame,
                       polymer: str,
                       target_col: str,
                       percentile_pct: float,
                       tail: str,
                       rules: list) -> dict:
    """
    Check physics rules for the extreme tail of a target distribution.

    Parameters
    ----------
    tail : 'low' or 'high'
    rules : list of (feature, direction, description)

    Returns a dict with pass counts, fail counts, and per-rule details.
    """
    sub  = df[df["Polymer"] == polymer].copy()
    tgt  = sub[target_col]

    # Identify extreme samples
    if tail == "low":
        threshold = tgt.quantile(percentile_pct / 100)
        extreme   = sub[tgt <= threshold]
    else:
        threshold = tgt.quantile(1 - percentile_pct / 100)
        extreme   = sub[tgt >= threshold]

    if len(extreme) == 0:
        return {"n_samples": 0, "rules": [], "overall_pass_rate": np.nan}

    # Compute polymer means from ALL samples (not just extremes)
    poly_means = {feat: sub[feat].mean() for feat, *_ in rules}

    rule_results = []
    for feat, direction, desc in rules:
        if feat not in sub.columns:
            rule_results.append({
                "feature": feat, "direction": direction,
                "description": desc, "pass_rate": np.nan,
                "n_pass": 0, "n_total": len(extreme),
                "status": "MISSING",
            })
            continue

        poly_mean = poly_means[feat]
        passes    = extreme[feat].apply(
            lambda v: _check_rule(v, poly_mean, direction)
        )
        n_pass    = int(passes.sum())
        n_total   = len(extreme)
        pass_rate = n_pass / n_total if n_total > 0 else np.nan

        rule_results.append({
            "feature"    : feat,
            "direction"  : direction,
            "description": desc,
            "poly_mean"  : poly_mean,
            "extreme_mean": float(extreme[feat].mean()),
            "n_pass"     : n_pass,
            "n_total"    : n_total,
            "pass_rate"  : pass_rate,
            "status"     : "PASS" if pass_rate >= 0.60 else
                           ("WARN" if pass_rate >= 0.40 else "FAIL"),
        })

    overall = np.nanmean([r["pass_rate"] for r in rule_results
                          if r["status"] != "MISSING"])

    return {
        "polymer"        : polymer,
        "target"         : target_col,
        "tail"           : tail,
        "n_extreme"      : len(extreme),
        "threshold"      : float(threshold),
        "percentile_pct" : percentile_pct,
        "rules"          : rule_results,
        "overall_pass_rate": float(overall) if not np.isnan(overall) else np.nan,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  REPORTING
# ──────────────────────────────────────────────────────────────────────────────

def _status_color(status: str) -> str:
    return {
        "PASS"   : GREEN,
        "WARN"   : YELLOW,
        "FAIL"   : RED,
        "MISSING": YELLOW,
    }.get(status, RESET)


def _status_icon(status: str) -> str:
    return {"PASS": "✔", "WARN": "~", "FAIL": "✗", "MISSING": "?"}.get(status, "?")


def print_banner(text: str, char: str = "═", width: int = 72) -> None:
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{BOLD}{text}{RESET}"
          f"{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")


def print_check_result(result: dict) -> None:
    """Print one tail check result in a readable format."""
    p      = result["polymer"]
    target = "Tg" if result["target"] == "Tg_degC" else "Dk"
    tail   = result["tail"].upper()
    n      = result["n_extreme"]
    thr    = result["threshold"]
    pct    = result["percentile_pct"]
    orate  = result["overall_pass_rate"]
    col    = POLY_COLOR.get(p, "")

    # Determine overall status color
    if np.isnan(orate):
        orate_str = "N/A"
        orate_col = YELLOW
    elif orate >= 0.60:
        orate_str = f"{orate*100:.1f}%"
        orate_col = GREEN
    elif orate >= 0.40:
        orate_str = f"{orate*100:.1f}%"
        orate_col = YELLOW
    else:
        orate_str = f"{orate*100:.1f}%"
        orate_col = RED

    print(f"\n  {col}{BOLD}{p}{RESET}  │  {target} {tail} tail  "
          f"(bottom {pct}%, threshold={thr:.2f}, n={n})")
    print(f"  {'─'*68}")
    print(f"  {'Feature':<28}  {'Direction':<13}  "
          f"{'Poly mean':>10}  {'Ext mean':>9}  {'Pass':>6}  {'Status':>6}")
    print(f"  {'─'*28}  {'─'*13}  {'─'*10}  {'─'*9}  {'─'*6}  {'─'*6}")

    for r in result["rules"]:
        sc = _status_color(r["status"])
        ic = _status_icon(r["status"])
        dir_label = ("↑ above mean" if r["direction"] == "above_mean"
                     else "↓ below mean")
        pm  = f"{r.get('poly_mean',  float('nan')):.4f}"
        em  = f"{r.get('extreme_mean', float('nan')):.4f}"
        pct_s = (f"{r['pass_rate']*100:.1f}%" if not np.isnan(r["pass_rate"])
                 else "N/A")
        print(f"  {r['feature']:<28}  {dir_label:<13}  "
              f"{pm:>10}  {em:>9}  {pct_s:>6}  "
              f"{sc}{BOLD}{ic} {r['status']}{RESET}")

    print(f"  {'─'*68}")
    print(f"  Overall pass rate: {orate_col}{BOLD}{orate_str}{RESET}")


def print_global_summary(all_results: list) -> None:
    """Print a compact one-line-per-polymer-tail summary table."""
    print_banner("TASK 7 — PHYSICS VALIDITY GLOBAL SUMMARY", width=72)

    print(f"\n  {'Polymer':<12}  {'Target':>6}  {'Tail':>6}  "
          f"{'N samp':>7}  {'Pass rate':>10}  {'Verdict':>8}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*10}  {'─'*8}")

    polymer_verdicts = {p: [] for p in POLYMERS}

    for r in all_results:
        p      = r["polymer"]
        target = "Tg" if r["target"] == "Tg_degC" else "Dk"
        tail   = r["tail"].upper()
        n      = r["n_extreme"]
        orate  = r["overall_pass_rate"]
        col    = POLY_COLOR.get(p, "")

        if np.isnan(orate):
            verdict = "N/A"
            vc = YELLOW
        elif orate >= 0.60:
            verdict = "VALID"
            vc = GREEN
        elif orate >= 0.40:
            verdict = "WARN"
            vc = YELLOW
        else:
            verdict = "FAIL"
            vc = RED

        polymer_verdicts[p].append(verdict)
        orate_s = f"{orate*100:.1f}%" if not np.isnan(orate) else "N/A"
        print(f"  {col}{BOLD}{p:<12}{RESET}  {target:>6}  {tail:>6}  "
              f"{n:>7}  {orate_s:>10}  {vc}{BOLD}{verdict:>8}{RESET}")

    # Overall verdict per polymer
    print(f"\n  {'─'*72}")
    print(f"  Per-polymer physics confidence:")
    for p, verdicts in polymer_verdicts.items():
        n_valid = verdicts.count("VALID")
        n_total = len(verdicts)
        col = POLY_COLOR.get(p, "")
        confidence = "HIGH" if n_valid == n_total else (
                     "MEDIUM" if n_valid >= n_total // 2 else "LOW")
        cc = GREEN if confidence == "HIGH" else (YELLOW if confidence == "MEDIUM" else RED)
        print(f"    {col}{BOLD}{p:<12}{RESET}: {n_valid}/{n_total} checks VALID "
              f"→ physics confidence = {cc}{BOLD}{confidence}{RESET}")

    print(f"\n  ℹ  PASS threshold ≥ 60% of extreme samples satisfy each rule")
    print(f"  ℹ  CoolingRate absent — ThermalDiffusivity used as processing proxy")
    print(f"  ℹ  Rules probe top/bottom {all_results[0]['percentile_pct']}% "
          f"samples by target value\n")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_banner(
        "TASK 7 — SOFT OUTLIER PHYSICS VALIDITY CHECK",
        char="█", width=72
    )
    print(f"\n  Verifying synthetic data follows real polymer physics ...")
    print(f"  Examining top & bottom 5% of samples by Tg and Dk.\n")

    # Load dataset
    with open("master_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    print(f"  ✔  Loaded: master_dataset.pkl  ({len(df)} rows)\n")

    PERCENTILE = 5.0   # examine bottom/top 5%

    all_results = []
    for polymer in POLYMERS:
        col = POLY_COLOR.get(polymer, "")
        print(f"  Checking {col}{BOLD}{polymer}{RESET} ...")

        # Tg — low tail
        r = check_target_rules(df, polymer, "Tg_degC", PERCENTILE,
                                "low",  TG_LOW_RULES)
        all_results.append(r)
        print_check_result(r)

        # Tg — high tail
        r = check_target_rules(df, polymer, "Tg_degC", PERCENTILE,
                                "high", TG_HIGH_RULES)
        all_results.append(r)
        print_check_result(r)

        # Dk — low tail
        r = check_target_rules(df, polymer, "Dk_1GHz", PERCENTILE,
                                "low",  DK_LOW_RULES)
        all_results.append(r)
        print_check_result(r)

        # Dk — high tail
        r = check_target_rules(df, polymer, "Dk_1GHz", PERCENTILE,
                                "high", DK_HIGH_RULES)
        all_results.append(r)
        print_check_result(r)

    # Global summary
    print_global_summary(all_results)

    # Save report
    with open("physics_check_report.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print(f"  💾  Saved: physics_check_report.pkl\n")
