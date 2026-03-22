"""
================================================================================
  Informatics-Driven Design of High-Performance Polymers for Satellite Protection
  A Comprehensive QSPR Pipeline for Thermal Endurance & Dielectric Stability
================================================================================
  Author  : Senior Materials Informatics Researcher
  Domain  : Polymer Science | Aerospace Materials | Machine Learning
================================================================================

TASK 6 ▶ Interactive Export Configuration

  Lets the user choose how many features to export per block in the final CSVs.
  Features within each block are ranked by MEAN WITHIN-CLASS VARIANCE so that
  the most informative features are always selected first.

  Blocks available:
    Block 1 — Structural descriptors   (max 40, RDKit / mock)
    Block 2 — Latent / polyBERT dims   (max 40, simulated BERT)
    Block 3 — Physics / morphological  (max 40, Gaussian priors)
    Block 4 — Morgan fingerprint bits  (max 64, ECFP4)

  Outputs: export_config.pkl  (read by task4_output_management.py)
           If this file is absent, task4 uses all features as defaults.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import warnings, pickle, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

RESET = "\033[0m"
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"

# ──────────────────────────────────────────────────────────────────────────────
#  FEATURE NAME LISTS  (must match task1 / task4 / task5)
# ──────────────────────────────────────────────────────────────────────────────
STRUCTURAL_FEATURE_NAMES = [
    "MolWt",           "HeavyAtomMolWt",  "ExactMolWt",      "NumHeavyAtoms",
    "NumRotatableBonds","NumRings",        "NumAromaticRings","NumAliphaticRings",
    "RingCount",       "FractionCSP3",    "NumHDonors",      "NumHAcceptors",
    "TPSA",            "MolLogP",         "MolMR",           "LabuteASA",
    "PEOE_VSA1",       "PEOE_VSA2",       "PEOE_VSA3",       "PEOE_VSA4",
    "SMR_VSA1",        "SMR_VSA2",        "SMR_VSA3",        "SlogP_VSA1",
    "SlogP_VSA2",      "SlogP_VSA3",      "NumValenceElectrons","NumRadicalElectrons",
    "fr_C_O",          "fr_NH0",          "fr_NH1",          "fr_ArN",
    "fr_Ar_COO",       "fr_ether",        "fr_ketone",       "fr_imide",
    "fr_amide",        "HallKierAlpha",   "Kappa1",          "Kappa2",
]
LATENT_FEATURE_NAMES   = [f"polyBERT_dim_{i+1:02d}" for i in range(40)]
PHYSICS_FEATURE_NAMES  = [
    "DegreeOfCrystallinity",   "CrystallinePhaseContent",  "AmorphousPhaseContent",
    "FreeVolumeFraction",      "ChainRigidityIndex",        "SegmentalMobility",
    "ThermalExpansionCoeff",   "HeatCapacity_Cp",           "ThermalDiffusivity",
    "GlassyModulus",           "DielectricPolarizability",  "ElectronicPolarizability",
    "IonicPolarizability",     "OrientationalPolarizability","DipoleMomentRepeat",
    "CurieWeissConstant",      "CrosslinkingDensity",       "EntanglementMolWt",
    "ContourLengthPerUnit",    "PersistenceLength",         "CharacteristicRatio",
    "ChainFlexibilityParam",   "Mw_kDa",                   "Mn_kDa",
    "PolyDisersityIndex",      "ZAverageMolWt",             "ViscosityAverageMolWt",
    "NumberAverageDPn",        "LamellaeThickness_nm",      "SpheruliteRadius_um",
    "CrystalThickness_nm",     "TieChainsPerArea",          "InterfacialThickness_nm",
    "MicrostructureOrder",     "PermittivityRealPart",      "PermittivityImaginaryPart",
    "TanDeltaDielectric",      "YoungModulus_GPa",          "TensileStrength_MPa",
    "ElongationBreak_pct",
]
MORGAN_FEATURE_NAMES   = [f"morgan_fp_{i:02d}" for i in range(64)]

BLOCK_DEFS = {
    "structural": {
        "label"   : "Structural descriptors  (RDKit / mock)",
        "cols"    : STRUCTURAL_FEATURE_NAMES,
        "max"     : 40,
        "default" : 40,
        "color"   : "\033[94m",
    },
    "latent": {
        "label"   : "Latent / polyBERT dims  (simulated BERT)",
        "cols"    : LATENT_FEATURE_NAMES,
        "max"     : 40,
        "default" : 40,
        "color"   : "\033[92m",
    },
    "physics": {
        "label"   : "Physics / morphological (Gaussian priors)",
        "cols"    : PHYSICS_FEATURE_NAMES,
        "max"     : 40,
        "default" : 40,
        "color"   : "\033[93m",
    },
    "morgan": {
        "label"   : "Morgan fingerprint bits (ECFP4, 64-bit)",
        "cols"    : MORGAN_FEATURE_NAMES,
        "max"     : 64,
        "default" : 64,
        "color"   : "\033[95m",
    },
}

POLYMERS = ["Polyimide", "PEEK", "PTFE"]

# ──────────────────────────────────────────────────────────────────────────────
#  FEATURE RANKING — top-N by mean within-class variance
# ──────────────────────────────────────────────────────────────────────────────

def rank_features_by_within_class_variance(df: pd.DataFrame,
                                           feature_cols: list) -> list:
    """
    Return feature_cols reordered from highest to lowest mean within-class
    variance. The first N features in the returned list are the most
    informative for within-class prediction.

    Within-class variance = variance of the feature within a single polymer
    class (ignoring between-class differences, which inflate global variance
    and can mask true predictive signal).
    """
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  {YELLOW}⚠  {len(missing)} features not found in dataset "
              f"(will be skipped): {missing[:3]}...{RESET}")

    scores = {}
    for col in available:
        per_poly_var = [df[df["Polymer"] == p][col].var() for p in POLYMERS]
        scores[col]  = float(np.nanmean(per_poly_var))

    ranked = sorted(available, key=lambda c: scores[c], reverse=True)
    return ranked


def rank_all_blocks(df: pd.DataFrame,
                    morgan_df: pd.DataFrame) -> dict:
    """Rank features in every block. Returns {block_key: [ranked_col_list]}."""
    ranked = {}

    # Structural, latent, physics from master_dataset
    for key in ("structural", "latent", "physics"):
        ranked[key] = rank_features_by_within_class_variance(
            df, BLOCK_DEFS[key]["cols"]
        )

    # Morgan from morgan_df (merge polymer labels in)
    if morgan_df is not None:
        mfp_cols   = BLOCK_DEFS["morgan"]["cols"]
        morgan_with_poly = morgan_df[["Polymer"] + mfp_cols].drop_duplicates("Polymer")
        # Compute within-class variance using the merged enriched view
        scores = {}
        for col in mfp_cols:
            if col not in morgan_df.columns:
                continue
            per_poly_var = [
                morgan_df[morgan_df["Polymer"] == p][col].var()
                for p in POLYMERS
            ]
            scores[col] = float(np.nanmean(per_poly_var))
        ranked["morgan"] = sorted(
            [c for c in mfp_cols if c in scores],
            key=lambda c: scores[c], reverse=True
        )
    else:
        ranked["morgan"] = BLOCK_DEFS["morgan"]["cols"]

    return ranked


# ──────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE PROMPT
# ──────────────────────────────────────────────────────────────────────────────

def _ask_int(prompt: str, lo: int, hi: int, default: int) -> int:
    """Prompt for an integer in [lo, hi]; returns default on empty input."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if lo <= val <= hi:
                return val
            print(f"  Please enter a number between {lo} and {hi}.")
        except ValueError:
            print("  Invalid input — please enter a whole number.")


def run_interactive_config(df: pd.DataFrame,
                           morgan_df: pd.DataFrame,
                           ranked: dict) -> dict:
    """
    Ask the user how many features to export per block.
    Returns a config dict.
    """
    print(f"\n{'─'*72}")
    print(f"  {BOLD}{CYAN}INTERACTIVE EXPORT CONFIGURATION{RESET}")
    print(f"{'─'*72}")
    print(f"\n  Features are ranked by mean within-class variance — the top N are")
    print(f"  the most informative for within-polymer prediction.\n")

    # Show block summaries
    for key, bdef in BLOCK_DEFS.items():
        cols_present = len([c for c in bdef["cols"] if c in ranked.get(key, [])])
        print(f"  {bdef['color']}{BOLD}Block: {bdef['label']}{RESET}")
        print(f"         Available: {cols_present} features  |  "
              f"Default export: {bdef['default']}  |  Max: {bdef['max']}")
    print()

    chosen = {}
    for key, bdef in BLOCK_DEFS.items():
        n_avail = len(ranked.get(key, []))
        limit   = min(bdef["max"], n_avail)
        default = min(bdef["default"], limit)
        n = _ask_int(
            f"  {bdef['color']}{BOLD}{bdef['label'][:38]}{RESET} — "
            f"how many to export? [1–{limit}, default={default}]: ",
            lo=1, hi=limit, default=default
        )
        chosen[key] = n

    return chosen


# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_config(chosen: dict, ranked: dict) -> dict:
    """
    Build the full config dict from user choices.
    selected_cols: the actual column names to export (in ranked order).
    """
    config = {
        "chosen_n"    : chosen,
        "selected_cols": {},
        "block_defs"  : {k: {"label": v["label"], "max": v["max"],
                              "default": v["default"]}
                         for k, v in BLOCK_DEFS.items()},
        "total_features": 0,
    }
    for key, n in chosen.items():
        cols = ranked[key][:n]
        config["selected_cols"][key] = cols

    config["total_features"] = sum(len(v) for v in config["selected_cols"].values())
    config["total_columns"]  = config["total_features"] + 2   # + 2 targets
    return config


def print_config_summary(config: dict) -> None:
    """Print a formatted summary of the chosen configuration."""
    print(f"\n{'─'*72}")
    print(f"  {BOLD}{GREEN}EXPORT CONFIGURATION SUMMARY{RESET}")
    print(f"{'─'*72}\n")

    cum = 1
    for key, cols in config["selected_cols"].items():
        bdef   = BLOCK_DEFS[key]
        n      = len(cols)
        end    = cum + n - 1
        pct    = n / bdef["max"] * 100
        print(f"  {bdef['color']}{BOLD}{bdef['label'][:42]}{RESET}")
        print(f"    Exporting : {n} / {bdef['max']} features  ({pct:.0f}%)")
        print(f"    CSV cols  : {cum} – {end}")
        print(f"    Top feat  : {cols[0]}  …  {cols[-1] if n > 1 else ''}")
        cum += n
        print()

    tf = config["total_features"]
    tc = config["total_columns"]
    print(f"  {'─'*40}")
    print(f"  Total features  : {tf}")
    print(f"  + 2 target cols : Tg_degC, Dk_1GHz")
    print(f"  Total CSV cols  : {tc}")
    print(f"  {'─'*40}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*72}")
    print(f"  {BOLD}TASK 6 — INTERACTIVE EXPORT CONFIGURATION{RESET}")
    print(f"{'='*72}\n")

    # Load master dataset
    print("  Loading master_dataset.pkl ...")
    with open("master_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    print(f"  ✔  master_dataset.pkl loaded  ({len(df)} rows)\n")

    # Load Morgan fingerprints
    morgan_df = None
    try:
        with open("morgan_fingerprints.pkl", "rb") as f:
            morgan_df = pickle.load(f)
        print(f"  ✔  morgan_fingerprints.pkl loaded  ({len(morgan_df)} rows)\n")
    except FileNotFoundError:
        print(f"  {YELLOW}⚠  morgan_fingerprints.pkl not found — "
              f"Morgan block will use default ordering.{RESET}\n")

    # Rank all blocks by within-class variance
    print("  Ranking features by within-class variance ...")
    ranked = rank_all_blocks(df, morgan_df)
    for key, cols in ranked.items():
        print(f"    {key:12}: top feature = {cols[0] if cols else 'N/A'}")
    print()

    # Interactive prompts
    chosen = run_interactive_config(df, morgan_df, ranked)

    # Build and display config
    config = build_config(chosen, ranked)
    print_config_summary(config)

    # Save
    with open("export_config.pkl", "wb") as f:
        pickle.dump(config, f)
    print(f"  💾  Saved: export_config.pkl  →  (read by task4_output_management.py)")
    print(f"\n  ℹ  Run task4_output_management.py to generate CSVs with this config.")
    print(f"     To reset to defaults, delete export_config.pkl before running task4.\n")
