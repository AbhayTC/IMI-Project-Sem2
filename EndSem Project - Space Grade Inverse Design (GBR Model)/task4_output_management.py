"""
================================================================================
  TASK 4 ▶ Output Management  (Revised — v2)

  CHANGES IN THIS VERSION (5 requirements addressed):

  1. OUTPUT FILE CONSOLIDATION
     Exactly three CSV files are produced:
       reference_leaderboard.csv    — best sample per reference polymer family,
                                      ranked by desirability_score.
       input_polymer_results.csv    — top-10 user-input samples, ranked by
                                      desirability_score, with physics
                                      consistency columns embedded.
       morgan_fingerprint_master.csv — left-join of 64-bit ECFP4 data (Task 5)
                                       with performance scores and target values.
     All three files include SMILES and a `rank` column.

  2. PHYSICS CONSISTENCY INTEGRATION (Task 7 → Task 4)
     For every row destined for input_polymer_results.csv the six physics
     consistency scores (Tg_high, Tg_low, Dk_low, Dk_high, Rad_high, Rad_low)
     are computed inline using the same sigmoid rule engine as task7 and appended
     as columns.  This lets you instantly verify whether a high-scoring design is
     chemically logical or a potential model hallucination.

  3. LEO-PRIORITY DESIRABILITY WEIGHTING
     compute_desirability_score() is re-weighted for Satellite Protection:
       Radiation Endurance  40 %  (↑ higher is better)
       Glass Transition Tg  40 %  (↑ higher is better; sigmoid at 200 °C)
       Dielectric Constant  20 %  (↓ lower is better)
     Old weights were 0.30·Tg + 0.25·Dk + 0.15·TML + 0.30·Rad.

  4. TOP-5 FEATURES PER BLOCK IN FINAL CSVs
     resolve_feature_cols() now respects the export_config.pkl produced by
     task6, which ranks features within each block by mean within-class
     variance.  If export_config.pkl is absent a built-in fallback selects the
     top-5 per block by that same ranking computed on the fly from master_df.
     Only those 15 features (5 structural + 5 latent + 5 physics) appear in the
     consolidated CSVs — keeping leaderboards focused and scannable.

  5. MORGAN FINGERPRINT SAMPLE_ID PREFIXING (Task 5 → Task 4)
     When building morgan_fingerprint_master.csv the Sample_ID column is
     inspected: reference polymer rows are left as-is (prefix already set by
     Task 1 per-polymer convention, e.g. "Po", "PE", "PT", "CE", "FP", "PB");
     user-input rows are given the prefix "USR_" if not already present.
     desirability_score is included in the merged file so the chemical bit
     pattern of top-ranked designs is immediately visible.

  Inputs : master_dataset.pkl
           morgan_fingerprints.pkl   (from task5)
           export_config.pkl         (from task6, optional)
           user_polymer.pkl          (optional)
  Outputs: outputs/reference_leaderboard.csv
           outputs/input_polymer_results.csv
           outputs/morgan_fingerprint_master.csv
================================================================================
"""
import warnings, math, pickle, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"

TARGETS = ["Tg_degC", "Dk_1GHz", "RadiationDose_MGy"]
TARGET_LABELS = {
    "Tg_degC"          : "Tg (°C)",
    "Dk_1GHz"          : "Dk",
    "RadiationDose_MGy": "Rad (MGy)",
}

STRUCTURAL_FEATURE_NAMES = [
    "MolWt","HeavyAtomMolWt","ExactMolWt","NumHeavyAtoms","NumRotatableBonds",
    "NumRings","NumAromaticRings","NumAliphaticRings","RingCount","FractionCSP3",
    "NumHDonors","NumHAcceptors","TPSA","MolLogP","MolMR","LabuteASA",
    "PEOE_VSA1","PEOE_VSA2","PEOE_VSA3","PEOE_VSA4","SMR_VSA1","SMR_VSA2",
    "SMR_VSA3","SlogP_VSA1","SlogP_VSA2","SlogP_VSA3","NumValenceElectrons",
    "NumRadicalElectrons","fr_C_O","fr_NH0","fr_NH1","fr_ArN","fr_Ar_COO",
    "fr_ether","fr_ketone","fr_imide","fr_amide","HallKierAlpha","Kappa1","Kappa2",
]
LATENT_FEATURE_NAMES  = [f"polyBERT_dim_{i+1:02d}" for i in range(40)]
PHYSICS_FEATURE_NAMES = [
    "DegreeOfCrystallinity","CrystallinePhaseContent","AmorphousPhaseContent",
    "FreeVolumeFraction","ChainRigidityIndex","SegmentalMobility",
    "ThermalExpansionCoeff","HeatCapacity_Cp","ThermalDiffusivity","GlassyModulus",
    "DielectricPolarizability","ElectronicPolarizability","IonicPolarizability",
    "OrientationalPolarizability","DipoleMomentRepeat","CurieWeissConstant",
    "CrosslinkingDensity","EntanglementMolWt","ContourLengthPerUnit",
    "PersistenceLength","CharacteristicRatio","ChainFlexibilityParam","Mw_kDa",
    "Mn_kDa","PolyDisersityIndex","ZAverageMolWt","ViscosityAverageMolWt",
    "NumberAverageDPn","LamellaeThickness_nm","SpheruliteRadius_um",
    "CrystalThickness_nm","TieChainsPerArea","InterfacialThickness_nm",
    "MicrostructureOrder","PermittivityRealPart","PermittivityImaginaryPart",
    "TanDeltaDielectric","YoungModulus_GPa","TensileStrength_MPa",
    "ElongationBreak_pct",
]
MORGAN_FEATURE_NAMES = [f"morgan_fp_{i:02d}" for i in range(64)]

# Number of top features to keep per block in consolidated CSVs (Requirement 4)
TOP_N_PER_BLOCK = 5


# ──────────────────────────────────────────────────────────────────────────────
#  REQUIREMENT 3 — LEO-PRIORITY DESIRABILITY SCORE
#  Weights: Tg 40 % | Rad 40 % | Dk 20 %
# ──────────────────────────────────────────────────────────────────────────────

def compute_desirability_score(row: pd.Series) -> float:
    """
    Composite LEO-priority desirability score [0–1].

    Tg  (higher is better, weight 0.40):
        Sigmoid centred at 200 °C; full score at ~350 °C.
    Rad (higher is better, weight 0.40):
        Linear: 1.0 at Rad ≥ 40 MGy, 0.0 at Rad ≤ 10 MGy.
    Dk  (lower  is better, weight 0.20):
        Linear: 1.0 at Dk ≤ 2.0, 0.0 at Dk ≥ 5.0.

    NaN values contribute 0.5 (neutral) to the weighted sum.
    """
    def safe(val, default=0.5):
        return default if (val is None or
                           (isinstance(val, float) and np.isnan(val))) else float(val)

    tg  = safe(row.get("Tg_degC",           np.nan))
    dk  = safe(row.get("Dk_1GHz",           np.nan))
    rad = safe(row.get("RadiationDose_MGy", np.nan))

    d_tg  = 1.0 / (1.0 + np.exp(-0.02 * (tg  - 200.0)))
    d_dk  = float(np.clip((5.0 - dk)   / (5.0  - 2.0),  0.0, 1.0))
    d_rad = float(np.clip((rad - 10.0) / (40.0 - 10.0), 0.0, 1.0))

    return round(0.40 * d_tg + 0.20 * d_dk + 0.40 * d_rad, 6)


def add_desirability_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with desirability_score and rank columns added."""
    df = df.copy()
    mask_real = df["is_dummy"] == False if "is_dummy" in df.columns else pd.Series(True, index=df.index)
    df.loc[mask_real, "desirability_score"] = df[mask_real].apply(
        compute_desirability_score, axis=1
    )
    if "is_dummy" in df.columns:
        df.loc[~mask_real, "desirability_score"] = np.nan
    df = df.sort_values("desirability_score", ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  REQUIREMENT 4 — TOP-5 FEATURES PER BLOCK
#  Ranked by mean within-class variance (same logic as task6)
# ──────────────────────────────────────────────────────────────────────────────

def _rank_by_within_class_variance(df: pd.DataFrame,
                                   feature_cols: list,
                                   polymers: list,
                                   top_n: int) -> list:
    """Return top_n features from feature_cols ranked by mean within-class variance."""
    available = [c for c in feature_cols if c in df.columns]
    scores = {}
    for col in available:
        per_poly_var = [df[df["Polymer"] == p][col].var() for p in polymers
                        if p in df["Polymer"].values]
        scores[col] = float(np.nanmean(per_poly_var)) if per_poly_var else 0.0
    ranked = sorted(available, key=lambda c: scores[c], reverse=True)
    return ranked[:top_n]


def resolve_feature_cols(master_df: pd.DataFrame,
                         config: dict | None) -> dict:
    """
    Return {block_name: [col_list]} for the top-5 features per block.

    If export_config.pkl is present (and was built with task6's within-class
    variance ranking), use the first TOP_N_PER_BLOCK cols from each block.
    Otherwise compute the ranking on the fly from master_df.

    Returns a dict with keys 'structural', 'latent', 'physics'.
    """
    # Polymer list for within-class variance (reference only, no dummies)
    if "is_reference" in master_df.columns:
        ref_df = master_df[master_df["is_reference"] == True]
    else:
        ref_df = master_df
    if "is_dummy" in ref_df.columns:
        ref_df = ref_df[ref_df["is_dummy"] == False]
    polymers = sorted(ref_df["Polymer"].dropna().unique().tolist())

    if config is not None:
        selected = config.get("selected_cols", {})
        result = {}
        for block, all_names in [("structural", STRUCTURAL_FEATURE_NAMES),
                                  ("latent",     LATENT_FEATURE_NAMES),
                                  ("physics",    PHYSICS_FEATURE_NAMES)]:
            cols = selected.get(block, all_names)
            result[block] = [c for c in cols if c in master_df.columns][:TOP_N_PER_BLOCK]
        return result

    # Fallback: compute on the fly
    result = {}
    for block, all_names in [("structural", STRUCTURAL_FEATURE_NAMES),
                              ("latent",     LATENT_FEATURE_NAMES),
                              ("physics",    PHYSICS_FEATURE_NAMES)]:
        result[block] = _rank_by_within_class_variance(
            master_df, all_names, polymers, TOP_N_PER_BLOCK
        )
    return result


def flat_feature_cols(block_cols: dict) -> list:
    """Flatten {block: [cols]} to a single ordered list."""
    return (block_cols.get("structural", []) +
            block_cols.get("latent", []) +
            block_cols.get("physics", []))


# ──────────────────────────────────────────────────────────────────────────────
#  REQUIREMENT 2 — PHYSICS CONSISTENCY SCORES (Task 7 rule engine, inline)
# ──────────────────────────────────────────────────────────────────────────────

# Rule definitions mirrored from task7_physics_check.py
_TG_HIGH_RULES = [
    ("ChainRigidityIndex",    "high", 1.0),
    ("NumAromaticRings",      "high", 0.8),
    ("DegreeOfCrystallinity", "high", 0.6),
    ("CrosslinkingDensity",   "high", 0.9),
    ("FreeVolumeFraction",    "low",  0.7),
]
_TG_LOW_RULES = [
    ("ChainFlexibilityParam", "high", 1.0),
    ("FractionCSP3",          "high", 0.8),
    ("FreeVolumeFraction",    "high", 0.7),
]
_DK_LOW_RULES = [
    ("FreeVolumeFraction",    "high", 1.0),
    ("DipoleMomentRepeat",    "low",  1.0),
    ("FractionCSP3",          "high", 0.6),
]
_DK_HIGH_RULES = [
    ("DielectricPolarizability", "high", 1.0),
    ("IonicPolarizability",      "high", 0.8),
    ("DipoleMomentRepeat",       "high", 1.0),
]
_RAD_HIGH_RULES = [
    ("NumAromaticRings",      "high", 1.0),
    ("DegreeOfCrystallinity", "high", 0.8),
    ("CrosslinkingDensity",   "high", 0.9),
]
_RAD_LOW_RULES = [
    ("FreeVolumeFraction",    "high", 1.0),
    ("NumAromaticRings",      "low",  1.0),
    ("SegmentalMobility",     "high", 0.8),
]

_PHYSICS_RULE_SETS = {
    "phys_Tg_high" : _TG_HIGH_RULES,
    "phys_Tg_low"  : _TG_LOW_RULES,
    "phys_Dk_low"  : _DK_LOW_RULES,
    "phys_Dk_high" : _DK_HIGH_RULES,
    "phys_Rad_high": _RAD_HIGH_RULES,
    "phys_Rad_low" : _RAD_LOW_RULES,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-2.0 * x))


def _score_one_rule_set(row: pd.Series,
                        rules: list,
                        global_means: dict,
                        global_stds: dict) -> float:
    """Compute a single physics consistency score for one row."""
    total_weight = sum(w for _, _, w in rules)
    score = 0.0
    for feat, direction, weight in rules:
        if feat not in row.index:
            continue
        val = row[feat]
        if pd.isna(val):
            continue
        std = global_stds.get(feat, 1.0) or 1.0
        z   = (float(val) - global_means.get(feat, 0.0)) / std
        if direction == "low":
            z = -z
        score += weight * _sigmoid(z)
    return round(score / total_weight, 5) if total_weight > 0 else 0.0


def add_physics_consistency_columns(df: pd.DataFrame,
                                    global_means: dict,
                                    global_stds: dict) -> pd.DataFrame:
    """
    Append six physics consistency score columns to df (row-wise computation).
    Columns added:
        phys_Tg_high, phys_Tg_low,
        phys_Dk_low,  phys_Dk_high,
        phys_Rad_high, phys_Rad_low
    """
    df = df.copy()
    for col_name, rules in _PHYSICS_RULE_SETS.items():
        df[col_name] = df.apply(
            lambda row, r=rules: _score_one_rule_set(row, r, global_means, global_stds),
            axis=1
        )
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  REQUIREMENT 1 — THREE CONSOLIDATED CSV EXPORTS
# ──────────────────────────────────────────────────────────────────────────────

def _build_base_columns(master_df: pd.DataFrame,
                        block_cols: dict) -> list:
    """Ordered column list: id | flags | targets | score | top-5×3 features."""
    id_cols   = ["rank", "Polymer", "Sample_ID", "SMILES"]
    flag_cols = [c for c in ("is_reference", "is_dummy") if c in master_df.columns]
    tgt_cols  = [t for t in TARGETS if t in master_df.columns]
    feat_cols = flat_feature_cols(block_cols)
    return id_cols + flag_cols + tgt_cols + ["desirability_score"] + feat_cols


# ── 1A. reference_leaderboard.csv ─────────────────────────────────────────────

def export_reference_leaderboard(master_df: pd.DataFrame,
                                 block_cols: dict,
                                 out_dir: str) -> str:
    """
    Best single sample per reference polymer family, ranked by desirability_score.
    Includes SMILES, rank, all target columns.
    """
    if "is_reference" in master_df.columns:
        ref_df = master_df[master_df["is_reference"] == True].copy()
    else:
        ref_df = master_df.copy()
    if "is_dummy" in ref_df.columns:
        ref_df = ref_df[ref_df["is_dummy"] == False]

    scored = add_desirability_column(ref_df)

    # Keep best row per polymer
    best_rows = (scored.groupby("Polymer", sort=False)
                       .apply(lambda g: g.nlargest(1, "desirability_score"))
                       .reset_index(drop=True))
    best_rows = best_rows.sort_values("desirability_score", ascending=False).reset_index(drop=True)
    best_rows["rank"] = range(1, len(best_rows) + 1)

    export_cols = _build_base_columns(master_df, block_cols)
    out_df = best_rows[[c for c in export_cols if c in best_rows.columns]]

    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, "reference_leaderboard.csv")
    out_df.to_csv(fpath, index=False)

    print(f"  💾  reference_leaderboard.csv  —  {len(out_df)} rows  "
          f"(1 best sample per polymer family)")
    for _, row in out_df.iterrows():
        print(f"    #{int(row['rank'])}  {row['Polymer']:<22}  "
              f"score={row.get('desirability_score', 0):.4f}  "
              f"Tg={row.get('Tg_degC', 0):.1f}°C  "
              f"Dk={row.get('Dk_1GHz', 0):.3f}  "
              f"Rad={row.get('RadiationDose_MGy', 0):.2f} MGy")
    return fpath


# ── 1B. input_polymer_results.csv ─────────────────────────────────────────────

def export_input_polymer_results(master_df: pd.DataFrame,
                                 block_cols: dict,
                                 user_meta: dict | None,
                                 global_means: dict,
                                 global_stds: dict,
                                 out_dir: str,
                                 top_n: int = 10) -> str | None:
    """
    Top-10 user-input samples ranked by desirability_score.
    Physics consistency scores (Requirement 2) are embedded as columns.
    """
    if user_meta is None:
        print(f"  ℹ  No user polymer found — input_polymer_results.csv skipped.")
        return None

    name = user_meta["name"]
    user_rows = master_df[master_df["Polymer"] == name].copy()
    if "is_dummy" in user_rows.columns:
        user_rows = user_rows[user_rows["is_dummy"] == False]

    if len(user_rows) == 0:
        print(f"  {YELLOW}⚠  No rows for '{name}' in dataset — "
              f"input_polymer_results.csv skipped.{RESET}")
        return None

    # Score, rank, trim to top-N
    scored = add_desirability_column(user_rows)
    top_df = scored.nlargest(top_n, "desirability_score").reset_index(drop=True)
    top_df["rank"] = range(1, len(top_df) + 1)

    # Attach physics consistency columns (Requirement 2)
    top_df = add_physics_consistency_columns(top_df, global_means, global_stds)

    physics_cols = list(_PHYSICS_RULE_SETS.keys())
    base_cols    = _build_base_columns(master_df, block_cols)
    export_cols  = base_cols + [c for c in physics_cols if c not in base_cols]
    out_df = top_df[[c for c in export_cols if c in top_df.columns]]

    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, "input_polymer_results.csv")
    out_df.to_csv(fpath, index=False)

    print(f"  💾  input_polymer_results.csv  —  {len(out_df)} rows  "
          f"(top {top_n} user samples + physics consistency scores)")
    print(f"\n  Top {min(top_n, len(out_df))} samples for '{name}':")
    print(f"  {'#':<4}  {'Score':>7}  {'Tg':>7}  {'Dk':>6}  {'Rad':>8}  "
          f"{'Tg↑':>6}  {'Dk↓':>6}  {'Rad↑':>6}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*8}  "
          f"{'─'*6}  {'─'*6}  {'─'*6}")
    for _, row in out_df.iterrows():
        print(f"  {int(row['rank']):<4}  "
              f"{row.get('desirability_score', 0):>7.4f}  "
              f"{row.get('Tg_degC', 0):>7.1f}  "
              f"{row.get('Dk_1GHz', 0):>6.3f}  "
              f"{row.get('RadiationDose_MGy', 0):>8.2f}  "
              f"{row.get('phys_Tg_high', 0):>6.3f}  "
              f"{row.get('phys_Dk_low', 0):>6.3f}  "
              f"{row.get('phys_Rad_high', 0):>6.3f}")
    print(f"\n  {CYAN}Physics columns guide:{RESET}")
    print(f"  phys_Tg_high  High = rigid backbone → high Tg expected  (good if Tg is high)")
    print(f"  phys_Tg_low   High = flexible chain → low Tg expected")
    print(f"  phys_Dk_low   High = low-polarisation structure → low Dk expected  (good if Dk is low)")
    print(f"  phys_Dk_high  High = polar structure → high Dk expected")
    print(f"  phys_Rad_high High = aromatic/crosslinked → radiation-resistant  (good if Rad is high)")
    print(f"  phys_Rad_low  High = aliphatic/mobile → radiation-sensitive")
    return fpath


# ── 1C. morgan_fingerprint_master.csv ─────────────────────────────────────────

def export_morgan_fingerprint_master(master_df: pd.DataFrame,
                                     morgan_df: pd.DataFrame | None,
                                     user_meta: dict | None,
                                     out_dir: str) -> str | None:
    """
    Left-join morgan fingerprint bits (Task 5) with performance scores and
    target values from master_df, keyed on [Polymer, Sample_ID].

    Requirement 5: user-input Sample_IDs are prefixed 'USR_' if not already.
    desirability_score is included so chemical bit patterns of top designs are visible.
    """
    if morgan_df is None:
        print(f"  ℹ  morgan_fingerprints.pkl not loaded — "
              f"morgan_fingerprint_master.csv skipped.")
        return None

    # Score the full master dataset
    if "is_dummy" in master_df.columns:
        real_df = master_df[master_df["is_dummy"] == False].copy()
    else:
        real_df = master_df.copy()

    scored = add_desirability_column(real_df)
    # Carry forward the global rank
    scored.rename(columns={"rank": "global_rank"}, inplace=True)

    # Requirement 5: prefix USR_ for user polymer Sample_IDs
    if user_meta:
        uname = user_meta["name"]
        is_user_mask = scored["Polymer"] == uname
        needs_prefix = is_user_mask & ~scored["Sample_ID"].str.startswith("USR_")
        scored.loc[needs_prefix, "Sample_ID"] = (
            "USR_" + scored.loc[needs_prefix, "Sample_ID"].astype(str)
        )
        # Mirror the same prefix into morgan_df so the join key matches
        morgan_user_mask = morgan_df["Polymer"] == uname
        needs_prefix_m   = morgan_user_mask & ~morgan_df["Sample_ID"].str.startswith("USR_")
        morgan_df = morgan_df.copy()
        morgan_df.loc[needs_prefix_m, "Sample_ID"] = (
            "USR_" + morgan_df.loc[needs_prefix_m, "Sample_ID"].astype(str)
        )

    # Columns to carry from master: identifiers + targets + score
    id_cols  = ["Polymer", "Sample_ID", "SMILES"]
    tgt_cols = [t for t in TARGETS if t in scored.columns]
    keep_cols = id_cols + tgt_cols + ["desirability_score", "global_rank"]
    master_slim = scored[[c for c in keep_cols if c in scored.columns]]

    # Left-join fingerprints
    merged = master_slim.merge(
        morgan_df[["Polymer", "Sample_ID"] + MORGAN_FEATURE_NAMES],
        on=["Polymer", "Sample_ID"],
        how="left",
    )
    merged = merged.sort_values("desirability_score", ascending=False).reset_index(drop=True)
    merged.insert(0, "rank", range(1, len(merged) + 1))

    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, "morgan_fingerprint_master.csv")
    merged.to_csv(fpath, index=False)

    n_matched = merged[MORGAN_FEATURE_NAMES[0]].notna().sum()
    print(f"  💾  morgan_fingerprint_master.csv  —  {len(merged)} rows  "
          f"({n_matched} with fingerprint bits, sorted best-first)")
    print(f"       Join keys: [Polymer, Sample_ID]  |  "
          f"Bit columns: morgan_fp_00 … morgan_fp_63")
    return fpath


# ──────────────────────────────────────────────────────────────────────────────
#  STATISTICS REPORT (unchanged, displayed to console only)
# ──────────────────────────────────────────────────────────────────────────────

def print_statistics(master_df: pd.DataFrame, user_meta: dict | None) -> None:
    real_df = master_df[master_df["is_dummy"] == False] \
              if "is_dummy" in master_df.columns else master_df

    print(f"\n  {'─'*78}")
    print(f"  {BOLD}Per-Polymer Target Statistics{RESET}")
    print(f"  {'─'*78}\n")

    header  = f"  {'Polymer':<22}  {'n':>5}"
    div_row = f"  {'─'*22}  {'─'*5}"
    for tgt in TARGETS:
        header  += f"  {TARGET_LABELS[tgt]:>14} ± σ"
        div_row += f"  {'─'*18}"
    print(header)
    print(div_row)

    for poly in sorted(real_df["Polymer"].unique()):
        rows    = real_df[real_df["Polymer"] == poly]
        is_user = user_meta and poly == user_meta["name"]
        color   = YELLOW if is_user else ""
        rst     = RESET  if is_user else ""
        line    = f"  {color}{poly:<22}{rst}  {len(rows):>5}"
        for tgt in TARGETS:
            if tgt in rows.columns:
                line += f"  {rows[tgt].mean():>10.3f} ± {rows[tgt].std():.3f}  "
            else:
                line += f"  {'N/A':>18}"
        if is_user:
            line += f"  {YELLOW}⚠ est.{RESET}"
        print(line)
    print()


# ──────────────────────────────────────────────────────────────────────────────
#  BANNER UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def print_banner(text: str, char: str = "═", width: int = 85) -> None:
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{BOLD}{text}{RESET}"
          f"{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_banner("TASK 4 — OUTPUT MANAGEMENT  (v2: 3-CSV Consolidation)", width=85)

    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load master dataset ──────────────────────────────────────────────────
    print("\n  Loading master_dataset.pkl ...")
    with open("master_dataset.pkl", "rb") as f:
        master_df = pickle.load(f)
    print(f"  ✔  {len(master_df)} rows loaded")

    # ── Load user polymer ────────────────────────────────────────────────────
    user_meta = None
    if os.path.exists("user_polymer.pkl"):
        with open("user_polymer.pkl", "rb") as f:
            user_meta = pickle.load(f)
        print(f"  ✔  User polymer: {BOLD}{user_meta['name']}{RESET}")

    # ── Load export config (task6) ───────────────────────────────────────────
    config = None
    if os.path.exists("export_config.pkl"):
        with open("export_config.pkl", "rb") as f:
            config = pickle.load(f)
        print(f"  ✔  export_config.pkl loaded  "
              f"({config.get('total_features', '?')} features configured by task6)")
    else:
        print(f"  ℹ  No export_config.pkl — computing within-class variance ranking on the fly.")

    # ── Load Morgan fingerprints (task5) ─────────────────────────────────────
    morgan_df = None
    if os.path.exists("morgan_fingerprints.pkl"):
        with open("morgan_fingerprints.pkl", "rb") as f:
            morgan_df = pickle.load(f)
        print(f"  ✔  morgan_fingerprints.pkl loaded  ({len(morgan_df)} rows)")
    else:
        print(f"  {YELLOW}⚠  morgan_fingerprints.pkl not found.  "
              f"Run task5_morgan_fingerprint.py first.{RESET}")

    # ── Resolve top-5 feature cols per block (Requirement 4) ─────────────────
    block_cols = resolve_feature_cols(master_df, config)
    total_feat = sum(len(v) for v in block_cols.values())
    print(f"\n  Feature selection (top {TOP_N_PER_BLOCK} per block, "
          f"{total_feat} total):")
    for block, cols in block_cols.items():
        print(f"    {block:<12}: {cols}")

    # ── Global means/stds for physics consistency scoring (Requirement 2) ────
    if "is_reference" in master_df.columns:
        base_for_stats = master_df[master_df["is_reference"] == True]
    else:
        base_for_stats = master_df
    if "is_dummy" in base_for_stats.columns:
        base_for_stats = base_for_stats[base_for_stats["is_dummy"] == False]
    num_df      = base_for_stats.select_dtypes(include=[np.number])
    global_means = num_df.mean().to_dict()
    global_stds  = num_df.std().to_dict()

    # ── Statistics to console ────────────────────────────────────────────────
    print_statistics(master_df, user_meta)

    # ── Export 1 of 3: reference_leaderboard.csv ─────────────────────────────
    print_banner("EXPORT 1/3 — REFERENCE LEADERBOARD", width=85)
    export_reference_leaderboard(master_df, block_cols, OUT_DIR)

    # ── Export 2 of 3: input_polymer_results.csv ─────────────────────────────
    print_banner("EXPORT 2/3 — INPUT POLYMER RESULTS (top-10 + physics checks)", width=85)
    export_input_polymer_results(
        master_df, block_cols, user_meta, global_means, global_stds, OUT_DIR, top_n=10
    )

    # ── Export 3 of 3: morgan_fingerprint_master.csv ──────────────────────────
    print_banner("EXPORT 3/3 — MORGAN FINGERPRINT MASTER", width=85)
    export_morgan_fingerprint_master(master_df, morgan_df, user_meta, OUT_DIR)

    print(f"\n  ✔  Task 4 complete.  Outputs in: {os.path.abspath(OUT_DIR)}/\n")
    print(f"  Files produced:")
    for fname in ("reference_leaderboard.csv",
                  "input_polymer_results.csv",
                  "morgan_fingerprint_master.csv"):
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {GREEN}✔{RESET}  {fname:<38}  {size_kb:.1f} KB")
        else:
            print(f"    {YELLOW}–{RESET}  {fname:<38}  (skipped — prerequisite missing)")
    print()
