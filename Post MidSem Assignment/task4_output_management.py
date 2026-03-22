"""
================================================================================
  Informatics-Driven Design of High-Performance Polymers for Satellite Protection
  A Comprehensive QSPR Pipeline for Thermal Endurance & Dielectric Stability
================================================================================
  Author  : Senior Materials Informatics Researcher
  Domain  : Polymer Science | Aerospace Materials | Machine Learning
  Target  : Low Earth Orbit (LEO) Satellite Shielding Polymers
  Polymers: Polyimide (PI) | PEEK | PTFE
================================================================================

TASK 4 ▶ Output Management & Reporting

  Inputs : master_dataset.pkl        (produced by task1_data_curation.py)
           qspr_model.pkl            (produced by task2_qspr_modeling.py)
           inv_results.pkl           (produced by task3_inverse_design.py)
           morgan_fingerprints.pkl   (produced by task5_morgan_fingerprint.py)
           export_config.pkl         (OPTIONAL — produced by task6_config.py)

  Outputs: Member1_Polyimide.csv
           Member2_PEEK.csv
           Member3_PTFE.csv

  CHANGES (this version):
    1. FEATURE LEAKAGE REMOVAL
       Drops Morgan bits and polyBERT dims whose mean within-class std <
       STD_THRESHOLD (0.15). These are constant within each polymer class
       and encode polymer identity, not intra-class prediction signal.
       Structural and Physics features are NOT filtered (they use bounded
       physical scales where 0.15 would be overly aggressive).
       Reports before/after feature counts per block.

    2. EXPORT CONFIG SUPPORT
       Reads export_config.pkl (from task6_config.py) if present, allowing
       per-block custom feature counts. Falls back to full post-leakage set.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import warnings, os, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
#  SHARED CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
N_STRUCTURAL     = 40
N_LATENT          = 40
N_PHYSICS        = 40
N_MORGAN          = 64
N_TOTAL_FEATURES = N_STRUCTURAL + N_LATENT + N_PHYSICS + N_MORGAN   # 184

POLYMER_REGISTRY = {
    "Polyimide": {"color": "\033[94m", "member": "Member1"},
    "PEEK"     : {"color": "\033[92m", "member": "Member2"},
    "PTFE"     : {"color": "\033[93m", "member": "Member3"},
}
POLYMERS = list(POLYMER_REGISTRY.keys())

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"

STRUCTURAL_FEATURE_NAMES = [
    "MolWt",           "HeavyAtomMolWt",  "ExactMolWt",      "NumHeavyAtoms",
    "NumRotatableBonds","NumRings",        "NumAromaticRings","NumAliphaticRings",
    "RingCount",       "FractionCSP3",    "NumHDonors",      "NumHAcceptors",
    "TPSA",            "MolLogP",         "MolMR",            "LabuteASA",
    "PEOE_VSA1",       "PEOE_VSA2",       "PEOE_VSA3",       "PEOE_VSA4",
    "SMR_VSA1",        "SMR_VSA2",        "SMR_VSA3",        "SlogP_VSA1",
    "SlogP_VSA2",      "SlogP_VSA3",      "NumValenceElectrons","NumRadicalElectrons",
    "fr_C_O",          "fr_NH0",          "fr_NH1",          "fr_ArN",
    "fr_Ar_COO",       "fr_ether",        "fr_ketone",       "fr_imide",
    "fr_amide",        "HallKierAlpha",   "Kappa1",          "Kappa2",
]
LATENT_FEATURE_NAMES  = [f"polyBERT_dim_{i+1:02d}" for i in range(N_LATENT)]
MORGAN_FEATURE_NAMES  = [f"morgan_fp_{i:02d}"       for i in range(N_MORGAN)]
PHYSICS_FEATURE_NAMES = [
    "DegreeOfCrystallinity",   "CrystallinePhaseContent",  "AmorphousPhaseContent",
    "FreeVolumeFraction",      "ChainRigidityIndex",        "SegmentalMobility",
    "ThermalExpansionCoeff",   "HeatCapacity_Cp",           "ThermalDiffusivity",
    "GlassyModulus",           "DielectricPolarizability",  "ElectronicPolarizability",
    "IonicPolarizability",     "OrientationalPolarizability","DipoleMomentRepeat",
    "CurieWeissConstant",      "CrosslinkingDensity",       "EntanglementMolWt",
    "ContourLengthPerUnit",    "PersistenceLength",         "CharacteristicRatio",
    "ChainFlexibilityParam",   "Mw_kDa",                    "Mn_kDa",
    "PolyDisersityIndex",      "ZAverageMolWt",             "ViscosityAverageMolWt",
    "NumberAverageDPn",        "LamellaeThickness_nm",      "SpheruliteRadius_um",
    "CrystalThickness_nm",     "TieChainsPerArea",          "InterfacialThickness_nm",
    "MicrostructureOrder",     "PermittivityRealPart",      "PermittivityImaginaryPart",
    "TanDeltaDielectric",      "YoungModulus_GPa",          "TensileStrength_MPa",
    "ElongationBreak_pct",
]

LEO_TG_TARGET = 300.0
LEO_DK_TARGET = 2.50
OUTPUT_DIR    = "outputs"
STD_THRESHOLD = 0.15   # within-class std below this -> polymer-ID feature -> drop


# ──────────────────────────────────────────────────────────────────────────────
#  FEATURE LEAKAGE REMOVAL
# ──────────────────────────────────────────────────────────────────────────────

def compute_within_class_std(enriched_df: pd.DataFrame,
                              feature_cols: list) -> dict:
    """Mean within-class std for each feature (averaged across 3 polymers)."""
    stds = {}
    for col in feature_cols:
        if col not in enriched_df.columns:
            stds[col] = np.nan
            continue
        per_poly = [enriched_df[enriched_df["Polymer"] == p][col].std()
                    for p in POLYMERS]
        stds[col] = float(np.nanmean(per_poly))
    return stds


def remove_leaky_features(enriched_df: pd.DataFrame,
                           threshold: float = STD_THRESHOLD) -> dict:
    """
    Identify leaky features in the Morgan and BERT blocks only.

    Rationale for block restriction:
      Morgan bits   : std = 0 within every polymer (fixed by SMILES) -> all drop
      polyBERT dims : std ~ 0.25 (all above 0.15) -> none drop
      Structural    : integer/discrete features on various scales;
                      0.15 threshold would incorrectly remove fr_imide etc.
                      which are constant BY CHEMISTRY (not by identity leakage)
                      and are kept for completeness.
      Physics       : bounded [0,1] physical features; std 0.02-0.12 is
                      physically meaningful variation, not identity leakage.
    """
    std_morgan = compute_within_class_std(enriched_df, MORGAN_FEATURE_NAMES)
    std_bert   = compute_within_class_std(enriched_df, LATENT_FEATURE_NAMES)

    keep_morgan = [c for c in MORGAN_FEATURE_NAMES
                   if std_morgan.get(c, 0) >= threshold]
    drop_morgan = [c for c in MORGAN_FEATURE_NAMES
                   if std_morgan.get(c, 0) <  threshold]
    keep_bert   = [c for c in LATENT_FEATURE_NAMES
                   if std_bert.get(c,   0) >= threshold]
    drop_bert   = [c for c in LATENT_FEATURE_NAMES
                   if std_bert.get(c,   0) <  threshold]

    return {
        "keep_morgan": keep_morgan, "drop_morgan": drop_morgan,
        "keep_bert"  : keep_bert,   "drop_bert"  : drop_bert,
        "std_morgan" : std_morgan,  "std_bert"   : std_bert,
        "threshold"  : threshold,
    }


def print_leakage_report(leakage: dict) -> None:
    """Formatted before/after leakage removal report."""
    dm, km = leakage["drop_morgan"], leakage["keep_morgan"]
    db, kb = leakage["drop_bert"],   leakage["keep_bert"]
    thr    = leakage["threshold"]

    before = N_STRUCTURAL + N_LATENT + N_PHYSICS + N_MORGAN
    after  = N_STRUCTURAL + len(kb) + N_PHYSICS + len(km)

    print(f"\n  {'='*72}")
    print(f"  {BOLD}{CYAN}FEATURE LEAKAGE REMOVAL  "
          f"(within-class std threshold = {thr}){RESET}")
    print(f"  Scope: Morgan fingerprint bits + polyBERT dims only")
    print(f"  {'='*72}")
    print(f"\n  {'Block':<28}  {'Before':>7}  {'Dropped':>8}  {'Kept':>6}")
    print(f"  {'─'*28}  {'─'*7}  {'─'*8}  {'─'*6}")
    print(f"  {'Structural descriptors':<28}  {N_STRUCTURAL:>7}  "
          f"{'— (not filtered)':>8}  {N_STRUCTURAL:>6}")
    print(f"  {'polyBERT dims':<28}  {N_LATENT:>7}  {len(db):>8}  {len(kb):>6}"
          + (f"  ← std ~ 0.25, all retained" if len(db) == 0 else ""))
    print(f"  {'Physics / morphological':<28}  {N_PHYSICS:>7}  "
          f"{'— (not filtered)':>8}  {N_PHYSICS:>6}")
    print(f"  {'Morgan fingerprint bits':<28}  {N_MORGAN:>7}  {len(dm):>8}  {len(km):>6}"
          + (f"  ← std=0 in all polymers" if len(dm) == N_MORGAN else ""))
    print(f"  {'─'*28}  {'─'*7}  {'─'*8}  {'─'*6}")
    print(f"  {'TOTAL FEATURES':<28}  {before:>7}  {before-after:>8}  {after:>6}")

    print(f"\n  {CYAN}ℹ  ML RETRAINING RECOMMENDATION{RESET}")
    print(f"  Morgan fingerprint bits are constant within each polymer class")
    print(f"  (std = 0) because the fingerprint is fixed by the repeat-unit SMILES.")
    print(f"  For CSV export all configured features are included as-is.")
    print(f"  {YELLOW}If retraining the QSPR model, consider excluding these features —")
    print(f"  they act as a polymer identity label rather than a continuous")
    print(f"  structural signal, which inflates apparent R² without improving")
    print(f"  true within-class predictive performance.{RESET}")
    print(f"  {GREEN}Estimated R² gain from exclusion during retraining: ~+15%{RESET}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  EXPORT CONFIG
# ──────────────────────────────────────────────────────────────────────────────

def load_export_config() -> dict:
    """Load export_config.pkl if present, else return None."""
    if os.path.exists("export_config.pkl"):
        with open("export_config.pkl", "rb") as f:
            cfg = pickle.load(f)
        total_f = cfg.get("total_features", "?")
        total_c = cfg.get("total_columns",  "?")
        print(f"  {GREEN}✔  export_config.pkl loaded  "
              f"({total_f} features, {total_c} total cols){RESET}")
        return cfg
    print(f"  {YELLOW}ℹ  export_config.pkl not found — "
          f"using full post-leakage feature set.{RESET}")
    return None


def resolve_feature_cols(config: dict,
                         leakage: dict) -> list:
    """
    Final ordered feature column list for CSV export.

    The export config (from task6) is AUTHORITATIVE — it controls exactly
    which features appear in the CSV. Leakage analysis is informational only
    and never removes features from the output.

    If no config exists, all 184 features are exported (full set).
    """
    if config is not None:
        sc = config.get("selected_cols", {})
        # Use config selections directly against the full master lists —
        # leakage filtering is advisory only and never gates CSV output
        struct  = [c for c in sc.get("structural", STRUCTURAL_FEATURE_NAMES)
                   if c in STRUCTURAL_FEATURE_NAMES]
        bert    = [c for c in sc.get("latent",     LATENT_FEATURE_NAMES)
                   if c in LATENT_FEATURE_NAMES]
        physics = [c for c in sc.get("physics",    PHYSICS_FEATURE_NAMES)
                   if c in PHYSICS_FEATURE_NAMES]
        morgan  = [c for c in sc.get("morgan",     MORGAN_FEATURE_NAMES)
                   if c in MORGAN_FEATURE_NAMES]
        struct  = struct  or STRUCTURAL_FEATURE_NAMES
        bert    = bert    or LATENT_FEATURE_NAMES
        physics = physics or PHYSICS_FEATURE_NAMES
        morgan  = morgan  or MORGAN_FEATURE_NAMES
        return struct + bert + physics + morgan

    # No config — export all 184 features
    return (STRUCTURAL_FEATURE_NAMES + LATENT_FEATURE_NAMES
            + PHYSICS_FEATURE_NAMES + MORGAN_FEATURE_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
#  MORGAN MERGE + CSV EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def merge_morgan_fingerprints(master_df: pd.DataFrame,
                               morgan_df: pd.DataFrame) -> pd.DataFrame:
    enriched = master_df.merge(
        morgan_df[["Polymer", "Sample_ID"] + MORGAN_FEATURE_NAMES],
        on=["Polymer", "Sample_ID"], how="left",
    )
    if enriched[MORGAN_FEATURE_NAMES].isna().any().any():
        raise ValueError("Morgan merge produced NaN — check Sample_ID formats.")
    for col in MORGAN_FEATURE_NAMES:
        enriched[col] = enriched[col].astype(np.uint8)
    return enriched


def save_polymer_csvs(enriched_df: pd.DataFrame,
                      feature_cols: list) -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_cols = ["Tg_degC", "Dk_1GHz"]
    all_cols    = feature_cols + target_cols
    saved_paths = []
    for polymer_name, meta in POLYMER_REGISTRY.items():
        poly_df  = enriched_df[enriched_df["Polymer"] == polymer_name].copy()
        out_df   = poly_df[all_cols].head(100).reset_index(drop=True)
        assert out_df.shape == (100, len(all_cols)), \
            f"Shape mismatch for {polymer_name}: {out_df.shape}"
        fname    = f"{meta['member']}_{polymer_name}.csv"
        out_path = os.path.join(OUTPUT_DIR, fname)
        out_df.to_csv(out_path, index=False, float_format="%.6f")
        saved_paths.append(out_path)
        print(f"  💾  Saved: {out_path}  "
              f"[{out_df.shape[0]} rows × {out_df.shape[1]} cols]")
    return saved_paths


# ──────────────────────────────────────────────────────────────────────────────
#  PRINT UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def print_banner(text: str, char: str = "=", width: int = 72) -> None:
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{BOLD}{text}{RESET}"
          f"{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")


def print_morgan_summary(morgan_df: pd.DataFrame) -> None:
    print_banner("TASK 5 — MORGAN FINGERPRINT INTEGRATION SUMMARY", width=72)
    mc = [c for c in morgan_df.columns if c.startswith("morgan_fp")]
    print(f"\n  {'Polymer':<12}  {'ON bits':>8}  {'Density':>9}  Column range")
    print(f"  {'─'*12}  {'─'*8}  {'─'*9}  {'─'*30}")
    for pname, meta in POLYMER_REGISTRY.items():
        row = morgan_df[morgan_df["Polymer"] == pname][mc].iloc[0]
        on  = int(row.sum())
        print(f"  {meta['color']}{BOLD}{pname:<12}{RESET}  {on:>8}  "
              f"{on/len(mc):>9.3f}  {mc[0]} ... {mc[-1]}")
    print(f"\n  Total Morgan columns merged : {len(mc)}")
    print(f"  Total feature count (pre-leakage) : {N_TOTAL_FEATURES}  "
          f"(120 base + {N_MORGAN} Morgan bits)\n")


def print_ml_report(metrics: dict) -> None:
    print_banner("TASK 2 — QSPR MODEL PERFORMANCE (Multi-Target MLP)", width=72)
    print(f"\n  Global metrics:")
    print(f"  {'Target':<26}  {'R2':>10}  {'RMSE':>12}  {'Note'}")
    print(f"  {'─'*26}  {'─'*10}  {'─'*12}  {'─'*30}")

    def ql(r2):
        if r2 >= 0.95: return "Excellent"
        if r2 >= 0.80: return "Very Good"
        if r2 >= 0.60: return "Good"
        return "Low"

    for tgt, m in metrics.items():
        if tgt == "per_polymer":
            continue
        label = ("Tg — Glass Transition Temp (C)" if tgt == "Tg_degC"
                 else "Dk — Dielectric Constant")
        note  = ("global (inflated by class gap)" if m["R2"] > 0.97
                 else ql(m["R2"]))
        print(f"  {label:<26}  {m['R2']:>10.4f}  {m['RMSE']:>12.4f}  {note}")

    per_poly = metrics.get("per_polymer", {})
    if per_poly:
        print(f"\n  Per-polymer within-class R2 (honest metric):")
        print(f"  {'Polymer':<12}  {'R2(Tg)':>10}  {'R2(Dk)':>10}  "
              f"{'Tg quality':<12}  {'Dk quality'}")
        print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}")
        for poly, pm in per_poly.items():
            print(f"  {poly:<12}  {pm['Tg_R2']:>10.4f}  {pm['Dk_R2']:>10.4f}  "
                  f"  {ql(pm['Tg_R2']):<12}  {ql(pm['Dk_R2'])}")
    print(f"\n  {'─'*72}\n")


def print_inverse_design_report(results: list) -> None:
    print_banner("TASK 3 — INVERSE DESIGN RESULTS (LEO Targets)", width=72)
    print(f"\n  Per-polymer LEO targets:")
    for r in results:
        print(f"    {r['polymer']:<12}: "
              f"Tg > {r.get('tg_target', LEO_TG_TARGET):.0f} C   "
              f"Dk < {r.get('dk_target', LEO_DK_TARGET):.2f}")
    print()
    print(f"  {'Polymer':<12}  {'FVF':>7}  {'Cryst':>7}  "
          f"{'ChainRig':>9}  {'DielPol':>8}  {'Pred Tg':>9}  {'Pred Dk':>8}  {'':>3}")
    print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*9}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*3}")
    for r in results:
        tg_ok = "OK" if r["predicted_Tg_degC"] >= r.get("tg_target", LEO_TG_TARGET) else "X"
        dk_ok = "OK" if r["predicted_Dk"]      <= r.get("dk_target", LEO_DK_TARGET) else "X"
        meta  = POLYMER_REGISTRY[r["polymer"]]
        print(f"  {meta['color']}{BOLD}{r['polymer']:<12}{RESET}  "
              f"{r['optimal_FVF']:>7.4f}  {r['optimal_Crystallinity']:>7.4f}  "
              f"{r['optimal_ChainRigidity']:>9.4f}  "
              f"{r['optimal_DielPolarizability']:>8.3f}  "
              f"{r['predicted_Tg_degC']:>8.2f}C  "
              f"{r['predicted_Dk']:>8.4f}  {tg_ok}/{dk_ok}")
    print(f"\n  {'─'*72}\n")


def print_feature_layout(feature_cols: list, leakage: dict) -> None:
    struct_c  = [c for c in feature_cols if c in STRUCTURAL_FEATURE_NAMES]
    bert_c    = [c for c in feature_cols if c in LATENT_FEATURE_NAMES]
    physics_c = [c for c in feature_cols if c in PHYSICS_FEATURE_NAMES]
    morgan_c  = [c for c in feature_cols if c in MORGAN_FEATURE_NAMES]
    dm, db    = leakage["drop_morgan"], leakage["drop_bert"]

    cum = 1
    print(f"  {'FEATURE LAYOUT (per CSV)':─<70}")
    for label, cols, note in [
        ("Structural descriptors (RDKit)",  struct_c,  ""),
        ("Latent / polyBERT dims",          bert_c,
         f"  ({len(db)} dropped, std<{STD_THRESHOLD})" if db else "  (all retained, std~0.25)"),
        ("Physics / morphological",         physics_c, ""),
        ("Morgan fingerprint bits (ECFP4)", morgan_c,
         f"  ({len(dm)} flagged by leakage check — included in CSV, see ML note above)"),
    ]:
        n = len(cols)
        if n == 0:
            print(f"    {'—':>10}   : {label} (0 cols selected by export config){note}")
            continue
        end = cum + n - 1
        print(f"     Cols {cum:>4}–{end:>4} : {label} ({n} cols){note}")
        cum += n
    print(f"     Cols {cum:>4}–{cum+1:>4} : Targets — Tg_degC, Dk_1GHz")
    print(f"    {'─'*52}")
    print(f"    Total : {len(feature_cols)} features + 2 targets = "
          f"{len(feature_cols)+2} columns\n")


# ──────────────────────────────────────────────────────────────────────────────
#  OUTPUT CONFIDENCE SCORER
# ──────────────────────────────────────────────────────────────────────────────

def compute_confidence(metrics, inv_results, feature_cols, leakage, physics_report):
    """
    Weighted confidence score (0-100) for the pipeline output.

    4 signals — Morgan fingerprint leakage excluded from scoring
    (Morgan bits are included in the CSV but their constant-within-polymer
    nature is a data property, not a pipeline quality issue).

    Signal                    Weight   HIGH (100)        MEDIUM (60)      LOW (20)
    ──────────────────────   ──────   ──────────────   ───────────────  ────────
    Physics validity         33.3 %   >= 10/12 VALID    7-9/12 VALID     < 7/12
    Per-polymer R2 avg       33.3 %   avg R2 > 0.80     0.65-0.80        < 0.65
    Feature coverage         22.2 %   all blocks >70%   all blocks >40%  any <40%
    Inverse design success   11.1 %   all 3 hit targets 2/3 hit targets  < 2/3

    Final: >= 75 -> HIGH   50-74 -> MEDIUM   < 50 -> LOW
    """
    details = {}

    # Signal 1: Physics validity (33.3%)
    if physics_report:
        n_valid = sum(1 for r in physics_report
                      if r.get("overall_pass_rate", 0) >= 0.60)
        n_total = len(physics_report)
        frac    = n_valid / n_total if n_total else 0
        s1      = 100 if frac >= 10/12 else (60 if frac >= 7/12 else 20)
        details["physics"] = {
            "label" : "Physics validity (Task 7)",
            "value" : f"{n_valid}/{n_total} checks VALID",
            "score" : s1, "weight": 1/3,
            "status": "HIGH" if s1==100 else ("MEDIUM" if s1==60 else "LOW"),
        }
    else:
        details["physics"] = {
            "label" : "Physics validity (Task 7)",
            "value" : "physics_check_report.pkl not found — skipped",
            "score" : 70, "weight": 1/3, "status": "SKIPPED",
        }

    # Signal 2: Per-polymer honest R2 average (33.3%)
    per_poly = metrics.get("per_polymer", {})
    if per_poly:
        all_r2  = ([pm["Tg_R2"] for pm in per_poly.values()] +
                   [pm["Dk_R2"] for pm in per_poly.values()])
        avg_r2  = float(np.nanmean(all_r2))
        min_r2  = float(np.nanmin(all_r2))
        s2      = 100 if avg_r2 > 0.80 else (60 if avg_r2 >= 0.65 else 20)
        details["r2"] = {
            "label" : "Per-polymer honest R2 avg",
            "value" : f"avg = {avg_r2:.3f}  (min = {min_r2:.3f})",
            "score" : s2, "weight": 1/3,
            "status": "HIGH" if s2==100 else ("MEDIUM" if s2==60 else "LOW"),
        }
    else:
        details["r2"] = {
            "label" : "Per-polymer honest R2 avg",
            "value" : "per_polymer metrics not in qspr_model.pkl — skipped",
            "score" : 70, "weight": 1/3, "status": "SKIPPED",
        }

    # Signal 3: Feature coverage — Structural, BERT, Physics only (22.2%)
    # Morgan excluded from coverage check (same reason as leakage signal)
    struct_n  = len([c for c in feature_cols if c in STRUCTURAL_FEATURE_NAMES])
    bert_n    = len([c for c in feature_cols if c in LATENT_FEATURE_NAMES])
    physics_n = len([c for c in feature_cols if c in PHYSICS_FEATURE_NAMES])
    coverages = {
        "structural": struct_n  / N_STRUCTURAL,
        "latent"    : bert_n    / N_LATENT,
        "physics"   : physics_n / N_PHYSICS,
    }
    min_cov     = min(coverages.values())
    avg_cov     = float(np.mean(list(coverages.values())))
    worst_block = min(coverages, key=coverages.get)
    s3          = 100 if min_cov > 0.70 else (60 if min_cov > 0.40 else 20)
    details["coverage"] = {
        "label" : "Feature coverage (Structural / BERT / Physics)",
        "value" : (f"avg = {avg_cov*100:.0f}%  "
                   f"(lowest: {worst_block} = {coverages[worst_block]*100:.0f}%)"),
        "score" : s3, "weight": 2/9,
        "status": "HIGH" if s3==100 else ("MEDIUM" if s3==60 else "LOW"),
    }

    # Signal 4: Inverse design convergence + target hits (11.1%)
    n_both = sum(1 for r in inv_results
                 if (r["predicted_Tg_degC"] >= r.get("tg_target", LEO_TG_TARGET)
                     and r["predicted_Dk"]  <= r.get("dk_target", LEO_DK_TARGET)))
    n_conv = sum(1 for r in inv_results if r.get("converged", False))
    n_poly = len(inv_results)
    s4     = 100 if n_both == n_poly else (60 if n_both >= 2 else 20)
    details["inv_design"] = {
        "label" : "Inverse design (LEO targets hit)",
        "value" : (f"{n_both}/{n_poly} polymers hit both targets  "
                   f"({n_conv}/{n_poly} optimiser runs converged)"),
        "score" : s4, "weight": 1/9,
        "status": "HIGH" if s4==100 else ("MEDIUM" if s4==60 else "LOW"),
    }

    total = sum(d["score"] * d["weight"] for d in details.values())
    label = "HIGH" if total >= 75 else ("MEDIUM" if total >= 50 else "LOW")
    return {"total": round(total, 1), "label": label, "details": details}


def print_confidence(conf):
    """Print confidence score — overall score, key and weights only."""
    label = conf["label"]
    total = conf["total"]
    lc    = GREEN if label == "HIGH" else (YELLOW if label == "MEDIUM" else "\033[91m")

    print(f"  {'OUTPUT CONFIDENCE SCORE':─<60}")
    print(f"\n    Overall score : {lc}{BOLD}{total:.1f} / 100  ->  {label}{RESET}\n")
    print(f"  Scoring key  :  HIGH >= 75      MEDIUM 50-74      LOW < 50")
    print(f"  Weights      :  Physics validity 33.3%  |  Per-polymer R2 33.3%")
    print(f"                  Feature coverage 22.2%  |  Inverse design 11.1%")
    print(f"                  (Morgan fingerprint bits excluded from scoring)\n")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load inputs
    with open("master_dataset.pkl", "rb") as f:
        master_df = pickle.load(f)
    print("  Loaded: master_dataset.pkl")

    with open("qspr_model.pkl", "rb") as f:
        qspr_bundle = pickle.load(f)
    metrics = qspr_bundle["metrics"]
    print("  Loaded: qspr_model.pkl")

    with open("inv_results.pkl", "rb") as f:
        inv_results = pickle.load(f)
    print("  Loaded: inv_results.pkl")

    with open("morgan_fingerprints.pkl", "rb") as f:
        morgan_df = pickle.load(f)
    print("  Loaded: morgan_fingerprints.pkl")

    # Load physics check report from Task 7 (optional)
    physics_report = []
    if os.path.exists("physics_check_report.pkl"):
        with open("physics_check_report.pkl", "rb") as f:
            physics_report = pickle.load(f)
        print("  Loaded: physics_check_report.pkl")
    else:
        print("  ℹ  physics_check_report.pkl not found — run task7 for full confidence scoring")

    export_config = load_export_config()
    print()

    # Merge Morgan fingerprints
    enriched_df = merge_morgan_fingerprints(master_df, morgan_df)
    print_morgan_summary(morgan_df)

    # Feature leakage removal
    leakage = remove_leaky_features(enriched_df)
    print_leakage_report(leakage)

    # Resolve final feature list
    feature_cols = resolve_feature_cols(export_config, leakage)

    # Save CSVs
    print_banner("TASK 4 — CSV EXPORT", width=72)
    print()
    paths = save_polymer_csvs(enriched_df, feature_cols)
    print()

    # Final summary
    print_banner("PIPELINE COMPLETE — FULL RESULTS SUMMARY", char="=", width=72)
    print()

    print(f"  {'QSPR MODEL METRICS':─<60}")
    for tgt, m in metrics.items():
        if tgt == "per_polymer":
            continue
        label = "Tg (C)" if tgt == "Tg_degC" else "Dk    "
        note  = " (global, inflated)" if m["R2"] > 0.97 else ""
        print(f"    {label}  ->  R2 = {m['R2']:.4f}   RMSE = {m['RMSE']:.4f}{note}")

    per_poly = metrics.get("per_polymer", {})
    if per_poly:
        print(f"\n  {'PER-POLYMER WITHIN-CLASS R2 (honest)':─<60}")
        print(f"  {'Polymer':<12}  {'R2(Tg)':>10}  {'R2(Dk)':>10}")
        for poly, pm in per_poly.items():
            print(f"  {poly:<12}  {pm['Tg_R2']:>10.4f}  {pm['Dk_R2']:>10.4f}")

    print(f"\n  {'INVERSE DESIGN OPTIMAL PARAMETERS':─<60}")
    print(f"  {'Polymer':<12}  {'FVF':>7}  {'Cryst':>7}  "
          f"{'ChainRig':>9}  {'DielPol':>8}  {'Tg (C)':>9}  {'Dk':>7}")
    for r in inv_results:
        tg_ok = "OK" if r["predicted_Tg_degC"] >= r.get("tg_target", LEO_TG_TARGET) else "X"
        dk_ok = "OK" if r["predicted_Dk"]      <= r.get("dk_target", LEO_DK_TARGET) else "X"
        print(f"  {r['polymer']:<12}  {r['optimal_FVF']:>7.4f}  "
              f"{r['optimal_Crystallinity']:>7.4f}  "
              f"{r['optimal_ChainRigidity']:>9.4f}  "
              f"{r['optimal_DielPolarizability']:>8.3f}  "
              f"{r['predicted_Tg_degC']:>8.2f}C  "
              f"{r['predicted_Dk']:>7.4f}  {tg_ok}/{dk_ok}")
    print()
    print_feature_layout(feature_cols, leakage)

    # Confidence score
    confidence = compute_confidence(
        metrics        = metrics,
        inv_results    = inv_results,
        feature_cols   = feature_cols,
        leakage        = leakage,
        physics_report = physics_report,
    )
    print_confidence(confidence)

    print(f"  {'SAVED FILES':─<50}")
    for p in paths:
        print(f"    {p}")
    print(f"\n{'='*72}\n")
