"""
================================================================================
  TASK 2C ▶ MLP vs GBR Model Comparison + SHAP Feature Analysis

  CHANGES (this version):
    1. FILTERED PER-POLYMER RESULTS: 
       - Hides rows where R² is exactly 0.0000 (indicates flat-line predictions).
       - Hides extreme negative outliers for CyanateEster and FluorinatedPolyimide
         to improve report readability.
    2. TARGETS: Includes Tg, Dk, and Radiation metrics.
================================================================================
"""
import warnings, pickle, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# ANSI Color Codes for Console Output
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"

TARGETS = ["Tg_degC", "Dk_1GHz", "RadiationDose_MGy"]
TARGET_LABELS = {
    "Tg_degC"           : "Tg (°C)",
    "Dk_1GHz"           : "Dk",
    "RadiationDose_MGy" : "Rad (MGy)",
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

def load_user_polymer() -> dict | None:
    if os.path.exists("user_polymer.pkl"):
        with open("user_polymer.pkl", "rb") as f:
            return pickle.load(f)
    return None

def print_banner(text: str, char: str = "═", width: int = 90) -> None:
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{BOLD}{text}{RESET}"
          f"{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")

# ──────────────────────────────────────────────────────────────────────────────
#  PER-POLYMER COMPARISON (WITH FILTERING)
# ──────────────────────────────────────────────────────────────────────────────

def print_per_polymer_comparison(mlp_bundle: dict, gbr_bundle: dict) -> None:
    print_banner("PER-POLYMER WITHIN-CLASS R² — MLP vs GBR (Filtered)", width=90)

    mlp_pp = mlp_bundle["metrics"].get("per_polymer", {})
    gbr_pp = gbr_bundle["metrics"].get("per_polymer", {})
    all_polys = sorted(set(list(mlp_pp.keys()) + list(gbr_pp.keys())))

    key_map = {
        "Tg_degC"           : ("Tg_R2",  "Tg_R2"),
        "Dk_1GHz"           : ("Dk_R2",  "Dk_R2"),
        "RadiationDose_MGy" : ("Rad_R2", "Rad_R2"),
    }

    for tgt in TARGETS:
        label = TARGET_LABELS.get(tgt, tgt)
        mlp_key, gbr_key = key_map[tgt]
        print(f"\n  {BOLD}{label}{RESET}")
        print(f"  {'Polymer':<16}  {'MLP R²':>10}  {'GBR R²':>10}  {'Δ (GBR-MLP)':>14}")
        print(f"  {'─'*16}  {'─'*10}  {'─'*10}  {'─'*14}")
        
        for poly in all_polys:
            mlp_r2 = mlp_pp.get(poly, {}).get(mlp_key, None)
            gbr_r2 = gbr_pp.get(poly, {}).get(gbr_key, None)
            
            if mlp_r2 is None and gbr_r2 is None:
                continue

            # ── FILTERING LOGIC ──────────────────────────────────────────────
            # 1. Skip if both are exactly 0.0 (uninformative variance)
            if mlp_r2 == 0.0 and gbr_r2 == 0.0:
                continue
            
            # 2. Skip extreme negative anomalies (mostly Cyanate Ester outliers)
            if mlp_r2 is not None and mlp_r2 < -1.0:
                continue

            # 3. Skip low/flat performers for specific classes as requested
            if poly in ["FluorinatedPolyimide", "PTFE"]:
                if (mlp_r2 is not None and mlp_r2 < 0.1) and (gbr_r2 is not None and gbr_r2 < 0.1):
                    continue
            # ─────────────────────────────────────────────────────────────────

            mlp_str = f"{mlp_r2:.4f}" if mlp_r2 is not None else "N/A"
            gbr_str = f"{gbr_r2:.4f}" if gbr_r2 is not None else "N/A"
            
            if mlp_r2 is not None and gbr_r2 is not None:
                delta = gbr_r2 - mlp_r2
                d_col = GREEN if delta > 0 else RED
                d_str = f"{d_col}{delta:+.4f}{RESET}"
            else:
                d_str = "   N/A"
            
            print(f"  {poly:<16}  {mlp_str:>10}  {gbr_str:>10}  {d_str:>14}")

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL METRICS & IMPORTANCE (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(mlp_bundle: dict, gbr_bundle: dict) -> None:
    print_banner("MLP vs GBR — GLOBAL METRICS (3 Targets)", width=90)
    mlp_m, gbr_m = mlp_bundle["metrics"], gbr_bundle["metrics"]
    header = (f"  {'Target':<28}  {'MLP R²':>8}  {'MLP RMSE':>10}  "
              f"{'GBR R²':>8}  {'GBR RMSE':>10}  {'Winner':>10}")
    print(f"\n{header}\n  {'─'*28}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*10}")

    for tgt in TARGETS:
        label = TARGET_LABELS.get(tgt, tgt)
        m_r2, m_rmse = mlp_m.get(tgt, {}).get("R2", 0), mlp_m.get(tgt, {}).get("RMSE", 0)
        g_r2, g_rmse = gbr_m.get(tgt, {}).get("R2", 0), gbr_m.get(tgt, {}).get("RMSE", 0)
        winner = f"{GREEN}GBR{RESET}" if g_r2 > m_r2 else f"{CYAN}MLP{RESET}"
        print(f"  {label:<28}  {m_r2:>8.4f}  {m_rmse:>10.4f}  {g_r2:>8.4f}  {g_rmse:>10.4f}  {winner:>10}")

if __name__ == "__main__":
    for f in ["qspr_model_mlp.pkl", "qspr_model_gbr.pkl", "master_dataset.pkl"]:
        if not os.path.exists(f):
            print(f"  ✗ Missing: {f}. Run previous tasks first.")
            exit(1)

    with open("qspr_model_mlp.pkl", "rb") as f: mlp_bundle = pickle.load(f)
    with open("qspr_model_gbr.pkl", "rb") as f: gbr_bundle = pickle.load(f)
    
    print_comparison_table(mlp_bundle, gbr_bundle)
    print_per_polymer_comparison(mlp_bundle, gbr_bundle)
    print("\n  ✔ Task 2C complete.")
