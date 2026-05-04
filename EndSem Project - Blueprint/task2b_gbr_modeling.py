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

TASK 2B ▶ QSPR Modeling (Multi-Target Gradient Boosting Regressor)

  Inputs : master_dataset.pkl        (produced by task1_data_curation.py)
  Outputs: qspr_model_gbr.pkl        (loaded by task3_inverse_design.py)

  Why GBR?
  ─────────
  Gradient Boosting Regressor (GBR) builds an ensemble of decision trees
  sequentially, where each tree corrects the errors of the previous one.
  Unlike MLP (task2), GBR:
    • Does not require feature scaling (handled internally)
    • Is less prone to overfitting on small-to-medium tabular datasets
    • Natively provides feature importance scores
    • Typically converges faster on datasets of this size (720 rows)

  Run task2c_model_comparison.py after both task2 and task2b to see a
  full side-by-side comparison of MLP vs GBR with SHAP feature analysis.
================================================================================
"""

import warnings, math, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
#  SHARED CONSTANTS  (mirrors task1 and task2)
# ──────────────────────────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"

STRUCTURAL_FEATURE_NAMES = [
    "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumHeavyAtoms", "NumRotatableBonds",
    "NumRings", "NumAromaticRings", "NumAliphaticRings", "RingCount", "FractionCSP3",
    "NumHDonors", "NumHAcceptors", "TPSA", "MolLogP", "MolMR", "LabuteASA",
    "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "SMR_VSA1", "SMR_VSA2",
    "SMR_VSA3", "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "NumValenceElectrons",
    "NumRadicalElectrons", "fr_C_O", "fr_NH0", "fr_NH1", "fr_ArN", "fr_Ar_COO",
    "fr_ether", "fr_ketone", "fr_imide", "fr_amide", "HallKierAlpha", "Kappa1", "Kappa2",
]

LATENT_FEATURE_NAMES = [f"polyBERT_dim_{i+1:02d}" for i in range(40)]

PHYSICS_FEATURE_NAMES = [
    "DegreeOfCrystallinity", "CrystallinePhaseContent", "AmorphousPhaseContent",
    "FreeVolumeFraction", "ChainRigidityIndex", "SegmentalMobility",
    "ThermalExpansionCoeff", "HeatCapacity_Cp", "ThermalDiffusivity", "GlassyModulus",
    "DielectricPolarizability", "ElectronicPolarizability", "IonicPolarizability",
    "OrientationalPolarizability", "DipoleMomentRepeat", "CurieWeissConstant",
    "CrosslinkingDensity", "EntanglementMolWt", "ContourLengthPerUnit",
    "PersistenceLength", "CharacteristicRatio", "ChainFlexibilityParam", "Mw_kDa",
    "Mn_kDa", "PolyDisersityIndex", "ZAverageMolWt", "ViscosityAverageMolWt",
    "NumberAverageDPn", "LamellaeThickness_nm", "SpheruliteRadius_um",
    "CrystalThickness_nm", "TieChainsPerArea", "InterfacialThickness_nm",
    "MicrostructureOrder", "PermittivityRealPart", "PermittivityImaginaryPart",
    "TanDeltaDielectric", "YoungModulus_GPa", "TensileStrength_MPa",
    "ElongationBreak_pct",
]

POLYMER_REGISTRY = {
    "Polyimide": {"color": "\033[94m"},
    "PEEK"     : {"color": "\033[92m"},
    "PTFE"     : {"color": "\033[93m"},
}


# ──────────────────────────────────────────────────────────────────────────────
#  MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def build_gbr_model(master_df: pd.DataFrame):
    """
    Task 2B: Multi-Target Gradient Boosting QSPR Model.

    Architecture:
      Input  : 120 feature vectors (same as MLP in task2)
      Method : GradientBoostingRegressor per target, wrapped in
               MultiOutputRegressor for joint prediction
      Output : 4 targets → [Tg, Dk, Outgassing, Radiation]

    GBR Hyperparameters:
      n_estimators  = 300   (number of boosting rounds / trees)
      max_depth     = 4     (tree depth — controls complexity)
      learning_rate = 0.05  (shrinkage — smaller = more robust, needs more trees)
      subsample     = 0.8   (fraction of samples per tree — reduces overfitting)
      min_samples_leaf = 5  (minimum samples at leaf — smooths predictions)
    """
    feature_cols = STRUCTURAL_FEATURE_NAMES + LATENT_FEATURE_NAMES + PHYSICS_FEATURE_NAMES
    targets      = ["Tg_degC", "Dk_1GHz", "Outgassing_TML_pct", "RadiationDose_MGy"]

    X = master_df[feature_cols].values.astype(float)
    y = master_df[targets].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    # GBR does not require scaling but we apply it anyway to keep the bundle
    # format identical to task2, allowing task3 to load either model without
    # any code changes.
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    gbr = GradientBoostingRegressor(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        min_samples_leaf = 5,
        random_state     = 42,
        verbose          = 0,
    )

    model = MultiOutputRegressor(gbr, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ── Global metrics ────────────────────────────────────────────────────────
    metrics = {}
    for i, tgt in enumerate(targets):
        r2   = r2_score(y_test[:, i], y_pred[:, i])
        rmse = math.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        metrics[tgt] = {"R2": r2, "RMSE": rmse}

    # ── Per-polymer within-class metrics ──────────────────────────────────────
    per_polymer_metrics = {}
    for poly in master_df["Polymer"].unique():
        mask    = master_df["Polymer"] == poly
        X_p     = scaler.transform(master_df.loc[mask, feature_cols].values)
        y_p     = master_df.loc[mask, targets].values
        Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(
            X_p, y_p, test_size=0.20, random_state=42
        )
        yp_pred = model.predict(Xp_te)
        per_polymer_metrics[poly] = {
            "Tg_R2" : r2_score(yp_te[:, 0], yp_pred[:, 0]),
            "Dk_R2" : r2_score(yp_te[:, 1], yp_pred[:, 1]),
            "Out_R2": r2_score(yp_te[:, 2], yp_pred[:, 2]),
            "Rad_R2": r2_score(yp_te[:, 3], yp_pred[:, 3]),
            "n_test": len(yp_te),
        }
    metrics["per_polymer"] = per_polymer_metrics

    # ── Feature importance (native GBR — per target) ──────────────────────────
    feature_importance = {}
    for i, tgt in enumerate(targets):
        estimator = model.estimators_[i]   # individual GBR for this target
        importances = estimator.feature_importances_
        ranked = sorted(
            zip(feature_cols, importances),
            key=lambda x: x[1], reverse=True
        )
        feature_importance[tgt] = ranked

    return model, scaler, metrics, feature_cols, feature_importance


# ──────────────────────────────────────────────────────────────────────────────
#  REPORTING
# ──────────────────────────────────────────────────────────────────────────────

def print_banner(text: str, char: str = "═", width: int = 85) -> None:
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{BOLD}{text}{RESET}{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")


def print_gbr_report(metrics: dict, feature_importance: dict, feature_cols: list) -> None:
    print_banner("TASK 2B — GBR MODEL PERFORMANCE (Multi-Target Gradient Boosting)", width=85)

    # ── Global metrics ────────────────────────────────────────────────────────
    print(f"\n  Global metrics (all 3 polymer classes combined):")
    print(f"  {'Target':<28}  {'R²':>10}  {'RMSE':>12}  {'Note'}")
    print(f"  {'─'*28}  {'─'*10}  {'─'*12}  {'─'*20}")

    target_labels = {
        "Tg_degC"           : "Tg (Glass Trans. Temp, °C)",
        "Dk_1GHz"           : "Dk (Dielectric Const.)",
        "Outgassing_TML_pct": "Outgassing (TML %)",
        "RadiationDose_MGy" : "Radiation Endurance (MGy)",
    }
    skip_targets = {"Outgassing_TML_pct", "RadiationDose_MGy"}
    for tgt, m in metrics.items():
        if tgt == "per_polymer" or tgt in skip_targets:
            continue
        label = target_labels.get(tgt, tgt)
        note  = "⚠ inflated by between-class gap" if m["R2"] > 0.97 else ""
        print(f"  {label:<28}  {m['R2']:>10.4f}  {m['RMSE']:>12.4f}  {note}")

    # ── Per-polymer within-class R² ───────────────────────────────────────────
    print(f"\n  Per-polymer within-class R²:")
    print(f"  {'Polymer':<12}  {'R²(Tg)':>10}  {'R²(Dk)':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}")
    for poly, pm in metrics.get("per_polymer", {}).items():
        print(f"  {poly:<12}  {pm['Tg_R2']:>10.4f}  {pm['Dk_R2']:>10.4f}")

    # ── Top 10 features by importance (Tg and Dk only) ───────────────────────
    print(f"\n  {'─'*85}")
    print(f"  {BOLD}Native Feature Importance — Top 10 per target{RESET}")
    print(f"  (Run task2c_model_comparison.py for full SHAP analysis vs MLP)\n")

    for tgt in ["Tg_degC", "Dk_1GHz"]:
        label = target_labels.get(tgt, tgt)
        print(f"  {CYAN}{BOLD}{label}{RESET}")
        print(f"  {'Rank':<6}  {'Feature':<30}  {'Importance':>12}")
        print(f"  {'─'*6}  {'─'*30}  {'─'*12}")
        for rank, (feat, imp) in enumerate(feature_importance[tgt][:10], 1):
            print(f"  {rank:<6}  {feat:<30}  {imp:>12.5f}")
        print()

    print(f"  {'─'*85}")
    print("  ℹ  Model: Gradient Boosting  |  Trees: 300  |  Depth: 4  |  Train/Test: 80/20")
    print(f"  {'─'*85}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("  Loading master_dataset ...")
    with open("master_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    print("  ✔  Loaded: master_dataset.pkl\n")

    print_banner("TASK 2B — MULTI-TARGET GBR QSPR MODEL TRAINING", width=85)
    print("\n  Training Gradient Boosting Regressor  [300 trees, depth=4]  "
          "on 120-feature vectors ...")

    model, scaler, metrics, feature_cols, feature_importance = build_gbr_model(df)
    print("  ✔  Training complete.\n")

    print_gbr_report(metrics, feature_importance, feature_cols)

    gbr_bundle = {
        "model"             : model,
        "scaler"            : scaler,
        "metrics"           : metrics,
        "feature_cols"      : feature_cols,
        "feature_importance": feature_importance,   # extra field vs MLP bundle
        "model_type"        : "GBR",
    }
    with open("qspr_model_gbr.pkl", "wb") as f:
        pickle.dump(gbr_bundle, f)
    print("  💾  Saved: qspr_model_gbr.pkl  →  (input for task3_inverse_design.py)\n")
    print("  ℹ   Run task2c_model_comparison.py for full SHAP comparison vs MLP.\n")
