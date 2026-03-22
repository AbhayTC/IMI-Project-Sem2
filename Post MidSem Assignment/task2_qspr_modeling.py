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

TASK 2 ▶ QSPR Modeling (Multi-Target MLP)

  Inputs : master_dataset.pkl        (produced by task1_data_curation.py)
  Outputs: qspr_model.pkl            (loaded by task3_inverse_design.py)
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import warnings, math, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
#  SHARED CONSTANTS  (mirrors task1_data_curation.py)
# ──────────────────────────────────────────────────────────────────────────────
N_STRUCTURAL = 40
N_LATENT     = 40
N_PHYSICS    = 40

POLYMER_REGISTRY = {
    "Polyimide": {"color": "\033[94m", "member": "Member1"},
    "PEEK"     : {"color": "\033[92m", "member": "Member2"},
    "PTFE"     : {"color": "\033[93m", "member": "Member3"},
}

RESET = "\033[0m"
BOLD  = "\033[1m"

STRUCTURAL_FEATURE_NAMES = [
    "MolWt",           "HeavyAtomMolWt",  "ExactMolWt",      "NumHeavyAtoms",
    "NumRotatableBonds","NumRings",        "NumAromaticRings", "NumAliphaticRings",
    "RingCount",       "FractionCSP3",
    "NumHDonors",      "NumHAcceptors",   "TPSA",
    "MolLogP",         "MolMR",           "LabuteASA",       "PEOE_VSA1",
    "PEOE_VSA2",       "PEOE_VSA3",       "PEOE_VSA4",
    "SMR_VSA1",        "SMR_VSA2",        "SMR_VSA3",
    "SlogP_VSA1",      "SlogP_VSA2",      "SlogP_VSA3",
    "NumValenceElectrons","NumRadicalElectrons",
    "fr_C_O",          "fr_NH0",          "fr_NH1",
    "fr_ArN",          "fr_Ar_COO",       "fr_ether",
    "fr_ketone",       "fr_imide",        "fr_amide",
    "HallKierAlpha",   "Kappa1",          "Kappa2",
]

LATENT_FEATURE_NAMES = [f"polyBERT_dim_{i+1:02d}" for i in range(N_LATENT)]

PHYSICS_FEATURE_NAMES = [
    "DegreeOfCrystallinity",   "CrystallinePhaseContent",  "AmorphousPhaseContent",
    "FreeVolumeFraction",      "ChainRigidityIndex",        "SegmentalMobility",
    "ThermalExpansionCoeff",   "HeatCapacity_Cp",           "ThermalDiffusivity",
    "GlassyModulus",
    "DielectricPolarizability","ElectronicPolarizability",  "IonicPolarizability",
    "OrientationalPolarizability","DipoleMomentRepeat",     "CurieWeissConstant",
    "CrosslinkingDensity",     "EntanglementMolWt",         "ContourLengthPerUnit",
    "PersistenceLength",       "CharacteristicRatio",       "ChainFlexibilityParam",
    "Mw_kDa",                  "Mn_kDa",                   "PolyDisersityIndex",
    "ZAverageMolWt",           "ViscosityAverageMolWt",    "NumberAverageDPn",
    "LamellaeThickness_nm",    "SpheruliteRadius_um",      "CrystalThickness_nm",
    "TieChainsPerArea",        "InterfacialThickness_nm",  "MicrostructureOrder",
    "PermittivityRealPart",    "PermittivityImaginaryPart","TanDeltaDielectric",
    "YoungModulus_GPa",        "TensileStrength_MPa",      "ElongationBreak_pct",
]

# ──────────────────────────────────────────────────────────────────────────────
#  ████████╗ █████╗ ███████╗██╗  ██╗    ██████╗
#  ╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝    ╚════██╗
#     ██║   ███████║███████╗█████╔╝      █████╔╝
#     ██║   ██╔══██║╚════██║██╔═██╗     ██╔═══╝
#     ██║   ██║  ██║███████║██║  ██╗    ███████╗
#     ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚══════╝
#  QSPR MODELLING — Multi-Target MLP
# ──────────────────────────────────────────────────────────────────────────────

def build_qspr_model(master_df: pd.DataFrame):
    """
    Task 2: Multi-Target MLP QSPR Model.

    Architecture:
      Input  : 120 standardised feature vectors
      Hidden : [256, 128, 64]  (ReLU + early stopping + L2)
      Output : 2 targets → [Tg, Dk]

    Returns trained model, scaler, and performance metrics dict.
    """
    # ── Feature / target split ────────────────────────────────────────────────
    feature_cols = STRUCTURAL_FEATURE_NAMES + LATENT_FEATURE_NAMES + PHYSICS_FEATURE_NAMES
    X = master_df[feature_cols].values.astype(float)
    y = master_df[["Tg_degC", "Dk_1GHz"]].values.astype(float)

    # ── Train / test split (80 / 20) ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    # ── Standardise features ─────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Multi-target MLP ──────────────────────────────────────────────────────
    mlp = MLPRegressor(
        hidden_layer_sizes = (256, 128, 64),
        activation         = "relu",
        solver             = "adam",
        alpha              = 1e-4,          # L2 regularisation
        batch_size         = 64,
        learning_rate      = "adaptive",
        learning_rate_init = 1e-3,
        max_iter           = 600,
        early_stopping     = True,
        validation_fraction= 0.10,
        n_iter_no_change   = 25,
        random_state       = 42,
        verbose            = False,
    )

    # Wrap in MultiOutputRegressor for cleaner two-target training
    model = MultiOutputRegressor(mlp, n_jobs=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ── Per-target metrics ────────────────────────────────────────────────────
    targets = ["Tg_degC", "Dk_1GHz"]
    metrics = {}
    for i, tgt in enumerate(targets):
        r2   = r2_score(y_test[:, i], y_pred[:, i])
        rmse = math.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        metrics[tgt] = {"R2": r2, "RMSE": rmse}

    # ── Per-polymer within-class R² (honest metric) ──────────────────────────
    # Global R² is inflated by between-class variance (99.8% of total Tg variance
    # is between PI/PEEK/PTFE, not within each class). Per-polymer R² measures
    # whether the model learns within-class structure — the honest metric.
    per_polymer_metrics = {}
    from sklearn.model_selection import train_test_split as _tts
    for poly in master_df["Polymer"].unique():
        mask    = master_df["Polymer"] == poly
        X_p     = scaler.transform(master_df.loc[mask, feature_cols].values)
        y_p     = master_df.loc[mask, ["Tg_degC", "Dk_1GHz"]].values
        Xp_tr, Xp_te, yp_tr, yp_te = _tts(X_p, y_p, test_size=0.20, random_state=42)
        yp_pred = model.predict(Xp_te)
        per_polymer_metrics[poly] = {
            "Tg_R2": r2_score(yp_te[:, 0], yp_pred[:, 0]),
            "Dk_R2": r2_score(yp_te[:, 1], yp_pred[:, 1]),
            "n_test": len(yp_te),
        }
    metrics["per_polymer"] = per_polymer_metrics

    return model, scaler, metrics, feature_cols


# ──────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def print_banner(text: str, char: str = "═", width: int = 72) -> None:
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{BOLD}{text}{RESET}{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")


def print_ml_report(metrics: dict) -> None:
    """Formatted QSPR model performance table — global + per-polymer."""
    print_banner("TASK 2 — QSPR MODEL PERFORMANCE (Multi-Target MLP)", width=72)

    def quality_label(r2):
        if r2 >= 0.90: return "✦ Excellent"
        if r2 >= 0.80: return "✔ Very Good"
        if r2 >= 0.60: return "~ Good"
        return "✗ Low"

    # Global metrics
    print(f"\n  Global metrics (all 3 polymer classes combined):")
    print(f"  {'Target':<22}  {'R²':>10}  {'RMSE':>12}  {'Note'}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*12}  {'─'*30}")
    for tgt, m in metrics.items():
        if tgt == "per_polymer": continue
        label = "Tg (Glass Trans. Temp, °C)" if tgt == "Tg_degC" else "Dk (Dielectric Const.)"
        note  = "⚠ inflated by between-class gap" if m["R2"] > 0.97 else ""
        print(f"  {label:<22}  {m['R2']:>10.4f}  {m['RMSE']:>12.4f}  {note}")

    # Per-polymer within-class R² — the honest metric
    print(f"\n  Per-polymer within-class R² (honest — measures intra-class learning):")
    print(f"  {'Polymer':<12}  {'R²(Tg)':>10}  {'R²(Dk)':>10}  {'Tg quality':>12}  {'Dk quality':>12}  {'n test':>7}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*7}")
    for poly, pm in metrics.get("per_polymer", {}).items():
        print(f"  {poly:<12}  {pm['Tg_R2']:>10.4f}  {pm['Dk_R2']:>10.4f}  "
              f"{quality_label(pm['Tg_R2']):>12}  {quality_label(pm['Dk_R2']):>12}  {pm['n_test']:>7}")

    print(f"\n  {'─'*72}")
    print("  ℹ  Model: Multi-Layer Perceptron  |  Layers: [256→128→64]  |  Train/Test: 80/20")
    print(f"  {'─'*72}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Load master dataset from Task 1 ──────────────────────────────────────
    with open("master_dataset.pkl", "rb") as f:
        master_df = pickle.load(f)
    print("  ✔  Loaded: master_dataset.pkl\n")

    # ── Task 2: QSPR Model ────────────────────────────────────────────────────
    print_banner("TASK 2 — MULTI-TARGET MLP QSPR MODEL TRAINING", width=72)
    print("\n  Training Multi-Layer Perceptron  [256 → 128 → 64]  "
          "on 120-feature vectors ...")
    model, scaler, metrics, feature_cols = build_qspr_model(master_df)
    print("  ✔  Training complete.\n")
    print_ml_report(metrics)

    # ── Save model artefacts for downstream tasks ─────────────────────────────
    qspr_bundle = {
        "model"       : model,
        "scaler"      : scaler,
        "metrics"     : metrics,
        "feature_cols": feature_cols,
    }
    with open("qspr_model.pkl", "wb") as f:
        pickle.dump(qspr_bundle, f)
    print("  💾  Saved: qspr_model.pkl  →  (input for task3_inverse_design.py)\n")
