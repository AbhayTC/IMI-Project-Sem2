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
    "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumHeavyAtoms", "NumRotatableBonds", "NumRings", "NumAromaticRings", "NumAliphaticRings", "RingCount", "FractionCSP3", "NumHDonors", "NumHAcceptors", "TPSA", "MolLogP", "MolMR", "LabuteASA", "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "SMR_VSA1", "SMR_VSA2", "SMR_VSA3", "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "NumValenceElectrons", "NumRadicalElectrons", "fr_C_O", "fr_NH0", "fr_NH1", "fr_ArN", "fr_Ar_COO", "fr_ether", "fr_ketone", "fr_imide", "fr_amide", "HallKierAlpha", "Kappa1", "Kappa2",
]

LATENT_FEATURE_NAMES = [f"polyBERT_dim_{i+1:02d}" for i in range(N_LATENT)]

PHYSICS_FEATURE_NAMES = [
    "DegreeOfCrystallinity", "CrystallinePhaseContent", "AmorphousPhaseContent", "FreeVolumeFraction", "ChainRigidityIndex", "SegmentalMobility", "ThermalExpansionCoeff", "HeatCapacity_Cp", "ThermalDiffusivity", "GlassyModulus", "DielectricPolarizability", "ElectronicPolarizability", "IonicPolarizability", "OrientationalPolarizability", "DipoleMomentRepeat", "CurieWeissConstant", "CrosslinkingDensity", "EntanglementMolWt", "ContourLengthPerUnit", "PersistenceLength", "CharacteristicRatio", "ChainFlexibilityParam", "Mw_kDa", "Mn_kDa", "PolyDisersityIndex", "ZAverageMolWt", "ViscosityAverageMolWt", "NumberAverageDPn", "LamellaeThickness_nm", "SpheruliteRadius_um", "CrystalThickness_nm", "TieChainsPerArea", "InterfacialThickness_nm", "MicrostructureOrder", "PermittivityRealPart", "PermittivityImaginaryPart", "TanDeltaDielectric", "YoungModulus_GPa", "TensileStrength_MPa", "ElongationBreak_pct",
]


def build_qspr_model(master_df: pd.DataFrame):
    """
    Task 2: Multi-Target MLP QSPR Model.
    Architecture:
      Input  : 120 standardised feature vectors
      Hidden : [256, 128, 64]
      Output : 4 targets → [Tg, Dk, Outgassing, Radiation]
    """
    feature_cols = STRUCTURAL_FEATURE_NAMES + LATENT_FEATURE_NAMES + PHYSICS_FEATURE_NAMES
    targets = ["Tg_degC", "Dk_1GHz", "Outgassing_TML_pct", "RadiationDose_MGy"]
    
    X = master_df[feature_cols].values.astype(float)
    y = master_df[targets].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    mlp = MLPRegressor(
        hidden_layer_sizes = (256, 128, 64),
        activation         = "relu",
        solver             = "adam",
        alpha              = 1e-4,
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

    model = MultiOutputRegressor(mlp, n_jobs=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {}
    for i, tgt in enumerate(targets):
        r2   = r2_score(y_test[:, i], y_pred[:, i])
        rmse = math.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        metrics[tgt] = {"R2": r2, "RMSE": rmse}

    per_polymer_metrics = {}
    from sklearn.model_selection import train_test_split as _tts
    for poly in master_df["Polymer"].unique():
        mask    = master_df["Polymer"] == poly
        X_p     = scaler.transform(master_df.loc[mask, feature_cols].values)
        y_p     = master_df.loc[mask, targets].values
        Xp_tr, Xp_te, yp_tr, yp_te = _tts(X_p, y_p, test_size=0.20, random_state=42)
        yp_pred = model.predict(Xp_te)
        
        per_polymer_metrics[poly] = {
            "Tg_R2": r2_score(yp_te[:, 0], yp_pred[:, 0]),
            "Dk_R2": r2_score(yp_te[:, 1], yp_pred[:, 1]),
            "Out_R2": r2_score(yp_te[:, 2], yp_pred[:, 2]),
            "Rad_R2": r2_score(yp_te[:, 3], yp_pred[:, 3]),
            "n_test": len(yp_te),
        }
    metrics["per_polymer"] = per_polymer_metrics

    return model, scaler, metrics, feature_cols

def print_banner(text: str, char: str = "═", width: int = 85) -> None:
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{BOLD}{text}{RESET}{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")

def print_ml_report(metrics: dict) -> None:
    print_banner("TASK 2 — QSPR MODEL PERFORMANCE (Multi-Target MLP)", width=85)

    print(f"\n  Global metrics (all 3 polymer classes combined):")
    print(f"  {'Target':<28}  {'R²':>10}  {'RMSE':>12}  {'Note'}")
    print(f"  {'─'*28}  {'─'*10}  {'─'*12}  {'─'*20}")
    
    target_labels = {
        "Tg_degC": "Tg (Glass Trans. Temp, °C)",
        "Dk_1GHz": "Dk (Dielectric Const.)",
        "Outgassing_TML_pct": "Outgassing (TML %)",
        "RadiationDose_MGy": "Radiation Endurance (MGy)"
    }
    
    skip_targets = {"Outgassing_TML_pct", "RadiationDose_MGy"}
    for tgt, m in metrics.items():
        if tgt == "per_polymer" or tgt in skip_targets: continue
        label = target_labels.get(tgt, tgt)
        note  = "⚠ inflated by between-class gap" if m["R2"] > 0.97 else ""
        print(f"  {label:<28}  {m['R2']:>10.4f}  {m['RMSE']:>12.4f}  {note}")

    print(f"\n  Per-polymer within-class R²:")
    print(f"  {'Polymer':<12}  {'R²(Tg)':>10}  {'R²(Dk)':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}")
    for poly, pm in metrics.get("per_polymer", {}).items():
        print(f"  {poly:<12}  {pm['Tg_R2']:>10.4f}  {pm['Dk_R2']:>10.4f}")

    print(f"\n  {'─'*85}")
    print("  ℹ  Model: Multi-Layer Perceptron  |  Layers: [256→128→64]  |  Train/Test: 80/20")
    print(f"  {'─'*85}\n")

if __name__ == "__main__":
    print("  Loading master_dataset ...")
    with open("master_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    print("  ✔  Loaded: master_dataset.pkl\n")

    print_banner("TASK 2 — MULTI-TARGET MLP QSPR MODEL TRAINING", width=85)
    print("\n  Training Multi-Layer Perceptron  [256 → 128 → 64]  "
          "on 120-feature vectors ...")
    model, scaler, metrics, feature_cols = build_qspr_model(df)
    print("  ✔  Training complete.\n")
    print_ml_report(metrics)

    qspr_bundle = {
        "model"       : model,
        "scaler"      : scaler,
        "metrics"     : metrics,
        "feature_cols": feature_cols,
    }
    with open("qspr_model_mlp.pkl", "wb") as f:
        pickle.dump(qspr_bundle, f)
    print("  💾  Saved: qspr_model_mlp.pkl  →  (input for task3_inverse_design.py)\n")
