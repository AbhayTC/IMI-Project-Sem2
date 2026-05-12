"""
================================================================================
  TASK 2B ▶ QSPR Modeling (Multi-Target Gradient Boosting Regressor)

  CHANGES (this version): identical change set to task2_qspr_modeling.py —
    1. All 4 targets (Tg, Dk, TML, Radiation) fully reported
    2. Training filtered to is_reference=True rows only
    3. Dummy validation table after training
    4. User polymer inference if user_polymer.pkl present
    5. Per-polymer R² for all 4 targets
    6. Feature importance table extended to TML and Radiation

  Inputs : master_dataset.pkl
  Outputs: qspr_model_gbr.pkl
================================================================================
"""
import warnings, math, pickle, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

np.random.seed(42)


def generate_sample_weights(
    y_train_classes: np.ndarray,
    target_class: str = "FluorinatedPolyimide",
    penalty_multiplier: float = 5.0,
) -> np.ndarray:
    """
    Return a per-sample weight array.  Samples belonging to `target_class`
    are upweighted by `penalty_multiplier` so the GBR pays extra attention
    to that polymer's region of feature space during tree construction.

    Parameters
    ----------
    y_train_classes     : 1-D array of polymer-class strings, aligned with X_train.
    target_class        : Polymer name to upweight (default: 'FluorinatedPolyimide').
    penalty_multiplier  : Weight applied to target-class rows (default: 5.0).
    """
    weights = np.ones(len(y_train_classes), dtype=float)
    target_indices = [
        i for i, p_class in enumerate(y_train_classes)
        if p_class == target_class
    ]
    if target_indices:
        weights[target_indices] *= penalty_multiplier
        print(f"  ℹ  Sample weighting: {len(target_indices)} '{target_class}' rows "
              f"upweighted ×{penalty_multiplier}")
    else:
        print(f"  {chr(10033)}  generate_sample_weights: '{target_class}' not found in "
              f"training split — all weights = 1.0")
    return weights


RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"

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

TARGETS = ["Tg_degC", "Dk_1GHz", "RadiationDose_MGy"]
TARGET_LABELS = {
    "Tg_degC"           : "Tg — Glass Transition Temp (°C)",
    "Dk_1GHz"           : "Dk — Dielectric Constant",
    "RadiationDose_MGy" : "Radiation Endurance (MGy)",
}
TARGET_PLAUSIBLE = {
    "Tg_degC"           : (-200.0, 600.0),
    "Dk_1GHz"           : (1.0, 10.0),
    "RadiationDose_MGy" : (0.0, 100.0),
}


def load_user_polymer() -> dict | None:
    if os.path.exists("user_polymer.pkl"):
        with open("user_polymer.pkl", "rb") as f:
            return pickle.load(f)
    return None


def print_banner(text: str, char: str = "═", width: int = 85) -> None:
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{BOLD}{text}{RESET}"
          f"{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")


def build_gbr_model(master_df: pd.DataFrame):
    feature_cols = STRUCTURAL_FEATURE_NAMES + LATENT_FEATURE_NAMES + PHYSICS_FEATURE_NAMES

    if "is_reference" in master_df.columns:
        train_df = master_df[master_df["is_reference"] == True].copy()
        n_excluded = len(master_df) - len(train_df)
        print(f"  ℹ  Training rows : {len(train_df)}  "
              f"(excluded {n_excluded} non-reference rows)")
    else:
        train_df = master_df.copy()

    X       = train_df[feature_cols].values.astype(float)
    y       = train_df[TARGETS].values.astype(float)
    classes = train_df["Polymer"].values          # class labels — split in sync with X/y

    # Split polymer class labels alongside X and y so weight array stays aligned
    X_train, X_test, y_train, y_test, classes_train, _ = train_test_split(
        X, y, classes, test_size=0.20, random_state=42, shuffle=True
    )

    # GBR is scale-invariant but we apply scaler for bundle format consistency with MLP
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    gbr = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42, verbose=0,
    )
    model = MultiOutputRegressor(gbr, n_jobs=-1)

    # Generate per-sample weights and pass through MultiOutputRegressor.
    # MultiOutputRegressor forwards sample_weight to each sub-estimator's fit().
    training_weights = generate_sample_weights(
        classes_train,
        target_class="FluorinatedPolyimide",
        penalty_multiplier=5.0,
    )
    model.fit(X_train, y_train, sample_weight=training_weights)

    y_pred = model.predict(X_test)

    metrics = {}
    for i, tgt in enumerate(TARGETS):
        r2   = r2_score(y_test[:, i], y_pred[:, i])
        rmse = math.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        metrics[tgt] = {"R2": r2, "RMSE": rmse}

    per_polymer_metrics = {}
    ref_polymers = train_df["Polymer"].unique()
    for poly in ref_polymers:
        mask = train_df["Polymer"] == poly
        X_p  = scaler.transform(train_df.loc[mask, feature_cols].values)
        y_p  = train_df.loc[mask, TARGETS].values
        Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(
            X_p, y_p, test_size=0.20, random_state=42
        )
        yp_pred = model.predict(Xp_te)
        per_polymer_metrics[poly] = {
            "Tg_R2"  : r2_score(yp_te[:, 0], yp_pred[:, 0]),
            "Dk_R2"  : r2_score(yp_te[:, 1], yp_pred[:, 1]),
            "Rad_R2" : r2_score(yp_te[:, 2], yp_pred[:, 2]),
            "n_test" : len(yp_te),
        }
    metrics["per_polymer"] = per_polymer_metrics

    # Feature importance (all 4 targets)
    feature_importance = {}
    for i, tgt in enumerate(TARGETS):
        estimator   = model.estimators_[i]
        importances = estimator.feature_importances_
        ranked = sorted(zip(feature_cols, importances),
                        key=lambda x: x[1], reverse=True)
        feature_importance[tgt] = ranked

    return model, scaler, metrics, feature_cols, feature_importance


def validate_dummy_samples(master_df: pd.DataFrame, model, scaler,
                            feature_cols: list) -> None:
    if "is_dummy" not in master_df.columns:
        return
    dummy_df = master_df[master_df["is_dummy"] == True]
    if len(dummy_df) == 0:
        return

    print_banner("DUMMY SAMPLE VALIDATION", width=85)
    print(f"\n  {BOLD}Purpose:{RESET} Confirm model does NOT clamp impossible inputs.\n")
    print(f"  {'Sample_ID':<12}  {'Description':<42}  {'Verdict'}")
    print(f"  {'─'*12}  {'─'*42}  {'─'*10}")

    X_dummy = dummy_df[feature_cols].values.astype(float)
    preds   = model.predict(scaler.transform(X_dummy))

    for j, (_, row) in enumerate(dummy_df.iterrows()):
        pred_row    = preds[j]
        out_of_range = []
        for k, tgt in enumerate(TARGETS):
            lo, hi = TARGET_PLAUSIBLE[tgt]
            if pred_row[k] < lo or pred_row[k] > hi:
                out_of_range.append(f"{tgt.split('_')[0]}={pred_row[k]:.2f}")

        label   = row.get("dummy_label", "")[:42]
        verdict = (f"{GREEN}{BOLD}✔ OOR{RESET}" if out_of_range
                   else f"{RED}{BOLD}✗ IN-RANGE (review model){RESET}")
        sid     = row.get("Sample_ID", f"DUMMY_{j+1:03d}")

        print(f"  {sid:<12}  {label:<42}  {verdict}")
        print(f"  {'':12}  Tg={pred_row[0]:.1f}  Dk={pred_row[1]:.3f}  "
              f"Rad={pred_row[2]:.2f}")
        if out_of_range:
            print(f"  {'':12}  {CYAN}OOR: {', '.join(out_of_range)}{RESET}\n")
        else:
            print()

    print(f"  OOR = Out Of Range — expected for physically impossible inputs.\n")


def infer_user_polymer(master_df: pd.DataFrame, model, scaler,
                       feature_cols: list, user_meta: dict) -> None:
    name      = user_meta["name"]
    user_rows = master_df[
        (master_df["Polymer"] == name) & (master_df.get("is_dummy", False) == False)
    ]
    if len(user_rows) == 0:
        return

    preds  = model.predict(scaler.transform(user_rows[feature_cols].values.astype(float)))
    pred_df = pd.DataFrame(preds, columns=TARGETS)

    print_banner(f"USER POLYMER INFERENCE — {name}", width=85)
    print(f"\n  {YELLOW}⚠  Extrapolated from reference polymer feature space — validate experimentally.{RESET}\n")
    print(f"  {'Target':<30}  {'Median':>10}  {'Mean':>10}  {'Std':>8}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*10}  {'─'*8}")
    for tgt in TARGETS:
        label = TARGET_LABELS[tgt]
        vals  = pred_df[tgt]
        print(f"  {label:<30}  {vals.median():>10.4f}  {vals.mean():>10.4f}  {vals.std():>8.4f}")
    print()


def print_gbr_report(metrics: dict, feature_importance: dict) -> None:
    print_banner("TASK 2B — GBR MODEL PERFORMANCE (3 Targets)", width=85)

    def ql(r2):
        if r2 >= 0.95: return f"{GREEN}Excellent{RESET}"
        if r2 >= 0.80: return f"{GREEN}Very Good{RESET}"
        if r2 >= 0.60: return f"{YELLOW}Good{RESET}"
        return f"{RED}Low{RESET}"

    print(f"\n  Global metrics:")
    print(f"  {'Target':<30}  {'R²':>10}  {'RMSE':>14}  {'Quality'}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*14}  {'─'*20}")
    for tgt in TARGETS:
        m     = metrics.get(tgt, {})
        label = TARGET_LABELS.get(tgt, tgt)
        note  = f"{YELLOW}⚠ inflated{RESET}" if m.get("R2", 0) > 0.97 else ql(m.get("R2", 0))
        print(f"  {label:<30}  {m.get('R2',0):>10.4f}  {m.get('RMSE',0):>14.4f}  {note}")

    # Feature importance — top 5 per target
    print(f"\n  {BOLD}Top-5 Feature Importance per Target{RESET}")
    for tgt in TARGETS:
        label = TARGET_LABELS.get(tgt, tgt)
        top   = feature_importance.get(tgt, [])[:5]
        feats = "  |  ".join(f"{f} ({v:.3f})" for f, v in top)
        print(f"\n  {CYAN}{label}{RESET}")
        print(f"  {feats}")

    print(f"\n  {'─'*85}")
    print("  ℹ  Model: GBR  |  Trees: 300  |  Depth: 4  |  Training: reference polymers only")
    print(f"  {'─'*85}\n")


if __name__ == "__main__":
    print("  Loading master_dataset.pkl ...")
    with open("master_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    print(f"  ✔  Loaded: {len(df)} rows\n")

    user_meta = load_user_polymer()
    if user_meta:
        print(f"  ✔  User polymer: {user_meta['name']}")

    print_banner("TASK 2B — MULTI-TARGET GBR QSPR MODEL TRAINING", width=85)
    print("\n  Training GBR [300 trees, depth=4] on 120-feature vectors ...")
    model, scaler, metrics, feature_cols, feature_importance = build_gbr_model(df)
    print("  ✔  Training complete.\n")

    print_gbr_report(metrics, feature_importance)

    if user_meta:
        infer_user_polymer(df, model, scaler, feature_cols, user_meta)

    gbr_bundle = {
        "model"             : model,
        "scaler"            : scaler,
        "metrics"           : metrics,
        "feature_cols"      : feature_cols,
        "feature_importance": feature_importance,
        "model_type"        : "GBR",
    }
    with open("qspr_model_gbr.pkl", "wb") as f:
        pickle.dump(gbr_bundle, f)
    print("  💾  Saved: qspr_model_gbr.pkl\n")
    print("  ℹ   Run task2c_model_comparison.py for SHAP comparison vs MLP.\n")
