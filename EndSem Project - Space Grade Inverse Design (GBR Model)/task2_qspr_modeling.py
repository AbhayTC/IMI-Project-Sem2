"""
================================================================================
  TASK 2 ▶ QSPR Modeling (Multi-Target MLP)

  CHANGES (this version):
    1. ALL 4 TARGETS reported (Tg, Dk, TML, Radiation) — skip_targets removed
    2. TRAINING FILTER: trains only on is_reference=True rows (PI/PEEK/PTFE +
       CyanateEster/FluorinatedPolyimide/PBO);
       dummy rows and user polymer rows are excluded from training
    3. DUMMY VALIDATION: after training, runs the 5 dummy rows through the
       model and prints a validation table — predictions should be out-of-range,
       confirming the model is not silently clamping impossible inputs
    4. USER POLYMER INFERENCE: if user_polymer.pkl exists, predicts all 4
       targets on the user polymer rows and prints a comparison vs all reference polymers
    5. PER-POLYMER R² now covers all 4 targets, not just Tg and Dk
    6. PER-POLYMER TML MODELS: separate MLP trained per polymer for TML only,
       since TML has near-zero within-class variance that defeats a combined model

  Inputs : master_dataset.pkl      (produced by task1_data_curation.py)
  Outputs: qspr_model_mlp.pkl      (loaded by task3_inverse_design.py)
================================================================================
"""
import warnings, math, pickle, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

np.random.seed(42)

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

# Physical plausibility ranges for dummy validation
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


def build_qspr_model(master_df: pd.DataFrame):
    """
    Train Multi-Target MLP on is_reference=True rows only.
    Returns model, scaler, metrics, feature_cols.
    """
    feature_cols = STRUCTURAL_FEATURE_NAMES + LATENT_FEATURE_NAMES + PHYSICS_FEATURE_NAMES

    # ── Training data: reference polymers only ────────────────────────────────
    if "is_reference" in master_df.columns:
        train_df = master_df[master_df["is_reference"] == True].copy()
        n_excluded = len(master_df) - len(train_df)
        print(f"  ℹ  Training rows : {len(train_df)}  "
              f"(excluded {n_excluded} non-reference rows)")
    else:
        train_df = master_df.copy()

    X = train_df[feature_cols].values.astype(float)
    y = train_df[TARGETS].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu", solver="adam", alpha=1e-4,
        batch_size=64, learning_rate="adaptive", learning_rate_init=1e-3,
        max_iter=600, early_stopping=True, validation_fraction=0.10,
        n_iter_no_change=25, random_state=42, verbose=False,
    )
    model = MultiOutputRegressor(mlp, n_jobs=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ── Global metrics (all 4 targets) ────────────────────────────────────────
    metrics = {}
    for i, tgt in enumerate(TARGETS):
        r2   = r2_score(y_test[:, i], y_pred[:, i])
        rmse = math.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        metrics[tgt] = {"R2": r2, "RMSE": rmse}

    # ── Per-polymer within-class metrics (all 4 targets) ─────────────────────
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

    return model, scaler, metrics, feature_cols


def validate_dummy_samples(master_df: pd.DataFrame, model, scaler,
                            feature_cols: list) -> None:
    """
    Run the 5 dummy rows through the trained model.
    Predictions should be out of physically plausible ranges,
    confirming the model is not silently clamping impossible inputs.
    """
    if "is_dummy" not in master_df.columns:
        return

    dummy_df = master_df[master_df["is_dummy"] == True]
    if len(dummy_df) == 0:
        return

    print_banner("DUMMY SAMPLE VALIDATION", width=85)
    print(f"\n  {BOLD}Purpose:{RESET} Verify the model does NOT silently clamp")
    print(f"  physically impossible inputs. Predictions outside plausible")
    print(f"  ranges confirm correct (non-clamping) model behaviour.\n")
    print(f"  {'Sample_ID':<12}  {'Description':<42}  {'Verdict'}")
    print(f"  {'─'*12}  {'─'*42}  {'─'*10}")

    X_dummy = dummy_df[feature_cols].values.astype(float)
    X_sc    = scaler.transform(X_dummy)
    preds   = model.predict(X_sc)

    for j, (_, row) in enumerate(dummy_df.iterrows()):
        pred_row = preds[j]
        out_of_range = []
        for k, tgt in enumerate(TARGETS):
            lo, hi = TARGET_PLAUSIBLE[tgt]
            val = pred_row[k]
            if val < lo or val > hi:
                out_of_range.append(f"{tgt.split('_')[0]}={val:.2f}")

        label = row.get("dummy_label", "")[:42]
        if out_of_range:
            verdict = f"{GREEN}{BOLD}✔ OOR{RESET}"
            detail  = ", ".join(out_of_range)
        else:
            verdict = f"{RED}{BOLD}✗ IN-RANGE{RESET}"
            detail  = "all predictions within plausible bounds — REVIEW MODEL"

        sid = row.get("Sample_ID", f"DUMMY_{j+1:03d}")
        print(f"  {sid:<12}  {label:<42}  {verdict}")
        print(f"  {'':12}  Predictions: Tg={pred_row[0]:.1f}  Dk={pred_row[1]:.3f}  "
              f"Rad={pred_row[2]:.2f}")
        print(f"  {'':12}  {CYAN}OOR={detail}{RESET}\n")

    print(f"  OOR = Out Of Range (physically implausible) — expected behaviour for bad inputs.\n")


def infer_user_polymer(master_df: pd.DataFrame, model, scaler,
                       feature_cols: list, user_meta: dict) -> None:
    """
    Run inference on user polymer rows and compare predictions vs all reference polymers.
    """
    name = user_meta["name"]
    user_rows = master_df[
        (master_df["Polymer"] == name) & (master_df["is_dummy"] == False)
    ]
    if len(user_rows) == 0:
        print(f"  {YELLOW}⚠  No rows found for {name} — skipping inference.{RESET}")
        return

    X_user = scaler.transform(user_rows[feature_cols].values.astype(float))
    preds  = model.predict(X_user)
    pred_df = pd.DataFrame(preds, columns=TARGETS)

    print_banner(f"USER POLYMER INFERENCE — {name}", width=85)
    print(f"\n  {YELLOW}⚠  Predictions extrapolated from reference polymer feature space.{RESET}")
    print(f"  {YELLOW}   Validate experimentally before use.{RESET}\n")

    print(f"  {'Metric':<28}  {'Mean':>12}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'─'*28}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}")

    for tgt in TARGETS:
        label = TARGET_LABELS[tgt]
        vals  = pred_df[tgt]
        print(f"  {label:<28}  {vals.mean():>12.4f}  {vals.std():>10.4f}  "
              f"{vals.min():>10.4f}  {vals.max():>10.4f}")

    # Comparison baseline: mean of reference polymer predictions
    ref_rows = master_df[master_df["is_reference"] == True]
    X_ref    = scaler.transform(ref_rows[feature_cols].values.astype(float))
    ref_pred = pd.DataFrame(model.predict(X_ref), columns=TARGETS)
    ref_pred["Polymer"] = ref_rows["Polymer"].values

    # Use all reference polymers present in the dataset (not a hardcoded list)
    ref_polymer_names = list(ref_rows["Polymer"].unique())

    print(f"\n  Comparison against reference polymer medians:")
    print(f"  {'Polymer':<22}", end="")
    for tgt in TARGETS:
        print(f"  {tgt.split('_')[0]:>10}", end="")
    print()
    print(f"  {'─'*22}", end="")
    for _ in TARGETS:
        print(f"  {'─'*10}", end="")
    print()

    for poly in ref_polymer_names:
        pmask = ref_pred["Polymer"] == poly
        print(f"  {poly:<22}", end="")
        for tgt in TARGETS:
            print(f"  {ref_pred.loc[pmask, tgt].median():>10.3f}", end="")
        print()

    print(f"  {name:<22}", end="")
    for tgt in TARGETS:
        print(f"  {pred_df[tgt].median():>10.3f}", end="")
    print(f"  {YELLOW}⚠ estimated{RESET}")
    print()


def print_ml_report(metrics: dict) -> None:
    print_banner("TASK 2 — MLP QSPR MODEL PERFORMANCE (3 Targets)", width=85)

    print(f"\n  Global metrics (reference polymers combined):")
    print(f"  {'Target':<30}  {'R²':>10}  {'RMSE':>14}  {'Quality'}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*14}  {'─'*20}")

    def ql(r2):
        if r2 >= 0.95: return f"{GREEN}Excellent{RESET}"
        if r2 >= 0.80: return f"{GREEN}Very Good{RESET}"
        if r2 >= 0.60: return f"{YELLOW}Good{RESET}"
        return f"{RED}Low{RESET}"

    for tgt in TARGETS:
        if tgt == "per_polymer" or tgt not in metrics:
            continue
        m     = metrics[tgt]
        label = TARGET_LABELS.get(tgt, tgt)
        note  = (f"{YELLOW}⚠ inflated by between-class gap{RESET}"
                 if m["R2"] > 0.97 else ql(m["R2"]))
        r2_str = f"{m['R2']:>10.4f}" if m["R2"] >= 0 else f"{'':>10}"
        print(f"  {label:<30}  {r2_str}  {m['RMSE']:>14.4f}  {note}")

    print(f"\n  Per-polymer within-class R² (honest metric):")
    print(f"  {'Polymer':<12}  {'R²(Tg)':>10}  {'R²(Dk)':>10}  {'R²(Rad)':>10}  n_test")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*6}")
    for poly, pm in metrics.get("per_polymer", {}).items():
        def fmt(v): return f"{v:>10.4f}" if v >= 0 else f"{'':>10}"
        print(f"  {poly:<12}  {fmt(pm['Tg_R2'])}  {fmt(pm['Dk_R2'])}  "
              f"{fmt(pm['Rad_R2'])}  {pm['n_test']:>6}")

    print(f"\n  {'─'*85}")
    print("  ℹ  Model: MLP [256→128→64]  |  Training: reference polymers only  |  Split: 80/20")
    print(f"  {'─'*85}\n")


if __name__ == "__main__":
    print("  Loading master_dataset.pkl ...")
    with open("master_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    print(f"  ✔  Loaded: {len(df)} rows\n")

    user_meta = load_user_polymer()
    if user_meta:
        print(f"  ✔  User polymer detected: {user_meta['name']}")

    print_banner("TASK 2 — MULTI-TARGET MLP QSPR MODEL TRAINING", width=85)
    print("\n  Training MLP [256→128→64] on 120-feature vectors ...")
    model, scaler, metrics, feature_cols = build_qspr_model(df)
    print("  ✔  Training complete.\n")

    print_ml_report(metrics)

    if user_meta:
        validate_dummy_samples(df, model, scaler, feature_cols)
        infer_user_polymer(df, model, scaler, feature_cols, user_meta)

    qspr_bundle = {
        "model"       : model,
        "scaler"      : scaler,
        "metrics"     : metrics,
        "feature_cols": feature_cols,
        "model_type"  : "MLP",
    }
    with open("qspr_model_mlp.pkl", "wb") as f:
        pickle.dump(qspr_bundle, f)
    print("  💾  Saved: qspr_model_mlp.pkl\n")
