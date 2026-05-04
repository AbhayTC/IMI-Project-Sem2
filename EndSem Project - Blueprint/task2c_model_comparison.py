"""
================================================================================
  Informatics-Driven Design of High-Performance Polymers for Satellite Protection
  A Comprehensive QSPR Pipeline for Thermal Endurance & Dielectric Stability
================================================================================
  Author  : Senior Materials Informatics Researcher
  Domain  : Polymer Science | Aerospace Materials | Machine Learning
================================================================================

TASK 2C ▶ Model Comparison & SHAP Feature Importance Analysis

  Inputs : master_dataset.pkl       (produced by task1_data_curation.py)
           qspr_model_mlp.pkl       (produced by task2_qspr_modeling.py)
           qspr_model_gbr.pkl       (produced by task2b_gbr_modeling.py)

  Outputs: model_comparison.csv     (R² / RMSE table for both models)
           shap_importance_mlp.csv  (SHAP values per feature per target, MLP)
           shap_importance_gbr.csv  (SHAP values per feature per target, GBR)

  What this file does:
  ─────────────────────
  1. Loads both trained QSPR models.
  2. Prints a side-by-side performance table (R², RMSE) for all 4 targets.
  3. Uses SHAP (SHapley Additive exPlanations) to compute feature importance
     for BOTH models independently — giving a true apples-to-apples comparison
     of which molecular features each model relies on.
  4. Prints the top 15 features per target per model.
  5. Exports all results to CSV for your assignment submission.

  Requires:
  ──────────
    pip install shap

  Note on SHAP compute time:
  ───────────────────────────
    GBR SHAP  : fast  (~5–15 seconds) — uses TreeExplainer
    MLP SHAP  : slower (~30–90 seconds) — uses KernelExplainer with sampling
    Progress messages are printed so you know it hasn't hung.
================================================================================
"""

import warnings, pickle, os, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── SHAP import with helpful error message ────────────────────────────────────
try:
    import shap
except ImportError:
    print("\n  ✖  SHAP is not installed.")
    print("     Install it with:  pip install shap")
    print("     Then re-run this file.\n")
    sys.exit(1)

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"

TARGET_LABELS = {
    "Tg_degC"           : "Tg  (Glass Trans. Temp °C)",
    "Dk_1GHz"           : "Dk  (Dielectric Const.)",
    "Outgassing_TML_pct": "Outgassing (TML %)",
    "RadiationDose_MGy" : "Radiation Endurance (MGy)",
}
TARGETS = list(TARGET_LABELS.keys())


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def print_banner(text: str, char: str = "═", width: int = 88) -> None:
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{BOLD}{text}{RESET}{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")


def load_bundle(path: str, label: str) -> dict:
    """Load a model bundle, exit with a clear message if missing."""
    if not os.path.exists(path):
        print(f"\n  {RED}✖  {label} not found at '{path}'{RESET}")
        print(f"     Run the corresponding training script first:")
        if "mlp" in path:
            print(f"       python task2_qspr_modeling.py")
        else:
            print(f"       python task2b_gbr_modeling.py")
        sys.exit(1)
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"  ✔  Loaded: {path}")
    return bundle


def prepare_test_set(master_df: pd.DataFrame, feature_cols: list):
    """Returns scaled X_test and y_test using the same split as training."""
    targets = TARGETS
    X = master_df[feature_cols].values.astype(float)
    y = master_df[targets].values.astype(float)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )
    return X_test, y_test


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — SIDE-BY-SIDE PERFORMANCE TABLE
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(mlp_metrics: dict, gbr_metrics: dict) -> pd.DataFrame:
    """Prints and returns a side-by-side R² / RMSE comparison table."""
    print_banner("SECTION 1 — PERFORMANCE COMPARISON  (MLP  vs  GBR)", width=88)

    header = (f"\n  {'Target':<28}  "
              f"{'MLP R²':>9}  {'MLP RMSE':>10}  "
              f"{'GBR R²':>9}  {'GBR RMSE':>10}  "
              f"{'Winner':>8}")
    print(header)
    print(f"  {'─'*28}  {'─'*9}  {'─'*10}  {'─'*9}  {'─'*10}  {'─'*8}")

    rows = []
    for tgt in TARGETS:
        if tgt not in mlp_metrics or tgt not in gbr_metrics:
            continue
        mlp_r2   = mlp_metrics[tgt]["R2"]
        mlp_rmse = mlp_metrics[tgt]["RMSE"]
        gbr_r2   = gbr_metrics[tgt]["R2"]
        gbr_rmse = gbr_metrics[tgt]["RMSE"]

        # Winner = higher R² (primary) then lower RMSE (tiebreak)
        if mlp_r2 > gbr_r2 + 0.005:
            winner = f"{GREEN}MLP{RESET}"
            winner_label = "MLP"
        elif gbr_r2 > mlp_r2 + 0.005:
            winner = f"{GREEN}GBR{RESET}"
            winner_label = "GBR"
        else:
            winner = f"{YELLOW}TIE{RESET}"
            winner_label = "TIE"

        label = TARGET_LABELS.get(tgt, tgt)
        print(f"  {label:<28}  "
              f"{mlp_r2:>9.4f}  {mlp_rmse:>10.4f}  "
              f"{gbr_r2:>9.4f}  {gbr_rmse:>10.4f}  "
              f"{winner:>8}")

        rows.append({
            "Target"   : tgt,
            "MLP_R2"   : round(mlp_r2,   4),
            "MLP_RMSE" : round(mlp_rmse, 4),
            "GBR_R2"   : round(gbr_r2,   4),
            "GBR_RMSE" : round(gbr_rmse, 4),
            "Winner"   : winner_label,
        })

    print(f"\n  {'─'*88}")
    print("  ℹ  Winner determined by R²; ties declared when difference < 0.005")
    print(f"  {'─'*88}")

    # Per-polymer within-class R² comparison
    print(f"\n  {'Per-polymer within-class R²':─<60}")
    print(f"  {'Polymer':<12}  {'MLP R²(Tg)':>12}  {'GBR R²(Tg)':>12}  "
          f"{'MLP R²(Dk)':>12}  {'GBR R²(Dk)':>12}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")

    mlp_pp = mlp_metrics.get("per_polymer", {})
    gbr_pp = gbr_metrics.get("per_polymer", {})
    for poly in mlp_pp:
        if poly not in gbr_pp:
            continue
        print(f"  {poly:<12}  "
              f"{mlp_pp[poly]['Tg_R2']:>12.4f}  {gbr_pp[poly]['Tg_R2']:>12.4f}  "
              f"{mlp_pp[poly]['Dk_R2']:>12.4f}  {gbr_pp[poly]['Dk_R2']:>12.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv("model_comparison.csv", index=False)
    print(f"\n  💾  Saved: model_comparison.csv\n")
    return df_out


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — SHAP FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_gbr(model, X_test_scaled: np.ndarray,
                     feature_cols: list) -> dict:
    """
    Compute SHAP values for GBR using TreeExplainer — fast and exact.
    Returns {target_name: [mean |SHAP| per feature ranked descending]}
    """
    print(f"  {CYAN}Computing SHAP for GBR (TreeExplainer) ...{RESET}")
    shap_ranked = {}

    for i, tgt in enumerate(TARGETS):
        estimator = model.estimators_[i]   # individual GBR per target
        explainer = shap.TreeExplainer(estimator)
        shap_vals = explainer.shap_values(X_test_scaled)   # shape (n, n_features)
        mean_abs  = np.abs(shap_vals).mean(axis=0)

        ranked = sorted(
            zip(feature_cols, mean_abs),
            key=lambda x: x[1], reverse=True
        )
        shap_ranked[tgt] = ranked
        print(f"    ✔  {TARGET_LABELS.get(tgt, tgt)}")

    return shap_ranked


def compute_shap_mlp(model, X_train_scaled: np.ndarray,
                     X_test_scaled: np.ndarray,
                     feature_cols: list) -> dict:
    """
    Compute SHAP values for MLP using KernelExplainer.
    KernelExplainer is model-agnostic (works on any black-box model) but
    is slower. We use a background summary (50 samples) to keep runtime
    reasonable while maintaining accuracy.
    """
    print(f"\n  {CYAN}Computing SHAP for MLP (KernelExplainer) ...{RESET}")
    print(f"  {YELLOW}  ⚠ This may take 30–90 seconds — please wait.{RESET}")

    shap_ranked = {}

    # Background dataset: 50 random training samples summarise the feature
    # distribution. KernelExplainer computes how much each feature shifts
    # the prediction away from the background average.
    background = shap.sample(X_train_scaled, 50, random_state=42)

    for i, tgt in enumerate(TARGETS):
        print(f"    → Computing {TARGET_LABELS.get(tgt, tgt)} ...")

        # Wrapper: predict only the i-th output column
        def predict_single_target(X, idx=i):
            return model.predict(X)[:, idx]

        explainer = shap.KernelExplainer(predict_single_target, background)

        # Use 100 test samples — enough for stable importance estimates
        n_explain  = min(100, len(X_test_scaled))
        X_explain  = X_test_scaled[:n_explain]
        shap_vals  = explainer.shap_values(X_explain, silent=True)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        ranked   = sorted(
            zip(feature_cols, mean_abs),
            key=lambda x: x[1], reverse=True
        )
        shap_ranked[tgt] = ranked
        print(f"    ✔  {TARGET_LABELS.get(tgt, tgt)}")

    return shap_ranked


def print_shap_table(shap_mlp: dict, shap_gbr: dict, top_n: int = 15) -> None:
    """Prints a side-by-side SHAP importance table for MLP and GBR."""
    print_banner(f"SECTION 2 — SHAP FEATURE IMPORTANCE  (Top {top_n})  MLP  vs  GBR",
                 width=88)

    print(f"\n  {BOLD}What is mean |SHAP|?{RESET}")
    print(f"  Each value is the average absolute SHAP contribution of that feature")
    print(f"  across all test predictions. Higher = more influential for that target.\n")

    for tgt in ["Tg_degC", "Dk_1GHz"]:
        label    = TARGET_LABELS.get(tgt, tgt)
        mlp_list = shap_mlp.get(tgt, [])
        gbr_list = shap_gbr.get(tgt, [])

        print(f"\n  {'─'*88}")
        print(f"  {BOLD}{CYAN}Target: {label}{RESET}")
        print(f"  {'─'*88}")
        print(f"  {'Rank':<5}  {'MLP Feature':<30}  {'MLP |SHAP|':>11}  "
              f"{'GBR Feature':<30}  {'GBR |SHAP|':>11}")
        print(f"  {'─'*5}  {'─'*30}  {'─'*11}  {'─'*30}  {'─'*11}")

        for rank in range(top_n):
            mlp_feat = mlp_list[rank][0] if rank < len(mlp_list) else "—"
            mlp_val  = f"{mlp_list[rank][1]:.5f}" if rank < len(mlp_list) else "—"
            gbr_feat = gbr_list[rank][0] if rank < len(gbr_list) else "—"
            gbr_val  = f"{gbr_list[rank][1]:.5f}" if rank < len(gbr_list) else "—"
            print(f"  {rank+1:<5}  {mlp_feat:<30}  {mlp_val:>11}  "
                  f"{gbr_feat:<30}  {gbr_val:>11}")


def save_shap_csvs(shap_mlp: dict, shap_gbr: dict,
                   feature_cols: list) -> None:
    """Export full SHAP importance tables to CSV."""
    for label, shap_dict, fname in [
        ("MLP", shap_mlp, "shap_importance_mlp.csv"),
        ("GBR", shap_gbr, "shap_importance_gbr.csv"),
    ]:
        rows = []
        for tgt, ranked in shap_dict.items():
            for rank, (feat, val) in enumerate(ranked, 1):
                rows.append({
                    "Model"    : label,
                    "Target"   : tgt,
                    "Rank"     : rank,
                    "Feature"  : feat,
                    "Mean_Abs_SHAP": round(float(val), 6),
                })
        pd.DataFrame(rows).to_csv(fname, index=False)
        print(f"  💾  Saved: {fname}")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_banner("TASK 2C — MODEL COMPARISON & SHAP FEATURE IMPORTANCE ANALYSIS",
                 width=88)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("\n  Loading inputs ...\n")
    with open("master_dataset.pkl", "rb") as f:
        master_df = pickle.load(f)
    print(f"  ✔  Loaded: master_dataset.pkl  ({len(master_df)} rows)")

    mlp_bundle = load_bundle("qspr_model_mlp.pkl", "MLP model")
    gbr_bundle = load_bundle("qspr_model_gbr.pkl", "GBR model")

    mlp_model    = mlp_bundle["model"]
    mlp_scaler   = mlp_bundle["scaler"]
    mlp_metrics  = mlp_bundle["metrics"]
    feature_cols = mlp_bundle["feature_cols"]   # same for both models

    gbr_model    = gbr_bundle["model"]
    gbr_scaler   = gbr_bundle["scaler"]
    gbr_metrics  = gbr_bundle["metrics"]

    # ── Prepare data ──────────────────────────────────────────────────────────
    X_raw = master_df[feature_cols].values.astype(float)
    y_raw = master_df[TARGETS].values.astype(float)
    X_train_raw, X_test_raw, _, _ = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, shuffle=True
    )
    X_train_mlp = mlp_scaler.transform(X_train_raw)
    X_test_mlp  = mlp_scaler.transform(X_test_raw)
    X_train_gbr = gbr_scaler.transform(X_train_raw)
    X_test_gbr  = gbr_scaler.transform(X_test_raw)

    # ── Section 1: Performance comparison ────────────────────────────────────
    print_comparison_table(mlp_metrics, gbr_metrics)

    # ── Section 2: SHAP feature importance ───────────────────────────────────
    shap_gbr = compute_shap_gbr(gbr_model, X_test_gbr, feature_cols)
    shap_mlp = compute_shap_mlp(mlp_model, X_train_mlp, X_test_mlp, feature_cols)

    print_shap_table(shap_mlp, shap_gbr, top_n=15)
    print()
    save_shap_csvs(shap_mlp, shap_gbr, feature_cols)

    print(f"\n  {'─'*88}")
    print(f"  {GREEN}{BOLD}Analysis complete.{RESET}")
    print(f"  Use the comparison above to decide which model to load in task3.")
    print(f"  {'─'*88}\n")
