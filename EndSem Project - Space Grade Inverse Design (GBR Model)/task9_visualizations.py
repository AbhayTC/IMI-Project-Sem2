"""
================================================================================
  TASK 9 ▶ Visualization Suite (5 Publication-Quality Graphs)

  Graphs produced:
    1. Inverse Design Parity Plot  — Forward Tg (true) vs Inverse Design Tg
       (predicted), with perfect-fit red reference line (y = x).
    2. Top-15 GBR Feature Importance — Horizontal bar chart colour-coded by
       feature block (Structural · Latent · Physics).
    3. GBR Absolute Error Histogram — Distribution of |y_pred − y_true| for
       Tg on the test set, with KDE overlay.
    4. Test MSE Comparison          — Test MSE vs training progress for both
       MLP and GBR on the Tg target (single overlaid plot).

  Inputs : master_dataset.pkl   (Task 1  output)
           qspr_model_mlp.pkl   (Task 2  output — MLP bundle)
           qspr_model_gbr.pkl   (Task 2b output — GBR bundle)
  Outputs: Graphs/task9_graph_*.png  (one file per graph, 300 dpi)

  Design notes
  ─────────────
  • Models are loaded from the pkl bundles produced by tasks 2 / 2b —
    no re-training occurs.  This guarantees graphs reflect the exact same
    models used throughout the pipeline.
  • The test split (random_state=42, 20 %) is re-applied to master_dataset
    so test-set graphs (2, 3, 4) use the same held-out rows the training
    tasks evaluated against.
  • Graph 4 (training curves) reads loss_curve_ / validation_scores_ from
    the loaded MLP estimators and re-runs staged_predict with the loaded GBR.
================================================================================
"""
import warnings, math, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
#  SHARED CONSTANTS  (verbatim from tasks 1 / 2 / 2b / 3)
# ──────────────────────────────────────────────────────────────────────────────
TARGETS = ["Tg_degC", "Dk_1GHz", "RadiationDose_MGy"]

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


POLYMER_COLORS = {
    "Polyimide"           : "#4e79d0",
    "PEEK"                : "#3dba72",
    "PTFE"                : "#e6b800",
    "CyanateEster"        : "#cc66cc",
    "FluorinatedPolyimide": "#40c8c8",
    "Polybenzoxazole"     : "#e8732a",
}

BLOCK_COLORS = {
    "Structural": "#5b9bd5",
    "Latent"    : "#70ad47",
    "Physics"   : "#ed7d31",
}

# ──────────────────────────────────────────────────────────────────────────────
#  PIPELINE DATA LOADER  (replaces synthesize_dataset + train_models)
# ──────────────────────────────────────────────────────────────────────────────

def load_pipeline_data(dataset_path: str, mlp_path: str, gbr_path: str) -> dict:
    """
    Load master_dataset.pkl, qspr_model_mlp.pkl, and qspr_model_gbr.pkl
    produced by tasks 1, 2, and 2b.  Reconstructs the same 80/20 test split
    used during training (random_state=42) so all test-set graphs are consistent.

    Returns a models_dict compatible with all graph functions.
    """
    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"  Loading {dataset_path} …")
    with open(dataset_path, "rb") as f:
        df = pickle.load(f)
    if "is_reference" in df.columns:
        df = df[df["is_reference"] == True].copy()
    print(f"  ✔  Dataset: {len(df)} reference rows\n")

    # ── Load MLP bundle (task 2) ──────────────────────────────────────────────
    print(f"  Loading {mlp_path} …")
    with open(mlp_path, "rb") as f:
        mlp_bundle = pickle.load(f)
    mlp_model   = mlp_bundle["model"]    # MultiOutputRegressor(MLPRegressor)
    mlp_scaler  = mlp_bundle["scaler"]
    feature_cols = mlp_bundle["feature_cols"]
    print(f"  ✔  MLP bundle loaded  ({len(feature_cols)} features)\n")

    # ── Load GBR bundle (task 2b) ─────────────────────────────────────────────
    print(f"  Loading {gbr_path} …")
    with open(gbr_path, "rb") as f:
        gbr_bundle = pickle.load(f)
    gbr_model  = gbr_bundle["model"]     # MultiOutputRegressor(GBR)
    gbr_scaler = gbr_bundle["scaler"]
    gbr_feat_importance = gbr_bundle["feature_importance"]   # dict tgt→ranked list
    print(f"  ✔  GBR bundle loaded\n")

    # ── Reconstruct the same test split ───────────────────────────────────────
    X = df[feature_cols].values.astype(float)
    y = df[TARGETS].values.astype(float)
    poly_labels = df["Polymer"].values

    X_tr, X_te, y_tr, y_te, pl_tr, pl_te = train_test_split(
        X, y, poly_labels, test_size=0.20, random_state=42, shuffle=True
    )

    # Both bundles share the same scaler (fitted on same split); use MLP's
    X_te_sc = mlp_scaler.transform(X_te)
    X_tr_sc = mlp_scaler.transform(X_tr)

    mlp_pred_te = mlp_model.predict(X_te_sc)
    gbr_pred_te = gbr_model.predict(X_te_sc)

    # ── MLP loss / validation curves from stored estimators ──────────────────
    mlp_loss_curves = [est.loss_curve_        for est in mlp_model.estimators_]
    mlp_val_curves  = [est.validation_scores_ for est in mlp_model.estimators_]

    # ── GBR staged MSE on both test and train sets ────────────────────────────
    gbr_estimators = gbr_model.estimators_   # list of GBR objects, one per target
    gbr_staged_te  = []
    gbr_staged_tr  = []
    for i, gbr_est in enumerate(gbr_estimators):
        staged_mse_te = [
            mean_squared_error(y_te[:, i], s_pred)
            for s_pred in gbr_est.staged_predict(X_te_sc)
        ]
        staged_mse_tr = [
            mean_squared_error(y_tr[:, i], s_pred)
            for s_pred in gbr_est.staged_predict(X_tr_sc)
        ]
        gbr_staged_te.append(staged_mse_te)
        gbr_staged_tr.append(staged_mse_tr)

    # ── Convert feature_importance to the list-of-tuples format graphs expect ─
    # gbr_bundle["feature_importance"] is dict{tgt: [(feat, val), …]}
    gbr_feat_imp = [gbr_feat_importance[tgt] for tgt in TARGETS]

    print("  ✔  Test split reconstructed and predictions computed.\n")
    return {
        "df"             : df,
        "feature_cols"   : feature_cols,
        "X_tr_sc"        : X_tr_sc,
        "X_te_sc"        : X_te_sc,
        "y_tr"           : y_tr,
        "y_te"           : y_te,
        "poly_te"        : pl_te,
        "mlp_model"      : mlp_model,
        "mlp_scaler"     : mlp_scaler,
        "mlp_pred_te"    : mlp_pred_te,
        "mlp_loss_curves": mlp_loss_curves,
        "mlp_val_curves" : mlp_val_curves,
        "gbr_model"      : gbr_model,
        "gbr_scaler"     : gbr_scaler,
        "gbr_pred_te"    : gbr_pred_te,
        "gbr_staged_te"  : gbr_staged_te,
        "gbr_staged_tr"  : gbr_staged_tr,
        "gbr_feat_imp"   : gbr_feat_imp,
        "scaler"         : mlp_scaler,   # alias used by simulate_inverse_design
        "gbr_models"     : gbr_estimators,  # alias used by simulate_inverse_design
    }



# ──────────────────────────────────────────────────────────────────────────────
#  INVERSE DESIGN SIMULATION  (mirrors task3 composite_score)
# ──────────────────────────────────────────────────────────────────────────────

def composite_score(tg, dk, rad) -> float:
    d_tg  = 1 / (1 + np.exp(-0.02 * (tg - 200)))
    d_dk  = float(np.clip((5.0 - dk)  / (5.0 - 2.0), 0, 1))
    d_rad = float(np.clip((rad - 10)  / (40.0 - 10.0), 0, 1))
    return 0.40 * d_tg + 0.30 * d_dk + 0.30 * d_rad


def simulate_inverse_design(models_dict: dict) -> pd.DataFrame:
    """
    Simulate Task 3 inverse design candidates using the loaded GBR.
    For each reference polymer, takes rows from master_dataset, perturbs
    features slightly to simulate SMILES mutation candidates, then predicts
    and scores with the GBR.  Returns a DataFrame with true_Tg and pred_Tg.
    """
    df           = models_dict["df"]
    gbr_models   = models_dict["gbr_models"]
    scaler       = models_dict["scaler"]
    feature_cols = models_dict["feature_cols"]
    rng          = np.random.RandomState(7)

    target_Tg = {
        "Polyimide"           : 300.0,
        "PEEK"                : 280.0,
        "PTFE"                :  50.0,
        "CyanateEster"        : 500.0,
        "FluorinatedPolyimide": 400.0,
        "Polybenzoxazole"     : 650.0,
    }

    rows = []
    for poly in df["Polymer"].unique():
        poly_df = df[df["Polymer"] == poly].sample(n=min(80, len(df[df["Polymer"]==poly])),
                                                    random_state=42)
        X_base  = poly_df[feature_cols].values.astype(float)
        true_tg = poly_df["Tg_degC"].values

        # Simulate mutations: small Gaussian perturbations to features
        for j in range(len(X_base)):
            for _ in range(3):  # 3 mutation candidates per sample
                noise = rng.normal(0, 0.03, size=X_base[j].shape) * np.abs(X_base[j])
                X_mut = X_base[j] + noise
                X_sc  = scaler.transform(X_mut.reshape(1, -1))
                pred_Tg  = gbr_models[0].predict(X_sc)[0]   # Tg estimator
                pred_Dk  = gbr_models[1].predict(X_sc)[0]
                pred_Rad = gbr_models[2].predict(X_sc)[0]
                score    = composite_score(pred_Tg, pred_Dk, pred_Rad)
                rows.append({
                    "Polymer"    : poly,
                    "true_Tg"    : true_tg[j],
                    "pred_Tg"    : pred_Tg,
                    "pred_Dk"    : pred_Dk,
                    "pred_Rad"   : pred_Rad,
                    "score"      : score,
                    "target_Tg"  : target_Tg[poly],
                })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
#  GRAPH STYLE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

STYLE = {
    "figure.facecolor" : "#0f1117",
    "axes.facecolor"   : "#161b27",
    "axes.edgecolor"   : "#2e3650",
    "axes.labelcolor"  : "#d0d8f0",
    "axes.titlecolor"  : "#e8ecfc",
    "xtick.color"      : "#8898c0",
    "ytick.color"      : "#8898c0",
    "grid.color"       : "#242d42",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.6,
    "text.color"       : "#d0d8f0",
    "legend.facecolor" : "#1a2035",
    "legend.edgecolor" : "#2e3650",
    "font.family"      : "DejaVu Sans",
}

def apply_style():
    for k, v in STYLE.items():
        plt.rcParams[k] = v
    plt.rcParams["axes.spines.top"]   = False
    plt.rcParams["axes.spines.right"] = False


# ──────────────────────────────────────────────────────────────────────────────
#  GRAPH 1 — Inverse Design Parity Plot  (Forward Tg vs Predicted Tg)
# ──────────────────────────────────────────────────────────────────────────────

def graph1_parity_plot(md: dict, out_path: str):
    """
    2 x 3 facet grid — one panel per polymer class.
    Each panel shows within-class scatter only, so R² and RMSE reflect
    the genuinely hard within-class prediction problem (no between-class
    variance inflation).
    """
    apply_style()

    y_true      = md["y_te"][:, 0]
    y_pred      = md["gbr_pred_te"][:, 0]
    poly_labels = md["poly_te"]

    polymers = sorted(np.unique(poly_labels))
    n_cols, n_rows = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 9))
    axes_flat = axes.flatten()

    for idx, poly in enumerate(polymers):
        ax    = axes_flat[idx]
        mask  = poly_labels == poly
        yt    = y_true[mask]
        yp    = y_pred[mask]
        color = POLYMER_COLORS.get(poly, "#aaaaaa")

        ax.scatter(yt, yp, c=color, alpha=0.70, s=28,
                   edgecolors="none", rasterized=True)

        pad = (yt.max() - yt.min()) * 0.08 + 5
        lo  = min(yt.min(), yp.min()) - pad
        hi  = max(yt.max(), yp.max()) + pad
        ax.plot([lo, hi], [lo, hi], color="#e84040", linewidth=1.6,
                linestyle="--", zorder=5)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        r2   = r2_score(yt, yp)
        rmse = math.sqrt(mean_squared_error(yt, yp))
        ax.text(0.05, 0.95,
                f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.1f} °C\nn = {mask.sum()}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                color="#c8d8ff",
                bbox=dict(boxstyle="round,pad=0.35", fc="#1a2035",
                          ec="#2e3650", alpha=0.85))

        ax.set_title(poly, fontsize=11, pad=6, color=color)
        ax.set_xlabel("True Tg  (°C)", fontsize=9, labelpad=4)
        ax.set_ylabel("GBR Predicted Tg  (°C)", fontsize=9, labelpad=4)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.30)

    for idx in range(len(polymers), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Graph 1 — Per-Polymer Parity Plot: True vs GBR-Predicted Tg\n"
        "Within-class R² and RMSE (honest metric — no between-class inflation)",
        fontsize=13, y=1.01, color="#e8ecfc",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Graph 1 saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  GRAPH 3 — Absolute Error Histogram for GBR Tg (Test Set)
# ──────────────────────────────────────────────────────────────────────────────

def graph3_error_histogram(md: dict, out_path: str):
    apply_style()

    y_te     = md["y_te"][:, 0]   # Tg column
    gbr_pred = md["gbr_pred_te"][:, 0]
    abs_err  = np.abs(y_te - gbr_pred)

    fig, ax = plt.subplots(figsize=(9, 6))

    n_bins   = 38
    counts, bin_edges, patches = ax.hist(
        abs_err, bins=n_bins, color="#5b9bd5", edgecolor="#0f1117",
        linewidth=0.4, alpha=0.82, label="Absolute error frequency"
    )

    ax.set_xlabel("Absolute Error  |Tg predicted − Tg true|  (°C)",
                  fontsize=12, labelpad=8)
    ax.set_ylabel("Frequency  (sample count)", fontsize=12, labelpad=8)
    ax.set_title("Graph 3 — GBR Tg Absolute Error Distribution\n"
                 "Test set (20 % hold-out) across all 6 reference polymer classes",
                 fontsize=13, pad=14, color="#e8ecfc")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Graph 3 saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  GRAPH 4 — Training Curves: MLP Validation Loss + GBR Staged Test MSE
# ──────────────────────────────────────────────────────────────────────────────

def graph4_training_curves(md: dict, out_path: str):
    apply_style()

    tgt_idx = 0   # Tg estimator

    # ── MLP: training loss curve (genuine MSE in °C² on training set) ─────────
    mlp_loss  = np.array(md["mlp_loss_curves"][tgt_idx])
    mlp_iters = np.arange(1, len(mlp_loss) + 1)

    # ── GBR: staged training MSE (°C² on training set) ───────────────────────
    gbr_mse     = np.array(md["gbr_staged_tr"][tgt_idx])
    gbr_n_trees = np.arange(1, len(gbr_mse) + 1)

    # ── Two side-by-side panels, sharey=False (independent y-limits) ─────────
    fig, (ax_mlp, ax_gbr) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # ── Left panel: MLP ───────────────────────────────────────────────────────
    ax_mlp.plot(mlp_iters, mlp_loss, color="#5b9bd5", linewidth=2.0, alpha=0.9)
    ax_mlp.set_xlabel("Iterations", fontsize=12, labelpad=8)
    ax_mlp.set_ylabel("MSE  (Tg, °C²)", fontsize=12, labelpad=8)
    ax_mlp.set_title(f"MLP [256 → 128 → 64]  —  Training Loss\n"
                     f"{len(mlp_iters)} Iterations  |  Tg target",
                     fontsize=11, pad=10, color="#e8ecfc")
    ax_mlp.set_yscale("log")
    ax_mlp.grid(True, alpha=0.35)
    ax_mlp.legend(
        handles=[Line2D([0], [0], color="#5b9bd5", linewidth=2)],
        labels=["MLP train MSE"],
        fontsize=9, framealpha=0.9,
    )

    # ── Right panel: GBR ──────────────────────────────────────────────────────
    ax_gbr.plot(gbr_n_trees, gbr_mse, color="#70ad47", linewidth=2.0,
                linestyle="--", alpha=0.9)
    ax_gbr.set_xlabel("Number of Boosting Trees", fontsize=12, labelpad=8)
    ax_gbr.set_ylabel("MSE  (Tg, °C²)", fontsize=12, labelpad=8)
    ax_gbr.set_title(f"GBR [300 trees, depth=4, lr=0.05]  —  Training MSE\n"
                     f"{len(gbr_n_trees)} trees  |  Tg target  |  80 % training set",
                     fontsize=11, pad=10, color="#e8ecfc")
    ax_gbr.set_yscale("log")
    ax_gbr.grid(True, alpha=0.35)
    ax_gbr.legend(
        handles=[Line2D([0], [0], color="#70ad47", linewidth=2, linestyle="--")],
        labels=["GBR train MSE"],
        fontsize=9, framealpha=0.9,
    )

    fig.suptitle("Graph 4 — Training MSE Curves: MLP (left) vs GBR (right)\n"
                 "Independent y-limits  |  Same units (MSE, °C²)  |  Log scale",
                 fontsize=13, y=1.02, color="#e8ecfc")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Graph 4 saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  GRAPH 5 — Top-15 GBR Feature Importance (colour-coded by block)
# ──────────────────────────────────────────────────────────────────────────────

def graph2_feature_importance(md: dict, out_path: str):
    apply_style()

    tgt_idx   = 0   # Tg estimator (most informative)
    feat_imp  = md["gbr_feat_imp"][tgt_idx]
    feat_imp_sorted = sorted(feat_imp, key=lambda x: x[1], reverse=True)[:15]

    names  = [f for f, _ in feat_imp_sorted]
    values = [v for _, v in feat_imp_sorted]

    # Determine block membership
    struct_set  = set(STRUCTURAL_FEATURE_NAMES)
    latent_set  = set(LATENT_FEATURE_NAMES)
    physics_set = set(PHYSICS_FEATURE_NAMES)

    bar_colors = []
    block_labels = []
    for feat in names:
        if feat in struct_set:
            bar_colors.append(BLOCK_COLORS["Structural"])
            block_labels.append("Structural")
        elif feat in latent_set:
            bar_colors.append(BLOCK_COLORS["Latent"])
            block_labels.append("Latent")
        else:
            bar_colors.append(BLOCK_COLORS["Physics"])
            block_labels.append("Physics")

    # Shorten long feature names for display
    def shorten(name: str) -> str:
        subs = {
            "polyBERT_dim_": "BERT_",
            "DegreeOfCrystallinity": "Crystallinity",
            "CrystallinePhaseContent": "CrystalPhase",
            "AmorphousPhaseContent": "AmorphPhase",
            "FreeVolumeFraction": "FreeVolume",
            "ChainRigidityIndex": "ChainRigidity",
            "DielectricPolarizability": "DielPolariz.",
            "ElectronicPolarizability": "ElecPolariz.",
            "CrosslinkingDensity": "CrosslinkDens.",
            "EntanglementMolWt": "EntangMolWt",
            "PersistenceLength": "PersistLen.",
            "ThermalExpansionCoeff": "ThermExpCoeff",
            "NumValenceElectrons": "ValElectrons",
            "PermittivityRealPart": "Permittivity_Re",
        }
        for long, short in subs.items():
            name = name.replace(long, short)
        return name

    display_names = [shorten(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, values, color=bar_colors, edgecolor="#0f1117",
                   linewidth=0.4, height=0.68)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.0003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8.5,
                color="#c8d8ff")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (mean decrease in impurity)", fontsize=11, labelpad=8)
    ax.set_ylabel("Feature Name", fontsize=11, labelpad=8)
    ax.set_title("Graph 2 — Top-15 GBR Feature Importances for Tg Prediction\n"
                 "Colour-coded by feature block: Structural · Latent (polyBERT) · Physics",
                 fontsize=13, pad=14, color="#e8ecfc")
    ax.grid(True, axis="x", alpha=0.35)

    # Legend for blocks
    legend_patches = [
        mpatches.Patch(color=BLOCK_COLORS["Structural"], label="Structural (RDKit descriptors)"),
        mpatches.Patch(color=BLOCK_COLORS["Latent"],     label="Latent (polyBERT dims)"),
        mpatches.Patch(color=BLOCK_COLORS["Physics"],    label="Physics (domain priors)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.9,
              loc="lower right", title="Feature Block", title_fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Graph 2 saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # ── Paths ─────────────────────────────────────────────────────────────────
    HERE       = os.path.dirname(os.path.abspath(__file__))
    GRAPHS_DIR = os.path.join(HERE, "Graphs")
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    def out(name): return os.path.join(GRAPHS_DIR, name)

    print("\n" + "═" * 72)
    print("  TASK 9 — VISUALIZATION SUITE  (4 graphs)")
    print("═" * 72 + "\n")
    print(f"  Output directory: {GRAPHS_DIR}\n")

    # ── Step 1: Load pipeline artefacts (dataset + trained models) ────────────
    models_dict = load_pipeline_data(
        dataset_path = os.path.join(HERE, "master_dataset.pkl"),
        mlp_path     = os.path.join(HERE, "qspr_model_mlp.pkl"),
        gbr_path     = os.path.join(HERE, "qspr_model_gbr.pkl"),
    )

    # ── Step 2: Simulate inverse design candidates ────────────────────────────
    print("  Simulating inverse design candidates …")
    inv_df = simulate_inverse_design(models_dict)
    print(f"  ✔  {len(inv_df)} candidate entries generated.\n")

    # ── Step 3: Render graphs ─────────────────────────────────────────────────
    print("  Rendering graphs …\n")

    graph1_parity_plot(models_dict,         out("task9_graph1_parity_plot.png"))
    graph2_feature_importance(models_dict,  out("task9_graph2_feature_importance.png"))
    graph3_error_histogram(models_dict,     out("task9_graph3_error_histogram.png"))
    graph4_training_curves(models_dict,     out("task9_graph4_training_curves.png"))

    print("\n" + "═" * 72)
    print(f"  ✔  Task 9 complete — 4 graphs saved to: {GRAPHS_DIR}")
    print("═" * 72 + "\n")
