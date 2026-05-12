"""
================================================================================
  TASK 2 ▶ QSPR Modeling (Multi-Target MLP — PyTorch)

  CHANGES (this version):
    1. ALL 3 TARGETS reported (Tg, Dk, Radiation) — skip_targets removed
    2. TRAINING FILTER: trains only on is_reference=True rows
    3. DUMMY VALIDATION: after training, runs dummy rows through the model
    4. USER POLYMER INFERENCE: if user_polymer.pkl exists, predicts all 3
       targets and prints a comparison vs all reference polymers
    5. PER-POLYMER R² covers all 3 targets
    6. PhysicsGuidedLoss: penalises predictions that deviate from Task-1
       heuristic baselines by more than `safe_margin` (normalised per-target)

  Inputs : master_dataset.pkl      (produced by task1_data_curation.py)
  Outputs: qspr_model_mlp.pkl      (loaded by task3_inverse_design.py)
================================================================================
"""
import warnings, math, pickle, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

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

# Corresponding Task-1 heuristic column names (if present in master_dataset).
# If a column is absent the code falls back to the per-polymer training mean.
HEURISTIC_COLS = {
    "Tg_degC"           : "Heuristic_Tg_degC",
    "Dk_1GHz"           : "Heuristic_Dk_1GHz",
    "RadiationDose_MGy" : "Heuristic_RadiationDose_MGy",
}

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

# ── Per-target safe margins for PhysicsGuidedLoss ─────────────────────────────
# Values are in *normalised* (z-score) units so the same loss function handles
# targets with very different raw scales (Tg ~hundreds vs Dk ~single digits).
# 0.5 σ  ≈  "predictions should stay within half a standard deviation of the
#            Task-1 heuristic baseline before incurring a physics penalty."
# Tune upward (e.g. 1.0) to relax, downward (e.g. 0.25) to tighten.
SAFE_MARGIN_SIGMA = 0.5          # same for all 3 targets; override per-target below if needed
PHYSICS_PENALTY_MULTIPLIER = 10.0


# ══════════════════════════════════════════════════════════════════════════════
#  1.  PHYSICS-GUIDED LOSS
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsGuidedLoss(nn.Module):
    """
    MSE + a soft penalty whenever predictions deviate from heuristic baselines
    by more than `safe_margin` (in the same units as predictions / baselines).

    Parameters
    ----------
    penalty_multiplier : float
        Weight applied to the physics penalty term (default 10.0).
    safe_margin : float
        Deviation threshold before the penalty activates (default 0.5 when
        working in z-score space — see normalisation in build_qspr_model).
    """
    def __init__(self, penalty_multiplier: float = 10.0, safe_margin: float = 0.5):
        super(PhysicsGuidedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.penalty_multiplier = penalty_multiplier
        self.safe_margin = safe_margin

    def forward(
        self,
        predictions: torch.Tensor,       # (batch, n_targets)
        targets: torch.Tensor,            # (batch, n_targets)
        heuristic_baselines: torch.Tensor # (batch, n_targets)  — z-score normalised
    ) -> torch.Tensor:
        standard_loss = self.mse(predictions, targets)

        deviation    = torch.abs(predictions - heuristic_baselines)
        penalty_mask = (deviation > self.safe_margin).float()
        physics_penalty = torch.mean(
            penalty_mask * deviation * self.penalty_multiplier
        )

        return standard_loss + physics_penalty


# ══════════════════════════════════════════════════════════════════════════════
#  2.  MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class PolymerMLP(nn.Module):
    """
    Feed-forward MLP: input_dim → 256 → 128 → 64 → n_targets.
    Mirrors the hidden-layer spec of the previous sklearn MLPRegressor.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(PolymerMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  3.  HEURISTIC BASELINE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _get_heuristics(df: pd.DataFrame, idx: pd.Index) -> np.ndarray:
    """
    Return a (len(idx), n_targets) array of heuristic baseline values.

    Strategy (per target):
      1. Use the Task-1 heuristic column if it exists in `df`.
      2. Fall back to the per-polymer training-set mean of that target.
      3. Last resort: global training mean.
    """
    baselines = np.zeros((len(idx), len(TARGETS)), dtype=float)
    sub = df.loc[idx]

    for j, tgt in enumerate(TARGETS):
        hcol = HEURISTIC_COLS[tgt]
        if hcol in df.columns:
            baselines[:, j] = sub[hcol].values.astype(float)
        elif "Polymer" in df.columns:
            # per-polymer mean of the actual target
            poly_means = (
                df.loc[idx, ["Polymer", tgt]]
                .groupby("Polymer")[tgt]
                .transform("mean")
            )
            baselines[:, j] = poly_means.values.astype(float)
        else:
            baselines[:, j] = df.loc[idx, tgt].mean()

    return baselines


# ══════════════════════════════════════════════════════════════════════════════
#  4.  SKLEARN-COMPATIBLE WRAPPER  (so the rest of the pipeline stays intact)
# ══════════════════════════════════════════════════════════════════════════════

class _TorchModelWrapper:
    """
    Thin wrapper that exposes sklearn-style `.predict(X)` so that
    validate_dummy_samples / infer_user_polymer need no changes.
    """
    def __init__(self, torch_model: PolymerMLP, target_scaler: StandardScaler):
        self.torch_model  = torch_model
        self.target_scaler = target_scaler
        self.torch_model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.FloatTensor(X)
            y_scaled = self.torch_model(t).numpy()
        return self.target_scaler.inverse_transform(y_scaled)


# ══════════════════════════════════════════════════════════════════════════════
#  5.  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_qspr_model(master_df: pd.DataFrame):
    """
    Train a PhysicsGuided multi-target MLP on is_reference=True rows only.
    Returns (model_wrapper, feature_scaler, metrics, feature_cols).

    The returned `model_wrapper` exposes sklearn-style .predict(X_scaled)
    so downstream code (validate / infer) is unchanged.
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

    X_raw = train_df[feature_cols].values.astype(float)
    y_raw = train_df[TARGETS].values.astype(float)

    # ── Heuristic baselines (full training set, before split) ─────────────────
    heur_raw = _get_heuristics(train_df, train_df.index)

    # ── Train / test split (indices kept for heuristic alignment) ─────────────
    idx_all = np.arange(len(train_df))
    idx_tr, idx_te = train_test_split(
        idx_all, test_size=0.20, random_state=42, shuffle=True
    )

    X_train_raw, X_test_raw = X_raw[idx_tr], X_raw[idx_te]
    y_train_raw, y_test_raw = y_raw[idx_tr], y_raw[idx_te]
    heur_train_raw           = heur_raw[idx_tr]

    # ── Feature scaling (StandardScaler on X) ─────────────────────────────────
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train_raw)
    X_test_scaled  = feature_scaler.transform(X_test_raw)

    # ── Target scaling — z-score so PhysicsGuidedLoss works in σ units ────────
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw)
    y_test_scaled  = target_scaler.transform(y_test_raw)

    # Scale heuristics with the same target scaler so safe_margin is in σ units
    heur_train_scaled = target_scaler.transform(heur_train_raw)

    # ── DataLoader — now carries heuristic baselines as a third tensor ─────────
    # (This is the updated TensorDataset described in the task spec.)
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_scaled),
        torch.FloatTensor(heur_train_scaled),   # ← ADDED: heuristic baselines
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # ── Model, optimiser, scheduler ───────────────────────────────────────────
    input_dim  = X_train_scaled.shape[1]
    output_dim = len(TARGETS)
    torch_model = PolymerMLP(input_dim, output_dim)

    optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    # ── Physics-guided loss (replaces plain nn.MSELoss) ───────────────────────
    criterion = PhysicsGuidedLoss(
        penalty_multiplier=PHYSICS_PENALTY_MULTIPLIER,
        safe_margin=SAFE_MARGIN_SIGMA,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0
    EARLY_STOP_PATIENCE = 25
    MAX_EPOCHS = 600

    X_val_t  = torch.FloatTensor(X_test_scaled)
    y_val_t  = torch.FloatTensor(y_test_scaled)
    # Use scaled heuristics on val set too (targets scaled from same scaler)
    heur_val_scaled = target_scaler.transform(_get_heuristics(train_df, train_df.index)[idx_te])
    heur_val_t = torch.FloatTensor(heur_val_scaled)

    print(f"  Training PyTorch MLP [256→128→64] with PhysicsGuidedLoss "
          f"(penalty×{PHYSICS_PENALTY_MULTIPLIER}, margin={SAFE_MARGIN_SIGMA}σ) ...")

    torch_model.train()
    for epoch in range(1, MAX_EPOCHS + 1):

        # ── Mini-batch pass — unpacks 3 items (X, y, heuristics) ──────────────
        for batch_X, batch_y, batch_heuristics in train_loader:
            optimizer.zero_grad()
            predictions = torch_model(batch_X)
            loss = criterion(predictions, batch_y, batch_heuristics)
            loss.backward()
            optimizer.step()

        # ── Validation loss (physics-guided, same criterion) ──────────────────
        torch_model.eval()
        with torch.no_grad():
            val_preds = torch_model(X_val_t)
            val_loss  = criterion(val_preds, y_val_t, heur_val_t).item()
        torch_model.train()

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            patience_count = 0
            best_state     = {k: v.clone() for k, v in torch_model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch} (val_loss={best_val_loss:.6f})")
                break

    # Restore best weights
    torch_model.load_state_dict(best_state)
    torch_model.eval()

    # ── Wrap for sklearn-compatible downstream calls ───────────────────────────
    model_wrapper = _TorchModelWrapper(torch_model, target_scaler)

    # ── Global metrics ────────────────────────────────────────────────────────
    with torch.no_grad():
        y_pred_scaled = torch_model(X_val_t).numpy()
    y_pred = target_scaler.inverse_transform(y_pred_scaled)

    metrics: dict = {}
    for i, tgt in enumerate(TARGETS):
        r2   = r2_score(y_test_raw[:, i], y_pred[:, i])
        rmse = math.sqrt(mean_squared_error(y_test_raw[:, i], y_pred[:, i]))
        metrics[tgt] = {"R2": r2, "RMSE": rmse}

    # ── Per-polymer within-class metrics ──────────────────────────────────────
    per_polymer_metrics: dict = {}
    ref_polymers = train_df["Polymer"].unique()
    for poly in ref_polymers:
        mask = (train_df["Polymer"] == poly).values
        X_p_sc = feature_scaler.transform(X_raw[mask])
        y_p    = y_raw[mask]
        if len(y_p) < 5:
            continue
        idx_pp_tr, idx_pp_te = train_test_split(
            np.arange(len(y_p)), test_size=0.20, random_state=42
        )
        with torch.no_grad():
            yp_pred_scaled = torch_model(
                torch.FloatTensor(X_p_sc[idx_pp_te])
            ).numpy()
        yp_pred = target_scaler.inverse_transform(yp_pred_scaled)
        yp_te   = y_p[idx_pp_te]
        per_polymer_metrics[poly] = {
            "Tg_R2"  : r2_score(yp_te[:, 0], yp_pred[:, 0]),
            "Dk_R2"  : r2_score(yp_te[:, 1], yp_pred[:, 1]),
            "Rad_R2" : r2_score(yp_te[:, 2], yp_pred[:, 2]),
            "n_test" : len(yp_te),
        }
    metrics["per_polymer"] = per_polymer_metrics

    return model_wrapper, feature_scaler, metrics, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

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


def validate_dummy_samples(master_df: pd.DataFrame, model, scaler,
                            feature_cols: list) -> None:
    """
    Run the dummy rows through the trained model.
    Predictions should be out of physically plausible ranges.
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

    X_dummy = scaler.transform(dummy_df[feature_cols].values.astype(float))
    preds   = model.predict(X_dummy)

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

    ref_rows = master_df[master_df["is_reference"] == True]
    X_ref    = scaler.transform(ref_rows[feature_cols].values.astype(float))
    ref_pred = pd.DataFrame(model.predict(X_ref), columns=TARGETS)
    ref_pred["Polymer"] = ref_rows["Polymer"].values
    ref_polymer_names   = list(ref_rows["Polymer"].unique())

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
    print("  ℹ  Model: PyTorch MLP [256→128→64] + PhysicsGuidedLoss  |  "
          "Training: reference polymers only  |  Split: 80/20")
    print(f"  {'─'*85}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("  Loading master_dataset.pkl ...")
    with open("master_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    print(f"  ✔  Loaded: {len(df)} rows\n")

    user_meta = load_user_polymer()
    if user_meta:
        print(f"  ✔  User polymer detected: {user_meta['name']}")

    print_banner("TASK 2 — MULTI-TARGET MLP QSPR MODEL TRAINING", width=85)
    print("\n  Training PyTorch MLP [256→128→64] on 120-feature vectors ...")
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
        "model_type"  : "PyTorch-MLP-PhysicsGuided",
    }
    with open("qspr_model_mlp.pkl", "wb") as f:
        pickle.dump(qspr_bundle, f)
    print("  💾  Saved: qspr_model_mlp.pkl\n")
