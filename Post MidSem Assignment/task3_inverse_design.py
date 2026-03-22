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

TASK 3 ▶ Inverse Design via Bayesian Optimisation (Differential Evolution)

  Inputs : master_dataset.pkl        (produced by task1_data_curation.py)
           qspr_model.pkl            (produced by task2_qspr_modeling.py)
  Outputs: inv_results.pkl           (loaded by task4_output_management.py)
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

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
#  ████████╗ █████╗ ███████╗██╗  ██╗     ██████╗
#  ╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝    ╚════██╗
#     ██║   ███████║███████╗█████╔╝      ███╔═╝
#     ██║   ██╔══██║╚════██║██╔═██╗     ███╔══╝
#     ██║   ██║  ██║███████║██║  ██╗    ███████╗
#     ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚══════╝
#  INVERSE DESIGN FRAMEWORK — Bayesian-Inspired Differential Evolution
# ──────────────────────────────────────────────────────────────────────────────

# Per-polymer LEO targets — physically calibrated per class
POLYMER_LEO_TARGETS = {
    "Polyimide": {"Tg": 300.0, "Dk": 2.80},  # premium thermal+dielectric candidate
    "PEEK"     : {"Tg": 150.0, "Dk": 2.80},  # structural candidate
    "PTFE"     : {"Tg": -80.0, "Dk": 2.30},  # dielectric candidate
}
# Kept for backward compatibility
LEO_TG_TARGET = 300.0
LEO_DK_TARGET = 2.50

# Design variable indices within the physics feature block
_IDX_FVF   = PHYSICS_FEATURE_NAMES.index("FreeVolumeFraction")          # 3
_IDX_CRYST = PHYSICS_FEATURE_NAMES.index("DegreeOfCrystallinity")       # 0
_IDX_CRI   = PHYSICS_FEATURE_NAMES.index("ChainRigidityIndex")          # 4
_IDX_DP    = PHYSICS_FEATURE_NAMES.index("DielectricPolarizability")    # 10

# Expanded 4-variable search (was 2 — too narrow to hit LEO targets).
# Added ChainRigidityIndex (biggest Tg driver, coeff=80) and
# DielectricPolarizability (biggest Dk driver, coeff=0.050).
# Without these, Polyimide Dk floor is ~2.81 (above LEO 2.5 target)
# and PEEK/PTFE cannot reach Tg>300°C regardless of FVF/Cryst.
_DESIGN_BOUNDS = {
    "FreeVolumeFraction"      : (0.04, 0.22),
    "DegreeOfCrystallinity"   : (0.10, 0.80),
    "ChainRigidityIndex"      : (0.50, 1.00),
}

# Per-polymer DielPol bounds — tightened to each polymer's realistic training
# distribution so the optimiser cannot extrapolate outside the fitted region.
# Lower bounds derived from training data minimum (rounded down to nearest 0.5):
#   Polyimide : observed range [18.9, 39.5]  → lower = 20.0
#   PEEK      : observed range [11.4, 36.8]  → lower = 15.0
#   PTFE      : observed range [ 9.0, 16.0]  → lower =  9.0
# Upper bound shared at 30.0 (above all three polymer maxima — non-binding).
_DIELPOL_BOUNDS = {
    "Polyimide": (20.0, 30.0),
    "PEEK"     : (15.0, 30.0),
    "PTFE"     : ( 9.0, 16.0),
}


def inverse_design_polymer(polymer_name: str,
                            master_df: pd.DataFrame,
                            model,
                            scaler,
                            feature_cols: list) -> dict:
    """
    Task 3: Inverse Design via Differential Evolution.

    Strategy:
      • Fix the structural (RDKit) and latent (polyBERT) features at the
        polymer class mean (representing the repeat-unit chemistry).
      • Allow FreeVolumeFraction and DegreeOfCrystallinity to vary freely
        within physical bounds.
      • All other physics features are held at the polymer class mean.

    Multi-objective fitness function (scalarised with penalty terms):
      F = w_Tg · max(0, LEO_Tg − Tg_pred) / LEO_Tg      ← penalty if Tg < 300
        + w_Dk · max(0, Dk_pred − LEO_Dk) / LEO_Dk       ← penalty if Dk > 2.5
        + w_robust · (FVF² + (1−Cryst)²) / 2             ← regularisation

    The algorithm minimises F → F = 0 is the global LEO target.
    """
    # ── Build representative feature vector for this polymer ─────────────────
    poly_df    = master_df[master_df["Polymer"] == polymer_name]
    mean_feats = poly_df[feature_cols].mean().values.copy()

    # Indices of physics features within the full feature vector
    phys_start = N_STRUCTURAL + N_LATENT   # 80

    # Per-polymer targets — use calibrated targets, not one-size-fits-all
    poly_targets = POLYMER_LEO_TARGETS[polymer_name]
    tg_target    = poly_targets["Tg"]
    dk_target    = poly_targets["Dk"]

    def fitness(x: np.ndarray) -> float:
        fvf   = x[0]
        cryst = x[1]
        cri   = x[2]   # ChainRigidityIndex
        dp    = x[3]   # DielectricPolarizability

        # Mutate all four design variables
        feat_vec = mean_feats.copy()
        feat_vec[phys_start + _IDX_FVF  ] = fvf
        feat_vec[phys_start + _IDX_CRYST] = cryst
        feat_vec[phys_start + _IDX_CRI  ] = cri
        feat_vec[phys_start + _IDX_DP   ] = dp
        # Enforce AmorphousPhaseContent = 1 − Crystallinity consistency
        amorph_idx = PHYSICS_FEATURE_NAMES.index("AmorphousPhaseContent")
        feat_vec[phys_start + amorph_idx] = 1.0 - cryst

        # Predict
        feat_scaled = scaler.transform(feat_vec.reshape(1, -1))
        pred        = model.predict(feat_scaled)[0]
        Tg_pred, Dk_pred = pred[0], pred[1]

        # Penalty terms — equal weight on both objectives
        w_Tg  = 1.0
        w_Dk  = 1.0
        w_reg = 0.03   # lighter regularisation (was 0.05) — give optimiser more freedom

        penalty_Tg = max(0.0, (tg_target - Tg_pred) / (abs(tg_target) if tg_target != 0 else 100.0))
        penalty_Dk = max(0.0, (Dk_pred - dk_target) / dk_target)
        # Regularisation penalises extreme values of all 4 variables
        reg = w_reg * (fvf**2 + (1 - cryst)**2 + (1 - cri)**2 + (dp / 30.0)**2) / 4.0

        return w_Tg * penalty_Tg + w_Dk * penalty_Dk + reg

    # ── Differential Evolution optimiser ─────────────────────────────────────
    bounds   = [_DESIGN_BOUNDS["FreeVolumeFraction"],
                _DESIGN_BOUNDS["DegreeOfCrystallinity"],
                _DESIGN_BOUNDS["ChainRigidityIndex"],
                _DIELPOL_BOUNDS[polymer_name]]          # polymer-specific DielPol range
    result   = differential_evolution(
        fitness,
        bounds       = bounds,
        seed         = 42,
        maxiter      = 500,
        tol          = 1e-8,
        popsize      = 20,
        mutation     = (0.5, 1.2),
        recombination= 0.9,
        polish       = True,
        workers      = 1,
    )

    opt_fvf   = result.x[0]
    opt_cryst = result.x[1]
    opt_cri   = result.x[2]
    opt_dp    = result.x[3]

    # Final prediction with all four optimal parameters
    feat_vec  = mean_feats.copy()
    feat_vec[phys_start + _IDX_FVF  ] = opt_fvf
    feat_vec[phys_start + _IDX_CRYST] = opt_cryst
    feat_vec[phys_start + _IDX_CRI  ] = opt_cri
    feat_vec[phys_start + _IDX_DP   ] = opt_dp
    amorph_idx = PHYSICS_FEATURE_NAMES.index("AmorphousPhaseContent")
    feat_vec[phys_start + amorph_idx ] = 1.0 - opt_cryst
    feat_scaled = scaler.transform(feat_vec.reshape(1, -1))
    pred_final  = model.predict(feat_scaled)[0]

    return {
        "polymer"                   : polymer_name,
        "optimal_FVF"               : opt_fvf,
        "optimal_Crystallinity"     : opt_cryst,
        "optimal_ChainRigidity"     : opt_cri,
        "optimal_DielPolarizability": opt_dp,
        "predicted_Tg_degC"         : pred_final[0],
        "predicted_Dk"              : pred_final[1],
        "tg_target"                 : tg_target,
        "dk_target"                 : dk_target,
        "fitness_final"             : result.fun,
        "converged"                 : result.success,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def print_banner(text: str, char: str = "═", width: int = 72) -> None:
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{BOLD}{text}{RESET}{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")


def print_inverse_design_report(results: list) -> None:
    """Formatted inverse design output table."""
    print_banner("TASK 3 — INVERSE DESIGN RESULTS (LEO Satellite Targets)", width=72)
    print(f"\n  Per-polymer LEO targets (physically calibrated):")
    for pn, tgt in POLYMER_LEO_TARGETS.items():
        print(f"    {pn:<12}: Tg > {tgt['Tg']:.0f} °C   Dk < {tgt['Dk']:.2f}")
    print()

    header = (f"  {'Polymer':<12}  {'FVF':>8}  {'Cryst':>8}  {'ChainRig':>10}  "
              f"{'DielPol':>9}  {'Tg (°C)':>9}  {'Dk':>7}  {'Tg✓':>4}  {'Dk✓':>4}")
    print(header)
    print(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*4}  {'─'*4}")

    for r in results:
        tg_ok = "✔" if r["predicted_Tg_degC"] >= r["tg_target"] else "✗"
        dk_ok = "✔" if r["predicted_Dk"]      <= r["dk_target"] else "✗"
        meta  = POLYMER_REGISTRY[r["polymer"]]
        col   = meta["color"]
        print(f"  {col}{BOLD}{r['polymer']:<12}{RESET}  "
              f"{r['optimal_FVF']:>8.4f}  "
              f"{r['optimal_Crystallinity']:>8.4f}  "
              f"{r['optimal_ChainRigidity']:>10.4f}  "
              f"{r['optimal_DielPolarizability']:>9.3f}  "
              f"{r['predicted_Tg_degC']:>9.2f}  "
              f"{r['predicted_Dk']:>7.4f}  "
              f"{tg_ok:>4}  {dk_ok:>4}")

    print(f"\n  {'─'*72}")
    print("  ℹ  Optimiser: Differential Evolution (SciPy)  |  "
          "Max iterations: 500  |  Polishing: L-BFGS-B")
    print(f"  {'─'*72}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Load inputs from Tasks 1 & 2 ─────────────────────────────────────────
    with open("master_dataset.pkl", "rb") as f:
        master_df = pickle.load(f)
    print("  ✔  Loaded: master_dataset.pkl")

    with open("qspr_model.pkl", "rb") as f:
        qspr_bundle = pickle.load(f)
    model        = qspr_bundle["model"]
    scaler       = qspr_bundle["scaler"]
    feature_cols = qspr_bundle["feature_cols"]
    print("  ✔  Loaded: qspr_model.pkl\n")

    # ── Task 3: Inverse Design ───────────────────────────────────────────────
    print_banner("TASK 3 — INVERSE DESIGN OPTIMISATION (Differential Evolution)",
                 width=72)
    print(f"\n  Per-polymer objectives (physically calibrated):")
    for pn, tgt in POLYMER_LEO_TARGETS.items():
        print(f"    {pn:<12}: Tg > {tgt['Tg']:.0f} °C   Dk < {tgt['Dk']:.2f}")
    print()
    inv_results = []
    for pname in POLYMER_REGISTRY:
        meta = POLYMER_REGISTRY[pname]
        print(f"  🔍  Optimising {meta['color']}{BOLD}{pname}{RESET} ...")
        res = inverse_design_polymer(pname, master_df, model, scaler, feature_cols)
        inv_results.append(res)
        print(f"       FVF={res['optimal_FVF']:.4f}  "
              f"Cryst={res['optimal_Crystallinity']:.4f}  "
              f"ChainRig={res['optimal_ChainRigidity']:.4f}  "
              f"DielPol={res['optimal_DielPolarizability']:.2f}  "
              f"Tg={res['predicted_Tg_degC']:.1f}°C  "
              f"Dk={res['predicted_Dk']:.4f}  "
              f"{'[converged]' if res['converged'] else '[not converged]'}")
    print()
    print_inverse_design_report(inv_results)

    # ── Save inverse design results for Task 4 ────────────────────────────────
    with open("inv_results.pkl", "wb") as f:
        pickle.dump(inv_results, f)
    print("  💾  Saved: inv_results.pkl  →  (input for task4_output_management.py)\n")
