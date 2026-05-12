"""
================================================================================
  TASK 3 ▶ Structural Inverse Design via RDKit Mutation

  CHANGES (this version):
    1. ALL 4 TARGETS in loss function (Tg, Dk, TML, Radiation)
       Normalised one-sided loss — each term dimensionless so no single
       target dominates by magnitude:
         L = 0.30·loss_Tg² + 0.25·loss_Dk² + 0.15·loss_TML² + 0.30·loss_Rad²
       where each loss_X is 0 when the target is met, >0 when it is not.
    2. TML and Radiation targets added to POLYMER_REGISTRY per polymer:
         Polyimide : TML ≤ 0.08 %   Rad ≥ 32 MGy
         PEEK      : TML ≤ 0.10 %   Rad ≥ 28 MGy
         PTFE      : TML ≤ 0.14 %   Rad ≥ 20 MGy
    3. USER POLYMER SUPPORT
       Loads user_polymer.pkl (written by task1); auto-adds the user polymer
       to the design loop using user-provided or model-estimated targets.
    4. REPORTING extended: top-10 candidate table now shows all 4 targets
       and a composite desirability score.
    5. inv_results.pkl updated to include TML and Radiation columns.

  Inputs : master_dataset.pkl, qspr_model_mlp.pkl (or qspr_model_gbr.pkl)
  Outputs: inv_results.pkl
================================================================================
"""
import warnings, pickle, random, os, contextlib
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions, DataStructs

try:
    from task1_data_curation import extract_structural_features_rdkit
except ImportError:
    print("  ⚠  Could not import task1_data_curation — ensure it is in the same folder.")

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"

# ──────────────────────────────────────────────────────────────────────────────
#  REFERENCE POLYMER REGISTRY (targets for all 4 properties)
# ──────────────────────────────────────────────────────────────────────────────
POLYMER_REGISTRY = {
    "Polyimide": {
        "color"      : "\033[94m",
        "member"     : "Member1",
        "target_Tg"  : 300.0,   
        "target_Dk"  : 3.30,
        "target_Rad" : 32.0,    
    },
    "PEEK": {
        "color"      : "\033[92m",
        "member"     : "Member2",
        "target_Tg"  : 280.0,
        "target_Dk"  : 3.25,
        "target_Rad" : 28.0,
    },
    "PTFE": {
        "color"      : "\033[93m",
        "member"     : "Member3",
        "target_Tg"  : 50.0,
        "target_Dk"  : 2.00,
        "target_Rad" : 20.0,
    },
    "CyanateEster": {
        "color"      : "\033[35m",
        "member"     : "Member4",
        "target_Tg"  : 500.0,
        "target_Dk"  : 2.70,
        "target_Rad" : 5.0,
    },
    "FluorinatedPolyimide": {
        "color"      : "\033[36m",
        "member"     : "Member5",
        "target_Tg"  : 400.0,
        "target_Dk"  : 2.60,
        "target_Rad" : 1.0,
    },
    "Polybenzoxazole": {
        "color"      : "\033[33m",
        "member"     : "Member6",
        "target_Tg"  : 650.0,
        "target_Dk"  : 2.50,
        "target_Rad" : 30.0,
    },
}

LOSS_WEIGHTS = {
    "Tg" : 0.40,
    "Dk" : 0.30,
    "Rad": 0.30,
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

# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_user_polymer() -> dict | None:
    if os.path.exists("user_polymer.pkl"):
        with open("user_polymer.pkl", "rb") as f:
            return pickle.load(f)
    return None

def load_best_model() -> dict:
    for fname in ("qspr_model_gbr.pkl", "qspr_model_mlp.pkl"):
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                bundle = pickle.load(f)
            return bundle
    raise FileNotFoundError("No QSPR model found. Run task2 or task2b first.")

def print_banner(text: str, char: str = "═", width: int = 88) -> None:
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{BOLD}{text}{RESET}"
          f"{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")

def composite_score(tg, dk, rad) -> float:
    d_tg  = 1 / (1 + np.exp(-0.02 * (tg - 200)))
    d_dk  = float(np.clip((5.0 - dk)  / (5.0 - 2.0), 0, 1))
    d_rad = float(np.clip((rad - 10)  / (40.0 - 10.0), 0, 1))
    return 0.40 * d_tg + 0.30 * d_dk + 0.30 * d_rad

# ──────────────────────────────────────────────────────────────────────────────
#  CANDIDATE GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_candidates(input_smiles: str, num_candidates: int = 50,
                        tanimoto_floor: float = 0.35) -> list:
    base_mol = Chem.MolFromSmiles(input_smiles)
    if base_mol is None:
        return []
    base_fp = AllChem.GetMorganFingerprintAsBitVect(base_mol, 2, nBits=2048)
    has_arom = any(a.GetIsAromatic() for a in base_mol.GetAtoms())

    mutations_aromatic = [
        "[cH1:1]>>[c:1](F)",
        "[cH1:1]>>[c:1](C(F)(F)F)",
        "[cH1:1]>>[c:1](OC(F)(F)F)",
        "[cH1:1]>>[c:1](F)[cH0:2]",
        "[cH1:1]>>[c:1](C#N)",
        "[cH1:1]>>[c:1](CC)",
        "[cH1:1]>>[c:1](OC)",
        "[cH1:1]>>[c:1](c1ccccc1)",
        "[cH1:1]>>[c:1](c1ccncc1)",
        "[cH1:1]>>[c:1](Cl)",
        "[cH1:1]>>[c:1](Br)",
        "[cH1:1]>>[c:1](N)",
        "[cH1:1]>>[c:1](S(=O)(=O)N)",
    ]
    mutations_aliphatic = [
        "[F:1]>>[Cl:1]",
        "[F:1]>>[Br:1]",
        "[F:1]>>[C:1]",
        "[F:1]>>[O:1]",
        "[F:1]>>[N:1]",
        "[F:1]>>[S:1]",
        "[C:1](F)(F)>>[C:1](F)(Cl)",
        "[C:1](F)(F)(F)>>[C:1](F)(F)(Cl)",
    ]
    rxn_strs = mutations_aromatic if has_arom else mutations_aliphatic
    rxns     = [rdChemReactions.ReactionFromSmarts(r) for r in rxn_strs]

    seen       = {input_smiles}
    candidates = [input_smiles]
    pool       = [base_mol]
    random.seed(42)

    attempts = 0
    while len(candidates) < num_candidates and attempts < 100_000:
        attempts += 1
        mol = pool[random.randint(0, len(pool) - 1)]
        rxn = rxns[random.randint(0, len(rxns) - 1)]
        prods = rxn.RunReactants((mol,))
        if not prods:
            continue
        new_mol = prods[random.randint(0, len(prods) - 1)][0]
        try:
            Chem.SanitizeMol(new_mol)
            new_smi = Chem.MolToSmiles(new_mol)
        except Exception:
            continue
        if new_smi in seen:
            continue
        new_fp = AllChem.GetMorganFingerprintAsBitVect(new_mol, 2, nBits=2048)
        if DataStructs.TanimotoSimilarity(base_fp, new_fp) >= tanimoto_floor:
            seen.add(new_smi)
            candidates.append(new_smi)
            pool.append(new_mol)

    return candidates

# ──────────────────────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION FOR CANDIDATES
# ──────────────────────────────────────────────────────────────────────────────

def build_candidate_features(smiles_list: list, polymer_name: str,
                              master_df: pd.DataFrame,
                              feature_cols: list) -> np.ndarray:
    try:
        # Suppress stdout to prevent print loops from task1_data_curation
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
            df_struct = extract_structural_features_rdkit(smiles_list, polymer_name)
    except Exception:
        df_struct = pd.DataFrame(
            np.zeros((len(smiles_list), len(STRUCTURAL_FEATURE_NAMES))),
            columns=STRUCTURAL_FEATURE_NAMES
        )

    poly_rows = master_df[master_df["Polymer"] == polymer_name]
    lp_cols   = LATENT_FEATURE_NAMES + PHYSICS_FEATURE_NAMES
    if len(poly_rows) > 0:
        lp_mean = poly_rows[lp_cols].mean()
    else:
        lp_mean = pd.Series(np.zeros(len(lp_cols)), index=lp_cols)

    df_lp = pd.DataFrame(
        [lp_mean.values] * len(smiles_list), columns=lp_cols
    )

    df_all = pd.concat([df_struct.reset_index(drop=True),
                        df_lp.reset_index(drop=True)], axis=1)

    for col in feature_cols:
        if col not in df_all.columns:
            df_all[col] = 0.0
    return df_all[feature_cols].values.astype(float)

# ──────────────────────────────────────────────────────────────────────────────
#  4-TARGET NORMALISED LOSS FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def compute_loss(pred: np.ndarray,
                 target_Tg: float, target_Dk: float,
                 target_Rad: float) -> float:
    pred_Tg, pred_Dk, pred_Rad = pred

    loss_Tg  = max(0.0, (target_Tg  - pred_Tg)  / max(abs(target_Tg),  1e-6))
    loss_Dk  = max(0.0, (pred_Dk    - target_Dk) / max(abs(target_Dk),  1e-6))
    loss_Rad = max(0.0, (target_Rad - pred_Rad)  / max(abs(target_Rad), 1e-6))

    return (LOSS_WEIGHTS["Tg"]  * loss_Tg**2
          + LOSS_WEIGHTS["Dk"]  * loss_Dk**2
          + LOSS_WEIGHTS["Rad"] * loss_Rad**2)

# ──────────────────────────────────────────────────────────────────────────────
#  INVERSE DESIGN ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def structural_inverse_design(polymer_name: str, base_smiles: str,
                               targets: dict, model, scaler,
                               feature_cols: list, master_df: pd.DataFrame,
                               num_candidates: int = 80,
                               top_k: int = 10,
                               color: str = "\033[0m") -> dict:
    t_Tg  = targets["target_Tg"]
    t_Dk  = targets["target_Dk"]
    t_Rad = targets["target_Rad"]

    _RELAXED_FLOOR_POLYMERS = {"FluorinatedPolyimide", "Polybenzoxazole", "PTFE"}
    tanimoto_floor  = 0.35 if polymer_name in _RELAXED_FLOOR_POLYMERS else 0.50
    candidate_budget = num_candidates * 2 if polymer_name in _RELAXED_FLOOR_POLYMERS \
                       else num_candidates

    candidates = generate_candidates(base_smiles, candidate_budget, tanimoto_floor)

    if not candidates:
        return {"polymer": polymer_name, "candidates": pd.DataFrame()}

    X = build_candidate_features(candidates, polymer_name, master_df, feature_cols)
    X_sc = scaler.transform(X)
    preds = model.predict(X_sc)

    rows = []
    for i, smi in enumerate(candidates):
        pred_Tg, pred_Dk, pred_Rad = preds[i]
        pred_Dk  = max(pred_Dk,  1.0)
        pred_Rad = max(pred_Rad, 0.0)
        clamped  = np.array([pred_Tg, pred_Dk, pred_Rad])
        loss  = compute_loss(clamped, t_Tg, t_Dk, t_Rad)
        score = composite_score(pred_Tg, pred_Dk, pred_Rad)
        rows.append({
            "Polymer"            : polymer_name,
            "SMILES"             : smi,
            "pred_Tg_degC"       : pred_Tg,
            "pred_Dk_1GHz"       : pred_Dk,
            "pred_Rad_MGy"       : pred_Rad,
            "loss"               : loss,
            "desirability_score" : score,
            "meets_Tg"           : pred_Tg  >= t_Tg,
            "meets_Dk"           : pred_Dk  <= t_Dk,
            "meets_Rad"          : pred_Rad >= t_Rad,
        })

    df_cands = (pd.DataFrame(rows)
                  .sort_values("desirability_score", ascending=False)
                  .reset_index(drop=True))
    df_cands["rank"] = df_cands.index + 1

    top = df_cands.head(top_k)
    n_all_met = int(df_cands[["meets_Tg","meets_Dk","meets_Rad"]].all(axis=1).sum())

    return {
        "polymer"     : polymer_name,
        "targets"     : targets,
        "candidates"  : df_cands,
        "top_k"       : top,
        "n_all_met"   : n_all_met,
    }

# ──────────────────────────────────────────────────────────────────────────────
#  USER POLYMER — AUTO TARGET ESTIMATION
# ──────────────────────────────────────────────────────────────────────────────

def estimate_user_targets_from_model(user_meta: dict, model, scaler,
                                     feature_cols: list,
                                     master_df: pd.DataFrame) -> dict:
    name = user_meta["name"]
    user_rows = master_df[master_df["Polymer"] == name]
    if len(user_rows) == 0:
        return {k: user_meta[k] for k in ("target_Tg","target_Dk","target_Rad")}

    X    = user_rows[feature_cols].values.astype(float)
    X_sc = scaler.transform(X)
    pred = model.predict(X_sc)
    med  = np.median(pred, axis=0)

    computed = {
        "target_Tg"  : med[0] * 1.10,
        "target_Dk"  : max(med[1] * 0.90, 1.5),
        "target_Rad" : med[2] * 1.15,
    }
    return computed

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("master_dataset.pkl", "rb") as f:
        master_df = pickle.load(f)

    bundle      = load_best_model()
    model       = bundle["model"]
    scaler      = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    user_meta = load_user_polymer()

    design_schedule = []

    for poly_name, meta in POLYMER_REGISTRY.items():
        base_smi = master_df[master_df["Polymer"] == poly_name]["SMILES"].iloc[0]
        design_schedule.append({
            "name"   : poly_name,
            "smiles" : base_smi,
            "color"  : meta["color"],
            "targets": {k: meta[k] for k in
                        ("target_Tg","target_Dk","target_Rad")},
        })

    if user_meta:
        name = user_meta["name"]
        if user_meta.get("target_Tg") and user_meta.get("target_Tg") != user_meta.get("tg_base") * 1.10:
            u_targets = {k: user_meta[k] for k in
                         ("target_Tg","target_Dk","target_Rad")}
        else:
            u_targets = estimate_user_targets_from_model(
                user_meta, model, scaler, feature_cols, master_df
            )
        user_smiles = master_df[master_df["Polymer"] == name]["SMILES"].iloc[0] \
            if name in master_df["Polymer"].values else user_meta["smiles"]
        design_schedule.append({
            "name"   : name,
            "smiles" : user_smiles,
            "color"  : user_meta.get("color", "\033[95m"),
            "targets": u_targets,
        })

    all_results = {}
    for entry in design_schedule:
        result = structural_inverse_design(
            polymer_name   = entry["name"],
            base_smiles    = entry["smiles"],
            targets        = entry["targets"],
            model          = model,
            scaler         = scaler,
            feature_cols   = feature_cols,
            master_df      = master_df,
            num_candidates = 80,
            top_k          = 10,
            color          = entry["color"],
        )
        all_results[entry["name"]] = result

    # ── Summary ───────────────────────────────────────────────────────────
    print_banner("INVERSE DESIGN SUMMARY", width=88)
    print(f"\n  {'Polymer':<18}  {'Candidates':>10}  "
          f"{'Best Score':>11}  {'Best Tg':>8}  {'Best Dk':>7}  "
          f"{'Best Rad':>9}")
    print(f"  {'─'*18}  {'─'*10}  {'─'*11}  {'─'*8}  {'─'*7}  {'─'*9}")

    for poly_name, res in all_results.items():
        if res["candidates"].empty:
            print(f"  {poly_name:<18}  {'—':>10}")
            continue
        top1 = res["candidates"].iloc[0]
        color = POLYMER_REGISTRY.get(poly_name, {}).get("color", "\033[95m")
        print(f"  {color}{poly_name:<18}{RESET}  "
              f"{len(res['candidates']):>10}  "
              f"{top1['desirability_score']:>11.4f}  "
              f"{top1['pred_Tg_degC']:>8.2f}  "
              f"{top1['pred_Dk_1GHz']:>7.3f}  "
              f"{top1['pred_Rad_MGy']:>9.2f}")

    # ── Save ──────────────────────────────────────────────────────────────
    with open("inv_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n  💾  Saved: inv_results.pkl\n")

    # ── Export top candidates to CSV ──────────────────────────────────────
    all_tops = []
    for res in all_results.values():
        if not res["candidates"].empty:
            all_tops.append(res["candidates"])
    if all_tops:
        out_df = pd.concat(all_tops, ignore_index=True)
        out_df = out_df.sort_values("desirability_score", ascending=False)
        out_df.to_csv("inverse_design_candidates.csv", index=False)
        print(f"  💾  Saved: inverse_design_candidates.csv  "
              f"({len(out_df)} candidates, sorted best-first)\n")
