"""
================================================================================
  TASK 3 ▶ Structural Inverse Design via RDKit Mutation (Plan A)
  
  Inputs : master_dataset.pkl, qspr_model.pkl
  Outputs: inv_results.pkl
================================================================================
"""
import warnings, pickle, random
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from rdkit import DataStructs

# Import the feature extractor from your Task 1 script to evaluate new SMILES
try:
    from task1_data_curation import extract_structural_features_rdkit
except ImportError:
    print("Error: Could not import task1_data_curation. Ensure it is in the same folder.")

# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS & REGISTRY
# ──────────────────────────────────────────────────────────────────────────────
POLYMER_REGISTRY = {
    "Polyimide": {"color": "\033[94m", "member": "Member1", "target_Tg": 300.0, "target_Dk": 2.80},
    "PEEK"     : {"color": "\033[92m", "member": "Member2", "target_Tg": 150.0, "target_Dk": 2.80},
    "PTFE"     : {"color": "\033[93m", "member": "Member3", "target_Tg": -80.0, "target_Dk": 2.30},
}

RESET = "\033[0m"
BOLD  = "\033[1m"

# ──────────────────────────────────────────────────────────────────────────────
#  GENERATION ENGINE (RDKit Mutations)
# ──────────────────────────────────────────────────────────────────────────────
def generate_candidates(input_smiles: str, num_candidates=50):
    """Generates structurally valid mutated SMILES within a >=0.60 Tanimoto constraint.

    Automatically selects aromatic mutation rules for ring-containing polymers
    (Polyimide, PEEK) and aliphatic rules for non-aromatic polymers (PTFE).
    """
    base_mol = Chem.MolFromSmiles(input_smiles)
    base_fp = AllChem.GetMorganFingerprintAsBitVect(base_mol, 2, nBits=2048)

    # Detect whether the base molecule has any aromatic atoms
    has_aromatics = any(atom.GetIsAromatic() for atom in base_mol.GetAtoms())

    if has_aromatics:
        # Aromatic mutation rules — for Polyimide and PEEK
        mutation_rules = [
            "[cH1:1]>>[c:1](F)",         # Fluorination
            "[cH1:1]>>[c:1](C(F)(F)F)",  # Trifluoromethylation
            "[cH1:1]>>[c:1](C)",         # Methylation
            "[cH1:1]>>[c:1](Cl)"         # Chlorination
        ]
    else:
        # Aliphatic mutation rules — for PTFE (no aromatic rings)
        # Mirrors the rules used in task1_data_curation.py for PTFE derivatives
        mutation_rules = [
            "[F:1]>>[Cl:1]",   # F → Cl  (halogen swap, increases polarizability)
            "[F:1]>>[Br:1]",   # F → Br  (heavier halogen)
            "[F:1]>>[C:1]",    # F → C   (defluorination)
            "[F:1]>>[O:1]",    # F → O   (introduces ether-like character)
        ]

    reactions = [rdChemReactions.ReactionFromSmarts(rxn) for rxn in mutation_rules]

    candidates = []
    generated_set = {input_smiles}
    current_pool = [base_mol]
    attempts = 0

    while len(candidates) < num_candidates and attempts < 10000:
        attempts += 1
        mol_to_mutate = random.choice(current_pool)
        rxn = random.choice(reactions)
        
        products = rxn.RunReactants((mol_to_mutate,))
        if not products: 
            continue
            
        new_mol = products[0][0]
        
        try:
            Chem.SanitizeMol(new_mol)
            new_smiles = Chem.MolToSmiles(new_mol)
        except Exception:
            continue
            
        if new_smiles in generated_set: 
            continue
            
        new_fp = AllChem.GetMorganFingerprintAsBitVect(new_mol, 2, nBits=2048)
        similarity = DataStructs.TanimotoSimilarity(base_fp, new_fp)

        # Relaxed constraint for better diversity
        if 0.60 <= similarity <= 0.99:
            generated_set.add(new_smiles)
            candidates.append({"SMILES": new_smiles, "Tanimoto": similarity})
            current_pool.append(new_mol)

    # --- SAFETY CATCH: PAD SHORTFALLS ---
    if len(candidates) < num_candidates:
        print(f"      ⚠ Could only generate {len(candidates)} unique mutations. Padding the rest.")
        shortfall = num_candidates - len(candidates)
        candidates.extend([{"SMILES": input_smiles, "Tanimoto": 1.0}] * shortfall)

    return candidates

# ──────────────────────────────────────────────────────────────────────────────
#  EVALUATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def structural_inverse_design(polymer_name: str, base_smiles: str, master_df: pd.DataFrame, model, scaler, feature_cols: list):
    
    poly_df = master_df[master_df["Polymer"] == polymer_name]
    base_features = poly_df[feature_cols].mean().values.copy()
    
    base_feat_scaled = scaler.transform(base_features.reshape(1, -1))
    base_pred = model.predict(base_feat_scaled)[0]
    base_Tg, base_Dk = base_pred[0], base_pred[1]

    print(f"      Base Tg={base_Tg:.1f}°C  Base Dk={base_Dk:.4f}")
    print(f"  --> Generating structural mutations for {polymer_name}...")

    candidates = generate_candidates(base_smiles, num_candidates=50)
    
    fallback_dict = {
        "polymer": polymer_name, "input_smiles": base_smiles, "winning_smiles": base_smiles,
        "predicted_Tg_degC": base_Tg, "predicted_Dk": base_Dk,
        "base_Tg": base_Tg, "base_Dk": base_Dk,
        "optimal_FVF": base_features[83], "optimal_Crystallinity": base_features[80],
        "optimal_ChainRigidity": base_features[84], "optimal_DielPolarizability": base_features[90],
        "converged": False
    }

    if not candidates:
        print("      ⚠ No valid structural mutations found. Outputting base.")
        return fallback_dict, []

    print(f"      Evaluated {len(candidates)} candidate structures.")
    
    best_candidate = None
    targets = POLYMER_REGISTRY[polymer_name]
    
    # HYBRID OPTIMIZATION: Start the score to beat at the base polymer's Dk
    best_fitness = base_Dk 
    
    evaluated_candidates = [] # Tracks all valid candidates for CSV export

    for cand in candidates:
        cand_smiles = cand["SMILES"]
        
        try:
            # FIX: Passed cand_smiles as a list and removed n_samples
            struct_df = extract_structural_features_rdkit([cand_smiles], polymer_name=polymer_name)
            new_struct_feats = struct_df.values[0]
        except Exception:
            continue 
            
        feat_vec = base_features.copy()
        feat_vec[:40] = new_struct_feats 
        
        feat_scaled = scaler.transform(feat_vec.reshape(1, -1))
        pred = model.predict(feat_scaled)[0]
        Tg_pred, Dk_pred = pred[0], pred[1]
        
        evaluated_candidates.append({
            "Polymer_Family": polymer_name,
            "Base_SMILES": base_smiles,
            "Candidate_SMILES": cand_smiles,
            "Tanimoto_Similarity": round(cand["Tanimoto"], 4),
            "Predicted_Tg_degC": round(Tg_pred, 2),
            "Predicted_Dk": round(Dk_pred, 4)
        })
        
        # --- HYBRID APPROACH: CONSTRAINED MINIMIZATION ---
        # 1. Thermal Constraint: Tg must survive space (>= target_Tg)
        # 2. Dielectric Goal: Dk must be at or below base_Dk + tolerance.
        #    A small tolerance (0.05) is applied because structural mutations
        #    primarily change structural features (indices 0-39), while Dk is
        #    also driven by physics features (indices 40-119) which are held
        #    fixed at the class mean during candidate evaluation. Without this
        #    buffer, valid PEEK candidates are systematically rejected even when
        #    their Dk is marginally above base due to feature-block separation.
        DK_GATE_TOLERANCE = 0.05
        if Tg_pred >= targets["target_Tg"] and Dk_pred <= base_Dk + DK_GATE_TOLERANCE:
            
            # We want the absolute lowest Dk possible
            fitness = Dk_pred
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_candidate = {
                    "polymer": polymer_name,
                    "input_smiles": base_smiles,
                    "winning_smiles": cand_smiles,
                    "Tanimoto_Similarity": cand["Tanimoto"],
                    "predicted_Tg_degC": Tg_pred,
                    "predicted_Dk": Dk_pred,
                    "tg_target": f">= {targets['target_Tg']}",
                    "dk_target": "Minimize",
                    "optimal_FVF": base_features[83],
                    "optimal_Crystallinity": base_features[80],
                    "optimal_ChainRigidity": base_features[84],
                    "optimal_DielPolarizability": base_features[90],
                    "converged": True
                }

    if not best_candidate:
        print("      ⚠ No candidates beat the base polymer. Outputting base.")
        return fallback_dict, evaluated_candidates

    best_candidate["base_Tg"] = base_Tg
    best_candidate["base_Dk"] = base_Dk

    return best_candidate, evaluated_candidates

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'═' * 72}")
    print(f"  {BOLD}TASK 3 — STRUCTURAL INVERSE DESIGN (HYBRID OPTIMIZATION){RESET}")
    print(f"{'═' * 72}\n")

    with open("master_dataset.pkl", "rb") as f:
        master_df = pickle.load(f)

    # ── Model selector ────────────────────────────────────────────────────────
    # Run task2c_model_comparison.py first to see a full performance + SHAP
    # comparison between the two models before making your choice.
    print(f"  {'─'*72}")
    print(f"  {BOLD}Select QSPR model for inverse design:{RESET}\n")
    print(f"  [1]  MLP  (Multi-Layer Perceptron)")
    print(f"       Neural network [256→128→64]. Better for complex non-linear")
    print(f"       patterns. Slower to train. File: qspr_model_mlp.pkl\n")
    print(f"  [2]  GBR  (Gradient Boosting Regressor)")
    print(f"       Ensemble of 300 decision trees. More stable on tabular data.")
    print(f"       Provides native feature importance. File: qspr_model_gbr.pkl\n")
    print(f"  ℹ   Run task2c_model_comparison.py to compare both before choosing.")
    print(f"  {'─'*72}")

    MODEL_FILES = {"1": "qspr_model_mlp.pkl", "2": "qspr_model_gbr.pkl"}
    MODEL_NAMES = {"1": "MLP", "2": "GBR"}

    while True:
        choice = input("\n  Enter choice (1 or 2): ").strip()
        if choice in MODEL_FILES:
            break
        print("  Please enter 1 or 2.")

    chosen_file = MODEL_FILES[choice]
    chosen_name = MODEL_NAMES[choice]

    import os
    if not os.path.exists(chosen_file):
        task_hint = "task2_qspr_modeling.py" if choice == "1" else "task2b_gbr_modeling.py"
        print(f"\n  ✖  {chosen_file} not found.")
        print(f"     Run '{task_hint}' first to train and save the model.\n")
        raise SystemExit(1)

    with open(chosen_file, "rb") as f:
        qspr_bundle = pickle.load(f)

    print(f"\n  ✔  Loaded {chosen_name} model from {chosen_file}\n")

    model        = qspr_bundle["model"]
    scaler       = qspr_bundle["scaler"]
    feature_cols = qspr_bundle["feature_cols"]

    inv_results = []
    master_candidate_log = []
    
    for pname, meta in POLYMER_REGISTRY.items():
        base_smiles = master_df[master_df["Polymer"] == pname]["SMILES"].iloc[0]
        print(f"\n  🔍 Optimising {meta['color']}{BOLD}{pname}{RESET} ...")
        print(f"  --> Base SMILES: {base_smiles}")
        
        # All polymers — including PTFE — now go through structural_inverse_design.
        # generate_candidates() auto-selects aliphatic rules for PTFE (no aromatic rings).
        res, cands = structural_inverse_design(pname, base_smiles, master_df, model, scaler, feature_cols)
        
        inv_results.append(res)
        master_candidate_log.extend(cands)
        
        print(f"  --> BEST NEW SMILES: {res['winning_smiles']}")
        print(f"      Best Tg={res['predicted_Tg_degC']:.1f}°C  Best Dk={res['predicted_Dk']:.4f}")

    with open("inv_results.pkl", "wb") as f:
        pickle.dump(inv_results, f)
    print("\n  💾 Saved: inv_results.pkl -> (input for task4_output_management.py)")
    
    if master_candidate_log:
        df_candidates = pd.DataFrame(master_candidate_log)
        df_candidates.to_csv("candidates.csv", index=False)
        print(f"  💾 Saved: candidates.csv -> (Contains {len(df_candidates)} evaluated structures)\n")
