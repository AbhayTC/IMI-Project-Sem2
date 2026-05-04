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

PIPELINE OVERVIEW
─────────────────
  Task 1 ▶ Data Curation & Feature Extraction   (Programs 1, 2, 3)
  Task 2 ▶ QSPR Modeling (Multi-Target MLP)
  Task 3 ▶ Inverse Design via Bayesian Optimisation
  Task 4 ▶ Output Management & Reporting
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import warnings, os, math, hashlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import differential_evolution, minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

np.random.seed(42)

# ── Optional RDKit (falls back to physics-derived mock descriptors) ──────────
try:
    from rdkit import RDLogger, Chem, DataStructs
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, rdChemReactions
    from rdkit.Chem.rdMolDescriptors import CalcTPSA
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
    print("✔  RDKit detected — using genuine molecular descriptors and derivation.")
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠  RDKit not installed — using deterministic mock descriptors.\n"
          "   Install via: pip install rdkit\n"
          "   (All downstream QSPR and inverse-design logic is fully functional.)")

# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS  &  POLYMER REGISTRY
# ──────────────────────────────────────────────────────────────────────────────
N_SAMPLES_PER_POLYMER = 240
N_STRUCTURAL          = 40
N_LATENT              = 40
N_PHYSICS             = 40
N_TOTAL_FEATURES      = N_STRUCTURAL + N_LATENT + N_PHYSICS   # 120

POLYMER_REGISTRY = {
    "Polyimide": {
        "label"  : "PI",
        "smiles" : "O=C1c2ccccc2C(=O)N1c3ccc(Oc4ccc(N5C(=O)c6ccccc6C5=O)cc4)cc3",
        "member" : "Member1",
        "color"  : "\033[94m",            # blue
    },
    "PEEK": {
        "label"  : "PEEK",
        "smiles" : "c1ccc(Oc2ccc(Oc3ccc(C(=O)c4ccc(Oc5ccc(Oc6ccc(C(=O)c7ccc(Oc8ccc(Oc9ccc(C(=O)c%10ccccc%10)cc9)cc8)cc7)cc6)cc5)cc4)cc3)cc2)cc1",
        "member" : "Member2",
        "color"  : "\033[92m",            # green
    },
    "PTFE": {
        "label"  : "PTFE",
        "smiles" : "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "member" : "Member3",
        "color"  : "\033[93m",            # yellow
    },
}

RESET = "\033[0m"
BOLD  = "\033[1m"


# ──────────────────────────────────────────────────────────────────────────────
#  SMILES DERIVATION (from v3_fixed)
# ──────────────────────────────────────────────────────────────────────────────
def generate_derivatives(base_smiles: str, polymer_class: str, num_needed: int = 240) -> list:
    """Generates strictly constrained derivatives maintaining >=0.60 Tanimoto similarity."""
    if not RDKIT_AVAILABLE:
        return [base_smiles] * num_needed

    mutations_aromatic = [
        "[cH1:1]>>[c:1](F)", "[cH1:1]>>[c:1](Cl)",
        "[cH1:1]>>[c:1](C)", "[cH1:1]>>[c:1](O)"
    ]
    mutations_aliphatic = [
        "[F:1]>>[Cl:1]", "[F:1]>>[C:1]",
        "[F:1]>>[O:1]", "[F:1]>>[Br:1]"
    ]
    
    aromatic_rxns = [rdChemReactions.ReactionFromSmarts(r) for r in mutations_aromatic]
    aliphatic_rxns = [rdChemReactions.ReactionFromSmarts(r) for r in mutations_aliphatic]

    base_mol = Chem.MolFromSmiles(base_smiles)
    if base_mol is None:
        raise ValueError(f"CRITICAL: RDKit could not parse the base SMILES for {polymer_class}. Check syntax.")
        
    base_fp = AllChem.GetMorganFingerprintAsBitVect(base_mol, 2, nBits=2048)
    
    rxns_to_use = aliphatic_rxns if polymer_class == "PTFE" else aromatic_rxns
    
    seen_smiles = {base_smiles}
    candidates = [base_smiles]
    pool = [base_mol]
    
    attempts = 0
    np.random.seed(42) # Deterministic generation
    while len(candidates) < num_needed and attempts < 100000:
        attempts += 1
        mol = pool[np.random.randint(0, len(pool))]
        rxn = rxns_to_use[np.random.randint(0, len(rxns_to_use))]
        
        prods = rxn.RunReactants((mol,))
        if not prods: continue
            
        prod_idx = np.random.randint(0, len(prods))
        new_mol = prods[prod_idx][0]
        
        try:
            Chem.SanitizeMol(new_mol)
            new_smi = Chem.MolToSmiles(new_mol)
        except Exception:
            continue
            
        if new_smi in seen_smiles: continue
            
        new_fp = AllChem.GetMorganFingerprintAsBitVect(new_mol, 2, nBits=2048)
        tanimoto = DataStructs.TanimotoSimilarity(base_fp, new_fp)
        
        # --- UPDATED CONSTRAINT: >= 0.60 ---
        if tanimoto >= 0.60:
            seen_smiles.add(new_smi)
            candidates.append(new_smi)
            pool.append(new_mol)
            
    # --- SAFETY CATCH: PAD SHORTFALLS ---
    if len(candidates) < num_needed:
        print(f"    ⚠ Could only generate {len(candidates)} unique derivatives for {polymer_class} (Tanimoto >= 0.60 too strict). Padding the rest.")
        shortfall = num_needed - len(candidates)
        candidates.extend([base_smiles] * shortfall)
            
    return candidates

# ──────────────────────────────────────────────────────────────────────────────
#  DATA CURATION & FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

# ─── PROGRAM 1 ─── Structural Features (RDKit Molecular Descriptors)  ────────

STRUCTURAL_FEATURE_NAMES = [
    # Topology & Size
    "MolWt",           "HeavyAtomMolWt",  "ExactMolWt",      "NumHeavyAtoms",
    "NumRotatableBonds","NumRings",        "NumAromaticRings", "NumAliphaticRings",
    "RingCount",       "FractionCSP3",
    # Hydrogen Bonding
    "NumHDonors",      "NumHAcceptors",   "TPSA",
    # Electronic/Polarity
    "MolLogP",         "MolMR",           "LabuteASA",       "PEOE_VSA1",
    "PEOE_VSA2",       "PEOE_VSA3",       "PEOE_VSA4",
    # Surface Area partitions
    "SMR_VSA1",        "SMR_VSA2",        "SMR_VSA3",
    "SlogP_VSA1",      "SlogP_VSA2",      "SlogP_VSA3",
    # Atom counts
    "NumValenceElectrons","NumRadicalElectrons",
    "fr_C_O",          "fr_NH0",          "fr_NH1",
    "fr_ArN",          "fr_Ar_COO",       "fr_ether",
    "fr_ketone",       "fr_imide",        "fr_amide",
    "HallKierAlpha",   "Kappa1",          "Kappa2",
]
assert len(STRUCTURAL_FEATURE_NAMES) == N_STRUCTURAL

_MOCK_STRUCTURAL_REFS = { # Truncated for brevity; same as original task1 context
    "Polyimide": { "MolWt": 720.7, "HeavyAtomMolWt": 714.3, "ExactMolWt": 720.1, "NumHeavyAtoms": 52, "NumRotatableBonds": 8, "NumRings": 6, "NumAromaticRings": 5, "NumAliphaticRings": 1, "RingCount": 6, "FractionCSP3": 0.04, "NumHDonors": 0, "NumHAcceptors": 5, "TPSA": 77.8, "MolLogP": 4.12, "MolMR": 188.4, "LabuteASA": 265.1, "PEOE_VSA1": 34.2, "PEOE_VSA2": 18.1, "PEOE_VSA3": 12.4, "PEOE_VSA4": 9.3, "SMR_VSA1": 22.5, "SMR_VSA2": 18.3, "SMR_VSA3": 8.7, "SlogP_VSA1": 20.1, "SlogP_VSA2": 14.6, "SlogP_VSA3": 11.2, "NumValenceElectrons": 200, "NumRadicalElectrons": 0, "fr_C_O": 4, "fr_NH0": 2, "fr_NH1": 0, "fr_ArN": 2, "fr_Ar_COO": 0, "fr_ether": 1, "fr_ketone": 0, "fr_imide": 2, "fr_amide": 0, "HallKierAlpha": -3.8, "Kappa1": 22.1, "Kappa2": 11.6 },
    "PEEK": { "MolWt": 480.5, "HeavyAtomMolWt": 475.1, "ExactMolWt": 480.1, "NumHeavyAtoms": 36, "NumRotatableBonds": 6, "NumRings": 4, "NumAromaticRings": 4, "NumAliphaticRings": 0, "RingCount": 4, "FractionCSP3": 0.00, "NumHDonors": 0, "NumHAcceptors": 3, "TPSA": 39.5, "MolLogP": 5.24, "MolMR": 136.2, "LabuteASA": 194.6, "PEOE_VSA1": 22.8, "PEOE_VSA2": 11.4, "PEOE_VSA3": 8.9, "PEOE_VSA4": 6.1, "SMR_VSA1": 15.3, "SMR_VSA2": 12.1, "SMR_VSA3": 5.8, "SlogP_VSA1": 13.7, "SlogP_VSA2": 10.2, "SlogP_VSA3": 8.4, "NumValenceElectrons": 136, "NumRadicalElectrons": 0, "fr_C_O": 3, "fr_NH0": 0, "fr_NH1": 0, "fr_ArN": 0, "fr_Ar_COO": 0, "fr_ether": 2, "fr_ketone": 1, "fr_imide": 0, "fr_amide": 0, "HallKierAlpha": -2.5, "Kappa1": 16.4, "Kappa2": 8.9 },
    "PTFE": { "MolWt": 338.0, "HeavyAtomMolWt": 336.0, "ExactMolWt": 337.9, "NumHeavyAtoms": 8, "NumRotatableBonds": 1, "NumRings": 0, "NumAromaticRings": 0, "NumAliphaticRings": 0, "RingCount": 0, "FractionCSP3": 1.00, "NumHDonors": 0, "NumHAcceptors": 6, "TPSA": 0.0, "MolLogP": 4.68, "MolMR": 48.2, "LabuteASA": 74.3, "PEOE_VSA1": 8.4, "PEOE_VSA2": 4.2, "PEOE_VSA3": 3.1, "PEOE_VSA4": 2.2, "SMR_VSA1": 5.6, "SMR_VSA2": 3.4, "SMR_VSA3": 1.8, "SlogP_VSA1": 4.8, "SlogP_VSA2": 3.2, "SlogP_VSA3": 2.1, "NumValenceElectrons": 56, "NumRadicalElectrons": 0, "fr_C_O": 0, "fr_NH0": 0, "fr_NH1": 0, "fr_ArN": 0, "fr_Ar_COO": 0, "fr_ether": 0, "fr_ketone": 0, "fr_imide": 0, "fr_amide": 0, "HallKierAlpha": 0.2, "Kappa1": 2.8, "Kappa2": 1.4 }
}
_MOCK_STRUCTURAL_NOISE = 0.05

def _seeded_noise(polymer_name: str, feature_name: str, sample_idx: int, scale: float) -> float:
    seed_str = f"{polymer_name}_{feature_name}_{sample_idx}"
    seed_int = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed_int)
    return rng.normal(0, scale)

def extract_structural_features_rdkit(smiles_list: list, polymer_name: str) -> pd.DataFrame:
    descriptor_map = {
        "MolWt"               : lambda m: Descriptors.MolWt(m),
        "HeavyAtomMolWt"      : lambda m: Descriptors.HeavyAtomMolWt(m),
        "ExactMolWt"          : lambda m: Descriptors.ExactMolWt(m),
        "NumHeavyAtoms"       : lambda m: m.GetNumHeavyAtoms(),
        "NumRotatableBonds"   : lambda m: rdMolDescriptors.CalcNumRotatableBonds(m),
        "NumRings"            : lambda m: rdMolDescriptors.CalcNumRings(m),
        "NumAromaticRings"    : lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
        "NumAliphaticRings"   : lambda m: rdMolDescriptors.CalcNumAliphaticRings(m),
        "RingCount"           : lambda m: rdMolDescriptors.CalcNumRings(m),
        "FractionCSP3"        : lambda m: rdMolDescriptors.CalcFractionCSP3(m),
        "NumHDonors"          : lambda m: rdMolDescriptors.CalcNumHBD(m),
        "NumHAcceptors"       : lambda m: rdMolDescriptors.CalcNumHBA(m),
        "TPSA"                : lambda m: CalcTPSA(m),
        "MolLogP"             : lambda m: Descriptors.MolLogP(m),
        "MolMR"               : lambda m: Descriptors.MolMR(m),
        "LabuteASA"           : lambda m: rdMolDescriptors.CalcLabuteASA(m),
        "PEOE_VSA1"           : lambda m: Descriptors.PEOE_VSA1(m),
        "PEOE_VSA2"           : lambda m: Descriptors.PEOE_VSA2(m),
        "PEOE_VSA3"           : lambda m: Descriptors.PEOE_VSA3(m),
        "PEOE_VSA4"           : lambda m: Descriptors.PEOE_VSA4(m),
        "SMR_VSA1"            : lambda m: Descriptors.SMR_VSA1(m),
        "SMR_VSA2"            : lambda m: Descriptors.SMR_VSA2(m),
        "SMR_VSA3"            : lambda m: Descriptors.SMR_VSA3(m),
        "SlogP_VSA1"          : lambda m: Descriptors.SlogP_VSA1(m),
        "SlogP_VSA2"          : lambda m: Descriptors.SlogP_VSA2(m),
        "SlogP_VSA3"          : lambda m: Descriptors.SlogP_VSA3(m),
        "NumValenceElectrons" : lambda m: Descriptors.NumValenceElectrons(m),
        "NumRadicalElectrons" : lambda m: Descriptors.NumRadicalElectrons(m),
        "fr_C_O"              : lambda m: Descriptors.fr_C_O(m),
        "fr_NH0"              : lambda m: Descriptors.fr_NH0(m),
        "fr_NH1"              : lambda m: Descriptors.fr_NH1(m),
        "fr_ArN"              : lambda m: Descriptors.fr_ArN(m),
        "fr_Ar_COO"           : lambda m: Descriptors.fr_Ar_COO(m),
        "fr_ether"            : lambda m: Descriptors.fr_ether(m),
        "fr_ketone"           : lambda m: Descriptors.fr_ketone(m),
        "fr_imide"            : lambda m: Descriptors.fr_imide(m),
        "fr_amide"            : lambda m: Descriptors.fr_amide(m),
        "HallKierAlpha"       : lambda m: rdMolDescriptors.CalcHallKierAlpha(m),
        "Kappa1"              : lambda m: rdMolDescriptors.CalcKappa1(m),
        "Kappa2"              : lambda m: rdMolDescriptors.CalcKappa2(m),
    }

    records = []
    for i, sm in enumerate(smiles_list):
        if i > 0 and i % 50 == 0:
            print(f"    [+] Processed {i}/{len(smiles_list)} RDKit descriptors for {polymer_name}...")
        mol = Chem.MolFromSmiles(sm)
        row = {}
        for feat in STRUCTURAL_FEATURE_NAMES:
            try:
                row[feat] = float(descriptor_map[feat](mol))
            except Exception:
                row[feat] = 0.0
        records.append(row)
    return pd.DataFrame(records, columns=STRUCTURAL_FEATURE_NAMES)

def extract_structural_features_mock(polymer_name: str, n_samples: int) -> pd.DataFrame:
    ref = _MOCK_STRUCTURAL_REFS[polymer_name]
    records = []
    for i in range(n_samples):
        row = {}
        for feat in STRUCTURAL_FEATURE_NAMES:
            bv    = ref[feat]
            scale = max(abs(bv) * _MOCK_STRUCTURAL_NOISE, 0.05)
            row[feat] = bv + _seeded_noise(polymer_name, feat, i, scale)
        records.append(row)
    return pd.DataFrame(records, columns=STRUCTURAL_FEATURE_NAMES)

def extract_structural_features(polymer_name: str, smiles_list: list) -> pd.DataFrame:
    if RDKIT_AVAILABLE:
        return extract_structural_features_rdkit(smiles_list, polymer_name)
    return extract_structural_features_mock(polymer_name, len(smiles_list))


# ─── PROGRAM 2 ─── Latent / Contextual Features (polyBERT Simulation) ────────

LATENT_FEATURE_NAMES = [f"polyBERT_dim_{i+1:02d}" for i in range(N_LATENT)]

_LATENT_ANCHORS = {
    "Polyimide": { "scale": 0.30, "aromatic_bias": 0.65, "polar_bias": 0.55, "backbone_bias": 0.45, "sequence_bias": 0.20 },
    "PEEK":      { "scale": 0.28, "aromatic_bias": 0.70, "polar_bias": 0.30, "backbone_bias": 0.60, "sequence_bias": 0.15 },
    "PTFE":      { "scale": 0.18, "aromatic_bias": -0.60, "polar_bias": -0.50, "backbone_bias": 0.20, "sequence_bias": 0.40 },
}

def extract_latent_features(polymer_name: str, smiles_list: list) -> pd.DataFrame:
    anch  = _LATENT_ANCHORS[polymer_name]
    scale = anch["scale"]

    base_anchor = np.array(
        [anch["aromatic_bias"]] * 10 +
        [anch["polar_bias"]   ] * 10 +
        [anch["backbone_bias"]] * 10 +
        [anch["sequence_bias"]] * 10
    )

    records = []
    for i, sm in enumerate(smiles_list):
        smiles_seed = int(hashlib.md5(sm.encode()).hexdigest(), 16) % (2**32)
        rng_base    = np.random.RandomState(smiles_seed)
        anchor      = base_anchor + rng_base.normal(0, 0.05, size=40)
        
        rng_sample = np.random.RandomState(smiles_seed + i * 137)
        vec = anchor + rng_sample.normal(0, scale, size=40)
        vec = np.clip(vec, -1.5, 1.5)
        records.append(dict(zip(LATENT_FEATURE_NAMES, vec)))

    return pd.DataFrame(records, columns=LATENT_FEATURE_NAMES)


# ─── PROGRAM 3 ─── Physics-Based / Morphological Features ────────────────────

PHYSICS_FEATURE_NAMES = [
    "DegreeOfCrystallinity",   "CrystallinePhaseContent",  "AmorphousPhaseContent",
    "FreeVolumeFraction",      "ChainRigidityIndex",        "SegmentalMobility",
    "ThermalExpansionCoeff",   "HeatCapacity_Cp",           "ThermalDiffusivity",
    "GlassyModulus",           "DielectricPolarizability","ElectronicPolarizability",  
    "IonicPolarizability",     "OrientationalPolarizability","DipoleMomentRepeat",     
    "CurieWeissConstant",      "CrosslinkingDensity",     "EntanglementMolWt",         
    "ContourLengthPerUnit",    "PersistenceLength",       "CharacteristicRatio",       
    "ChainFlexibilityParam",   "Mw_kDa",                  "Mn_kDa",                   
    "PolyDisersityIndex",      "ZAverageMolWt",           "ViscosityAverageMolWt",    
    "NumberAverageDPn",        "LamellaeThickness_nm",    "SpheruliteRadius_um",      
    "CrystalThickness_nm",     "TieChainsPerArea",        "InterfacialThickness_nm",  
    "MicrostructureOrder",     "PermittivityRealPart",    "PermittivityImaginaryPart",
    "TanDeltaDielectric",      "YoungModulus_GPa",        "TensileStrength_MPa",      
    "ElongationBreak_pct",
]
assert len(PHYSICS_FEATURE_NAMES) == N_PHYSICS

_PHYSICS_PRIORS = {
    "Polyimide": dict( DegreeOfCrystallinity=(0.35, 0.06), CrystallinePhaseContent=(0.35, 0.05), AmorphousPhaseContent=(0.65, 0.05), FreeVolumeFraction=(0.11, 0.02), ChainRigidityIndex=(0.82, 0.06), SegmentalMobility=(0.18, 0.03), ThermalExpansionCoeff=(3.2e-5, 4e-6), HeatCapacity_Cp=(1.05, 0.08), ThermalDiffusivity=(1.8e-7, 2e-8), GlassyModulus=(3.4, 0.3), DielectricPolarizability=(28.5, 2.0), ElectronicPolarizability=(22.1, 1.5), IonicPolarizability=(3.8, 0.4), OrientationalPolarizability=(2.6, 0.3), DipoleMomentRepeat=(4.8, 0.5), CurieWeissConstant=(320.0, 20.0), CrosslinkingDensity=(0.008, 0.002), EntanglementMolWt=(5800.0, 500.0), ContourLengthPerUnit=(1.48, 0.10), PersistenceLength=(12.5, 1.5), CharacteristicRatio=(8.2, 0.8), ChainFlexibilityParam=(0.22, 0.03), Mw_kDa=(85.0, 12.0), Mn_kDa=(42.0, 6.0), PolyDisersityIndex=(2.05, 0.25), ZAverageMolWt=(130.0, 18.0), ViscosityAverageMolWt=(78.0, 10.0), NumberAverageDPn=(210.0, 30.0), LamellaeThickness_nm=(12.0, 2.0), SpheruliteRadius_um=(3.5, 0.8), CrystalThickness_nm=(18.0, 3.0), TieChainsPerArea=(1.8e14, 2e13), InterfacialThickness_nm=(4.5, 0.6), MicrostructureOrder=(0.55, 0.06), PermittivityRealPart=(3.5, 0.2), PermittivityImaginaryPart=(0.08, 0.01), TanDeltaDielectric=(0.022, 0.003), YoungModulus_GPa=(3.1, 0.3), TensileStrength_MPa=(185.0, 20.0), ElongationBreak_pct=(35.0, 5.0) ),
    "PEEK": dict( DegreeOfCrystallinity=(0.42, 0.07), CrystallinePhaseContent=(0.42, 0.06), AmorphousPhaseContent=(0.58, 0.06), FreeVolumeFraction=(0.09, 0.02), ChainRigidityIndex=(0.88, 0.05), SegmentalMobility=(0.14, 0.02), ThermalExpansionCoeff=(4.7e-5, 5e-6), HeatCapacity_Cp=(1.32, 0.10), ThermalDiffusivity=(2.5e-7, 3e-8), GlassyModulus=(3.7, 0.4), DielectricPolarizability=(25.2, 2.2), ElectronicPolarizability=(19.8, 1.6), IonicPolarizability=(2.9, 0.3), OrientationalPolarizability=(2.5, 0.3), DipoleMomentRepeat=(3.6, 0.4), CurieWeissConstant=(180.0, 15.0), CrosslinkingDensity=(0.004, 0.001), EntanglementMolWt=(8000.0, 700.0), ContourLengthPerUnit=(1.62, 0.12), PersistenceLength=(9.8, 1.2), CharacteristicRatio=(10.5, 1.0), ChainFlexibilityParam=(0.16, 0.02), Mw_kDa=(95.0, 15.0), Mn_kDa=(48.0, 7.0), PolyDisersityIndex=(2.00, 0.22), ZAverageMolWt=(148.0, 22.0), ViscosityAverageMolWt=(88.0, 12.0), NumberAverageDPn=(220.0, 35.0), LamellaeThickness_nm=(15.0, 2.5), SpheruliteRadius_um=(8.5, 1.5), CrystalThickness_nm=(22.0, 4.0), TieChainsPerArea=(2.1e14, 2.5e13), InterfacialThickness_nm=(5.8, 0.7), MicrostructureOrder=(0.68, 0.07), PermittivityRealPart=(3.3, 0.2), PermittivityImaginaryPart=(0.06, 0.01), TanDeltaDielectric=(0.003, 0.0005), YoungModulus_GPa=(3.8, 0.4), TensileStrength_MPa=(210.0, 25.0), ElongationBreak_pct=(30.0, 4.0) ),
    "PTFE": dict( DegreeOfCrystallinity=(0.60, 0.08), CrystallinePhaseContent=(0.60, 0.07), AmorphousPhaseContent=(0.40, 0.07), FreeVolumeFraction=(0.13, 0.02), ChainRigidityIndex=(0.65, 0.07), SegmentalMobility=(0.28, 0.04), ThermalExpansionCoeff=(1.1e-4, 1e-5), HeatCapacity_Cp=(1.02, 0.07), ThermalDiffusivity=(2.5e-7, 3e-8), GlassyModulus=(0.55, 0.08), DielectricPolarizability=(12.8, 1.2), ElectronicPolarizability=(12.0, 1.0), IonicPolarizability=(0.5, 0.1), OrientationalPolarizability=(0.3, 0.05), DipoleMomentRepeat=(0.0, 0.05), CurieWeissConstant=(0.0, 5.0), CrosslinkingDensity=(0.0005, 0.0002), EntanglementMolWt=(12000.0, 1000.0), ContourLengthPerUnit=(1.28, 0.08), PersistenceLength=(1.1, 0.2), CharacteristicRatio=(5.8, 0.6), ChainFlexibilityParam=(0.52, 0.05), Mw_kDa=(3200.0, 400.0), Mn_kDa=(500.0, 60.0), PolyDisersityIndex=(6.50, 0.80), ZAverageMolWt=(8500.0, 1000.0), ViscosityAverageMolWt=(2800.0, 350.0), NumberAverageDPn=(5000.0, 600.0), LamellaeThickness_nm=(25.0, 4.0), SpheruliteRadius_um=(30.0, 5.0), CrystalThickness_nm=(32.0, 5.0), TieChainsPerArea=(3.5e14, 4e13), InterfacialThickness_nm=(8.0, 1.0), MicrostructureOrder=(0.75, 0.08), PermittivityRealPart=(2.0, 0.10), PermittivityImaginaryPart=(0.001, 0.0002), TanDeltaDielectric=(0.0002, 0.00003), YoungModulus_GPa=(0.55, 0.06), TensileStrength_MPa=(32.0, 5.0), ElongationBreak_pct=(300.0, 40.0) ),
}

def extract_physics_features(polymer_name: str, n_samples: int) -> pd.DataFrame:
    priors = _PHYSICS_PRIORS[polymer_name]
    seed   = int(hashlib.md5(f"physics_{polymer_name}".encode()).hexdigest(), 16) % (2**32)
    rng    = np.random.RandomState(seed)

    records = []
    for _ in range(n_samples):
        row = {}
        for feat in PHYSICS_FEATURE_NAMES:
            mu, sigma = priors[feat]
            val = rng.normal(mu, sigma)
            if feat in ("DegreeOfCrystallinity", "CrystallinePhaseContent", "AmorphousPhaseContent", "FreeVolumeFraction", "MicrostructureOrder"):
                val = np.clip(val, 0.0, 1.0)
            elif feat in ("CrosslinkingDensity", "DipoleMomentRepeat", "PermittivityImaginaryPart", "TanDeltaDielectric", "PolyDisersityIndex"):
                val = max(val, 0.0)
            row[feat] = val
        row["AmorphousPhaseContent"] = 1.0 - row["DegreeOfCrystallinity"]
        records.append(row)

    return pd.DataFrame(records, columns=PHYSICS_FEATURE_NAMES)


# ─── MASTER DATASET BUILDER ──────────────────────────────────────────────────

def build_master_dataset() -> pd.DataFrame:
    all_dfs = []
    for polymer_name, meta in POLYMER_REGISTRY.items():
        print(f"  ▶ Extracting features for {meta['color']}{BOLD}{polymer_name}{RESET} ...")
        base_smiles = meta["smiles"]
        n_samples = N_SAMPLES_PER_POLYMER

        if RDKIT_AVAILABLE:
            smiles_list = generate_derivatives(base_smiles, polymer_name, n_samples)
        else:
            smiles_list = [base_smiles] * n_samples

        df_struct  = extract_structural_features(polymer_name, smiles_list)
        df_latent  = extract_latent_features(polymer_name, smiles_list)
        df_physics = extract_physics_features(polymer_name, n_samples)

        df_poly = pd.concat([df_struct, df_latent, df_physics], axis=1)
        df_poly.insert(0, "Polymer",   polymer_name)
        df_poly.insert(1, "SMILES",    smiles_list)
        df_poly.insert(2, "Sample_ID", [f"{polymer_name[:2]}_{i+1:04d}" for i in range(n_samples)])
        all_dfs.append(df_poly)

    master = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  ✔  Master dataset shape: {master.shape}  "
          f"({len(master)} samples × {master.shape[1]} columns)\n")
    return master


# ──────────────────────────────────────────────────────────────────────────────
#  QSPR TARGET GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_targets(master_df: pd.DataFrame) -> pd.DataFrame:
    df = master_df.copy()

    # --- Tg ---
    alpha, beta, gamma, delta, eps = 280.0, 18.0, 600.0, 80.0, 1500.0
    tg_base = { "Polyimide": 240.0, "PEEK": 200.0, "PTFE": 10.0 }
    rng_tg = np.random.RandomState(101)

    # --- Dk ---
    a, b, c, d = 0.065, 8.0, 2.0, 1.0
    dk_base = { "Polyimide": 1.8, "PEEK": 1.6, "PTFE": 1.6 }
    rng_dk = np.random.RandomState(202)

    # --- Outgassing & Radiation Random States ---
    rng_out = np.random.RandomState(303)
    rng_rad = np.random.RandomState(404)

    Tg_vals, Dk_vals, Outgassing_vals, Radiation_vals = [], [], [], []

    for _, row in df.iterrows():
        # Tg
        base_t = tg_base[row["Polymer"]]
        Tg = (base_t + alpha * row["ChainRigidityIndex"] + beta * row["NumAromaticRings"] - gamma * row["FreeVolumeFraction"] + delta * row["DegreeOfCrystallinity"] + eps * row["CrosslinkingDensity"]) + rng_tg.normal(0, 4.0)
        Tg_vals.append(Tg)

        # Dk
        base_d = dk_base[row["Polymer"]]
        Dk = (base_d + a * row["DielectricPolarizability"] - b * row["FreeVolumeFraction"] + c * row["DegreeOfCrystallinity"] - d * row["FractionCSP3"]) + rng_dk.normal(0, 0.05)
        Dk_vals.append(max(Dk, 1.5))

        # Outgassing
        out = 0.153 + 1.2*(row["FreeVolumeFraction"] - 0.09) - 0.30*(row["DegreeOfCrystallinity"] - 0.42) + rng_out.normal(0, 0.01)
        Outgassing_vals.append(max(0.0, out))

        # Radiation
        rad = 25.0 + 8.0*(row["NumAromaticRings"] - 4.0) - 50.0*(row["FreeVolumeFraction"] - 0.09) + rng_rad.normal(0, 0.5)
        Radiation_vals.append(max(0.0, rad))

    df["Tg_degC"] = Tg_vals
    df["Dk_1GHz"] = Dk_vals
    df["Outgassing_TML_pct"] = Outgassing_vals
    df["RadiationDose_MGy"] = Radiation_vals

    return df

import pickle
if __name__ == "__main__":
    print("Starting Task 1: Complete Data Curation (120 features)...")
    df_master = build_master_dataset()
    df_final = compute_targets(df_master)
    with open("master_dataset.pkl", "wb") as f:
        pickle.dump(df_final, f)
    print("Successfully saved master_dataset.pkl with", df_final.shape)
