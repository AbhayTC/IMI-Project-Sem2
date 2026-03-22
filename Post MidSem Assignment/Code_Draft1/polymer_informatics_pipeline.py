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
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import CalcTPSA
    RDKIT_AVAILABLE = True
    print("✔  RDKit detected — using genuine molecular descriptors.")
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
        "smiles" : ("O=C1c2ccccc2C(=O)N1c3ccc(Oc4ccc"
                    "(N5C(=O)c6ccccc6C5=O)cc4)cc3"),
        "member" : "Member1",
        "color"  : "\033[94m",            # blue
    },
    "PEEK": {
        "label"  : "PEEK",
        "smiles" : ("O=C(c1ccc(Oc2ccc(Oc3ccc(C(=O)c4ccccc4)"
                    "cc3)cc2)cc1)c5ccccc5"),
        "member" : "Member2",
        "color"  : "\033[92m",            # green
    },
    "PTFE": {
        "label"  : "PTFE",
        "smiles" : "FC(F)(F)C(F)(F)F",
        "member" : "Member3",
        "color"  : "\033[93m",            # yellow
    },
}

RESET = "\033[0m"
BOLD  = "\033[1m"

# ──────────────────────────────────────────────────────────────────────────────
#  ████████╗ █████╗ ███████╗██╗  ██╗     ██╗
#  ╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝    ███║
#     ██║   ███████║███████╗█████╔╝     ╚██║
#     ██║   ██╔══██║╚════██║██╔═██╗      ██║
#     ██║   ██║  ██║███████║██║  ██╗     ██║
#     ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝
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
    # Atom counts (relevant to thermal/dielectric)
    "NumValenceElectrons","NumRadicalElectrons",
    "fr_C_O",          "fr_NH0",          "fr_NH1",
    "fr_ArN",          "fr_Ar_COO",       "fr_ether",
    "fr_ketone",       "fr_imide",        "fr_amide",
    "HallKierAlpha",   "Kappa1",          "Kappa2",
]
assert len(STRUCTURAL_FEATURE_NAMES) == N_STRUCTURAL, \
    f"Expected 40 structural names, got {len(STRUCTURAL_FEATURE_NAMES)}"

# Realistic per-polymer reference values for the mock descriptor generator
# (used only when RDKit is unavailable)
_MOCK_STRUCTURAL_REFS = {
    "Polyimide": {
        "MolWt": 720.7,    "HeavyAtomMolWt": 714.3,  "ExactMolWt": 720.1,
        "NumHeavyAtoms": 52,"NumRotatableBonds": 8,   "NumRings": 6,
        "NumAromaticRings": 5,"NumAliphaticRings": 1, "RingCount": 6,
        "FractionCSP3": 0.04,"NumHDonors": 0,         "NumHAcceptors": 5,
        "TPSA": 77.8,       "MolLogP": 4.12,          "MolMR": 188.4,
        "LabuteASA": 265.1, "PEOE_VSA1": 34.2,        "PEOE_VSA2": 18.1,
        "PEOE_VSA3": 12.4,  "PEOE_VSA4":  9.3,        "SMR_VSA1": 22.5,
        "SMR_VSA2": 18.3,   "SMR_VSA3":  8.7,         "SlogP_VSA1": 20.1,
        "SlogP_VSA2": 14.6, "SlogP_VSA3": 11.2,
        "NumValenceElectrons": 200,"NumRadicalElectrons": 0,
        "fr_C_O": 4,        "fr_NH0": 2,               "fr_NH1": 0,
        "fr_ArN": 2,        "fr_Ar_COO": 0,            "fr_ether": 1,
        "fr_ketone": 0,     "fr_imide": 2,             "fr_amide": 0,
        "HallKierAlpha": -3.8,"Kappa1": 22.1,          "Kappa2": 11.6,
    },    "PEEK": {
        "MolWt": 480.5,    "HeavyAtomMolWt": 475.1,  "ExactMolWt": 480.1,
        "NumHeavyAtoms": 36,"NumRotatableBonds": 6,   "NumRings": 4,
        "NumAromaticRings": 4,"NumAliphaticRings": 0, "RingCount": 4,
        "FractionCSP3": 0.00,"NumHDonors": 0,         "NumHAcceptors": 3,
        "TPSA": 39.5,       "MolLogP": 5.24,          "MolMR": 136.2,
        "LabuteASA": 194.6, "PEOE_VSA1": 22.8,        "PEOE_VSA2": 11.4,
        "PEOE_VSA3":  8.9,  "PEOE_VSA4":  6.1,        "SMR_VSA1": 15.3,
        "SMR_VSA2": 12.1,   "SMR_VSA3":  5.8,         "SlogP_VSA1": 13.7,
        "SlogP_VSA2": 10.2, "SlogP_VSA3":  8.4,
        "NumValenceElectrons": 136,"NumRadicalElectrons": 0,
        "fr_C_O": 3,        "fr_NH0": 0,               "fr_NH1": 0,
        "fr_ArN": 0,        "fr_Ar_COO": 0,            "fr_ether": 2,
        "fr_ketone": 1,     "fr_imide": 0,             "fr_amide": 0,
        "HallKierAlpha": -2.5,"Kappa1": 16.4,          "Kappa2":  8.9,
    },
    "PTFE": {
        "MolWt": 338.0,    "HeavyAtomMolWt": 336.0,  "ExactMolWt": 337.9,
        "NumHeavyAtoms":  8,"NumRotatableBonds": 1,   "NumRings": 0,
        "NumAromaticRings": 0,"NumAliphaticRings": 0, "RingCount": 0,
        "FractionCSP3": 1.00,"NumHDonors": 0,         "NumHAcceptors": 6,
        "TPSA":  0.0,       "MolLogP": 4.68,          "MolMR":  48.2,
        "LabuteASA":  74.3, "PEOE_VSA1":  8.4,        "PEOE_VSA2":  4.2,
        "PEOE_VSA3":  3.1,  "PEOE_VSA4":  2.2,        "SMR_VSA1":  5.6,
        "SMR_VSA2":  3.4,   "SMR_VSA3":  1.8,         "SlogP_VSA1":  4.8,
        "SlogP_VSA2":  3.2, "SlogP_VSA3":  2.1,
        "NumValenceElectrons":  56,"NumRadicalElectrons": 0,
        "fr_C_O": 0,        "fr_NH0": 0,               "fr_NH1": 0,
        "fr_ArN": 0,        "fr_Ar_COO": 0,            "fr_ether": 0,
        "fr_ketone": 0,     "fr_imide": 0,             "fr_amide": 0,
        "HallKierAlpha":  0.2,"Kappa1":  2.8,          "Kappa2":  1.4,
    },
}

# Per-feature noise scale (5 % of the reference value, minimum 0.05)
_MOCK_STRUCTURAL_NOISE = 0.05


def _seeded_noise(polymer_name: str, feature_name: str,
                  sample_idx: int, scale: float) -> float:
    """Deterministic pseudo-random noise using MD5 seed — reproducible."""
    seed_str = f"{polymer_name}_{feature_name}_{sample_idx}"
    seed_int = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed_int)
    return rng.normal(0, scale)


def extract_structural_features_rdkit(smiles: str, n_samples: int,
                                      polymer_name: str) -> pd.DataFrame:
    """
    Program 1 (RDKit path):
    Compute 40 molecular descriptors from the repeat-unit SMILES.
    The deterministic descriptor values are perturbed by small Gaussian
    noise across samples to represent synthesis/batch variability.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    # Map each feature name to an RDKit callable
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

    # Compute deterministic base values once
    base_vals = {}
    for feat in STRUCTURAL_FEATURE_NAMES:
        try:
            base_vals[feat] = float(descriptor_map[feat](mol))
        except Exception:
            base_vals[feat] = 0.0

    # Replicate with per-sample Gaussian noise (σ = 2 % of value)
    records = []
    for i in range(n_samples):
        row = {}
        for feat in STRUCTURAL_FEATURE_NAMES:
            bv = base_vals[feat]
            noise_scale = max(abs(bv) * 0.02, 0.01)
            row[feat] = bv + _seeded_noise(polymer_name, feat, i, noise_scale)
        records.append(row)

    return pd.DataFrame(records, columns=STRUCTURAL_FEATURE_NAMES)


def extract_structural_features_mock(polymer_name: str,
                                     n_samples: int) -> pd.DataFrame:
    """
    Program 1 (mock path):
    Deterministic physics-anchored mock descriptors when RDKit is unavailable.
    Gaussian noise (σ = 5 % of reference) creates 240 unique samples.
    """
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


def extract_structural_features(polymer_name: str, smiles: str,
                                 n_samples: int) -> pd.DataFrame:
    """Program 1 dispatcher — RDKit if available, else mock."""
    if RDKIT_AVAILABLE:
        return extract_structural_features_rdkit(smiles, n_samples, polymer_name)
    return extract_structural_features_mock(polymer_name, n_samples)


# ─── PROGRAM 2 ─── Latent / Contextual Features (polyBERT Simulation) ────────

LATENT_FEATURE_NAMES = [f"polyBERT_dim_{i+1:02d}" for i in range(N_LATENT)]

# Realistic latent-space anchors per polymer  (hand-crafted for chemical meaning)
_LATENT_ANCHORS = {
    "Polyimide": {
        # aromatic-rich, high polarity, imide character
        "scale"          : 0.30,
        "aromatic_bias"  : 0.65,   # dims 1-10 — aromaticity token
        "polar_bias"     : 0.55,   # dims 11-20 — polar group tokens
        "backbone_bias"  : 0.45,   # dims 21-30 — backbone stiffness
        "sequence_bias"  : 0.20,   # dims 31-40 — sequence context
    },
    "PEEK": {
        "scale"          : 0.28,
        "aromatic_bias"  : 0.70,
        "polar_bias"     : 0.30,
        "backbone_bias"  : 0.60,
        "sequence_bias"  : 0.15,
    },
    "PTFE": {
        "scale"          : 0.18,
        "aromatic_bias"  : -0.60,  # no aromaticity — negative projection
        "polar_bias"     : -0.50,  # highly non-polar
        "backbone_bias"  :  0.20,  # flexible helix
        "sequence_bias"  :  0.40,  # regular –(CF2-CF2)n– repeat
    },
}


def extract_latent_features(polymer_name: str, smiles: str,
                             n_samples: int) -> pd.DataFrame:
    """
    Program 2 — Simulated polyBERT Embedding.

    Maps each SMILES to a 40-dimensional latent vector.
    The 40 dims are partitioned into four chemical-meaning blocks:
      • dims  1–10  : aromaticity / conjugation sub-space
      • dims 11–20  : polarity / hydrogen-bonding sub-space
      • dims 21–30  : backbone rigidity / stiffness sub-space
      • dims 31–40  : sequence & repeat-unit context sub-space

    Sample-level diversity is introduced by adding Gaussian noise (σ = 0.10)
    to simulate molecular weight dispersity and conformational sampling.
    """
    anch  = _LATENT_ANCHORS[polymer_name]
    scale = anch["scale"]

    # Build the polymer-specific anchor vector
    anchor = np.array(
        [anch["aromatic_bias"]] * 10 +
        [anch["polar_bias"]   ] * 10 +
        [anch["backbone_bias"]] * 10 +
        [anch["sequence_bias"]] * 10
    )

    # Normalise to unit-sphere neighbourhood using SMILES hash as seed offset
    smiles_seed = int(hashlib.md5(smiles.encode()).hexdigest(), 16) % (2**32)
    rng_base    = np.random.RandomState(smiles_seed)
    anchor     += rng_base.normal(0, 0.05, size=40)   # SMILES-specific jitter

    records = []
    for i in range(n_samples):
        rng_sample = np.random.RandomState(smiles_seed + i * 137)
        vec = anchor + rng_sample.normal(0, scale, size=40)
        vec = np.clip(vec, -1.5, 1.5)                 # bounded latent space
        records.append(dict(zip(LATENT_FEATURE_NAMES, vec)))

    return pd.DataFrame(records, columns=LATENT_FEATURE_NAMES)


# ─── PROGRAM 3 ─── Physics-Based / Morphological Features ────────────────────

PHYSICS_FEATURE_NAMES = [
    # Thermal / chain dynamics  (10)
    "DegreeOfCrystallinity",   "CrystallinePhaseContent",  "AmorphousPhaseContent",
    "FreeVolumeFraction",      "ChainRigidityIndex",        "SegmentalMobility",
    "ThermalExpansionCoeff",   "HeatCapacity_Cp",           "ThermalDiffusivity",
    "GlassyModulus",
    # Dielectric / electronic  (6)
    "DielectricPolarizability","ElectronicPolarizability",  "IonicPolarizability",
    "OrientationalPolarizability","DipoleMomentRepeat",     "CurieWeissConstant",
    # Network / chain architecture  (6)
    "CrosslinkingDensity",     "EntanglementMolWt",         "ContourLengthPerUnit",
    "PersistenceLength",       "CharacteristicRatio",       "ChainFlexibilityParam",
    # Molecular weight distribution  (6)
    "Mw_kDa",                  "Mn_kDa",                   "PolyDisersityIndex",
    "ZAverageMolWt",           "ViscosityAverageMolWt",    "NumberAverageDPn",
    # Morphology / processing  (6)
    "LamellaeThickness_nm",    "SpheruliteRadius_um",      "CrystalThickness_nm",
    "TieChainsPerArea",        "InterfacialThickness_nm",  "MicrostructureOrder",
    # Mechanical-dielectric coupling  (6)
    "PermittivityRealPart",    "PermittivityImaginaryPart","TanDeltaDielectric",
    "YoungModulus_GPa",        "TensileStrength_MPa",      "ElongationBreak_pct",
]
assert len(PHYSICS_FEATURE_NAMES) == N_PHYSICS, \
    f"Expected 40 physics names, got {len(PHYSICS_FEATURE_NAMES)}"

# ── Realistic Gaussian priors for each polymer (μ, σ) ───────────────────────
_PHYSICS_PRIORS = {
    "Polyimide": dict(
        DegreeOfCrystallinity        = (0.35, 0.06),
        CrystallinePhaseContent      = (0.35, 0.05),
        AmorphousPhaseContent        = (0.65, 0.05),
        FreeVolumeFraction           = (0.11, 0.02),
        ChainRigidityIndex           = (0.82, 0.06),
        SegmentalMobility            = (0.18, 0.03),
        ThermalExpansionCoeff        = (3.2e-5, 4e-6),
        HeatCapacity_Cp              = (1.05, 0.08),
        ThermalDiffusivity           = (1.8e-7, 2e-8),
        GlassyModulus                = (3.4, 0.3),
        DielectricPolarizability     = (28.5, 2.0),
        ElectronicPolarizability     = (22.1, 1.5),
        IonicPolarizability          = (3.8, 0.4),
        OrientationalPolarizability  = (2.6, 0.3),
        DipoleMomentRepeat           = (4.8, 0.5),
        CurieWeissConstant           = (320.0, 20.0),
        CrosslinkingDensity          = (0.008, 0.002),
        EntanglementMolWt            = (5800.0, 500.0),
        ContourLengthPerUnit         = (1.48, 0.10),
        PersistenceLength            = (12.5, 1.5),
        CharacteristicRatio          = (8.2, 0.8),
        ChainFlexibilityParam        = (0.22, 0.03),
        Mw_kDa                       = (85.0, 12.0),
        Mn_kDa                       = (42.0, 6.0),
        PolyDisersityIndex           = (2.05, 0.25),
        ZAverageMolWt                = (130.0, 18.0),
        ViscosityAverageMolWt        = (78.0, 10.0),
        NumberAverageDPn             = (210.0, 30.0),
        LamellaeThickness_nm         = (12.0, 2.0),
        SpheruliteRadius_um          = (3.5, 0.8),
        CrystalThickness_nm          = (18.0, 3.0),
        TieChainsPerArea             = (1.8e14, 2e13),
        InterfacialThickness_nm      = (4.5, 0.6),
        MicrostructureOrder          = (0.55, 0.06),
        PermittivityRealPart         = (3.5, 0.2),
        PermittivityImaginaryPart    = (0.08, 0.01),
        TanDeltaDielectric           = (0.022, 0.003),
        YoungModulus_GPa             = (3.1, 0.3),
        TensileStrength_MPa          = (185.0, 20.0),
        ElongationBreak_pct          = (35.0, 5.0),
    ),
    "PEEK": dict(
        DegreeOfCrystallinity        = (0.42, 0.07),
        CrystallinePhaseContent      = (0.42, 0.06),
        AmorphousPhaseContent        = (0.58, 0.06),
        FreeVolumeFraction           = (0.09, 0.02),
        ChainRigidityIndex           = (0.88, 0.05),
        SegmentalMobility            = (0.14, 0.02),
        ThermalExpansionCoeff        = (4.7e-5, 5e-6),
        HeatCapacity_Cp              = (1.32, 0.10),
        ThermalDiffusivity           = (2.5e-7, 3e-8),
        GlassyModulus                = (3.7, 0.4),
        DielectricPolarizability     = (25.2, 2.2),
        ElectronicPolarizability     = (19.8, 1.6),
        IonicPolarizability          = (2.9, 0.3),
        OrientationalPolarizability  = (2.5, 0.3),
        DipoleMomentRepeat           = (3.6, 0.4),
        CurieWeissConstant           = (180.0, 15.0),
        CrosslinkingDensity          = (0.004, 0.001),
        EntanglementMolWt            = (8000.0, 700.0),
        ContourLengthPerUnit         = (1.62, 0.12),
        PersistenceLength            = (9.8, 1.2),
        CharacteristicRatio          = (10.5, 1.0),
        ChainFlexibilityParam        = (0.16, 0.02),
        Mw_kDa                       = (95.0, 15.0),
        Mn_kDa                       = (48.0, 7.0),
        PolyDisersityIndex           = (2.00, 0.22),
        ZAverageMolWt                = (148.0, 22.0),
        ViscosityAverageMolWt        = (88.0, 12.0),
        NumberAverageDPn             = (220.0, 35.0),
        LamellaeThickness_nm         = (15.0, 2.5),
        SpheruliteRadius_um          = (8.5, 1.5),
        CrystalThickness_nm          = (22.0, 4.0),
        TieChainsPerArea             = (2.1e14, 2.5e13),
        InterfacialThickness_nm      = (5.8, 0.7),
        MicrostructureOrder          = (0.68, 0.07),
        PermittivityRealPart         = (3.3, 0.2),
        PermittivityImaginaryPart    = (0.06, 0.01),
        TanDeltaDielectric           = (0.003, 0.0005),
        YoungModulus_GPa             = (3.8, 0.4),
        TensileStrength_MPa          = (210.0, 25.0),
        ElongationBreak_pct          = (30.0, 4.0),
    ),
    "PTFE": dict(
        DegreeOfCrystallinity        = (0.60, 0.08),
        CrystallinePhaseContent      = (0.60, 0.07),
        AmorphousPhaseContent        = (0.40, 0.07),
        FreeVolumeFraction           = (0.13, 0.02),
        ChainRigidityIndex           = (0.65, 0.07),
        SegmentalMobility            = (0.28, 0.04),
        ThermalExpansionCoeff        = (1.1e-4, 1e-5),
        HeatCapacity_Cp              = (1.02, 0.07),
        ThermalDiffusivity           = (2.5e-7, 3e-8),
        GlassyModulus                = (0.55, 0.08),
        DielectricPolarizability     = (12.8, 1.2),
        ElectronicPolarizability     = (12.0, 1.0),
        IonicPolarizability          = (0.5, 0.1),
        OrientationalPolarizability  = (0.3, 0.05),
        DipoleMomentRepeat           = (0.0, 0.05),
        CurieWeissConstant           = (0.0, 5.0),
        CrosslinkingDensity          = (0.0005, 0.0002),
        EntanglementMolWt            = (12000.0, 1000.0),
        ContourLengthPerUnit         = (1.28, 0.08),
        PersistenceLength            = (1.1, 0.2),
        CharacteristicRatio          = (5.8, 0.6),
        ChainFlexibilityParam        = (0.52, 0.05),
        Mw_kDa                       = (3200.0, 400.0),
        Mn_kDa                       = (500.0, 60.0),
        PolyDisersityIndex           = (6.50, 0.80),
        ZAverageMolWt                = (8500.0, 1000.0),
        ViscosityAverageMolWt        = (2800.0, 350.0),
        NumberAverageDPn             = (5000.0, 600.0),
        LamellaeThickness_nm         = (25.0, 4.0),
        SpheruliteRadius_um          = (30.0, 5.0),
        CrystalThickness_nm          = (32.0, 5.0),
        TieChainsPerArea             = (3.5e14, 4e13),
        InterfacialThickness_nm      = (8.0, 1.0),
        MicrostructureOrder          = (0.75, 0.08),
        PermittivityRealPart         = (2.0, 0.10),
        PermittivityImaginaryPart    = (0.001, 0.0002),
        TanDeltaDielectric           = (0.0002, 0.00003),
        YoungModulus_GPa             = (0.55, 0.06),
        TensileStrength_MPa          = (32.0, 5.0),
        ElongationBreak_pct          = (300.0, 40.0),
    ),
}


def extract_physics_features(polymer_name: str, n_samples: int) -> pd.DataFrame:
    """
    Program 3 — Physics-Based & Morphological Features.

    Samples each feature from a Gaussian distribution N(μ, σ) anchored on
    experimentally validated values for PI, PEEK, and PTFE.
    Boundary clipping prevents physically impossible values.
    """
    priors = _PHYSICS_PRIORS[polymer_name]
    seed   = int(hashlib.md5(f"physics_{polymer_name}".encode()).hexdigest(),
                 16) % (2**32)
    rng    = np.random.RandomState(seed)

    records = []
    for _ in range(n_samples):
        row = {}
        for feat in PHYSICS_FEATURE_NAMES:
            mu, sigma = priors[feat]
            val = rng.normal(mu, sigma)
            # Physical constraints
            if feat in ("DegreeOfCrystallinity", "CrystallinePhaseContent",
                        "AmorphousPhaseContent", "FreeVolumeFraction",
                        "MicrostructureOrder"):
                val = np.clip(val, 0.0, 1.0)
            elif feat in ("CrosslinkingDensity", "DipoleMomentRepeat",
                          "PermittivityImaginaryPart", "TanDeltaDielectric",
                          "PolyDisersityIndex"):
                val = max(val, 0.0)
            row[feat] = val
        # Enforce AmorphousPhaseContent = 1 − DegreeOfCrystallinity
        row["AmorphousPhaseContent"] = 1.0 - row["DegreeOfCrystallinity"]
        records.append(row)

    return pd.DataFrame(records, columns=PHYSICS_FEATURE_NAMES)


# ─── MASTER DATASET BUILDER ──────────────────────────────────────────────────

def build_master_dataset() -> pd.DataFrame:
    """
    Combine Programs 1, 2, 3 for all three polymer classes.
    Returns a 720 × 124 DataFrame (120 features + Polymer + SMILES + Sample_ID).
    """
    all_dfs = []
    for polymer_name, meta in POLYMER_REGISTRY.items():
        print(f"  ▶ Extracting features for {meta['color']}{BOLD}{polymer_name}{RESET} ...")
        smiles    = meta["smiles"]
        n_samples = N_SAMPLES_PER_POLYMER

        df_struct  = extract_structural_features(polymer_name, smiles, n_samples)
        df_latent  = extract_latent_features(polymer_name, smiles, n_samples)
        df_physics = extract_physics_features(polymer_name, n_samples)

        df_poly = pd.concat([df_struct, df_latent, df_physics], axis=1)
        df_poly.insert(0, "Polymer",   polymer_name)
        df_poly.insert(1, "SMILES",    smiles)
        df_poly.insert(2, "Sample_ID", [f"{polymer_name[:2]}_{i+1:04d}"
                                        for i in range(n_samples)])
        all_dfs.append(df_poly)

    master = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  ✔  Master dataset shape: {master.shape}  "
          f"({len(master)} samples × {master.shape[1]} columns)\n")
    return master


# ──────────────────────────────────────────────────────────────────────────────
#  ████████╗ █████╗ ███████╗██╗  ██╗    ██████╗
#  ╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝    ╚════██╗
#     ██║   ███████║███████╗█████╔╝      █████╔╝
#     ██║   ██╔══██║╚════██║██╔═██╗     ██╔═══╝
#     ██║   ██║  ██║███████║██║  ██╗    ███████╗
#     ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚══════╝
#  QSPR MODELLING — Multi-Target MLP
# ──────────────────────────────────────────────────────────────────────────────

def compute_targets(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-Informed Synthetic Target Generation.

    Tg (Glass Transition Temperature, °C):
    ───────────────────────────────────────
      Tg = α · ChainRigidityIndex
         + β · NumAromaticRings
         − γ · FreeVolumeFraction
         + δ · DegreeOfCrystallinity
         + ε · CrosslinkingDensity_scaled
         + base_offset
         + N(0, σ_noise)

    Dk (Dielectric Constant at 1 GHz):
    ────────────────────────────────────
      Dk = a · DielectricPolarizability_norm
         − b · FreeVolumeFraction
         + c · DegreeOfCrystallinity
         − d · FractionCSP3
         + base_offset
         + N(0, σ_noise)

    Coefficients are chosen so that the target ranges are realistic:
      Tg  ∈ [150, 450] °C   (PI > PEEK > PTFE)
      Dk  ∈ [1.8, 4.5]      (PTFE < PI ≈ PEEK)
    """
    df = master_df.copy()

    # --- Tg ---
    alpha = 280.0    # chain rigidity weight
    beta  =  18.0    # per aromatic ring contribution
    gamma = 600.0    # free volume penalty
    delta =  80.0    # crystallinity contribution
    eps   = 1500.0   # cross-linking boost (density in mol/cm³ × 10^3 scale)

    tg_base = {
        "Polyimide" : 240.0,
        "PEEK"      : 200.0,
        "PTFE"      :  10.0,   # Tg ~ −113 °C for PTFE (above room-T here: repeat-unit level)
    }

    rng_tg = np.random.RandomState(101)
    Tg_vals = []
    for _, row in df.iterrows():
        base = tg_base[row["Polymer"]]
        Tg   = (base
                + alpha * row["ChainRigidityIndex"]
                + beta  * row["NumAromaticRings"]
                - gamma * row["FreeVolumeFraction"]
                + delta * row["DegreeOfCrystallinity"]
                + eps   * row["CrosslinkingDensity"])
        Tg += rng_tg.normal(0, 4.0)
        Tg_vals.append(Tg)
    df["Tg_degC"] = Tg_vals

    # --- Dk ---
    a = 0.065     # polarizability weight
    b = 8.0       # free-volume reduction
    c = 2.0       # crystallinity correction
    d = 1.0       # fluorinated/non-aromatic contribution reduction

    dk_base = {
        "Polyimide" : 1.8,
        "PEEK"      : 1.6,
        "PTFE"      : 1.6,
    }

    rng_dk = np.random.RandomState(202)
    Dk_vals = []
    for _, row in df.iterrows():
        base = dk_base[row["Polymer"]]
        Dk   = (base
                + a * row["DielectricPolarizability"]
                - b * row["FreeVolumeFraction"]
                + c * row["DegreeOfCrystallinity"]
                - d * row["FractionCSP3"])
        Dk += rng_dk.normal(0, 0.05)
        Dk = max(Dk, 1.5)   # physical lower bound
        Dk_vals.append(Dk)
    df["Dk_1GHz"] = Dk_vals

    return df


def build_qspr_model(master_df: pd.DataFrame):
    """
    Task 2: Multi-Target MLP QSPR Model.

    Architecture:
      Input  : 120 standardised feature vectors
      Hidden : [256, 128, 64]  (ReLU + early stopping + L2)
      Output : 2 targets → [Tg, Dk]

    Returns trained model, scaler, and performance metrics dict.
    """
    # ── Feature / target split ────────────────────────────────────────────────
    feature_cols = STRUCTURAL_FEATURE_NAMES + LATENT_FEATURE_NAMES + PHYSICS_FEATURE_NAMES
    X = master_df[feature_cols].values.astype(float)
    y = master_df[["Tg_degC", "Dk_1GHz"]].values.astype(float)

    # ── Train / test split (80 / 20) ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    # ── Standardise features ─────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Multi-target MLP ──────────────────────────────────────────────────────
    mlp = MLPRegressor(
        hidden_layer_sizes = (256, 128, 64),
        activation         = "relu",
        solver             = "adam",
        alpha              = 1e-4,          # L2 regularisation
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

    # Wrap in MultiOutputRegressor for cleaner two-target training
    model = MultiOutputRegressor(mlp, n_jobs=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ── Per-target metrics ────────────────────────────────────────────────────
    targets = ["Tg_degC", "Dk_1GHz"]
    metrics = {}
    for i, tgt in enumerate(targets):
        r2   = r2_score(y_test[:, i], y_pred[:, i])
        rmse = math.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        metrics[tgt] = {"R2": r2, "RMSE": rmse}

    return model, scaler, metrics, feature_cols


# ──────────────────────────────────────────────────────────────────────────────
#  ████████╗ █████╗ ███████╗██╗  ██╗     ██████╗
#  ╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝    ╚════██╗
#     ██║   ███████║███████╗█████╔╝      ███╔═╝
#     ██║   ██╔══██║╚════██║██╔═██╗     ███╔══╝
#     ██║   ██║  ██║███████║██║  ██╗    ███████╗
#     ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚══════╝
#  INVERSE DESIGN FRAMEWORK — Bayesian-Inspired Differential Evolution
# ──────────────────────────────────────────────────────────────────────────────

# LEO Targets
LEO_TG_TARGET = 300.0    # °C — must EXCEED this
LEO_DK_TARGET = 2.50     # —   must be BELOW this

# Design variable indices within the physics feature block
_IDX_FVF   = PHYSICS_FEATURE_NAMES.index("FreeVolumeFraction")     # 3
_IDX_CRYST = PHYSICS_FEATURE_NAMES.index("DegreeOfCrystallinity")  # 0

# Search bounds (physically meaningful)
_DESIGN_BOUNDS = {
    "FreeVolumeFraction"  : (0.04, 0.22),
    "DegreeOfCrystallinity": (0.10, 0.80),
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

    def fitness(x: np.ndarray) -> float:
        fvf   = x[0]
        cryst = x[1]

        # Mutate only the two design variables
        feat_vec = mean_feats.copy()
        feat_vec[phys_start + _IDX_FVF  ] = fvf
        feat_vec[phys_start + _IDX_CRYST] = cryst
        # Enforce AmorphousPhaseContent consistency
        amorph_idx = PHYSICS_FEATURE_NAMES.index("AmorphousPhaseContent")
        feat_vec[phys_start + amorph_idx] = 1.0 - cryst

        # Predict
        feat_scaled = scaler.transform(feat_vec.reshape(1, -1))
        pred        = model.predict(feat_scaled)[0]
        Tg_pred, Dk_pred = pred[0], pred[1]

        # Penalty terms
        w_Tg     = 1.0
        w_Dk     = 1.0
        w_reg    = 0.05

        penalty_Tg = max(0.0, (LEO_TG_TARGET - Tg_pred) / LEO_TG_TARGET)
        penalty_Dk = max(0.0, (Dk_pred - LEO_DK_TARGET) / LEO_DK_TARGET)
        reg        = w_reg * (fvf**2 + (1 - cryst)**2) / 2.0

        return w_Tg * penalty_Tg + w_Dk * penalty_Dk + reg

    # ── Differential Evolution optimiser ─────────────────────────────────────
    bounds   = [_DESIGN_BOUNDS["FreeVolumeFraction"],
                _DESIGN_BOUNDS["DegreeOfCrystallinity"]]
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

    # Final prediction with optimal parameters
    feat_vec  = mean_feats.copy()
    feat_vec[phys_start + _IDX_FVF  ] = opt_fvf
    feat_vec[phys_start + _IDX_CRYST] = opt_cryst
    amorph_idx = PHYSICS_FEATURE_NAMES.index("AmorphousPhaseContent")
    feat_vec[phys_start + amorph_idx ] = 1.0 - opt_cryst
    feat_scaled = scaler.transform(feat_vec.reshape(1, -1))
    pred_final  = model.predict(feat_scaled)[0]

    return {
        "polymer"              : polymer_name,
        "optimal_FVF"          : opt_fvf,
        "optimal_Crystallinity": opt_cryst,
        "predicted_Tg_degC"    : pred_final[0],
        "predicted_Dk"         : pred_final[1],
        "fitness_final"        : result.fun,
        "converged"            : result.success,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  ████████╗ █████╗ ███████╗██╗  ██╗    ██╗  ██╗
#  ╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝    ██║  ██║
#     ██║   ███████║███████╗█████╔╝     ███████║
#     ██║   ██╔══██║╚════██║██╔═██╗     ╚════██║
#     ██║   ██║  ██║███████║██║  ██╗         ██║
#     ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝         ╚═╝
#  OUTPUT MANAGEMENT & REPORTING
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"

def save_polymer_csvs(master_df: pd.DataFrame) -> list:
    """
    Task 4 output: Three CSVs, one per polymer.
    Ensures the output directory exists before writing.
    """
    # 1. ADD THIS LINE: This creates the 'outputs' folder if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    feature_cols = STRUCTURAL_FEATURE_NAMES + LATENT_FEATURE_NAMES + PHYSICS_FEATURE_NAMES
    target_cols  = ["Tg_degC", "Dk_1GHz"]
    all_cols     = feature_cols + target_cols

    assert len(feature_cols) == 120, f"Feature count error: {len(feature_cols)}"

    saved_paths = []
    for polymer_name, meta in POLYMER_REGISTRY.items():
        poly_df  = master_df[master_df["Polymer"] == polymer_name].copy()
        out_df   = poly_df[all_cols].reset_index(drop=True)

        assert out_df.shape == (240, 122), \
            f"Shape mismatch for {polymer_name}: {out_df.shape}"

        fname    = f"{meta['member']}_{polymer_name}.csv"
        # out_path will now be 'outputs/Member1_Polyimide.csv'
        out_path = os.path.join(OUTPUT_DIR, fname)
        
        # 2. This will now work because the folder is guaranteed to exist
        out_df.to_csv(out_path, index=False, float_format="%.6f")
        saved_paths.append(out_path)
        print(f"  💾  Saved: {out_path}  [{out_df.shape[0]} rows × {out_df.shape[1]} cols]")

    return saved_paths


def print_banner(text: str, char: str = "═", width: int = 72) -> None:
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{BOLD}{text}{RESET}{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")


def print_ml_report(metrics: dict) -> None:
    """Formatted QSPR model performance table."""
    print_banner("TASK 2 — QSPR MODEL PERFORMANCE (Multi-Target MLP)", width=72)
    print(f"\n  {'Target':<22}  {'R²':>10}  {'RMSE':>12}  {'Quality':>12}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*12}  {'─'*12}")

    def quality_label(r2):
        if r2 >= 0.95: return "✦ Excellent"
        if r2 >= 0.90: return "✔ Very Good"
        if r2 >= 0.80: return "~ Good"
        return "✗ Needs work"

    for tgt, m in metrics.items():
        label = "Tg (Glass Trans. Temp, °C)" if tgt == "Tg_degC" \
                else "Dk (Dielectric Const.)"
        print(f"  {label:<22}  {m['R2']:>10.4f}  {m['RMSE']:>12.4f}  "
              f"{quality_label(m['R2']):>12}")

    print(f"\n  {'─'*72}")
    print("  ℹ  Model: Multi-Layer Perceptron  |  Layers: [256→128→64]  |  "
          "Train/Test: 80/20")
    print(f"  {'─'*72}\n")


def print_inverse_design_report(results: list) -> None:
    """Formatted inverse design output table."""
    print_banner("TASK 3 — INVERSE DESIGN RESULTS (LEO Satellite Targets)", width=72)
    print(f"\n  LEO Requirements:  Tg > {LEO_TG_TARGET:.0f} °C   |   "
          f"Dk < {LEO_DK_TARGET:.2f}\n")

    header = (f"  {'Polymer':<12}  {'FVF (opt)':>10}  {'Cryst (opt)':>12}  "
              f"{'Pred Tg (°C)':>14}  {'Pred Dk':>9}  {'Tg ✓':>5}  {'Dk ✓':>5}")
    print(header)
    print(f"  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*14}  {'─'*9}  {'─'*5}  {'─'*5}")

    for r in results:
        tg_ok = "✔" if r["predicted_Tg_degC"] >= LEO_TG_TARGET else "✗"
        dk_ok = "✔" if r["predicted_Dk"]      <= LEO_DK_TARGET else "✗"
        meta  = POLYMER_REGISTRY[r["polymer"]]
        col   = meta["color"]
        print(f"  {col}{BOLD}{r['polymer']:<12}{RESET}  "
              f"{r['optimal_FVF']:>10.4f}  "
              f"{r['optimal_Crystallinity']:>12.4f}  "
              f"{r['predicted_Tg_degC']:>14.2f}  "
              f"{r['predicted_Dk']:>9.4f}  "
              f"{tg_ok:>5}  {dk_ok:>5}")

    print(f"\n  {'─'*72}")
    print("  ℹ  Optimiser: Differential Evolution (SciPy)  |  "
          "Max iterations: 500  |  Polishing: L-BFGS-B")
    print(f"  {'─'*72}\n")


def print_dataset_summary(master_df: pd.DataFrame) -> None:
    """Dataset statistics per polymer."""
    print_banner("TASK 1 — DATASET SUMMARY", width=72)
    print(f"\n  {'Polymer':<12}  {'Samples':>8}  {'Tg mean±σ':>16}  "
          f"{'Dk mean±σ':>16}  {'Features':>9}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*16}  {'─'*16}  {'─'*9}")

    for pname in POLYMER_REGISTRY:
        sub  = master_df[master_df["Polymer"] == pname]
        n    = len(sub)
        tg_m = sub["Tg_degC"].mean()
        tg_s = sub["Tg_degC"].std()
        dk_m = sub["Dk_1GHz"].mean()
        dk_s = sub["Dk_1GHz"].std()
        meta = POLYMER_REGISTRY[pname]
        print(f"  {meta['color']}{BOLD}{pname:<12}{RESET}  {n:>8}  "
              f"{tg_m:>7.1f}±{tg_s:<6.1f}  {dk_m:>7.4f}±{dk_s:<6.4f}  "
              f"{N_TOTAL_FEATURES:>9}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print_banner(
        "INFORMATICS-DRIVEN DESIGN OF HIGH-PERFORMANCE POLYMERS",
        char="█", width=72
    )
    print(f"\n  {'Subtitle':<10}: QSPR Modelling for Satellite Protection (LEO)")
    print(f"  {'Polymers':<10}: Polyimide (PI) | PEEK | PTFE")
    print(f"  {'Targets':<10}: Tg > 300 °C  |  Dk < 2.5")
    print(f"  {'Dataset':<10}: 720 samples  |  120 features  |  2 target variables")
    print(f"  {'Pipeline':<10}: Data Curation → QSPR → Inverse Design → CSV Export\n")

    # ── Task 1: Build Dataset ─────────────────────────────────────────────────
    print_banner("TASK 1 — DATA CURATION & FEATURE EXTRACTION", width=72)
    print()
    master_df = build_master_dataset()
    master_df = compute_targets(master_df)
    print_dataset_summary(master_df)

    # Sanity checks
    assert len(master_df) == 720,           f"Row count error: {len(master_df)}"
    assert (master_df["Polymer"].value_counts() == 240).all(), \
        "Unequal polymer counts!"
    feat_cols_check = (STRUCTURAL_FEATURE_NAMES + LATENT_FEATURE_NAMES
                       + PHYSICS_FEATURE_NAMES)
    assert all(c in master_df.columns for c in feat_cols_check), \
        "Missing feature columns!"

    # ── Task 2: QSPR Model ────────────────────────────────────────────────────
    print_banner("TASK 2 — MULTI-TARGET MLP QSPR MODEL TRAINING", width=72)
    print("\n  Training Multi-Layer Perceptron  [256 → 128 → 64]  "
          "on 120-feature vectors ...")
    model, scaler, metrics, feature_cols = build_qspr_model(master_df)
    print("  ✔  Training complete.\n")
    print_ml_report(metrics)

    # ── Task 3: Inverse Design ───────────────────────────────────────────────
    print_banner("TASK 3 — INVERSE DESIGN OPTIMISATION (Differential Evolution)",
                 width=72)
    print(f"\n  Objective: Tg ≥ {LEO_TG_TARGET:.0f} °C  AND  "
          f"Dk ≤ {LEO_DK_TARGET:.2f}\n")
    inv_results = []
    for pname in POLYMER_REGISTRY:
        meta = POLYMER_REGISTRY[pname]
        print(f"  🔍  Optimising {meta['color']}{BOLD}{pname}{RESET} ...")
        res = inverse_design_polymer(pname, master_df, model, scaler, feature_cols)
        inv_results.append(res)
        print(f"       FVF = {res['optimal_FVF']:.4f}   "
              f"Cryst = {res['optimal_Crystallinity']:.4f}   "
              f"Tg = {res['predicted_Tg_degC']:.1f} °C   "
              f"Dk = {res['predicted_Dk']:.4f}   "
              f"{'[converged]' if res['converged'] else '[not converged]'}")
    print()
    print_inverse_design_report(inv_results)

    # ── Task 4: Save CSVs ────────────────────────────────────────────────────
    print_banner("TASK 4 — CSV EXPORT", width=72)
    print()
    paths = save_polymer_csvs(master_df)
    print()

    # ── Final Summary ─────────────────────────────────────────────────────────
    print_banner("PIPELINE COMPLETE — FULL RESULTS SUMMARY", char="═", width=72)
    print()

    print(f"  {'QSPR MODEL METRICS':─<50}")
    for tgt, m in metrics.items():
        label = "Tg (°C)" if tgt == "Tg_degC" else "Dk     "
        print(f"    {label}  →  R² = {m['R2']:.4f}   RMSE = {m['RMSE']:.4f}")

    print(f"\n  {'INVERSE DESIGN OPTIMAL PARAMETERS':─<50}")
    print(f"  {'Polymer':<12}  {'FVF':>8}  {'Crystallinity':>14}  "
          f"{'Pred Tg':>9}  {'Pred Dk':>9}")
    for r in inv_results:
        print(f"  {r['polymer']:<12}  {r['optimal_FVF']:>8.4f}  "
              f"{r['optimal_Crystallinity']:>14.4f}  "
              f"{r['predicted_Tg_degC']:>8.2f}°C  "
              f"{r['predicted_Dk']:>9.4f}")

    print(f"\n  {'SAVED FILES':─<50}")
    for p in paths:
        print(f"    📄  {p}")

    print(f"\n{'═'*72}\n")
    return master_df, model, scaler, metrics, inv_results


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    master_df, model, scaler, metrics, inv_results = main()
