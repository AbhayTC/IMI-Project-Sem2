"""
================================================================================
  Informatics-Driven Design of High-Performance Polymers for Satellite Protection
  A Comprehensive QSPR Pipeline for Thermal Endurance & Dielectric Stability
================================================================================
  TASK 1 ▶ Data Curation

  CHANGES (this version):
    1. GENERIC POLYMER SUPPORT
       CLI prompt at startup. User enters any SMILES string; the pipeline
       auto-names it (Poly_<sha256[:8]>), estimates all 40 physics priors
       from RDKit descriptors as proxies, and generates 240 derivative
       samples alongside PI/PEEK/PTFE. PI/PEEK/PTFE anchor the model;
       the user polymer is inference-only in Tasks 2/2b.

    2. PHYSICS PRIOR ESTIMATION
       estimate_physics_priors_from_smiles() maps RDKit descriptors to
       physics features via documented proxy rules. Key mappings:
         FractionCSP3      → crystallinity (higher sp3 → more crystalline)
         AromaticRingDensity → chain rigidity
         MolLogP (inv)     → free volume fraction
         HBD+HBA count     → dielectric polarizability
         TPSA/MolWt        → dipole moment repeat
       Estimated features printed with ⚠ ESTIMATED tag so users know
       which values are inferred rather than measured.

    3. Tg / Dk BASELINE CHOICE
       User chooses: (1) enter own Tg and Dk, or (2) RDKit heuristic
       (Van Krevelen-inspired group-contribution estimate).

    4. DUMMY SAMPLES
       5 rows with physically impossible feature values injected at the
       bottom of master_dataset. Flagged is_dummy=True. Task 2/2b trains
       only on is_reference=True rows, then runs dummies through the
       trained model to verify it produces out-of-range predictions,
       confirming it isn't silently clamping bad inputs.

    5. COLUMN FLAGS
       is_reference : True for PI / PEEK / PTFE rows (used for training)
       is_dummy     : True for the 5 sanity-check rows
       Both are False for the user polymer rows (inference only).

  Inputs : None (generates synthetic data from SMILES + priors)
  Outputs: master_dataset.pkl   (loaded by all downstream tasks)
           user_polymer.pkl      (loaded by tasks 2, 3, 4, 7, 8)
================================================================================
"""

import warnings, hashlib, os, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

np.random.seed(42)

# ── Optional RDKit ────────────────────────────────────────────────────────────
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import (AllChem, Descriptors, rdMolDescriptors,
                             rdChemReactions, DataStructs)
    from rdkit.Chem.rdMolDescriptors import CalcTPSA
    RDLogger.DisableLog('rdApp.warning')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("  !! RDKit not installed — using mock descriptors.\n"
          "     Install via: pip install rdkit")

# ──────────────────────────────────────────────────────────────────────────────
#  SHARED CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
N_STRUCTURAL         = 40
N_LATENT             = 40
N_PHYSICS            = 40
N_SAMPLES_PER_POLYMER = 240

RESET  = "\033[0m"
BOLD   = "\033[1m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"

# Reference polymers — these anchor the model
POLYMER_REGISTRY = {
    "Polyimide": {
        "smiles": "O=C1OC(=O)c2ccc1cc2-c1ccc(Oc2ccc(cc2)N2C(=O)c3ccc(cc3)C2=O)cc1",
        "color" : "\033[94m",
        "member": "Member1",
    },
    "PEEK": {
        "smiles": "O=C(c1ccc(Oc2ccc(cc2)Oc2ccc(cc2)C(=O)c2ccccc2)cc1)c1ccccc1",
        "color" : "\033[92m",
        "member": "Member2",
    },
    "PTFE": {
        "smiles": "FC(F)(F)C(F)(F)F",
        "color" : "\033[93m",
        "member": "Member3",
    },
    # ── Low-Dk / low-outgassing expansion ────────────────────────────────────
    "CyanateEster": {
        # Bisphenol-E cyanate ester monomer repeat unit (difunctional)
        "smiles": "N#COc1ccc(C(C)(CC)c2ccc(OC#N)cc2)cc1",
        "color" : "\033[35m",   # magenta
        "member": "Member4",
    },
    "FluorinatedPolyimide": {
        # 6FDA-based fluorinated polyimide repeat unit
        "smiles": "O=C1OC(=O)c2cc(C(F)(F)F)ccc21",
        "color" : "\033[36m",   # cyan
        "member": "Member5",
    },
    "Polybenzoxazole": {
        # PBO (poly-p-phenylene benzobisoxazole) repeat unit
        "smiles": "c1ccc2oc(-c3ccc4oc(-c5ccccc5)nc4c3)nc2c1",
        "color" : "\033[33m",   # dark yellow / brown
        "member": "Member6",
    },
}

# Colors reserved by reference polymers: blue, green, yellow
_USER_COLOR_PALETTE = ["\033[95m", "\033[96m", "\033[91m", "\033[97m"]


# ──────────────────────────────────────────────────────────────────────────────
#  AUTO-NAMING & COLOR
# ──────────────────────────────────────────────────────────────────────────────

def auto_name_from_smiles(smiles: str) -> str:
    """Generate a reproducible short name from SMILES via SHA-256."""
    h = hashlib.sha256(smiles.encode()).hexdigest()[:8]
    return f"Poly_{h}"


def auto_color(index: int = 0) -> str:
    return _USER_COLOR_PALETTE[index % len(_USER_COLOR_PALETTE)]


# ──────────────────────────────────────────────────────────────────────────────
#  Tg / Dk ESTIMATION FROM SMILES
# ──────────────────────────────────────────────────────────────────────────────

def estimate_tg_from_smiles(smiles: str) -> float:
    """
    Van Krevelen-inspired Tg heuristic from SMILES.
    Returns estimated glass transition temperature (°C).
    """
    if not RDKIT_AVAILABLE:
        # Token-based fallback
        arom = smiles.count('c') // 6
        return float(np.clip(80 + 50 * arom, -100, 420))

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 100.0

    frac_csp3  = rdMolDescriptors.CalcFractionCSP3(mol)
    n_arom     = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_hbd      = rdMolDescriptors.CalcNumHBD(mol)
    n_rot      = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_heavy    = mol.GetNumHeavyAtoms()
    rot_frac   = n_rot / max(n_heavy, 1)

    tg = (-50.0
          + 150.0 * (1.0 - frac_csp3)   # chain stiffness from sp2 fraction
          + 20.0  * n_arom               # aromatic ring stiffening
          + 15.0  * n_hbd                # H-bond stiffening
          - 80.0  * rot_frac)            # flexibility penalty
    return float(np.clip(tg, -150.0, 450.0))


def estimate_dk_from_smiles(smiles: str) -> float:
    """
    Estimate dielectric constant from molecular polarity indicators.
    Range: 1.5 (fluoropolymer) — 6.0 (highly polar).
    """
    if not RDKIT_AVAILABLE:
        return 3.0

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 3.0

    tpsa       = CalcTPSA(mol)
    mol_wt     = Descriptors.MolWt(mol)
    frac_csp3  = rdMolDescriptors.CalcFractionCSP3(mol)
    n_hba      = rdMolDescriptors.CalcNumHBA(mol)

    dk = (2.0
          + 0.025 * (tpsa / max(mol_wt, 1)) * 100
          + 0.3   * frac_csp3
          + 0.08  * n_hba)
    return float(np.clip(dk, 1.5, 6.0))


# ──────────────────────────────────────────────────────────────────────────────
#  PHYSICS PRIOR ESTIMATION
# ──────────────────────────────────────────────────────────────────────────────

PHYSICS_FEATURE_NAMES = [
    "DegreeOfCrystallinity",    "CrystallinePhaseContent",  "AmorphousPhaseContent",
    "FreeVolumeFraction",       "ChainRigidityIndex",        "SegmentalMobility",
    "ThermalExpansionCoeff",    "HeatCapacity_Cp",           "ThermalDiffusivity",
    "GlassyModulus",            "DielectricPolarizability",  "ElectronicPolarizability",
    "IonicPolarizability",      "OrientationalPolarizability","DipoleMomentRepeat",
    "CurieWeissConstant",       "CrosslinkingDensity",       "EntanglementMolWt",
    "ContourLengthPerUnit",     "PersistenceLength",         "CharacteristicRatio",
    "ChainFlexibilityParam",    "Mw_kDa",                   "Mn_kDa",
    "PolyDisersityIndex",       "ZAverageMolWt",             "ViscosityAverageMolWt",
    "NumberAverageDPn",         "LamellaeThickness_nm",      "SpheruliteRadius_um",
    "CrystalThickness_nm",      "TieChainsPerArea",          "InterfacialThickness_nm",
    "MicrostructureOrder",      "PermittivityRealPart",      "PermittivityImaginaryPart",
    "TanDeltaDielectric",       "YoungModulus_GPa",          "TensileStrength_MPa",
    "ElongationBreak_pct",
]
assert len(PHYSICS_FEATURE_NAMES) == N_PHYSICS

_PHYSICS_PRIORS = {
    "Polyimide": dict(
        DegreeOfCrystallinity=(0.35,0.06), CrystallinePhaseContent=(0.35,0.05),
        AmorphousPhaseContent=(0.65,0.05), FreeVolumeFraction=(0.11,0.02),
        ChainRigidityIndex=(0.82,0.06),    SegmentalMobility=(0.18,0.03),
        ThermalExpansionCoeff=(3.2e-5,4e-6), HeatCapacity_Cp=(1.05,0.08),
        ThermalDiffusivity=(1.8e-7,2e-8),  GlassyModulus=(3.4,0.3),
        DielectricPolarizability=(28.5,2.0), ElectronicPolarizability=(22.1,1.5),
        IonicPolarizability=(3.8,0.4),     OrientationalPolarizability=(2.6,0.3),
        DipoleMomentRepeat=(4.8,0.5),      CurieWeissConstant=(320.0,20.0),
        CrosslinkingDensity=(0.008,0.002), EntanglementMolWt=(5800.0,500.0),
        ContourLengthPerUnit=(1.48,0.10),  PersistenceLength=(12.5,1.5),
        CharacteristicRatio=(8.2,0.8),     ChainFlexibilityParam=(0.22,0.03),
        Mw_kDa=(85.0,12.0),               Mn_kDa=(42.0,6.0),
        PolyDisersityIndex=(2.05,0.25),    ZAverageMolWt=(130.0,18.0),
        ViscosityAverageMolWt=(78.0,10.0), NumberAverageDPn=(210.0,30.0),
        LamellaeThickness_nm=(12.0,2.0),   SpheruliteRadius_um=(3.5,0.8),
        CrystalThickness_nm=(18.0,3.0),    TieChainsPerArea=(1.8e14,2e13),
        InterfacialThickness_nm=(4.5,0.6), MicrostructureOrder=(0.55,0.06),
        PermittivityRealPart=(3.5,0.2),    PermittivityImaginaryPart=(0.08,0.01),
        TanDeltaDielectric=(0.022,0.003),  YoungModulus_GPa=(3.1,0.3),
        TensileStrength_MPa=(185.0,20.0),  ElongationBreak_pct=(35.0,5.0),
    ),
    "PEEK": dict(
        DegreeOfCrystallinity=(0.42,0.07), CrystallinePhaseContent=(0.42,0.06),
        AmorphousPhaseContent=(0.58,0.06), FreeVolumeFraction=(0.09,0.02),
        ChainRigidityIndex=(0.88,0.05),    SegmentalMobility=(0.14,0.02),
        ThermalExpansionCoeff=(4.7e-5,5e-6), HeatCapacity_Cp=(1.32,0.10),
        ThermalDiffusivity=(2.5e-7,3e-8),  GlassyModulus=(3.7,0.4),
        DielectricPolarizability=(25.2,2.2), ElectronicPolarizability=(19.8,1.6),
        IonicPolarizability=(2.9,0.3),     OrientationalPolarizability=(2.5,0.3),
        DipoleMomentRepeat=(3.6,0.4),      CurieWeissConstant=(180.0,15.0),
        CrosslinkingDensity=(0.004,0.001), EntanglementMolWt=(8000.0,700.0),
        ContourLengthPerUnit=(1.62,0.12),  PersistenceLength=(9.8,1.2),
        CharacteristicRatio=(10.5,1.0),    ChainFlexibilityParam=(0.16,0.02),
        Mw_kDa=(95.0,15.0),               Mn_kDa=(48.0,7.0),
        PolyDisersityIndex=(2.00,0.22),    ZAverageMolWt=(148.0,22.0),
        ViscosityAverageMolWt=(88.0,12.0), NumberAverageDPn=(220.0,35.0),
        LamellaeThickness_nm=(15.0,2.5),   SpheruliteRadius_um=(8.5,1.5),
        CrystalThickness_nm=(22.0,4.0),    TieChainsPerArea=(2.1e14,2.5e13),
        InterfacialThickness_nm=(5.8,0.7), MicrostructureOrder=(0.68,0.07),
        PermittivityRealPart=(3.3,0.2),    PermittivityImaginaryPart=(0.06,0.01),
        TanDeltaDielectric=(0.003,0.0005), YoungModulus_GPa=(3.8,0.4),
        TensileStrength_MPa=(210.0,25.0),  ElongationBreak_pct=(30.0,4.0),
    ),
    "PTFE": dict(
        DegreeOfCrystallinity=(0.60,0.08), CrystallinePhaseContent=(0.60,0.07),
        AmorphousPhaseContent=(0.40,0.07), FreeVolumeFraction=(0.13,0.02),
        ChainRigidityIndex=(0.65,0.07),    SegmentalMobility=(0.28,0.04),
        ThermalExpansionCoeff=(1.1e-4,1e-5), HeatCapacity_Cp=(1.02,0.07),
        ThermalDiffusivity=(2.5e-7,3e-8),  GlassyModulus=(0.55,0.08),
        DielectricPolarizability=(12.8,1.2), ElectronicPolarizability=(12.0,1.0),
        IonicPolarizability=(0.5,0.1),     OrientationalPolarizability=(0.3,0.05),
        DipoleMomentRepeat=(0.0,0.05),     CurieWeissConstant=(0.0,5.0),
        CrosslinkingDensity=(0.0005,0.0002), EntanglementMolWt=(12000.0,1000.0),
        ContourLengthPerUnit=(1.28,0.08),  PersistenceLength=(1.1,0.2),
        CharacteristicRatio=(5.8,0.6),     ChainFlexibilityParam=(0.52,0.05),
        Mw_kDa=(3200.0,400.0),            Mn_kDa=(500.0,60.0),
        PolyDisersityIndex=(6.50,0.80),    ZAverageMolWt=(8500.0,1000.0),
        ViscosityAverageMolWt=(2800.0,350.0), NumberAverageDPn=(5000.0,600.0),
        LamellaeThickness_nm=(25.0,4.0),   SpheruliteRadius_um=(30.0,5.0),
        CrystalThickness_nm=(32.0,5.0),    TieChainsPerArea=(3.5e14,4e13),
        InterfacialThickness_nm=(8.0,1.0), MicrostructureOrder=(0.75,0.08),
        PermittivityRealPart=(2.0,0.10),   PermittivityImaginaryPart=(0.001,0.0002),
        TanDeltaDielectric=(0.0002,0.00003), YoungModulus_GPa=(0.55,0.06),
        TensileStrength_MPa=(32.0,5.0),    ElongationBreak_pct=(300.0,40.0),
    ),
    # ── Low-Dk / low-outgassing polymer families ──────────────────────────────
    #
    # CyanateEster: triazine network former; Dk ~2.7, near-zero outgassing,
    #   high crosslink density, moderate crystallinity, rigid aromatic backbone.
    "CyanateEster": dict(
        DegreeOfCrystallinity=(0.25,0.05), CrystallinePhaseContent=(0.25,0.04),
        AmorphousPhaseContent=(0.75,0.04), FreeVolumeFraction=(0.14,0.02),
        ChainRigidityIndex=(0.80,0.06),    SegmentalMobility=(0.16,0.03),
        ThermalExpansionCoeff=(5.5e-5,6e-6), HeatCapacity_Cp=(1.15,0.09),
        ThermalDiffusivity=(1.9e-7,2e-8),  GlassyModulus=(3.2,0.3),
        DielectricPolarizability=(18.5,1.8), ElectronicPolarizability=(15.2,1.4),
        IonicPolarizability=(1.8,0.3),     OrientationalPolarizability=(1.5,0.2),
        DipoleMomentRepeat=(2.2,0.4),      CurieWeissConstant=(120.0,15.0),
        CrosslinkingDensity=(0.045,0.008), EntanglementMolWt=(2800.0,400.0),
        ContourLengthPerUnit=(1.38,0.10),  PersistenceLength=(14.0,2.0),
        CharacteristicRatio=(7.5,0.8),     ChainFlexibilityParam=(0.18,0.03),
        Mw_kDa=(22.0,4.0),                Mn_kDa=(11.0,2.0),
        PolyDisersityIndex=(2.10,0.30),    ZAverageMolWt=(38.0,6.0),
        ViscosityAverageMolWt=(19.0,3.5),  NumberAverageDPn=(55.0,10.0),
        LamellaeThickness_nm=(8.0,1.5),    SpheruliteRadius_um=(2.0,0.5),
        CrystalThickness_nm=(12.0,2.0),    TieChainsPerArea=(2.5e14,3e13),
        InterfacialThickness_nm=(3.5,0.5), MicrostructureOrder=(0.50,0.06),
        PermittivityRealPart=(2.7,0.15),   PermittivityImaginaryPart=(0.012,0.002),
        TanDeltaDielectric=(0.005,0.001),  YoungModulus_GPa=(3.5,0.3),
        TensileStrength_MPa=(70.0,10.0),   ElongationBreak_pct=(2.5,0.5),
    ),
    #
    # FluorinatedPolyimide: 6FDA-based; Dk ~2.5–2.8, –CF3 groups lower
    #   polarizability and moisture uptake; retains PI thermal stability.
    "FluorinatedPolyimide": dict(
        DegreeOfCrystallinity=(0.18,0.04), CrystallinePhaseContent=(0.18,0.04),
        AmorphousPhaseContent=(0.82,0.04), FreeVolumeFraction=(0.17,0.02),
        ChainRigidityIndex=(0.78,0.06),    SegmentalMobility=(0.20,0.03),
        ThermalExpansionCoeff=(4.2e-5,5e-6), HeatCapacity_Cp=(1.10,0.08),
        ThermalDiffusivity=(1.7e-7,2e-8),  GlassyModulus=(2.8,0.3),
        DielectricPolarizability=(16.0,1.6), ElectronicPolarizability=(13.5,1.3),
        IonicPolarizability=(1.2,0.2),     OrientationalPolarizability=(1.3,0.2),
        DipoleMomentRepeat=(1.8,0.3),      CurieWeissConstant=(80.0,12.0),
        CrosslinkingDensity=(0.006,0.002), EntanglementMolWt=(6200.0,600.0),
        ContourLengthPerUnit=(1.52,0.10),  PersistenceLength=(13.0,1.8),
        CharacteristicRatio=(7.8,0.8),     ChainFlexibilityParam=(0.20,0.03),
        Mw_kDa=(90.0,14.0),               Mn_kDa=(44.0,7.0),
        PolyDisersityIndex=(2.08,0.28),    ZAverageMolWt=(138.0,20.0),
        ViscosityAverageMolWt=(82.0,11.0), NumberAverageDPn=(200.0,30.0),
        LamellaeThickness_nm=(9.0,1.5),    SpheruliteRadius_um=(2.5,0.6),
        CrystalThickness_nm=(14.0,2.5),    TieChainsPerArea=(1.6e14,2e13),
        InterfacialThickness_nm=(4.0,0.5), MicrostructureOrder=(0.48,0.06),
        PermittivityRealPart=(2.65,0.12),  PermittivityImaginaryPart=(0.010,0.002),
        TanDeltaDielectric=(0.004,0.001),  YoungModulus_GPa=(2.6,0.3),
        TensileStrength_MPa=(155.0,18.0),  ElongationBreak_pct=(20.0,4.0),
    ),
    #
    # Polybenzoxazole: rigid-rod aromatic heterocycle; exceptional thermal
    #   stability (Td >500 °C), low dielectric loss, very low outgassing.
    "Polybenzoxazole": dict(
        DegreeOfCrystallinity=(0.30,0.06), CrystallinePhaseContent=(0.30,0.05),
        AmorphousPhaseContent=(0.70,0.05), FreeVolumeFraction=(0.10,0.02),
        ChainRigidityIndex=(0.95,0.03),    SegmentalMobility=(0.06,0.02),
        ThermalExpansionCoeff=(1.5e-5,2e-6), HeatCapacity_Cp=(1.08,0.08),
        ThermalDiffusivity=(2.2e-7,3e-8),  GlassyModulus=(4.5,0.4),
        DielectricPolarizability=(22.0,2.0), ElectronicPolarizability=(18.5,1.6),
        IonicPolarizability=(2.0,0.3),     OrientationalPolarizability=(1.5,0.2),
        DipoleMomentRepeat=(2.8,0.4),      CurieWeissConstant=(200.0,20.0),
        CrosslinkingDensity=(0.003,0.001), EntanglementMolWt=(9500.0,900.0),
        ContourLengthPerUnit=(1.72,0.12),  PersistenceLength=(45.0,6.0),
        CharacteristicRatio=(18.0,2.0),    ChainFlexibilityParam=(0.08,0.02),
        Mw_kDa=(110.0,18.0),              Mn_kDa=(54.0,8.0),
        PolyDisersityIndex=(2.05,0.25),    ZAverageMolWt=(170.0,25.0),
        ViscosityAverageMolWt=(100.0,14.0), NumberAverageDPn=(260.0,38.0),
        LamellaeThickness_nm=(18.0,3.0),   SpheruliteRadius_um=(5.0,1.0),
        CrystalThickness_nm=(25.0,4.0),    TieChainsPerArea=(2.8e14,3e13),
        InterfacialThickness_nm=(5.0,0.7), MicrostructureOrder=(0.72,0.07),
        PermittivityRealPart=(3.1,0.15),   PermittivityImaginaryPart=(0.018,0.003),
        TanDeltaDielectric=(0.006,0.001),  YoungModulus_GPa=(4.8,0.4),
        TensileStrength_MPa=(280.0,30.0),  ElongationBreak_pct=(4.0,1.0),
    ),
}


def _generic_physics_priors() -> dict:
    """
    Fallback priors: average of all reference polymers with 1.5× wider uncertainty.
    Used as the base for unknown polymers before RDKit overrides are applied.
    """
    priors = {}
    for feat in PHYSICS_FEATURE_NAMES:
        mus    = [_PHYSICS_PRIORS[p][feat][0] for p in ("Polyimide", "PEEK", "PTFE",
                                                         "CyanateEster", "FluorinatedPolyimide",
                                                         "Polybenzoxazole")]
        sigmas = [_PHYSICS_PRIORS[p][feat][1] for p in ("Polyimide", "PEEK", "PTFE",
                                                          "CyanateEster", "FluorinatedPolyimide",
                                                          "Polybenzoxazole")]
        priors[feat] = (float(np.mean(mus)), float(np.mean(sigmas) * 1.5))
    return priors


def estimate_physics_priors_from_smiles(smiles: str) -> dict:
    """
    Estimate all 40 physics priors from SMILES using RDKit descriptor proxies.

    Proxy mapping (key features):
      FractionCSP3          → DegreeOfCrystallinity (higher sp3 = more crystalline)
      AromaticRingDensity   → ChainRigidityIndex    (aromatic backbone = rigid)
      MolLogP (inverted)    → FreeVolumeFraction     (lower logP = more polar/dense)
      HBD + HBA             → DielectricPolarizability (H-bond groups dominate)
      TPSA / MolWt          → DipoleMomentRepeat     (polar surface ~ dipole)
      FractionCSP3          → ChainFlexibilityParam  (sp3 = flexible)

    Returns dict matching _PHYSICS_PRIORS format: {feature: (mean, std)}.
    Estimated values are labelled with ⚠ ESTIMATED in the printed summary.
    """
    priors = _generic_physics_priors()   # start from generic average

    if not RDKIT_AVAILABLE:
        return priors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return priors

    frac_csp3    = rdMolDescriptors.CalcFractionCSP3(mol)
    n_arom       = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_heavy      = mol.GetNumHeavyAtoms()
    mol_logp     = Descriptors.MolLogP(mol)
    tpsa         = CalcTPSA(mol)
    mol_wt       = Descriptors.MolWt(mol)
    n_hbd        = rdMolDescriptors.CalcNumHBD(mol)
    n_hba        = rdMolDescriptors.CalcNumHBA(mol)
    n_rot        = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom_density = n_arom / max(n_heavy, 1)

    # ── Core estimated quantities ────────────────────────────────────────────
    crystallinity = float(np.clip(0.20 + 0.60 * frac_csp3, 0.05, 0.85))
    chain_rigidity = float(np.clip(0.30 + 8.0 * arom_density, 0.10, 0.95))
    # Free volume: lower logP (more polar) → denser packing → less free volume
    logp_norm = float(np.clip((mol_logp + 2) / 10, 0, 1))
    free_vol  = float(np.clip(0.05 + 0.10 * logp_norm, 0.04, 0.22))
    diel_pol  = float(np.clip(10.0 + 1.5 * (n_hbd + n_hba) + 1.0 * n_arom, 8.0, 38.0))
    dipole    = float(np.clip(tpsa / max(mol_wt, 1) * 25, 0.0, 8.0))
    flex_param = float(np.clip(0.10 + 0.45 * frac_csp3, 0.05, 0.65))
    rot_frac   = n_rot / max(n_heavy, 1)
    seg_mob    = float(np.clip(0.05 + 0.40 * rot_frac, 0.05, 0.55))

    # ── Override the 12 most impactful features ──────────────────────────────
    overrides = {
        "DegreeOfCrystallinity"    : (crystallinity,          0.07),
        "CrystallinePhaseContent"  : (crystallinity,          0.06),
        "AmorphousPhaseContent"    : (1.0 - crystallinity,    0.06),
        "ChainRigidityIndex"       : (chain_rigidity,         0.07),
        "SegmentalMobility"        : (seg_mob,                0.04),
        "FreeVolumeFraction"       : (free_vol,               0.02),
        "DielectricPolarizability" : (diel_pol,               2.5),
        "ElectronicPolarizability" : (diel_pol * 0.78,        2.0),
        "IonicPolarizability"      : (max(diel_pol * 0.08, 0.3), 0.3),
        "DipoleMomentRepeat"       : (dipole,                 0.5),
        "ChainFlexibilityParam"    : (flex_param,             0.04),
        "MicrostructureOrder"      : (float(np.clip(chain_rigidity * 0.8, 0.1, 0.90)), 0.07),
    }
    priors.update(overrides)

    # Preserve physics constraint: amorphous + crystalline = 1
    priors["AmorphousPhaseContent"] = (1.0 - priors["DegreeOfCrystallinity"][0],
                                       priors["DegreeOfCrystallinity"][1])
    return priors


def print_estimated_priors(priors: dict, polymer_name: str) -> None:
    """Print the estimated physics priors with ESTIMATED tags for overridden ones."""
    generic = _generic_physics_priors()
    estimated_keys = {
        "DegreeOfCrystallinity", "CrystallinePhaseContent", "AmorphousPhaseContent",
        "ChainRigidityIndex", "SegmentalMobility", "FreeVolumeFraction",
        "DielectricPolarizability", "ElectronicPolarizability", "IonicPolarizability",
        "DipoleMomentRepeat", "ChainFlexibilityParam", "MicrostructureOrder",
    }
    print(f"\n  {BOLD}Physics priors for {polymer_name}{RESET}  "
          f"({YELLOW}⚠ ESTIMATED{RESET} = inferred from SMILES):")
    print(f"  {'Feature':<32}  {'Mean':>12}  {'Std':>10}  {'Source'}")
    print(f"  {'─'*32}  {'─'*12}  {'─'*10}  {'─'*10}")
    for feat in PHYSICS_FEATURE_NAMES:
        mu, sigma = priors[feat]
        tag = f"{YELLOW}⚠ ESTIMATED{RESET}" if feat in estimated_keys else "generic avg"
        print(f"  {feat:<32}  {mu:>12.4g}  {sigma:>10.4g}  {tag}")


# ──────────────────────────────────────────────────────────────────────────────
#  DUMMY SAMPLE GENERATION
# ──────────────────────────────────────────────────────────────────────────────

DUMMY_CONFIGS = [
    {
        "Sample_ID"   : "DUMMY_001",
        "dummy_label" : "Impossible free volume + negative crystallinity",
        "overrides"   : {
            "FreeVolumeFraction"      : 9.99,    # physically max ~0.25
            "DegreeOfCrystallinity"   : -5.0,    # must be 0-1
            "CrystallinePhaseContent" : -3.0,
            "AmorphousPhaseContent"   : 8.0,
        },
    },
    {
        "Sample_ID"   : "DUMMY_002",
        "dummy_label" : "Extreme chain rigidity + negative dipole",
        "overrides"   : {
            "ChainRigidityIndex"  : 15.0,    # must be 0-1
            "DipoleMomentRepeat"  : -500.0,  # must be ≥ 0
            "SegmentalMobility"   : -8.0,    # must be 0-1
        },
    },
    {
        "Sample_ID"   : "DUMMY_003",
        "dummy_label" : "Extreme dielectric + negative permittivity",
        "overrides"   : {
            "DielectricPolarizability" : 9999.0,
            "PermittivityRealPart"     : -50.0,  # must be > 0
            "TanDeltaDielectric"       : -1.0,   # must be ≥ 0
        },
    },
    {
        "Sample_ID"   : "DUMMY_004",
        "dummy_label" : "Negative chain lengths + impossible PDI",
        "overrides"   : {
            "PersistenceLength"    : -100.0,   # must be > 0
            "PolyDisersityIndex"   : -10.0,    # must be ≥ 1
            "Mw_kDa"              : -500.0,   # must be > 0
            "ContourLengthPerUnit" : -5.0,
        },
    },
    {
        "Sample_ID"   : "DUMMY_005",
        "dummy_label" : "All-zero features (missing data simulation)",
        "overrides"   : "ALL_ZEROS",
    },
]


def generate_dummy_samples(reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 5 dummy rows with physically impossible feature values.
    Use Polyimide mean as the starting point (before overrides).
    Targets are set to sentinel values (9999 / -999) that are obviously
    wrong — the trained model's predictions on these rows should also be
    out-of-range, confirming it isn't silently clamping bad inputs.
    """
    feat_cols = STRUCTURAL_FEATURE_NAMES + LATENT_FEATURE_NAMES + PHYSICS_FEATURE_NAMES
    pi_mean   = reference_df[reference_df["Polymer"] == "Polyimide"][feat_cols].mean()

    rows = []
    for cfg in DUMMY_CONFIGS:
        row = pi_mean.to_dict()

        if cfg["overrides"] == "ALL_ZEROS":
            for f in feat_cols:
                row[f] = 0.0
        else:
            row.update(cfg["overrides"])

        row["Polymer"]           = "DUMMY"
        row["SMILES"]            = "C"          # methane — trivially valid
        row["Sample_ID"]         = cfg["Sample_ID"]
        row["is_dummy"]          = True
        row["is_reference"]      = False
        row["dummy_label"]       = cfg["dummy_label"]
        row["Tg_degC"]           = 9999.0
        row["Dk_1GHz"]           = -999.0
        row["RadiationDose_MGy"] = -999.0
        rows.append(row)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI PROMPT — USER POLYMER
# ──────────────────────────────────────────────────────────────────────────────

def prompt_user_polymer() -> dict | None:
    """
    Interactive CLI prompt for a user-defined polymer.
    Returns a metadata dict or None if the user skips.

    The returned dict contains everything downstream tasks need:
      name, smiles, tg_base, dk_base, physics_priors,
      target_Tg, target_Dk, target_Rad,
      color, member, is_reference (False).
    """
    print(f"\n{'─'*72}")
    print(f"  {BOLD}CUSTOM POLYMER INPUT{RESET}")
    print(f"{'─'*72}")
    print("  PI / PEEK / PTFE will anchor model training.")
    print("  Your polymer will be evaluated via inference (no retraining).")
    print()
    choice = input("  Add a custom polymer? [y/N]: ").strip().lower()
    if choice not in ("y", "yes"):
        return None

    # ── SMILES ───────────────────────────────────────────────────────────────
    while True:
        smiles = input("\n  Enter SMILES string: ").strip()
        if not smiles:
            print("  ✗ Empty SMILES — try again.")
            continue
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print("  ✗ RDKit could not parse that SMILES — try again.")
                continue
            n_atoms = mol.GetNumHeavyAtoms()
            print(f"  ✔ Valid SMILES  ({n_atoms} heavy atoms, "
                  f"{rdMolDescriptors.CalcNumAromaticRings(mol)} aromatic rings)")
        break

    name = auto_name_from_smiles(smiles)
    print(f"  Auto-generated name: {BOLD}{name}{RESET}")

    # ── Tg baseline ──────────────────────────────────────────────────────────
    print(f"\n  Baseline Tg (glass transition temperature):")
    print(f"    [1] Enter my own value")
    print(f"    [2] RDKit structural estimate (Van Krevelen heuristic)")
    tg_choice = input("  Choice [1/2, default=2]: ").strip() or "2"

    if tg_choice == "1":
        try:
            tg_base = float(input("  Baseline Tg (°C): ").strip())
        except ValueError:
            print("  ✗ Invalid — using RDKit estimate.")
            tg_base = estimate_tg_from_smiles(smiles)
    else:
        tg_base = estimate_tg_from_smiles(smiles)
        print(f"  Estimated Tg ≈ {CYAN}{tg_base:.1f}°C{RESET}  {YELLOW}(⚠ heuristic){RESET}")

    # ── Dk baseline ──────────────────────────────────────────────────────────
    print(f"\n  Baseline Dk (dielectric constant at 1 GHz):")
    print(f"    [1] Enter my own value")
    print(f"    [2] RDKit structural estimate")
    dk_choice = input("  Choice [1/2, default=2]: ").strip() or "2"

    if dk_choice == "1":
        try:
            dk_base = float(input("  Baseline Dk: ").strip())
        except ValueError:
            print("  ✗ Invalid — using RDKit estimate.")
            dk_base = estimate_dk_from_smiles(smiles)
    else:
        dk_base = estimate_dk_from_smiles(smiles)
        print(f"  Estimated Dk ≈ {CYAN}{dk_base:.2f}{RESET}  {YELLOW}(⚠ heuristic){RESET}")

    # ── Inverse design targets ────────────────────────────────────────────────
    print(f"\n  Inverse design targets for {BOLD}{name}{RESET}:")
    print(f"    [1] I will specify all three targets manually")
    print(f"    [2] Auto: 10% improvement over estimated baseline")
    tgt_choice = input("  Choice [1/2, default=2]: ").strip() or "2"

    if tgt_choice == "1":
        try:
            target_Tg     = float(input("  Target Tg ≥ (°C) [min]: ").strip())
            target_Tg_max = float(input("  Target Tg ≤ (°C) [max]: ").strip())
            target_Dk     = float(input("  Target Dk ≤          : ").strip())
            target_Rad    = float(input("  Target Rad ≥ (MGy)   : ").strip())
            if target_Tg_max <= target_Tg:
                print(f"  {YELLOW}⚠  Tg max must be > Tg min — setting max = min + 100°C.{RESET}")
                target_Tg_max = target_Tg + 100.0
        except ValueError:
            print("  ✗ Invalid — using auto targets.")
            tgt_choice = "2"

    if tgt_choice == "2":
        target_Tg     = tg_base * 1.10
        target_Tg_max = tg_base * 1.40   # upper bound: +40% above baseline
        target_Dk     = max(dk_base * 0.90, 1.5)
        target_Rad    = 22.0

    # ── Physics priors ───────────────────────────────────────────────────────
    print(f"\n  Estimating physics priors from SMILES ...")
    priors = estimate_physics_priors_from_smiles(smiles)
    print_estimated_priors(priors, name)

    meta = {
        "name"          : name,
        "smiles"        : smiles,
        "tg_base"       : tg_base,
        "dk_base"       : dk_base,
        "physics_priors": priors,
        "target_Tg"     : target_Tg,
        "target_Tg_max" : target_Tg_max,
        "target_Dk"     : target_Dk,
        "target_Rad"    : target_Rad,
        "color"         : auto_color(0),
        "member"        : "UserPolymer",
        "is_reference"  : False,
    }

    print(f"\n  {GREEN}✔  {BOLD}{name}{RESET}{GREEN} registered.{RESET}")
    print(f"     Baseline : Tg = {tg_base:.1f}°C   Dk = {dk_base:.2f}")
    print(f"     Targets  : Tg ∈ [{target_Tg:.1f}, {target_Tg_max:.1f}]°C  "
          f"Dk ≤ {target_Dk:.2f}  Rad ≥ {target_Rad:.1f} MGy")
    print(f"  {YELLOW}⚠  Physics priors are ESTIMATED — validate experimentally.{RESET}")

    return meta


# ──────────────────────────────────────────────────────────────────────────────
#  DERIVATIVE GENERATION  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

STRUCTURAL_FEATURE_NAMES = [
    "MolWt","HeavyAtomMolWt","ExactMolWt","NumHeavyAtoms","NumRotatableBonds",
    "NumRings","NumAromaticRings","NumAliphaticRings","RingCount","FractionCSP3",
    "NumHDonors","NumHAcceptors","TPSA","MolLogP","MolMR","LabuteASA",
    "PEOE_VSA1","PEOE_VSA2","PEOE_VSA3","PEOE_VSA4","SMR_VSA1","SMR_VSA2",
    "SMR_VSA3","SlogP_VSA1","SlogP_VSA2","SlogP_VSA3","NumValenceElectrons",
    "NumRadicalElectrons","fr_C_O","fr_NH0","fr_NH1","fr_ArN","fr_Ar_COO",
    "fr_ether","fr_ketone","fr_imide","fr_amide","HallKierAlpha","Kappa1","Kappa2",
]
assert len(STRUCTURAL_FEATURE_NAMES) == N_STRUCTURAL

LATENT_FEATURE_NAMES = [f"polyBERT_dim_{i+1:02d}" for i in range(N_LATENT)]


def generate_derivatives(base_smiles: str, polymer_class: str,
                          num_needed: int) -> list:
    """Generate SMILES derivatives via RDKit reaction SMARTS mutations."""
    mutations_aromatic = [
        "[cH1:1]>>[c:1](F)",         "[cH1:1]>>[c:1](Cl)",
        "[cH1:1]>>[c:1](Br)",        "[cH1:1]>>[c:1](C)",
        "[cH1:1]>>[c:1](OC)",        "[cH1:1]>>[c:1](C(F)(F)F)",
        "[cH1:1]>>[c:1](C#N)",       "[cH1:1]>>[c:1](N)",
        "[cH1:1]>>[c:1](CC)",        "[cH1:1]>>[c:1](S(=O)(=O)N)",
    ]
    mutations_aliphatic = [
        "[F:1]>>[Cl:1]", "[F:1]>>[Br:1]", "[F:1]>>[C:1]",
        "[F:1]>>[O:1]",  "[F:1]>>[N:1]",  "[F:1]>>[S:1]",
        "[C:1](F)(F)>>[C:1](F)(Cl)",
        "[C:1](F)(F)(F)>>[C:1](F)(F)(Cl)",
    ]
    aromatic_rxns  = [rdChemReactions.ReactionFromSmarts(r) for r in mutations_aromatic]
    aliphatic_rxns = [rdChemReactions.ReactionFromSmarts(r) for r in mutations_aliphatic]

    base_mol = Chem.MolFromSmiles(base_smiles)
    if base_mol is None:
        raise ValueError(f"Cannot parse SMILES for {polymer_class}: {base_smiles}")

    _morgan_gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    base_fp     = _morgan_gen.GetFingerprint(base_mol)
    rxns_to_use = aliphatic_rxns if polymer_class == "PTFE" else aromatic_rxns

    # also detect aromatics for user polymers
    if polymer_class not in ("Polyimide", "PEEK", "PTFE"):
        has_arom = any(a.GetIsAromatic() for a in base_mol.GetAtoms())
        rxns_to_use = aromatic_rxns if has_arom else aliphatic_rxns

    seen = {base_smiles}
    candidates = [base_smiles]
    pool = [base_mol]
    MULTI_SITE_PROB = 0.30
    np.random.seed(42)

    attempts = 0
    while len(candidates) < num_needed and attempts < 150000:
        attempts += 1
        mol = pool[np.random.randint(0, len(pool))]
        rxn = rxns_to_use[np.random.randint(0, len(rxns_to_use))]
        prods = rxn.RunReactants((mol,))
        if not prods:
            continue
        new_mol = prods[np.random.randint(0, len(prods))][0]
        try:
            Chem.SanitizeMol(new_mol)
            new_smi = Chem.MolToSmiles(new_mol)
        except Exception:
            continue
        if new_smi in seen:
            continue
        new_fp   = _morgan_gen.GetFingerprint(new_mol)
        tanimoto = DataStructs.TanimotoSimilarity(base_fp, new_fp)
        if tanimoto >= 0.50:
            seen.add(new_smi)
            candidates.append(new_smi)
            pool.append(new_mol)
            if np.random.random() < MULTI_SITE_PROB and len(candidates) < num_needed:
                rxn2  = rxns_to_use[np.random.randint(0, len(rxns_to_use))]
                p2    = rxn2.RunReactants((new_mol,))
                if p2:
                    mol2 = p2[np.random.randint(0, len(p2))][0]
                    try:
                        Chem.SanitizeMol(mol2)
                        smi2 = Chem.MolToSmiles(mol2)
                    except Exception:
                        smi2 = None
                    if smi2 and smi2 not in seen:
                        fp2  = _morgan_gen.GetFingerprint(mol2)
                        sim2 = DataStructs.TanimotoSimilarity(base_fp, fp2)
                        if sim2 >= 0.50:
                            seen.add(smi2)
                            candidates.append(smi2)
                            pool.append(mol2)

    if len(candidates) < num_needed:
        print(f"    ⚠ {polymer_class}: {len(candidates)} / {num_needed} derivatives generated.")
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

_MOCK_STRUCTURAL_REFS = {
    "Polyimide": {"MolWt":720.7,"HeavyAtomMolWt":714.3,"ExactMolWt":720.1,"NumHeavyAtoms":52,"NumRotatableBonds":8,"NumRings":6,"NumAromaticRings":5,"NumAliphaticRings":1,"RingCount":6,"FractionCSP3":0.04,"NumHDonors":0,"NumHAcceptors":5,"TPSA":77.8,"MolLogP":4.12,"MolMR":188.4,"LabuteASA":265.1,"PEOE_VSA1":34.2,"PEOE_VSA2":18.1,"PEOE_VSA3":12.4,"PEOE_VSA4":9.3,"SMR_VSA1":22.5,"SMR_VSA2":18.3,"SMR_VSA3":8.7,"SlogP_VSA1":20.1,"SlogP_VSA2":14.6,"SlogP_VSA3":11.2,"NumValenceElectrons":200,"NumRadicalElectrons":0,"fr_C_O":4,"fr_NH0":2,"fr_NH1":0,"fr_ArN":2,"fr_Ar_COO":0,"fr_ether":1,"fr_ketone":0,"fr_imide":2,"fr_amide":0,"HallKierAlpha":-3.8,"Kappa1":22.1,"Kappa2":11.6},
    "PEEK":      {"MolWt":480.5,"HeavyAtomMolWt":475.1,"ExactMolWt":480.1,"NumHeavyAtoms":36,"NumRotatableBonds":6,"NumRings":4,"NumAromaticRings":4,"NumAliphaticRings":0,"RingCount":4,"FractionCSP3":0.00,"NumHDonors":0,"NumHAcceptors":3,"TPSA":39.5,"MolLogP":5.24,"MolMR":136.2,"LabuteASA":194.6,"PEOE_VSA1":22.8,"PEOE_VSA2":11.4,"PEOE_VSA3":8.9,"PEOE_VSA4":6.1,"SMR_VSA1":15.3,"SMR_VSA2":12.1,"SMR_VSA3":5.8,"SlogP_VSA1":13.7,"SlogP_VSA2":10.2,"SlogP_VSA3":8.4,"NumValenceElectrons":136,"NumRadicalElectrons":0,"fr_C_O":3,"fr_NH0":0,"fr_NH1":0,"fr_ArN":0,"fr_Ar_COO":0,"fr_ether":2,"fr_ketone":1,"fr_imide":0,"fr_amide":0,"HallKierAlpha":-2.5,"Kappa1":16.4,"Kappa2":8.9},
    "PTFE":      {"MolWt":338.0,"HeavyAtomMolWt":336.0,"ExactMolWt":337.9,"NumHeavyAtoms":8,"NumRotatableBonds":1,"NumRings":0,"NumAromaticRings":0,"NumAliphaticRings":0,"RingCount":0,"FractionCSP3":1.00,"NumHDonors":0,"NumHAcceptors":6,"TPSA":0.0,"MolLogP":4.68,"MolMR":48.2,"LabuteASA":74.3,"PEOE_VSA1":8.4,"PEOE_VSA2":4.2,"PEOE_VSA3":3.1,"PEOE_VSA4":2.2,"SMR_VSA1":5.6,"SMR_VSA2":3.4,"SMR_VSA3":1.8,"SlogP_VSA1":4.8,"SlogP_VSA2":3.2,"SlogP_VSA3":2.1,"NumValenceElectrons":56,"NumRadicalElectrons":0,"fr_C_O":0,"fr_NH0":0,"fr_NH1":0,"fr_ArN":0,"fr_Ar_COO":0,"fr_ether":0,"fr_ketone":0,"fr_imide":0,"fr_amide":0,"HallKierAlpha":0.2,"Kappa1":2.8,"Kappa2":1.4},
}


def _seeded_noise(polymer_name, feature_name, sample_idx, scale):
    seed_str = f"{polymer_name}_{feature_name}_{sample_idx}"
    seed_int = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    return np.random.RandomState(seed_int).normal(0, scale)


def extract_structural_features_rdkit(smiles_list: list,
                                      polymer_name: str) -> pd.DataFrame:
    descriptor_map = {
        "MolWt"                : lambda m: Descriptors.MolWt(m),
        "HeavyAtomMolWt"       : lambda m: Descriptors.HeavyAtomMolWt(m),
        "ExactMolWt"           : lambda m: Descriptors.ExactMolWt(m),
        "NumHeavyAtoms"        : lambda m: m.GetNumHeavyAtoms(),
        "NumRotatableBonds"    : lambda m: rdMolDescriptors.CalcNumRotatableBonds(m),
        "NumRings"             : lambda m: rdMolDescriptors.CalcNumRings(m),
        "NumAromaticRings"     : lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
        "NumAliphaticRings"    : lambda m: rdMolDescriptors.CalcNumAliphaticRings(m),
        "RingCount"            : lambda m: rdMolDescriptors.CalcNumRings(m),
        "FractionCSP3"         : lambda m: rdMolDescriptors.CalcFractionCSP3(m),
        "NumHDonors"           : lambda m: rdMolDescriptors.CalcNumHBD(m),
        "NumHAcceptors"        : lambda m: rdMolDescriptors.CalcNumHBA(m),
        "TPSA"                 : lambda m: CalcTPSA(m),
        "MolLogP"              : lambda m: Descriptors.MolLogP(m),
        "MolMR"                : lambda m: Descriptors.MolMR(m),
        "LabuteASA"            : lambda m: rdMolDescriptors.CalcLabuteASA(m),
        "PEOE_VSA1"            : lambda m: Descriptors.PEOE_VSA1(m),
        "PEOE_VSA2"            : lambda m: Descriptors.PEOE_VSA2(m),
        "PEOE_VSA3"            : lambda m: Descriptors.PEOE_VSA3(m),
        "PEOE_VSA4"            : lambda m: Descriptors.PEOE_VSA4(m),
        "SMR_VSA1"             : lambda m: Descriptors.SMR_VSA1(m),
        "SMR_VSA2"             : lambda m: Descriptors.SMR_VSA2(m),
        "SMR_VSA3"             : lambda m: Descriptors.SMR_VSA3(m),
        "SlogP_VSA1"           : lambda m: Descriptors.SlogP_VSA1(m),
        "SlogP_VSA2"           : lambda m: Descriptors.SlogP_VSA2(m),
        "SlogP_VSA3"           : lambda m: Descriptors.SlogP_VSA3(m),
        "NumValenceElectrons"  : lambda m: Descriptors.NumValenceElectrons(m),
        "NumRadicalElectrons"  : lambda m: Descriptors.NumRadicalElectrons(m),
        "fr_C_O"               : lambda m: Descriptors.fr_C_O(m),
        "fr_NH0"               : lambda m: Descriptors.fr_NH0(m),
        "fr_NH1"               : lambda m: Descriptors.fr_NH1(m),
        "fr_ArN"               : lambda m: Descriptors.fr_ArN(m),
        "fr_Ar_COO"            : lambda m: Descriptors.fr_Ar_COO(m),
        "fr_ether"             : lambda m: Descriptors.fr_ether(m),
        "fr_ketone"            : lambda m: Descriptors.fr_ketone(m),
        "fr_imide"             : lambda m: Descriptors.fr_imide(m),
        "fr_amide"             : lambda m: Descriptors.fr_amide(m),
        "HallKierAlpha"        : lambda m: rdMolDescriptors.CalcHallKierAlpha(m),
        "Kappa1"               : lambda m: rdMolDescriptors.CalcKappa1(m),
        "Kappa2"               : lambda m: rdMolDescriptors.CalcKappa2(m),
    }
    records = []
    for i, sm in enumerate(smiles_list):
        if i > 0 and i % 50 == 0:
            print(f"    [+] {i}/{len(smiles_list)} RDKit descriptors for {polymer_name}...")
        mol = Chem.MolFromSmiles(sm)
        row = {}
        for feat in STRUCTURAL_FEATURE_NAMES:
            try:
                row[feat] = float(descriptor_map[feat](mol))
            except Exception:
                row[feat] = 0.0
        records.append(row)
    return pd.DataFrame(records, columns=STRUCTURAL_FEATURE_NAMES)


def extract_structural_features_mock(polymer_name: str,
                                     n_samples: int) -> pd.DataFrame:
    """Mock fallback using reference means + seeded Gaussian noise."""
    ref = _MOCK_STRUCTURAL_REFS.get(polymer_name, _MOCK_STRUCTURAL_REFS["Polyimide"])
    records = []
    for i in range(n_samples):
        row = {}
        for feat in STRUCTURAL_FEATURE_NAMES:
            bv    = ref[feat]
            scale = max(abs(bv) * 0.05, 0.05)
            row[feat] = bv + _seeded_noise(polymer_name, feat, i, scale)
        records.append(row)
    return pd.DataFrame(records, columns=STRUCTURAL_FEATURE_NAMES)


def extract_structural_features(polymer_name: str,
                                smiles_list: list) -> pd.DataFrame:
    if RDKIT_AVAILABLE:
        return extract_structural_features_rdkit(smiles_list, polymer_name)
    return extract_structural_features_mock(polymer_name, len(smiles_list))


# ── Latent features ───────────────────────────────────────────────────────────

_LATENT_ANCHORS = {
    "Polyimide": {"scale":0.30,"aromatic_bias":0.65,"polar_bias":0.55,"backbone_bias":0.45,"sequence_bias":0.20},
    "PEEK"     : {"scale":0.28,"aromatic_bias":0.70,"polar_bias":0.30,"backbone_bias":0.60,"sequence_bias":0.15},
    "PTFE"     : {"scale":0.18,"aromatic_bias":-0.60,"polar_bias":-0.50,"backbone_bias":0.20,"sequence_bias":0.40},
}


def extract_latent_features(polymer_name: str,
                            smiles_list: list,
                            user_smiles: str = None) -> pd.DataFrame:
    """
    For reference polymers: use the fixed per-class anchors.
    For user polymers: derive a unique anchor from the base SMILES hash
    so that the embedding is reproducible but distinct from PI/PEEK/PTFE.
    """
    if polymer_name in _LATENT_ANCHORS:
        anch = _LATENT_ANCHORS[polymer_name]
        base_anchor = np.array(
            [anch["aromatic_bias"]] * 10 + [anch["polar_bias"]] * 10 +
            [anch["backbone_bias"]] * 10 + [anch["sequence_bias"]] * 10
        )
        scale = anch["scale"]
    else:
        # User polymer: anchor from SMILES hash → unique but deterministic
        ref_smi = user_smiles or (smiles_list[0] if smiles_list else "C")
        h = int(hashlib.sha256(ref_smi.encode()).hexdigest(), 16)
        rng_anchor = np.random.RandomState(h % (2**32))
        base_anchor = rng_anchor.uniform(-0.5, 0.5, size=40)
        scale = 0.25

    records = []
    for i, sm in enumerate(smiles_list):
        smiles_seed = int(hashlib.md5(sm.encode()).hexdigest(), 16) % (2**32)
        rng_base    = np.random.RandomState(smiles_seed)
        anchor      = base_anchor + rng_base.normal(0, 0.05, size=40)
        rng_sample  = np.random.RandomState(smiles_seed + i * 137)
        vec         = np.clip(anchor + rng_sample.normal(0, scale, size=40), -1.5, 1.5)
        records.append(dict(zip(LATENT_FEATURE_NAMES, vec)))
    return pd.DataFrame(records, columns=LATENT_FEATURE_NAMES)


# ── Physics features ──────────────────────────────────────────────────────────

def extract_physics_features(polymer_name: str, n_samples: int,
                              custom_priors: dict = None) -> pd.DataFrame:
    """
    Sample physics features from Gaussian priors.
    Uses custom_priors if provided (user polymer), else _PHYSICS_PRIORS lookup.
    """
    priors = custom_priors if custom_priors else _PHYSICS_PRIORS[polymer_name]
    seed   = int(hashlib.md5(f"physics_{polymer_name}".encode()).hexdigest(), 16) % (2**32)
    rng    = np.random.RandomState(seed)

    clip_01 = {"DegreeOfCrystallinity","CrystallinePhaseContent","AmorphousPhaseContent",
                "FreeVolumeFraction","MicrostructureOrder"}
    clip_pos = {"CrosslinkingDensity","DipoleMomentRepeat","PermittivityImaginaryPart",
                "TanDeltaDielectric","PolyDisersityIndex"}

    records = []
    for _ in range(n_samples):
        row = {}
        for feat in PHYSICS_FEATURE_NAMES:
            mu, sigma = priors[feat]
            val = rng.normal(mu, sigma)
            if feat in clip_01:
                val = float(np.clip(val, 0.0, 1.0))
            elif feat in clip_pos:
                val = max(val, 0.0)
            row[feat] = val
        row["AmorphousPhaseContent"] = 1.0 - row["DegreeOfCrystallinity"]
        records.append(row)
    return pd.DataFrame(records, columns=PHYSICS_FEATURE_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
#  MASTER DATASET BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_master_dataset(user_polymer_meta: dict = None) -> pd.DataFrame:
    """
    Build the master dataset.
    - PI / PEEK / PTFE: is_reference=True, is_dummy=False
    - User polymer    : is_reference=False, is_dummy=False  (if provided)
    - Dummy samples   : is_reference=False, is_dummy=True   (5 rows at bottom)
    """
    all_dfs = []

    # ── Reference polymers ───────────────────────────────────────────────────
    for polymer_name, meta in POLYMER_REGISTRY.items():
        print(f"  ▶ {meta['color']}{BOLD}{polymer_name}{RESET} ...")
        base_smiles = meta["smiles"]
        n = N_SAMPLES_PER_POLYMER

        smiles_list = (generate_derivatives(base_smiles, polymer_name, n)
                       if RDKIT_AVAILABLE else [base_smiles] * n)
        # Pad if derivative generation fell short (e.g. PTFE has few mutable sites)
        if len(smiles_list) < n:
            smiles_list = (smiles_list * ((n // len(smiles_list)) + 1))[:n]

        df_s = extract_structural_features(polymer_name, smiles_list)
        df_l = extract_latent_features(polymer_name, smiles_list)
        df_p = extract_physics_features(polymer_name, n)

        df_poly = pd.concat([df_s, df_l, df_p], axis=1)
        df_poly.insert(0, "Polymer",      polymer_name)
        df_poly.insert(1, "SMILES",       smiles_list)
        df_poly.insert(2, "Sample_ID",    [f"{polymer_name[:2]}_{i+1:04d}" for i in range(n)])
        df_poly["is_reference"] = True
        df_poly["is_dummy"]     = False
        df_poly["dummy_label"]  = ""
        all_dfs.append(df_poly)

    # ── User polymer ─────────────────────────────────────────────────────────
    if user_polymer_meta:
        name   = user_polymer_meta["name"]
        smiles = user_polymer_meta["smiles"]
        priors = user_polymer_meta["physics_priors"]
        n      = N_SAMPLES_PER_POLYMER

        print(f"  ▶ {user_polymer_meta['color']}{BOLD}{name}{RESET} "
              f"{YELLOW}(⚠ estimated priors){RESET} ...")

        smiles_list = (generate_derivatives(smiles, name, n)
                       if RDKIT_AVAILABLE else [smiles] * n)
        # Pad if derivative generation fell short
        if len(smiles_list) < n:
            smiles_list = (smiles_list * ((n // len(smiles_list)) + 1))[:n]

        df_s = extract_structural_features(name, smiles_list)
        df_l = extract_latent_features(name, smiles_list, user_smiles=smiles)
        df_p = extract_physics_features(name, n, custom_priors=priors)

        df_user = pd.concat([df_s, df_l, df_p], axis=1)
        df_user.insert(0, "Polymer",   name)
        df_user.insert(1, "SMILES",    smiles_list)
        df_user.insert(2, "Sample_ID", [f"USR_{i+1:04d}" for i in range(n)])
        df_user["is_reference"] = False
        df_user["is_dummy"]     = False
        df_user["dummy_label"]  = ""
        all_dfs.append(df_user)

    master = pd.concat(all_dfs, ignore_index=True)

    # ── Dummy samples (appended last) ────────────────────────────────────────
    dummy_df = generate_dummy_samples(master)
    # Align columns
    for col in master.columns:
        if col not in dummy_df.columns:
            dummy_df[col] = np.nan
    dummy_df = dummy_df[master.columns]

    master = pd.concat([master, dummy_df], ignore_index=True)

    print(f"\n  ✔  Master dataset shape: {master.shape}  "
          f"({len(master)} samples × {master.shape[1]} columns)\n")
    return master


# ──────────────────────────────────────────────────────────────────────────────
#  TARGET GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_targets(master_df: pd.DataFrame,
                    user_polymer_meta: dict = None) -> pd.DataFrame:
    """
    Compute all four targets for every row.
    Dummy rows keep their sentinel target values (9999 / -999).
    User polymer uses user-provided or estimated Tg/Dk baselines.
    """
    df = master_df.copy()

    tg_base = {
        "Polyimide"          : 240.0,
        "PEEK"               : 200.0,
        "PTFE"               :  10.0,
        # Low-Dk / low-outgassing families
        "CyanateEster"       : 290.0,   # fully cured triazine network; Tg ~250–350 °C
        "FluorinatedPolyimide": 280.0,  # 6FDA-based; Tg ~250–320 °C
        "Polybenzoxazole"    : 400.0,   # rigid-rod PBO; Tg typically >400 °C
    }
    dk_base = {
        "Polyimide"          : 1.8,
        "PEEK"               : 1.6,
        "PTFE"               : 1.6,
        # Low-Dk families — principal design targets
        "CyanateEster"       : 0.7,    # Dk ~2.7; offset ~0.7 from formula base of 2.0
        "FluorinatedPolyimide": 0.55,  # Dk ~2.5–2.8; lower than standard PI
        "Polybenzoxazole"    : 1.1,    # Dk ~3.1; low loss despite aromatic rigidity
    }

    if user_polymer_meta:
        n = user_polymer_meta["name"]
        tg_base[n] = user_polymer_meta["tg_base"]
        dk_base[n] = user_polymer_meta["dk_base"]

    alpha, beta, gamma, delta, eps = 280.0, 18.0, 600.0, 80.0, 1500.0
    a, b, c, d = 0.065, 8.0, 2.0, 1.0
    rng_tg  = np.random.RandomState(101)
    rng_dk  = np.random.RandomState(202)
    rng_rad = np.random.RandomState(404)

    Tg_vals, Dk_vals, Rad_vals = [], [], []

    for _, row in df.iterrows():
        # Skip dummies — keep sentinel values
        if row.get("is_dummy", False):
            Tg_vals.append(9999.0)
            Dk_vals.append(-999.0)
            Rad_vals.append(-999.0)
            continue

        poly  = row["Polymer"]
        base_t = tg_base.get(poly, 150.0)   # fallback for any unknown polymer
        base_d = dk_base.get(poly, 3.0)

        tg = (base_t
              + alpha * row["ChainRigidityIndex"]
              + beta  * row["NumAromaticRings"]
              - gamma * row["FreeVolumeFraction"]
              + delta * row["DegreeOfCrystallinity"]
              + eps   * row["CrosslinkingDensity"]
              ) + rng_tg.normal(0, 4.0)

        dk = (base_d
              + a * row["DielectricPolarizability"]
              - b * row["FreeVolumeFraction"]
              + c * row["DegreeOfCrystallinity"]
              - d * row["FractionCSP3"]
              ) + rng_dk.normal(0, 0.05)

        rad = (25.0
               + 8.0  * (row["NumAromaticRings"] - 4.0)
               - 50.0 * (row["FreeVolumeFraction"] - 0.09)
               ) + rng_rad.normal(0, 0.5)

        Tg_vals.append(tg)
        Dk_vals.append(max(dk, 1.5))
        Rad_vals.append(max(rad, 0.0))

    df["Tg_degC"]           = Tg_vals
    df["Dk_1GHz"]           = Dk_vals
    df["RadiationDose_MGy"] = Rad_vals
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'═'*72}")
    print(f"  {BOLD}TASK 1 — DATA CURATION{RESET}")
    print(f"{'═'*72}\n")

    # ── User polymer prompt ───────────────────────────────────────────────────
    user_meta = prompt_user_polymer()

    # ── Build dataset ─────────────────────────────────────────────────────────
    print(f"\n  Building master dataset ...")
    df_master = build_master_dataset(user_polymer_meta=user_meta)

    # ── Compute targets ───────────────────────────────────────────────────────
    print("  Computing QSPR targets ...")
    df_final = compute_targets(df_master, user_polymer_meta=user_meta)

    # ── Dummy row summary ─────────────────────────────────────────────────────
    n_ref   = int(df_final["is_reference"].sum())
    n_user  = int((~df_final["is_reference"] & ~df_final["is_dummy"]).sum())
    n_dummy = int(df_final["is_dummy"].sum())
    print(f"\n  Dataset composition:")
    print(f"    Reference polymers : {n_ref} rows  (PI + PEEK + PTFE + CyanateEster + FluorinatedPI + PBO — used for training)")
    if n_user:
        print(f"    User polymer       : {n_user} rows  ({user_meta['name']} — inference only)")
    print(f"    Dummy rows         : {n_dummy} rows  (sanity checks — excluded from training)")
    print(f"    Total              : {len(df_final)} rows × {df_final.shape[1]} cols")

    # ── Save master dataset ───────────────────────────────────────────────────
    with open("master_dataset.pkl", "wb") as f:
        pickle.dump(df_final, f)
    print(f"\n  💾  Saved: master_dataset.pkl")

    # ── Save user polymer metadata ────────────────────────────────────────────
    if user_meta:
        with open("user_polymer.pkl", "wb") as f:
            pickle.dump(user_meta, f)
        print(f"  💾  Saved: user_polymer.pkl  →  (read by tasks 2, 3, 4, 7, 8)")
    else:
        # Remove stale user_polymer.pkl if no user polymer this run
        if os.path.exists("user_polymer.pkl"):
            os.remove("user_polymer.pkl")
            print("  🗑   Removed stale user_polymer.pkl")

    print(f"\n  ✔  Task 1 complete.\n")
