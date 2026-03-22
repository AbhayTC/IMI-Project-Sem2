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

TASK 5 ▶ Morgan Fingerprint Generation (ECFP4, radius=2, nBits=64)

  Computes circular Morgan fingerprints for the repeat-unit SMILES of each
  polymer. The fingerprint encodes all circular substructures up to radius 2
  (ECFP4), capturing local chemical environment up to 4 bonds from each atom.

  The same bit vector is replicated across all 240 samples per polymer because
  the fingerprint is a property of the repeat-unit SMILES (fixed chemistry),
  not of individual morphological/processing samples.

  RDKit path  : uses AllChem.GetMorganFingerprintAsBitVect (genuine ECFP4)
  Fallback    : chemistry-anchored deterministic mock. Shared-region RNG
                strategy ensures PI<->PEEK Tanimoto >> PI<->PTFE, matching
                the real structural relationship.

  Output schema (morgan_fingerprints.pkl):
    A pandas DataFrame with 720 rows and 66 columns:
      - Polymer    : polymer class name (str)
      - Sample_ID  : matches task1 Sample_ID (str)
      - morgan_fp_00 ... morgan_fp_63 : 64 binary ECFP4 bits (uint8)

  This schema is consumed directly by task4_output_management.py, which
  left-joins on [Polymer, Sample_ID] to append the 64 bit columns.

  Outputs: morgan_fingerprints.pkl  (loaded by task4_output_management.py)
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import warnings, hashlib, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Optional RDKit ────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
    print("OK  RDKit detected -- using genuine ECFP4 Morgan fingerprints.")
except ImportError:
    RDKIT_AVAILABLE = False
    print("!!  RDKit not installed -- using chemistry-anchored mock fingerprints.\n"
          "    Install via: pip install rdkit\n"
          "    (Mock preserves PI<->PEEK similarity > PI/PEEK<->PTFE.)")

# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
MORGAN_BITS   = 64     # Must match N_MORGAN in task4_output_management.py
MORGAN_RADIUS = 2      # radius=2 -> ECFP4

# Column names that task4 expects
MORGAN_FEATURE_NAMES = [f"morgan_fp_{i:02d}" for i in range(MORGAN_BITS)]

POLYMER_REGISTRY = {
    "Polyimide": {
        "smiles" : ("O=C1c2ccccc2C(=O)N1c3ccc(Oc4ccc"
                    "(N5C(=O)c6ccccc6C5=O)cc4)cc3"),
        "color"  : "\033[94m",
        "member" : "Member1",
    },
    "PEEK": {
        "smiles" : ("O=C(c1ccc(Oc2ccc(Oc3ccc(C(=O)c4ccccc4)"
                    "cc3)cc2)cc1)c5ccccc5"),
        "color"  : "\033[92m",
        "member" : "Member2",
    },
    "PTFE": {
        "smiles" : "FC(F)(F)C(F)(F)F",
        "color"  : "\033[93m",
        "member" : "Member3",
    },
}

N_SAMPLES_PER_POLYMER = 240

RESET = "\033[0m"
BOLD  = "\033[1m"

# ──────────────────────────────────────────────────────────────────────────────
#  ████████╗  █████╗ ███████╗██╗  ██╗    ███████╗
#  ╚══██╔══╝ ██╔══██╗██╔════╝██║ ██╔╝    ██╔════╝
#     ██║    ███████║███████╗█████╔╝     ███████╗
#     ██║    ██╔══██║╚════██║██╔═██╗     ╚════██║
#     ██║    ██║  ██║███████║██║  ██╗    ███████║
#     ╚═╝    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚══════╝
#  MORGAN FINGERPRINT GENERATION
# ──────────────────────────────────────────────────────────────────────────────

# ── Chemistry-Anchored Bit Regions for Mock Fingerprints ─────────────────────
#
# The 64 bits are partitioned into 7 chemically distinct regions. Within each
# region, a SHARED random draw (seeded on the region name, not the SMILES)
# decides which bit positions are ON. Polymers sharing a motif (e.g. benzene)
# tend to share the same ON bits within that region, reproducing the key
# property of real Morgan fingerprints: PI<->PEEK Tanimoto >> PI/PEEK<->PTFE.
#
# Region breakdown (64 bits total):
#   0 – 11 : Benzene ring substructures   (PI + PEEK, absent PTFE)
#   12– 19 : General aromatic fragments   (PI + PEEK, absent PTFE)
#   20– 27 : Ether linkage substructures  (PI + PEEK, absent PTFE)
#   28– 35 : Imide ring substructures     (PI only)
#   36– 41 : Aryl-ketone substructures    (PEEK only)
#   42– 55 : CF2/CF3 fluorocarbon bits    (PTFE only)
#   56– 63 : Misc aliphatic/connectivity  (all, low probability)
_BIT_REGIONS = {
    # name             start  end   P_PI   P_PEEK  P_PTFE
    "benzene"      : (  0,  12,  0.70,   0.75,   0.00),
    "aromatic"     : ( 12,  20,  0.60,   0.50,   0.00),
    "ether"        : ( 20,  28,  0.55,   0.62,   0.00),
    "imide"        : ( 28,  36,  0.80,   0.00,   0.00),
    "aryl_ketone"  : ( 36,  42,  0.08,   0.85,   0.00),
    "fluorocarbon" : ( 42,  56,  0.00,   0.00,   0.70),
    "misc"         : ( 56,  64,  0.14,   0.10,   0.08),
}


def _mock_morgan_fingerprint(polymer_name: str) -> np.ndarray:
    """
    Chemistry-anchored deterministic mock Morgan fingerprint (64-bit).

    Within each chemical bit region the random draws are seeded on the REGION
    NAME so polymers sharing a motif share the same ON bit positions. A polymer
    absent from a region (p=0) always gets all-zero bits there.
    """
    prob_table = {
        "Polyimide": {k: v[2] for k, v in _BIT_REGIONS.items()},
        "PEEK"     : {k: v[3] for k, v in _BIT_REGIONS.items()},
        "PTFE"     : {k: v[4] for k, v in _BIT_REGIONS.items()},
    }

    bits = np.zeros(MORGAN_BITS, dtype=np.uint8)

    for region_name, (start, end, *_) in _BIT_REGIONS.items():
        # Shared RNG -- identical draws for ALL polymers within this region
        region_seed  = int(hashlib.md5(region_name.encode()).hexdigest(), 16) % (2**32)
        region_rng   = np.random.RandomState(region_seed)
        shared_draws = region_rng.random(end - start)

        # Bit is ON if polymer probability > the shared random draw
        p = prob_table[polymer_name][region_name]
        bits[start:end] = (shared_draws < p).astype(np.uint8)

    return bits


def compute_morgan_fingerprint_rdkit(smiles: str) -> np.ndarray:
    """RDKit path: genuine ECFP4 as numpy uint8 array (64 bits)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    fp  = AllChem.GetMorganFingerprintAsBitVect(
              mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS)
    arr = np.zeros(MORGAN_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_morgan_fingerprint(polymer_name: str, smiles: str) -> np.ndarray:
    """Dispatcher -- RDKit if available, chemistry-anchored mock otherwise."""
    if RDKIT_AVAILABLE:
        return compute_morgan_fingerprint_rdkit(smiles)
    return _mock_morgan_fingerprint(polymer_name)


# ── Tanimoto Similarity ───────────────────────────────────────────────────────

def tanimoto_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Tanimoto (Jaccard) for binary bit vectors: |A AND B| / |A OR B|.
    Returns 0.0 if both vectors are all-zero.
    """
    intersection = int(np.sum(a & b))
    union        = int(np.sum(a | b))
    return intersection / union if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  OUTPUT BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_morgan_dataframe(fp_dict: dict) -> pd.DataFrame:
    """
    Build the 720-row DataFrame that task4 expects.

    Schema:
      - Polymer    (str)           : polymer class name
      - Sample_ID  (str)           : matches task1 Sample_ID format  e.g. 'Po_0001'
      - morgan_fp_00 ... morgan_fp_63 (uint8) : 64 ECFP4 bit columns

    The same bit vector is replicated 240 times per polymer because the
    fingerprint is a property of the repeat-unit SMILES (fixed chemistry).
    Sample_IDs are generated to exactly match task1_data_curation.py:
      Polyimide -> 'Po_0001' ... 'Po_0240'
      PEEK      -> 'PE_0001' ... 'PE_0240'
      PTFE      -> 'PT_0001' ... 'PT_0240'
    """
    records = []
    for polymer_name in POLYMER_REGISTRY:
        fp   = fp_dict[polymer_name]              # (64,) uint8
        prefix = polymer_name[:2]                  # 'Po', 'PE', 'PT'
        for i in range(N_SAMPLES_PER_POLYMER):
            row = {
                "Polymer"   : polymer_name,
                "Sample_ID" : f"{prefix}_{i+1:04d}",
            }
            for j, col in enumerate(MORGAN_FEATURE_NAMES):
                row[col] = int(fp[j])
            records.append(row)

    df = pd.DataFrame(records)
    for col in MORGAN_FEATURE_NAMES:
        df[col] = df[col].astype(np.uint8)
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  REPORTING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def print_banner(text: str, char: str = "=", width: int = 72) -> None:
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{BOLD}{text}{RESET}{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")


def print_fingerprint_report(fp_dict: dict) -> None:
    """Formatted Morgan fingerprint report with Tanimoto similarity matrix."""
    print_banner(
        f"TASK 5 -- MORGAN FINGERPRINT REPORT  "
        f"(ECFP{MORGAN_RADIUS*2}, nBits={MORGAN_BITS})",
        width=72
    )

    # Per-polymer stats
    print(f"\n  {'Polymer':<12}  {'SMILES (truncated)':<38}  "
          f"{'ON bits':>8}  {'Density':>9}  {'OFF bits':>9}")
    print(f"  {'─'*12}  {'─'*38}  {'─'*8}  {'─'*9}  {'─'*9}")
    for pname, meta in POLYMER_REGISTRY.items():
        fp     = fp_dict[pname]
        on     = int(fp.sum())
        smiles = meta["smiles"]
        smiles_short = smiles[:35] + "..." if len(smiles) > 35 else smiles
        print(f"  {meta['color']}{BOLD}{pname:<12}{RESET}  "
              f"{smiles_short:<38}  {on:>8}  {on/MORGAN_BITS:>9.3f}  "
              f"{MORGAN_BITS-on:>9}")

    # Tanimoto similarity matrix
    names = list(fp_dict.keys())
    print(f"\n  Tanimoto Similarity Matrix "
          f"(ECFP{MORGAN_RADIUS*2}, T=1 identical, T=0 no overlap):")
    print(f"  {'':12}  " + "  ".join(f"{n:>12}" for n in names))
    print(f"  {'─'*12}  " + "  ".join(["─"*12] * len(names)))
    for a in names:
        meta = POLYMER_REGISTRY[a]
        row  = "  ".join(
            f"{tanimoto_similarity(fp_dict[a], fp_dict[b]):>12.4f}"
            for b in names
        )
        print(f"  {meta['color']}{BOLD}{a:<12}{RESET}  {row}")

    # Interpretation
    pairs = [
        ("Polyimide", "PEEK",
         "Both aromatic --> shared substructures expected"),
        ("Polyimide", "PTFE",
         "Aromatic/imide vs fluorocarbon --> near-zero overlap"),
        ("PEEK",      "PTFE",
         "Aromatic/ketone vs fluorocarbon --> near-zero overlap"),
    ]
    print(f"\n  Chemical interpretation:")
    for a, b, note in pairs:
        t = tanimoto_similarity(fp_dict[a], fp_dict[b])
        print(f"    {a} <-> {b:<12}: T = {t:.4f}  ({note})")

    print(f"\n  Method  : "
          f"{'RDKit AllChem.GetMorganFingerprintAsBitVect' if RDKIT_AVAILABLE else 'Chemistry-anchored mock (shared-region RNG)'}")
    print(f"  Radius  : {MORGAN_RADIUS}  (ECFP{MORGAN_RADIUS*2})")
    print(f"  nBits   : {MORGAN_BITS}")
    print(f"  Columns : morgan_fp_00 ... morgan_fp_{MORGAN_BITS-1:02d}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_banner(
        "TASK 5 -- MORGAN FINGERPRINT GENERATION",
        char="*", width=72
    )
    print(f"\n  Computing ECFP{MORGAN_RADIUS*2} fingerprints "
          f"(radius={MORGAN_RADIUS}, nBits={MORGAN_BITS}) for 3 polymers ...\n")

    # ── Compute one fingerprint vector per polymer ────────────────────────────
    fp_dict = {}
    for pname, meta in POLYMER_REGISTRY.items():
        fp = compute_morgan_fingerprint(pname, meta["smiles"])
        fp_dict[pname] = fp
        print(f"  {meta['color']}{BOLD}{pname:<12}{RESET}  "
              f"ON bits: {int(fp.sum()):>3} / {MORGAN_BITS}  "
              f"(density = {fp.sum()/MORGAN_BITS:.3f})")

    # ── Build the 720-row DataFrame in the schema task4 expects ───────────────
    morgan_df = build_morgan_dataframe(fp_dict)

    # ── Verify shape and schema ───────────────────────────────────────────────
    assert morgan_df.shape == (720, 66), \
        f"Shape error: {morgan_df.shape} (expected (720, 66))"
    assert list(morgan_df.columns[:2]) == ["Polymer", "Sample_ID"], \
        "First two columns must be Polymer and Sample_ID"
    assert list(morgan_df.columns[2:]) == MORGAN_FEATURE_NAMES, \
        "Morgan bit columns do not match MORGAN_FEATURE_NAMES"
    assert morgan_df[MORGAN_FEATURE_NAMES].isin([0, 1]).all().all(), \
        "Non-binary values found in fingerprint columns"
    print(f"\n  Schema verified: {morgan_df.shape[0]} rows x "
          f"{morgan_df.shape[1]} cols "
          f"(Polymer + Sample_ID + {MORGAN_BITS} bits)")

    # ── Print report ──────────────────────────────────────────────────────────
    print_fingerprint_report(fp_dict)

    # ── Save as pkl (plain DataFrame — consumed directly by task4) ────────────
    with open("morgan_fingerprints.pkl", "wb") as f:
        pickle.dump(morgan_df, f)
    print("  Saved: morgan_fingerprints.pkl  -->  "
          "(input for task4_output_management.py)\n")
    print(f"  PKL format : pandas DataFrame  |  "
          f"Rows: {morgan_df.shape[0]}  |  Cols: {morgan_df.shape[1]}")
    print(f"  Bit cols   : morgan_fp_00 ... morgan_fp_{MORGAN_BITS-1:02d}")
    print(f"  Merge key  : [Polymer, Sample_ID]\n")
