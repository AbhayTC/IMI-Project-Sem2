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
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import warnings, hashlib, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
    print("RDKit detected printing Morgan fingerprints data.")
except ImportError:
    RDKIT_AVAILABLE = False
    print("!!  RDKit not installed -- using chemistry-anchored mock fingerprints.\n"
          "    Install via: pip install rdkit")

# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
MORGAN_BITS   = 64
MORGAN_RADIUS = 2
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
        # FIXED: FC(F)(F)C(F)(F)F is hexafluoroethane — not a valid PTFE repeat unit.
        # Correct PTFE repeat unit: tetrafluoroethylene monomer FC(F)=C(F)F
        "smiles" : "FC(F)=C(F)F",
        "color"  : "\033[93m",
        "member" : "Member3",
    },
}

N_SAMPLES_PER_POLYMER = 240
RESET = "\033[0m"
BOLD  = "\033[1m"

# ──────────────────────────────────────────────────────────────────────────────
#  BIT REGION MAP FOR MOCK FINGERPRINTS
# ──────────────────────────────────────────────────────────────────────────────
_BIT_REGIONS = {
    "benzene"      : (  0,  12,  0.70,   0.75,   0.00),
    "aromatic"     : ( 12,  20,  0.60,   0.50,   0.00),
    "ether"        : ( 20,  28,  0.55,   0.62,   0.00),
    "imide"        : ( 28,  36,  0.80,   0.00,   0.00),
    "aryl_ketone"  : ( 36,  42,  0.08,   0.85,   0.00),
    "fluorocarbon" : ( 42,  56,  0.00,   0.00,   0.70),
    "misc"         : ( 56,  64,  0.14,   0.10,   0.08),
}


def _mock_morgan_fingerprint(polymer_name: str) -> np.ndarray:
    prob_table = {
        "Polyimide": {k: v[2] for k, v in _BIT_REGIONS.items()},
        "PEEK"     : {k: v[3] for k, v in _BIT_REGIONS.items()},
        "PTFE"     : {k: v[4] for k, v in _BIT_REGIONS.items()},
    }
    bits = np.zeros(MORGAN_BITS, dtype=np.uint8)
    for region_name, (start, end, *_) in _BIT_REGIONS.items():
        region_seed  = int(hashlib.md5(region_name.encode()).hexdigest(), 16) % (2**32)
        region_rng   = np.random.RandomState(region_seed)
        shared_draws = region_rng.random(end - start)
        p = prob_table[polymer_name][region_name]
        bits[start:end] = (shared_draws < p).astype(np.uint8)
    return bits


def compute_morgan_fingerprint_rdkit(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    fp  = AllChem.GetMorganFingerprintAsBitVect(
              mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS)
    arr = np.zeros(MORGAN_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_morgan_fingerprint(polymer_name: str, smiles: str) -> np.ndarray:
    if RDKIT_AVAILABLE:
        return compute_morgan_fingerprint_rdkit(smiles)
    return _mock_morgan_fingerprint(polymer_name)


# ──────────────────────────────────────────────────────────────────────────────
#  TANIMOTO
# ──────────────────────────────────────────────────────────────────────────────

def tanimoto_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto (Jaccard) for binary bit vectors: |A AND B| / |A OR B|."""
    intersection = int(np.sum(a & b))
    union        = int(np.sum(a | b))
    return intersection / union if union > 0 else 0.0


def interpret_tanimoto(score: float) -> str:
    if   score >= 0.85: return "Very High  — near-identical substructure"
    elif score >= 0.65: return "High       — significant structural overlap"
    elif score >= 0.40: return "Moderate   — partial scaffold similarity"
    elif score >= 0.20: return "Low        — limited shared features"
    else:               return "Very Low   — structurally dissimilar"


# ──────────────────────────────────────────────────────────────────────────────
#  FINGERPRINT DISPLAY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def bits_to_hex(fp: np.ndarray) -> str:
    """Convert a uint8 bit array to a compact hex string (4 bits per hex char)."""
    bit_str = "".join(str(b) for b in fp)
    pad     = (4 - len(bit_str) % 4) % 4
    bit_str = bit_str + "0" * pad
    return "".join(format(int(bit_str[i:i+4], 2), "x") for i in range(0, len(bit_str), 4))


def format_bit_grid(fp: np.ndarray, cols: int = 16) -> list[str]:
    """
    Format the 64-bit fingerprint as a grid of rows.
    Each row shows: bit index range | visual bar | ON count.
    '█' = bit ON, '░' = bit OFF.
    """
    lines = []
    n = len(fp)
    for row_start in range(0, n, cols):
        chunk   = fp[row_start: row_start + cols]
        bar     = "".join("█" if b else "░" for b in chunk)
        on_ct   = int(chunk.sum())
        end_idx = min(row_start + cols - 1, n - 1)
        lines.append(f"    Bits {row_start:02d}-{end_idx:02d}  [{bar}]  ON:{on_ct:>2}/{len(chunk)}")
    return lines


def print_separator(char: str = "─", width: int = 72) -> None:
    print(f"  {char * width}")


def print_banner(text: str, char: str = "═", width: int = 74) -> None:
    print(f"\n  {char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"  {char}{' ' * pad}{BOLD}{text}{RESET}{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"  {char * width}")


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — MORGAN FINGERPRINT REPORT
# ──────────────────────────────────────────────────────────────────────────────

def print_fingerprint_report(fp_dict: dict) -> None:
    W = 72
    print_banner(
        f"SECTION 1 — MORGAN FINGERPRINTS  "
        f"(ECFP{MORGAN_RADIUS*2}, radius={MORGAN_RADIUS}, nBits={MORGAN_BITS})"
    )
    method_str = ("RDKit AllChem.GetMorganFingerprintAsBitVect"
                  if RDKIT_AVAILABLE else "Chemistry-anchored mock (shared-region RNG)")
    print(f"\n  Method  : {method_str}")
    print(f"  Radius  : {MORGAN_RADIUS}  →  ECFP{MORGAN_RADIUS * 2} (encodes up to {MORGAN_RADIUS * 2} bonds from each atom)")
    print(f"  nBits   : {MORGAN_BITS}  (64-bit folded bit vector)\n")

    for pname, meta in POLYMER_REGISTRY.items():
        fp      = fp_dict[pname]
        on      = int(fp.sum())
        off     = MORGAN_BITS - on
        density = on / MORGAN_BITS
        hex_str = bits_to_hex(fp)
        bit_str = "".join(str(b) for b in fp)
        color   = meta["color"]

        print(f"  ┌── {color}{BOLD}{pname}{RESET} {'─'*(W - 4 - len(pname))}┐")
        print(f"  │  Full Name  : {POLYMER_REGISTRY[pname].get('full_name', pname):<55}│")
        smiles = meta["smiles"]
        smiles_lines = [smiles[i:i+55] for i in range(0, len(smiles), 55)]
        print(f"  │  SMILES     : {smiles_lines[0]:<55}│")
        for sl in smiles_lines[1:]:
            print(f"  │               {sl:<55}│")
        print(f"  │  Bits ON    : {on:<8}  Bits OFF : {off:<8}  Density : {density:.4f}       │")
        print(f"  │  Hex FP     : {hex_str:<55}│")
        print(f"  │  Bit String :                                                        │")
        for grid_line in format_bit_grid(fp, cols=16):
            print(f"  │  {grid_line:<69}│")
        # Show ON bit indices
        on_indices = np.where(fp == 1)[0].tolist()
        idx_str = ", ".join(str(i) for i in on_indices)
        idx_lines = [idx_str[i:i+55] for i in range(0, len(idx_str), 55)] if idx_str else ["(none)"]
        print(f"  │  ON Indices : {idx_lines[0]:<55}│")
        for il in idx_lines[1:]:
            print(f"  │               {il:<55}│")
        print(f"  └{'─'*W}┘")
        print()


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — TANIMOTO SIMILARITY REPORT
# ──────────────────────────────────────────────────────────────────────────────

def print_tanimoto_report(fp_dict: dict) -> None:
    W  = 72
    names = list(fp_dict.keys())

    # Pre-compute all pairwise scores
    score_matrix = {}
    for a in names:
        for b in names:
            score_matrix[(a, b)] = tanimoto_similarity(fp_dict[a], fp_dict[b])

    print_banner("SECTION 2 — TANIMOTO SIMILARITY ANALYSIS")

    # ── 2a: Pairwise Table ────────────────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  {'2a. Pairwise Tanimoto Scores':^{W}}")
    print(f"  {'─'*W}\n")

    pairs = [
        ("Polyimide", "PEEK"),
        ("Polyimide", "PTFE"),
        ("PEEK",      "PTFE"),
    ]

    print(f"  {'Pair':<30} {'|A∩B|':>6}  {'|A∪B|':>6}  {'Tanimoto':>10}   Interpretation")
    print(f"  {'─'*28} {'─'*6}  {'─'*6}  {'─'*10}   {'─'*38}")

    pair_scores = {}
    for a, b in pairs:
        fa, fb       = fp_dict[a], fp_dict[b]
        intersection = int(np.sum(fa & fb))
        union        = int(np.sum(fa | fb))
        score        = intersection / union if union > 0 else 0.0
        pair_scores[(a, b)] = score
        label        = f"{a}  ↔  {b}"
        bar_len      = int(score * 30)
        bar          = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {label:<30} {intersection:>6}  {union:>6}  {score:>10.6f}   {interpret_tanimoto(score)}")
        print(f"  {'':30} {'':6}  {'':6}  [{bar}]")
        print()

    # ── 2b: Similarity Matrix ─────────────────────────────────────────────────
    print(f"  {'─'*W}")
    print(f"  {'2b. Full Similarity Matrix (Tanimoto)':^{W}}")
    print(f"  {'─'*W}\n")

    col_w = 14
    print(f"  {'':>14}", end="")
    for n in names:
        print(f"  {n:^{col_w}}", end="")
    print()
    print(f"  {'─'*14}", end="")
    for _ in names:
        print(f"  {'─'*col_w}", end="")
    print()

    for ni in names:
        color = POLYMER_REGISTRY[ni]["color"]
        print(f"  {color}{BOLD}{ni:>14}{RESET}", end="")
        for nj in names:
            val = score_matrix[(ni, nj)]
            cell = "1.000000" if ni == nj else f"{val:.6f}"
            print(f"  {cell:^{col_w}}", end="")
        print()
    print()

    # ── 2c: Ranked Bar Chart ──────────────────────────────────────────────────
    print(f"  {'─'*W}")
    print(f"  {'2c. Ranked Similarity (High → Low)':^{W}}")
    print(f"  {'─'*W}\n")

    ranked = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, ((a, b), score) in enumerate(ranked, 1):
        bar_len = int(score * 40)
        bar     = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  #{rank}  {a}  ↔  {b}")
        print(f"       |A∩B| / |A∪B| = {int(np.sum(fp_dict[a] & fp_dict[b]))} / "
              f"{int(np.sum(fp_dict[a] | fp_dict[b]))}  →  T = {score:.6f}")
        print(f"       [{bar}]  {score*100:.2f}%")
        print(f"       {interpret_tanimoto(score)}")
        print()

    # ── 2d: Chemical Insight ──────────────────────────────────────────────────
    print(f"  {'─'*W}")
    print(f"  {'2d. Chemical Interpretation':^{W}}")
    print(f"  {'─'*W}\n")

    insights = {
        ("Polyimide", "PEEK"): [
            "Both are high-performance aromatic engineering polymers.",
            "Share para-phenylene rings, carbonyl (C=O) groups, and ether (-O-) linkages.",
            "PEEK's ether-ketone backbone and PI's imide rings cause partial divergence.",
            "Moderate-to-low Tanimoto at ECFP4 due to distinct ring environments.",
        ],
        ("Polyimide", "PTFE"): [
            "PTFE (C2F4) is a fully fluorinated alkene — no aromatic content.",
            "Polyimide is a complex aromatic heterocyclic system.",
            "No shared Morgan substructures at radius=2; near-zero score expected.",
        ],
        ("PEEK", "PTFE"): [
            "PEEK is aromatic with carbonyl and ether groups; PTFE is purely fluorinated.",
            "No overlapping ECFP4 circular fragments at any radius.",
            "Near-zero Tanimoto is chemically expected.",
        ],
    }
    for (a, b), lines in insights.items():
        sim = pair_scores.get((a, b), pair_scores.get((b, a), 0.0))
        print(f"  {a}  ↔  {b}   (T = {sim:.6f})")
        for line in lines:
            print(f"      • {line}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
#  DATAFRAME BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_morgan_dataframe(fp_dict: dict) -> pd.DataFrame:
    records = []
    for polymer_name in POLYMER_REGISTRY:
        fp     = fp_dict[polymer_name]
        prefix = polymer_name[:2]
        for i in range(N_SAMPLES_PER_POLYMER):
            row = {"Polymer": polymer_name, "Sample_ID": f"{prefix}_{i+1:04d}"}
            for j, col in enumerate(MORGAN_FEATURE_NAMES):
                row[col] = int(fp[j])
            records.append(row)
    df = pd.DataFrame(records)
    for col in MORGAN_FEATURE_NAMES:
        df[col] = df[col].astype(np.uint8)
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_banner(
        "TASK 5 — MORGAN FINGERPRINT GENERATION & TANIMOTO ANALYSIS",
        char="*", width=74
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
              f"(density = {fp.sum()/MORGAN_BITS:.3f})  "
              f"hex: {bits_to_hex(fp)}")

    # ── Section 1: Fingerprint display ───────────────────────────────────────
    print_fingerprint_report(fp_dict)

    # ── Section 2: Tanimoto similarity display ───────────────────────────────
    print_tanimoto_report(fp_dict)

    # ── Build and validate DataFrame ──────────────────────────────────────────
    morgan_df = build_morgan_dataframe(fp_dict)

    assert morgan_df.shape == (720, 66), \
        f"Shape error: {morgan_df.shape} (expected (720, 66))"
    assert list(morgan_df.columns[:2]) == ["Polymer", "Sample_ID"]
    assert list(morgan_df.columns[2:]) == MORGAN_FEATURE_NAMES
    assert morgan_df[MORGAN_FEATURE_NAMES].isin([0, 1]).all().all()

    print(f"\n  Schema verified: {morgan_df.shape[0]} rows × "
          f"{morgan_df.shape[1]} cols "
          f"(Polymer + Sample_ID + {MORGAN_BITS} bits)")

    # ── Save pkl ──────────────────────────────────────────────────────────────
    with open("morgan_fingerprints.pkl", "wb") as f:
        pickle.dump(morgan_df, f)
    print(f"  Saved : morgan_fingerprints.pkl")
    print(f"  Rows  : {morgan_df.shape[0]}   Cols : {morgan_df.shape[1]}")
    print(f"  Bits  : morgan_fp_00 ... morgan_fp_{MORGAN_BITS-1:02d}")
    print(f"  Key   : [Polymer, Sample_ID]\n")
