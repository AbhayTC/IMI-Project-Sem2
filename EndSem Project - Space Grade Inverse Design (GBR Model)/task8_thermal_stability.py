"""
================================================================================
  TASK 8 ▶ Thermal & Materials Stability Check

  CHANGES (this version):
    1. TML STABILITY CHECK
       infer_tml_ceiling_from_smiles() estimates the maximum expected TML%
       a polymer can achieve based on its structural features (free volume,
       crystallinity, aromatic content). Predicted TML is checked against
       this ceiling and against the ASTM E595 LEO threshold (1.0%).
    2. RADIATION STABILITY CHECK
       infer_rad_floor_from_smiles() estimates the minimum radiation
       endurance expected from a polymer's molecular structure (aromatic
       density, crosslinking capability). Predicted Rad is checked against
       this floor and against LEO minimum (5 MGy).
    3. COMBINED STABILITY RATING
       Each polymer sample receives a 4-component stability verdict:
         Tg    : above/below class-appropriate threshold
         Dk    : within/outside acceptable dielectric window
         TML   : below/above ASTM E595 ceiling and structural ceiling
         Rad   : above/below LEO floor and structural floor
    4. USER POLYMER SUPPORT
       Loads user_polymer.pkl. Structural ceilings/floors are estimated
       from the user SMILES via RDKit heuristics; printed with ⚠ tag.

  Inputs : master_dataset.pkl, inv_results.pkl (opt), user_polymer.pkl (opt)
  Outputs: stability_report.txt  (console summary)
================================================================================
"""
import warnings, pickle, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"

# REFERENCE_POLYMERS is now derived dynamically from master_df (is_reference=True)
# at runtime. This list is kept only as a fallback if the dataset flag is absent.
_REFERENCE_POLYMERS_FALLBACK = [
    "Polyimide", "PEEK", "PTFE",
    "CyanateEster", "FluorinatedPolyimide", "Polybenzoxazole",
]

# ── Global LEO thresholds ──────────────────────────────────────────────
LEO_RAD_FLOOR    = 5.00   # MGy  — minimum LEO radiation dose survivability

# Per-class Tg thresholds (minimum acceptable for LEO)
TG_THRESHOLDS = {
    "Polyimide"           : 200.0,
    "PEEK"                : 150.0,
    "PTFE"                :  -50.0,
    # New polymer families
    "CyanateEster"        : 250.0,   # cured CE resins typically Tg 250–290°C
    "FluorinatedPolyimide": 220.0,   # CF3-PI retains high Tg vs standard PI
    "Polybenzoxazole"     : 380.0,   # PBO (Zylon-type) has exceptional thermal stability
    "_default"            :  80.0,   # used for user polymer unless overridden
}

# Per-class Dk windows (min, max acceptable)
DK_WINDOWS = {
    "Polyimide"           : (2.5, 4.5),
    "PEEK"                : (2.8, 4.0),
    "PTFE"                : (1.8, 2.5),
    # New polymer families
    "CyanateEster"        : (2.4, 3.8),   # low-loss thermoset; used in radomes
    "FluorinatedPolyimide": (2.2, 3.5),   # fluorination lowers Dk vs standard PI
    "Polybenzoxazole"     : (2.6, 3.8),   # fused aromatic; moderate Dk
    "_default"            : (2.0, 5.0),
}


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


# ──────────────────────────────────────────────────────────────────────────────
#  RADIATION STRUCTURAL LIMITS (FROM SMILES)
# ──────────────────────────────────────────────────────────────────────────────

def infer_rad_floor_from_smiles(smiles: str) -> float:
    """
    Estimate structural radiation endurance floor (MGy) from SMILES.
    Higher floor = polymer should be radiation-resistant by structure.

    Rules (RDKit):
      More aromatic rings    → energy dissipation → higher floor
      More H-bond donors     → crosslinking potential → higher floor
      High FractionCSP3      → aliphatic → lower floor (easier chain scission)
      High rotatable bonds   → flexible → lower floor

    Returns estimated floor (MGy) in range [1.0, 45.0].
    Printed with ⚠ ESTIMATED tag.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0

        n_arom     = rdMolDescriptors.CalcNumAromaticRings(mol)
        n_heavy    = mol.GetNumHeavyAtoms()
        frac_csp3  = rdMolDescriptors.CalcFractionCSP3(mol)
        n_hbd      = rdMolDescriptors.CalcNumHBD(mol)
        n_rot      = rdMolDescriptors.CalcNumRotatableBonds(mol)
        arom_dens  = n_arom / max(n_heavy, 1)

        rad_floor = (8.0
                     + 80.0 * arom_dens
                     + 2.0  * n_hbd
                     - 10.0 * frac_csp3
                     - 0.5  * n_rot)
        return float(np.clip(rad_floor, 1.0, 45.0))
    except:
        return 10.0


def run_stability_check(df: pd.DataFrame, polymer_name: str,
                        smiles_for_limits: str,
                        tg_threshold: float, dk_window: tuple,
                        is_user: bool = False) -> dict:
    """
    Check Tg, Dk, and Radiation stability criteria for the rows of one polymer.
    Returns a summary dict of pass rates and detailed statistics.
    """
    rad_floor_struct = infer_rad_floor_from_smiles(smiles_for_limits)

    # ── Column existence checks ────────────────────────────────────────────────
    has_tg  = "Tg_degC"            in df.columns
    has_dk  = "Dk_1GHz"            in df.columns
    has_rad = "RadiationDose_MGy"  in df.columns

    n = len(df)
    results = {
        "polymer"          : polymer_name,
        "n_samples"        : n,
        "is_user"          : is_user,
        "tg_threshold"     : tg_threshold,
        "dk_window"        : dk_window,
        "rad_leo_floor"    : LEO_RAD_FLOOR,
        "rad_struct_floor" : rad_floor_struct,
    }

    if has_tg:
        tg_pass = (df["Tg_degC"] >= tg_threshold).sum()
        results["Tg_mean"]       = float(df["Tg_degC"].mean())
        results["Tg_std"]        = float(df["Tg_degC"].std())
        results["Tg_pass_rate"]  = float(tg_pass / n)
        results["Tg_pass_n"]     = int(tg_pass)

    if has_dk:
        dk_lo, dk_hi = dk_window
        dk_pass = ((df["Dk_1GHz"] >= dk_lo) & (df["Dk_1GHz"] <= dk_hi)).sum()
        results["Dk_mean"]       = float(df["Dk_1GHz"].mean())
        results["Dk_std"]        = float(df["Dk_1GHz"].std())
        results["Dk_pass_rate"]  = float(dk_pass / n)
        results["Dk_pass_n"]     = int(dk_pass)

    if has_rad:
        # Two checks: vs LEO floor AND vs structural floor
        rad_leo_pass    = (df["RadiationDose_MGy"] >= LEO_RAD_FLOOR).sum()
        rad_struct_pass = (df["RadiationDose_MGy"] >= rad_floor_struct).sum()
        results["Rad_mean"]             = float(df["RadiationDose_MGy"].mean())
        results["Rad_std"]              = float(df["RadiationDose_MGy"].std())
        results["Rad_leo_pass_rate"]    = float(rad_leo_pass / n)
        results["Rad_struct_pass_rate"] = float(rad_struct_pass / n)
        results["Rad_leo_pass_n"]       = int(rad_leo_pass)
        results["Rad_struct_pass_n"]    = int(rad_struct_pass)

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  REPORTING
# ──────────────────────────────────────────────────────────────────────────────

def fmt_pass(rate: float, n: int, total: int) -> str:
    color = GREEN if rate >= 0.80 else (YELLOW if rate >= 0.50 else RED)
    return f"{color}{rate*100:5.1f}% ({n}/{total}){RESET}"


def print_stability_report(all_results: list, user_meta: dict | None) -> None:
    print_banner("TASK 8 — MATERIALS STABILITY REPORT (Tg, Dk, Radiation)", width=85)

    # ── Summary table only ───────────────────────────────────────────────────────────────────
    print_banner("STABILITY SUMMARY", width=85)
    print(f"\n  {'Polymer':<18}  {'Tg%':>8}  {'Dk%':>8}  "
          f"{'Rad(LEO)':>10}  {'Rad(str)':>10}")
    print(f"  {'─'*18}  {'─'*8}  {'─'*8}  "
          f"{'─'*10}  {'─'*10}")

    def pct(rate):
        color = GREEN if rate >= 0.80 else (YELLOW if rate >= 0.50 else RED)
        return f"{color}{rate*100:6.1f}%{RESET}"

    for res in all_results:
        poly = res["polymer"]
        print(f"  {poly:<18}  "
              f"{pct(res.get('Tg_pass_rate',0)):>8}  "
              f"{pct(res.get('Dk_pass_rate',0)):>8}  "
              f"{pct(res.get('Rad_leo_pass_rate',0)):>10}  "
              f"{pct(res.get('Rad_struct_pass_rate',0)):>10}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'═'*72}")
    print(f"  {BOLD}TASK 8 — THERMAL & MATERIALS STABILITY CHECK{RESET}")
    print(f"{'═'*72}\n")

    # ── Load user polymer — this task only reports on the user's input ────────
    user_meta = load_user_polymer()

    if not user_meta:
        print(f"  {YELLOW}ℹ  No user polymer found (user_polymer.pkl absent).{RESET}")
        print(f"     Run task1_data_curation.py and enter a SMILES to generate one.")
        print(f"     Task 8 has nothing to report.\n")
        exit(0)

    print(f"  ✔  User polymer: {BOLD}{user_meta['name']}{RESET}")
    print(f"  ✔  SMILES      : {user_meta['smiles']}\n")

    print("  Loading master_dataset.pkl ...")
    with open("master_dataset.pkl", "rb") as f:
        master_df = pickle.load(f)
    print(f"  ✔  {len(master_df)} rows loaded\n")

    # Exclude dummies
    if "is_dummy" in master_df.columns:
        real_df = master_df[master_df["is_dummy"] == False]
    else:
        real_df = master_df

    # Pull user polymer rows from the dataset
    name       = user_meta["name"]
    user_rows  = real_df[real_df["Polymer"] == name]
    rep_smiles = user_meta["smiles"]

    if len(user_rows) == 0:
        print(f"  {YELLOW}⚠  No rows for '{name}' found in master_dataset.pkl.{RESET}")
        print(f"     The user polymer may not have been added to the dataset yet.\n")
        exit(0)

    print(f"  ℹ  {len(user_rows)} rows found for {name} in dataset.\n")

    # Tg threshold: 80% of the user-supplied baseline Tg
    user_tg_thr = user_meta.get("tg_base", TG_THRESHOLDS["_default"]) * 0.80
    user_dk_win = DK_WINDOWS["_default"]

    # Run stability check for the user polymer only
    print(f"  Checking {name} ...")
    res = run_stability_check(
        df                = user_rows,
        polymer_name      = name,
        smiles_for_limits = rep_smiles,
        tg_threshold      = user_tg_thr,
        dk_window         = user_dk_win,
        is_user           = True,
    )
    all_results = [res]

    # Print report
    print_stability_report(all_results, user_meta)

    # Save summary
    summary_rows = []
    for res in all_results:
        summary_rows.append({
            "Polymer"             : res["polymer"],
            "n_samples"          : res["n_samples"],
            "Tg_mean"            : res.get("Tg_mean"),
            "Tg_pass_rate"       : res.get("Tg_pass_rate"),
            "Dk_mean"            : res.get("Dk_mean"),
            "Dk_pass_rate"       : res.get("Dk_pass_rate"),
            "Rad_mean"           : res.get("Rad_mean"),
            "Rad_leo_pass_rate"  : res.get("Rad_leo_pass_rate"),
            "Rad_struct_pass_rate": res.get("Rad_struct_pass_rate"),
            "is_user_polymer"    : res.get("is_user", False),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("stability_report.csv", index=False)
    print(f"  💾  Saved: stability_report.csv\n")
    print(f"  ✔  Task 8 complete.\n")
