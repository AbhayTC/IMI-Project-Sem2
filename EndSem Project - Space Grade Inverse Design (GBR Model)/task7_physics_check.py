"""
================================================================================
  TASK 7 ▶ Physics Consistency Check

  CHANGES (this version):
    1. USER-INPUT INTEGRATION:
       Automatically detects and appends the user-defined polymer from 
       user_polymer.pkl to the consistency report.
    2. TARGET-SPECIFIC COLOURS:
       - Tg columns (Thermal)     : Cyan   (↑ and ↓)
       - Dk columns (Dielectric)  : Green  (↑ and ↓)
       - Rad columns (Radiation)  : Yellow (↑ and ↓)
    3. CLEAN REPORTING: 
       Removed the misleading "Score = fraction..." legend. The table now 
       functions as a "Chemical Signature" analysis tool.
================================================================================
"""
import warnings, pickle, os
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ANSI Color Codes
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"

REFERENCE_POLYMERS = [
    "Polyimide", "PEEK", "PTFE",
    "CyanateEster", "FluorinatedPolyimide", "Polybenzoxazole",
]

def load_user_polymer() -> dict | None:
    """Loads metadata for the custom polymer if it exists."""
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
#  PHYSICS RULE DEFINITIONS (Logic remains unchanged)
# ──────────────────────────────────────────────────────────────────────────────

TG_HIGH_RULES = [
    ("ChainRigidityIndex",       "high", 1.0, "Rigid backbone → high Tg"),
    ("NumAromaticRings",         "high", 0.8, "Aromatic rings stiffen chain"),
    ("DegreeOfCrystallinity",    "high", 0.6, "Crystallinity raises Tg"),
    ("CrosslinkingDensity",      "high", 0.9, "Crosslinks restrict motion"),
    ("FreeVolumeFraction",       "low",  0.7, "Low free vol → less mobility"),
]
TG_LOW_RULES = [
    ("ChainFlexibilityParam",    "high", 1.0, "Flexible chains → low Tg"),
    ("FractionCSP3",             "high", 0.8, "High sp3 → aliphatic flexibility"),
    ("FreeVolumeFraction",       "high", 0.7, "High free vol → plasticising"),
]

DK_LOW_RULES = [
    ("FreeVolumeFraction",       "high", 1.0, "Air voids reduce Dk"),
    ("DipoleMomentRepeat",       "low",  1.0, "Less dipole → less polarisation"),
    ("FractionCSP3",             "high", 0.6, "High sp3 → low Dk"),
]
DK_HIGH_RULES = [
    ("DielectricPolarizability", "high", 1.0, "High polarisability → high Dk"),
    ("IonicPolarizability",      "high", 0.8, "Ionic groups dominate Dk"),
    ("DipoleMomentRepeat",       "high", 1.0, "Strong dipole → high Dk"),
]

RAD_HIGH_RULES = [
    ("NumAromaticRings",         "high", 1.0, "Aromatic resonance energy dissipation"),
    ("DegreeOfCrystallinity",    "high", 0.8, "Ordered packing resists scission"),
    ("CrosslinkingDensity",      "high", 0.9, "Crosslinks maintain integrity"),
]
RAD_LOW_RULES = [
    ("FreeVolumeFraction",       "high", 1.0, "Free vol radical pathways"),
    ("NumAromaticRings",         "low",  1.0, "Weak energy dissipation"),
    ("SegmentalMobility",        "high", 0.8, "Damage propagation via mobility"),
]

ALL_RULE_SETS = {
    "Tg_high": TG_HIGH_RULES, "Tg_low": TG_LOW_RULES,
    "Dk_low" : DK_LOW_RULES,  "Dk_high": DK_HIGH_RULES,
    "Rad_high": RAD_HIGH_RULES, "Rad_low": RAD_LOW_RULES,
}

# ──────────────────────────────────────────────────────────────────────────────
#  CHECK ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def compute_rule_scores(df: pd.DataFrame, rules: list, global_means: dict, global_stds: dict) -> pd.Series:
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-2.0 * x))
    total_weight = sum(w for _, _, w, _ in rules)
    scores = np.zeros(len(df))
    for feat, direction, weight, _ in rules:
        if feat not in df.columns: continue
        z = (df[feat].values - global_means[feat]) / (global_stds[feat] or 1.0)
        if direction == "low": z = -z
        scores += weight * sigmoid(z)
    return pd.Series(scores / total_weight, index=df.index)

def check_physics(master_df: pd.DataFrame, polymer_name: str, global_means: dict, global_stds: dict) -> dict:
    mask = (master_df["Polymer"] == polymer_name)
    if "is_dummy" in master_df.columns: mask &= (master_df["is_dummy"] == False)
    df = master_df[mask]
    if len(df) == 0: return {}
    return {name: float(compute_rule_scores(df, rules, global_means, global_stds).mean()) 
            for name, rules in ALL_RULE_SETS.items()}

# ──────────────────────────────────────────────────────────────────────────────
#  REPORTING
# ──────────────────────────────────────────────────────────────────────────────

def fmt(v, color): return f"{color}{v:.3f}{RESET}"

def print_physics_report(all_polymers: list, results: dict, user_meta: dict | None) -> None:
    print_banner("TASK 7 — PHYSICS CONSISTENCY CHECK", width=85)

    # ── Target-Specific Colored Header ────────────────────────────────────────
    header = (f"  {'Polymer':<18}  "
              f"{CYAN}{'Tg↑':>8}  {'Tg↓':>8}{RESET}  "
              f"{GREEN}{'Dk↓':>8}  {'Dk↑':>8}{RESET}  "
              f"{YELLOW}{'Rad↑':>8}  {'Rad↓':>8}{RESET}")
    print(f"\n{header}")
    print(f"  {'─'*18}  " + "  ".join(['─'*8]*6))

    for poly in all_polymers:
        r = results.get(poly, {})
        if not r: continue
        tag = f"  {BOLD}(user){RESET}" if (user_meta and poly == user_meta["name"]) else ""
        
        # Displaying columns with Cyan (Thermal), Green (Dielectric), Yellow (Radiation)
        print(f"  {poly:<18}  "
              f"{fmt(r.get('Tg_high',0), CYAN):>8}  "
              f"{fmt(r.get('Tg_low',0), CYAN):>8}  "
              f"{fmt(r.get('Dk_low',0), GREEN):>8}  "
              f"{fmt(r.get('Dk_high',0), GREEN):>8}  "
              f"{fmt(r.get('Rad_high',0), YELLOW):>8}  "
              f"{fmt(r.get('Rad_low',0), YELLOW):>8}{tag}")

    
    print()

if __name__ == "__main__":
    if not os.path.exists("master_dataset.pkl"):
        print("  ✗ Error: master_dataset.pkl not found. Run Task 1 first.")
        exit(1)

    with open("master_dataset.pkl", "rb") as f:
        master_df = pickle.load(f)

    user_meta = load_user_polymer()
    all_polymers = REFERENCE_POLYMERS.copy()
    
    # Dynamically include user polymer if it exists
    if user_meta and user_meta["name"] not in all_polymers:
        all_polymers.append(user_meta["name"])

    # Global baseline calculation (Reference rows only)
    real_df = master_df[master_df["is_reference"] == True] if "is_reference" in master_df.columns else master_df
    num_df  = real_df.select_dtypes(include=[np.number])
    g_means, g_stds = num_df.mean().to_dict(), num_df.std().to_dict()

    results_by_poly = {p: check_physics(master_df, p, g_means, g_stds) for p in all_polymers}
    print_physics_report(all_polymers, results_by_poly, user_meta)
