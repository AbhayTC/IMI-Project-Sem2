import pandas as pd
import numpy as np
import os

# =========================
# SETTINGS
# =========================
BASE_DIR = "/home/vishnu-sai/Polymer_Project_Submission"

polymer_files = [
    os.path.join(BASE_DIR, "Polyimide_feature_bank.csv"),
    os.path.join(BASE_DIR, "PEEK_feature_bank.csv"),
    os.path.join(BASE_DIR, "PTFE_feature_bank.csv")
]

# =========================
# COLUMNS USED TO BUILD PSEUDO-MORGAN FEATURES
# (must exist in your datasets)
# =========================
base_cols = [
    "Mol_Length_Index",
    "Atom_Count_Score",
    "Ring_Complexity",
    "Aromaticity_Index",
    "Heteroatom_Fraction",
    "Flexibility_Score",
    "Rigidity_Score"
]

# =========================
# FUNCTION TO CREATE 5 MORGAN-LIKE FEATURES
# =========================
def add_pseudo_morgan(df):
    for col in base_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for fingerprint generation: {col}")

    # Create 5 compressed fingerprint-like binary bits
    df["MorganFP_1"] = (
        (df["Mol_Length_Index"] * 7 + df["Atom_Count_Score"] * 3 + df["Ring_Complexity"] * 11)
        .round().astype(int) % 2
    )

    df["MorganFP_2"] = (
        (df["Aromaticity_Index"] * 100 + df["Heteroatom_Fraction"] * 200)
        .round().astype(int) % 2
    )

    df["MorganFP_3"] = (
        (df["Flexibility_Score"] * 10 + df["Rigidity_Score"] * 9)
        .round().astype(int) % 2
    )

    df["MorganFP_4"] = (
        (df["Mol_Length_Index"] * 5 + df["Aromaticity_Index"] * 50 + df["Flexibility_Score"] * 4)
        .round().astype(int) % 2
    )

    df["MorganFP_5"] = (
        (df["Atom_Count_Score"] * 13 + df["Rigidity_Score"] * 6 + df["Heteroatom_Fraction"] * 100)
        .round().astype(int) % 2
    )

    return df

# =========================
# PROCESS FILES
# =========================
for file_path in polymer_files:
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    # If already 45 columns and Morgan columns exist, skip safely
    if all(col in df.columns for col in ["MorganFP_1", "MorganFP_2", "MorganFP_3", "MorganFP_4", "MorganFP_5"]):
        print(f"⚠ Morgan features already exist in: {os.path.basename(file_path)}")
        continue

    if df.shape[1] != 40:
        raise ValueError(f"{os.path.basename(file_path)} must have exactly 40 columns before update. Found {df.shape[1]}")

    df = add_pseudo_morgan(df)

    # final safety check
    if df.shape[1] != 45:
        raise ValueError(f"Failed to create 45 columns for {os.path.basename(file_path)}. Found {df.shape[1]}")

    df.to_csv(file_path, index=False)
    print(f"✅ Updated {os.path.basename(file_path)} to 45 columns.")

print("\n🎯 All polymer datasets updated successfully.")
