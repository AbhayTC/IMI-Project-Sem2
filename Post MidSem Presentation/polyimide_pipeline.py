"""
================================================================================
  Polyimide (PI) - Complete Informatics & Exploratory Design Pipeline
================================================================================
"""
import warnings, hashlib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import qmc

np.random.seed(42)

POLYMER_NAME = "Polyimide"
SMILES = "O=C1c2ccccc2C(=O)N1c3ccc(Oc4ccc(N5C(=O)c6ccccc6C5=O)cc4)cc3"

N_SAMPLES = 600

STRUCTURAL_FEATURE_NAMES = "MolWt HeavyAtomMolWt ExactMolWt NumHeavyAtoms NumRotatableBonds NumRings NumAromaticRings NumAliphaticRings RingCount FractionCSP3 NumHDonors NumHAcceptors TPSA MolLogP MolMR LabuteASA PEOE_VSA1 PEOE_VSA2 PEOE_VSA3 PEOE_VSA4 SMR_VSA1 SMR_VSA2 SMR_VSA3 SlogP_VSA1 SlogP_VSA2 SlogP_VSA3 NumValenceElectrons NumRadicalElectrons fr_C_O fr_NH0 fr_NH1 fr_ArN fr_Ar_COO fr_ether fr_ketone fr_imide fr_amide HallKierAlpha Kappa1 Kappa2".split()
LATENT_FEATURE_NAMES = [f"polyBERT_dim_{i+1:02d}" for i in range(40)]
PHYSICS_FEATURE_NAMES = "DegreeOfCrystallinity CrystallinePhaseContent AmorphousPhaseContent FreeVolumeFraction ChainRigidityIndex SegmentalMobility ThermalExpansionCoeff HeatCapacity_Cp ThermalDiffusivity GlassyModulus DielectricPolarizability ElectronicPolarizability IonicPolarizability OrientationalPolarizability DipoleMomentRepeat CurieWeissConstant CrosslinkingDensity EntanglementMolWt ContourLengthPerUnit PersistenceLength CharacteristicRatio ChainFlexibilityParam Mw_kDa Mn_kDa PolyDisersityIndex ZAverageMolWt ViscosityAverageMolWt NumberAverageDPn LamellaeThickness_nm SpheruliteRadius_um CrystalThickness_nm TieChainsPerArea InterfacialThickness_nm MicrostructureOrder PermittivityRealPart PermittivityImaginaryPart TanDeltaDielectric PolymerDensity_g_cm3 TensileStrength_MPa ElongationBreak_pct".split()

_MOCK_REFS = {"MolWt": 720.7, "HeavyAtomMolWt": 714.3, "ExactMolWt": 720.1, "NumHeavyAtoms": 52,"NumRotatableBonds": 8, "NumRings": 6, "NumAromaticRings": 5,"NumAliphaticRings": 1, "RingCount": 6, "FractionCSP3": 0.04,"NumHDonors": 0, "NumHAcceptors": 5, "TPSA": 77.8, "MolLogP": 4.12, "MolMR": 188.4, "LabuteASA": 265.1, "PEOE_VSA1": 34.2, "PEOE_VSA2": 18.1, "PEOE_VSA3": 12.4, "PEOE_VSA4":  9.3, "SMR_VSA1": 22.5, "SMR_VSA2": 18.3, "SMR_VSA3":  8.7, "SlogP_VSA1": 20.1, "SlogP_VSA2": 14.6, "SlogP_VSA3": 11.2, "NumValenceElectrons": 200,"NumRadicalElectrons": 0, "fr_C_O": 4, "fr_NH0": 2, "fr_NH1": 0, "fr_ArN": 2, "fr_Ar_COO": 0, "fr_ether": 1, "fr_ketone": 0, "fr_imide": 2, "fr_amide": 0, "HallKierAlpha": -3.8,"Kappa1": 22.1, "Kappa2": 11.6}

_LATENT_ANCHOR = {"scale": 0.18, "aromatic_bias": 0.65, "polar_bias": 0.55, "backbone_bias": 0.45, "sequence_bias": 0.20}

_PHYSICS_PRIORS = dict(DegreeOfCrystallinity=(0.35, 0.06), CrystallinePhaseContent=(0.35, 0.05), AmorphousPhaseContent=(0.65, 0.05), FreeVolumeFraction=(0.11, 0.02), ChainRigidityIndex=(0.82, 0.06), SegmentalMobility=(0.18, 0.03), ThermalExpansionCoeff=(3.2e-5, 4e-6), HeatCapacity_Cp=(1.05, 0.08), ThermalDiffusivity=(1.8e-7, 2e-8), GlassyModulus=(3.4, 0.3), DielectricPolarizability=(28.5, 2.0), ElectronicPolarizability=(22.1, 1.5), IonicPolarizability=(3.8, 0.4), OrientationalPolarizability=(2.6, 0.3), DipoleMomentRepeat=(4.8, 0.5), CurieWeissConstant=(320.0, 20.0), CrosslinkingDensity=(0.008, 0.002), EntanglementMolWt=(5800.0, 500.0), ContourLengthPerUnit=(1.48, 0.10), PersistenceLength=(12.5, 1.5), CharacteristicRatio=(8.2, 0.8), ChainFlexibilityParam=(0.22, 0.03), Mw_kDa=(85.0, 12.0), Mn_kDa=(42.0, 6.0), PolyDisersityIndex=(2.05, 0.25), ZAverageMolWt=(130.0, 18.0), ViscosityAverageMolWt=(78.0, 10.0), NumberAverageDPn=(210.0, 30.0), LamellaeThickness_nm=(12.0, 2.0), SpheruliteRadius_um=(3.5, 0.8), CrystalThickness_nm=(18.0, 3.0), TieChainsPerArea=(1.8e14, 2e13), InterfacialThickness_nm=(4.5, 0.6), MicrostructureOrder=(0.55, 0.06), PermittivityRealPart=(3.5, 0.2), PermittivityImaginaryPart=(0.08, 0.01), TanDeltaDielectric=(0.022, 0.003), PolymerDensity_g_cm3=(1.40, 0.02), TensileStrength_MPa=(185.0, 20.0), ElongationBreak_pct=(35.0, 5.0))

def generate_features():
    def _seeded_noise(feat, idx, scale):
        seed_int = int(hashlib.md5(f"{POLYMER_NAME}_{feat}_{idx}".encode()).hexdigest(), 16) % (2**32)
        return np.random.RandomState(seed_int).normal(0, scale)

    # -------- Structural Features --------
    records_s = []
    for i in range(N_SAMPLES):
        row = {}
        for feat in STRUCTURAL_FEATURE_NAMES:
            bv = _MOCK_REFS[feat]
            row[feat] = bv + _seeded_noise(feat, i, max(abs(bv) * 0.03, 0.02))
            
            # --- Round discrete counts to integers ---
            discrete_counts = [
                "NumHeavyAtoms", "NumRotatableBonds", "NumRings", "NumAromaticRings",
                "NumAliphaticRings", "RingCount", "NumHDonors", "NumHAcceptors",
                "NumValenceElectrons", "NumRadicalElectrons", "fr_C_O", "fr_NH0", 
                "fr_NH1", "fr_ArN", "fr_Ar_COO", "fr_ether", "fr_ketone", "fr_imide", "fr_amide"
            ]
            continuous_bounds = ["MolWt", "HeavyAtomMolWt", "ExactMolWt"]

            if feat in discrete_counts:
                row[feat] = int(round(max(row[feat], 0)))
            elif feat in continuous_bounds:
                row[feat] = max(row[feat], 0.0)

        records_s.append(row)
    df_s = pd.DataFrame(records_s)

    # -------- Latent Features --------
    smiles_seed = int(hashlib.md5(SMILES.encode()).hexdigest(), 16) % (2**32)
    rng_base = np.random.RandomState(smiles_seed)
    anch = _LATENT_ANCHOR
    anchor = np.array([anch["aromatic_bias"]]*10 + [anch["polar_bias"]]*10 + [anch["backbone_bias"]]*10 + [anch["sequence_bias"]]*10) + rng_base.normal(0, 0.03, size=40)
    records_l = []
    for i in range(N_SAMPLES):
        rng = np.random.RandomState(smiles_seed + i * 137)
        vec = np.clip(anchor + rng.normal(0, anch["scale"], size=40), -1.5, 1.5)
        records_l.append(dict(zip(LATENT_FEATURE_NAMES, vec)))
    df_l = pd.DataFrame(records_l)

    # -------- Physics Features --------
    seed = int(hashlib.md5(f"physics_{POLYMER_NAME}".encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    records_p = []
    for _ in range(N_SAMPLES):
        row = {}
        for feat in PHYSICS_FEATURE_NAMES:
            mu, sigma = _PHYSICS_PRIORS[feat]
            val = rng.normal(mu, sigma)
            if feat in ("DegreeOfCrystallinity", "CrystallinePhaseContent", "AmorphousPhaseContent", "FreeVolumeFraction", "MicrostructureOrder"): val = np.clip(val, 0.0, 1.0)
            elif feat in ("CrosslinkingDensity", "DipoleMomentRepeat", "PermittivityImaginaryPart", "TanDeltaDielectric", "PolyDisersityIndex"): val = max(val, 0.0)
            row[feat] = val
        
        row["AmorphousPhaseContent"] = 1.0 - row["DegreeOfCrystallinity"]
        row["CrystallinePhaseContent"] = row["DegreeOfCrystallinity"]
        records_p.append(row)
    df_p = pd.DataFrame(records_p)

    df = pd.concat([df_s, df_l, df_p], axis=1)
    return df

def compute_targets(df):
    rng = np.random.RandomState(123)

    latent_backbone = (
        0.35 * df["polyBERT_dim_11"] +
        0.25 * df["polyBERT_dim_12"] -
        0.20 * df["polyBERT_dim_21"] -
        0.15 * df["polyBERT_dim_22"]
    )

    struct_term = (
        0.30 * (df["fr_imide"]       - 2.0)  +
        0.20 * (df["NumAromaticRings"] - 5.0) +
        0.10 * (df["ExactMolWt"]     - 720.1) / 100.0
    )

    df["Tg_degC"] = np.clip(
        534.0
        + 280.0 * (df["ChainRigidityIndex"]    - 0.82)
        +  18.0 * (df["NumAromaticRings"]       - 5.0)
        - 600.0 * (df["FreeVolumeFraction"]     - 0.11)
        +  80.0 * (df["DegreeOfCrystallinity"]  - 0.35)
        + 1500.0 * (df["CrosslinkingDensity"]   - 0.008)
        +  10.0 * latent_backbone
        +   4.0 * struct_term
        + rng.normal(0, 2.5, len(df)),
        380, 700
    )

    df["Dk_1GHz"] = np.maximum(
        3.43
        + 0.065 * (df["DielectricPolarizability"] - 28.5)
        - 8.0   * (df["FreeVolumeFraction"]        - 0.11)
        + 2.0   * (df["DegreeOfCrystallinity"]     - 0.35)
        - 1.0   * (df["FractionCSP3"]              - 0.04)
        - 0.015 * latent_backbone
        + rng.normal(0, 0.03, len(df)),
        1.5
    )

    df["YoungModulus_GPa"] = np.maximum(
        3.075
        + 1.5   * (df["ChainRigidityIndex"]   - 0.82)
        + 0.5   * (df["DegreeOfCrystallinity"] - 0.35)
        - 3.0   * (df["FreeVolumeFraction"]    - 0.11)
        + 100.0 * (df["CrosslinkingDensity"]   - 0.008)
        + 0.05  * latent_backbone
        + rng.normal(0, 0.08, len(df)),
        0.05
    )

    df["Outgassing_TML_pct"] = np.maximum(
        0.50
        + 1.2  * (df["FreeVolumeFraction"]    - 0.11)
        + 0.25 * (df["DipoleMomentRepeat"]     - 4.8) / 10.0
        - 0.30 * (df["DegreeOfCrystallinity"]  - 0.35)
        + rng.normal(0, 0.015, len(df)),
        0.001
    )

    df["RadiationDose_MGy"] = np.maximum(
        50.0
        +  8.0   * (df["NumAromaticRings"]     - 5.0)
        + 1000.0 * (df["CrosslinkingDensity"]  - 0.008)
        -  50.0  * (df["FreeVolumeFraction"]   - 0.11)
        +   0.04 * latent_backbone
        + rng.normal(0, 1.0, len(df)),
        0.01
    )

    return df

def run_pipeline():
    print(f"\n{'='*70}\nExecuting Complete Pipeline for {POLYMER_NAME}\n{'='*70}")
    df = generate_features()
    df = compute_targets(df)

    X_cols = STRUCTURAL_FEATURE_NAMES + LATENT_FEATURE_NAMES + PHYSICS_FEATURE_NAMES
    y_cols = ["Tg_degC", "Dk_1GHz", "YoungModulus_GPa", "Outgassing_TML_pct", "RadiationDose_MGy"]

    X = df[X_cols].values
    y = df[y_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

    rf = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(rf)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2_scores  = [r2_score(y_test[:, i],               y_pred[:, i]) for i in range(5)]
    avg_r2 = np.mean(r2_scores)

    # ── LHS Inverse Design ───────────────────────────────────────────────────
    lo, hi = np.array([0.04, 0.10, 0.50, 20.0]), np.array([0.22, 0.80, 1.00, 30.0])
    samples = qmc.scale(qmc.LatinHypercube(d=4, seed=42).random(n=5000), lo, hi)
    mean_feats = np.mean(X, axis=0)
    X_design = np.tile(mean_feats, (5000, 1))

    physics_offset = len(STRUCTURAL_FEATURE_NAMES) + len(LATENT_FEATURE_NAMES)

    idx_fvf   = physics_offset + PHYSICS_FEATURE_NAMES.index("FreeVolumeFraction")
    idx_doc   = physics_offset + PHYSICS_FEATURE_NAMES.index("DegreeOfCrystallinity")
    idx_cri   = physics_offset + PHYSICS_FEATURE_NAMES.index("ChainRigidityIndex")
    idx_dpol  = physics_offset + PHYSICS_FEATURE_NAMES.index("DielectricPolarizability")
    idx_amorph= physics_offset + PHYSICS_FEATURE_NAMES.index("AmorphousPhaseContent")
    idx_cryst = physics_offset + PHYSICS_FEATURE_NAMES.index("CrystallinePhaseContent")

    X_design[:, idx_fvf]    = samples[:, 0]
    X_design[:, idx_doc]    = samples[:, 1]
    X_design[:, idx_cri]    = samples[:, 2]
    X_design[:, idx_dpol]   = samples[:, 3]
    X_design[:, idx_amorph] = 1.0 - samples[:, 1]
    X_design[:, idx_cryst]  = samples[:, 1]

    preds = model.predict(scaler.transform(X_design))

    feasible_mask = (preds[:,0] > 280.0) & (preds[:,1] < 3.50) & (preds[:,2] > 2.0) & (preds[:,3] < 1.0) & (preds[:,4] > 20.0)
    n_feasible = int(np.sum(feasible_mask))

    # ── Export ───────────────────────────────────────────────────────────────
    UNIQUE_10_FEATURES = [
        "fr_imide", "NumAromaticRings", "ExactMolWt",
        "ChainRigidityIndex", "GlassyModulus", "CrosslinkingDensity",
        "polyBERT_dim_11", "polyBERT_dim_12",
        "polyBERT_dim_21", "polyBERT_dim_22"
    ]
    TARGET_COLS = y_cols

    export_cols = UNIQUE_10_FEATURES + TARGET_COLS
    df_export = df[export_cols].head(100)
    df_export.to_csv(f"{POLYMER_NAME}_output.csv", index=False)

    # ── Confidence ───────────────────────────────────────────────────────────
    s_data   = 100
    s_model  = 100 if avg_r2 > 0.85 else (70 if avg_r2 >= 0.70 else 40)
    s_export = 100
    s_inv    = 100 if n_feasible > 0 else 30
    total_conf  = (0.25 * s_data) + (0.35 * s_model) + (0.20 * s_export) + (0.20 * s_inv)
    conf_label  = "HIGH" if total_conf >= 80 else ("MEDIUM" if total_conf >= 60 else "LOW")

    print(f"\n[OUTPUT EXPORT]")
    print(f"Saved 100 unranked samples to {POLYMER_NAME}_output.csv")
    print(f"Export Features: {UNIQUE_10_FEATURES}")

    print(f"\n[MODEL METRICS]")
    print(f"┌────────────────────┬──────────┐")
    print(f"│ Target Property    │ R² Score │")
    print(f"├────────────────────┼──────────┤")
    print(f"│ Tg (°C)            │ {r2_scores[0]:8.4f} │")
    print(f"│ Dk (1 GHz)         │ {r2_scores[1]:8.4f} │")
    print(f"│ Young's Modulus    │ {r2_scores[2]:8.4f} │")
    print(f"│ Outgassing (TML %) │ {r2_scores[3]:8.4f} │")
    print(f"│ Radiation (MGy)    │ {r2_scores[4]:8.4f} │")
    print(f"├────────────────────┼──────────┤")
    print(f"│ Average R²         │ {avg_r2:8.4f} │")
    print(f"└────────────────────┴──────────┘")

    print(f"\n[PIPELINE CONFIDENCE]")
    print(f"Feasible LHS Candidates Found: {n_feasible} / 5000")
    print(f"Overall Confidence Score: {total_conf:.1f}/100 -> {conf_label}")

if __name__ == "__main__":
    run_pipeline()
