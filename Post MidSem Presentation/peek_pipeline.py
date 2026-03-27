"""
================================================================================
  PEEK - Complete Informatics & Exploratory Design Pipeline
================================================================================
"""
import warnings, hashlib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import qmc

np.random.seed(42)

POLYMER_NAME = "PEEK"
SMILES = "O=C(c1ccc(Oc2ccc(Oc3ccc(C(=O)c4ccccc4)cc3)cc2)cc1)c5ccccc5"

N_SAMPLES = 600

STRUCTURAL_FEATURE_NAMES = "MolWt HeavyAtomMolWt ExactMolWt NumHeavyAtoms NumRotatableBonds NumRings NumAromaticRings NumAliphaticRings RingCount FractionCSP3 NumHDonors NumHAcceptors TPSA MolLogP MolMR LabuteASA PEOE_VSA1 PEOE_VSA2 PEOE_VSA3 PEOE_VSA4 SMR_VSA1 SMR_VSA2 SMR_VSA3 SlogP_VSA1 SlogP_VSA2 SlogP_VSA3 NumValenceElectrons NumRadicalElectrons fr_C_O fr_NH0 fr_NH1 fr_ArN fr_Ar_COO fr_ether fr_ketone fr_imide fr_amide HallKierAlpha Kappa1 Kappa2".split()
LATENT_FEATURE_NAMES = [f"polyBERT_dim_{i+1:02d}" for i in range(40)]
PHYSICS_FEATURE_NAMES = "DegreeOfCrystallinity CrystallinePhaseContent AmorphousPhaseContent FreeVolumeFraction ChainRigidityIndex SegmentalMobility ThermalExpansionCoeff HeatCapacity_Cp ThermalDiffusivity GlassyModulus DielectricPolarizability ElectronicPolarizability IonicPolarizability OrientationalPolarizability DipoleMomentRepeat CurieWeissConstant CrosslinkingDensity EntanglementMolWt ContourLengthPerUnit PersistenceLength CharacteristicRatio ChainFlexibilityParam Mw_kDa Mn_kDa PolyDisersityIndex ZAverageMolWt ViscosityAverageMolWt NumberAverageDPn LamellaeThickness_nm SpheruliteRadius_um CrystalThickness_nm TieChainsPerArea InterfacialThickness_nm MicrostructureOrder PermittivityRealPart PermittivityImaginaryPart TanDeltaDielectric PolymerDensity_g_cm3 TensileStrength_MPa ElongationBreak_pct".split()

_MOCK_REFS = {"MolWt": 480.5, "HeavyAtomMolWt": 475.1, "ExactMolWt": 480.1, "NumHeavyAtoms": 36,"NumRotatableBonds": 6, "NumRings": 4, "NumAromaticRings": 4,"NumAliphaticRings": 0, "RingCount": 4, "FractionCSP3": 0.00,"NumHDonors": 0, "NumHAcceptors": 3, "TPSA": 39.5, "MolLogP": 5.24, "MolMR": 136.2, "LabuteASA": 194.6, "PEOE_VSA1": 22.8, "PEOE_VSA2": 11.4, "PEOE_VSA3":  8.9, "PEOE_VSA4":  6.1, "SMR_VSA1": 15.3, "SMR_VSA2": 12.1, "SMR_VSA3":  5.8, "SlogP_VSA1": 13.7, "SlogP_VSA2": 10.2, "SlogP_VSA3":  8.4, "NumValenceElectrons": 136,"NumRadicalElectrons": 0, "fr_C_O": 3, "fr_NH0": 0, "fr_NH1": 0, "fr_ArN": 0, "fr_Ar_COO": 0, "fr_ether": 2, "fr_ketone": 1, "fr_imide": 0, "fr_amide": 0, "HallKierAlpha": -2.5,"Kappa1": 16.4, "Kappa2":  8.9}

_LATENT_ANCHOR = {"scale": 0.18, "aromatic_bias": 0.70, "polar_bias": 0.30, "backbone_bias": 0.60, "sequence_bias": 0.15}

_PHYSICS_PRIORS = dict(DegreeOfCrystallinity=(0.42, 0.07), CrystallinePhaseContent=(0.42, 0.06), AmorphousPhaseContent=(0.58, 0.06), FreeVolumeFraction=(0.09, 0.02), ChainRigidityIndex=(0.88, 0.05), SegmentalMobility=(0.14, 0.02), ThermalExpansionCoeff=(4.7e-5, 5e-6), HeatCapacity_Cp=(1.32, 0.10), ThermalDiffusivity=(2.5e-7, 3e-8), GlassyModulus=(3.7, 0.4), DielectricPolarizability=(25.2, 2.2), ElectronicPolarizability=(19.8, 1.6), IonicPolarizability=(2.9, 0.3), OrientationalPolarizability=(2.5, 0.3), DipoleMomentRepeat=(3.6, 0.4), CurieWeissConstant=(180.0, 15.0), CrosslinkingDensity=(0.004, 0.001), EntanglementMolWt=(8000.0, 700.0), ContourLengthPerUnit=(1.62, 0.12), PersistenceLength=(9.8, 1.2), CharacteristicRatio=(10.5, 1.0), ChainFlexibilityParam=(0.16, 0.02), Mw_kDa=(95.0, 15.0), Mn_kDa=(48.0, 7.0), PolyDisersityIndex=(2.00, 0.22), ZAverageMolWt=(148.0, 22.0), ViscosityAverageMolWt=(88.0, 12.0), NumberAverageDPn=(220.0, 35.0), LamellaeThickness_nm=(15.0, 2.5), SpheruliteRadius_um=(8.5, 1.5), CrystalThickness_nm=(22.0, 4.0), TieChainsPerArea=(2.1e14, 2.5e13), InterfacialThickness_nm=(5.8, 0.7), MicrostructureOrder=(0.68, 0.07), PermittivityRealPart=(3.3, 0.2), PermittivityImaginaryPart=(0.06, 0.01), TanDeltaDielectric=(0.003, 0.0005), PolymerDensity_g_cm3=(1.29, 0.02), TensileStrength_MPa=(210.0, 25.0), ElongationBreak_pct=(30.0, 4.0))

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
            
        row["AmorphousPhaseContent"]    = 1.0 - row["DegreeOfCrystallinity"]
        row["CrystallinePhaseContent"]  = row["DegreeOfCrystallinity"]
        records_p.append(row)
    df_p = pd.DataFrame(records_p)

    df = pd.concat([df_s, df_l, df_p], axis=1)
    return df

def compute_targets(df):
    rng = np.random.RandomState(123)

    latent_backbone = (
        0.35 * df["polyBERT_dim_01"] +
        0.25 * df["polyBERT_dim_02"] -
        0.20 * df["polyBERT_dim_31"] -
        0.15 * df["polyBERT_dim_32"]
    )

    struct_term = (
        0.30 * (df["fr_ether"]         - 2.0) +
        0.20 * (df["fr_ketone"]        - 1.0) +
        0.10 * (df["NumAromaticRings"] - 4.0)
    )

    df["Tg_degC"] = np.clip(
        504.0
        + 280.0  * (df["ChainRigidityIndex"]    - 0.88)
        +  18.0  * (df["NumAromaticRings"]       - 4.0)
        - 600.0  * (df["FreeVolumeFraction"]     - 0.09)
        +  80.0  * (df["DegreeOfCrystallinity"]  - 0.42)
        + 1500.0 * (df["CrosslinkingDensity"]    - 0.004)
        +  10.0  * latent_backbone
        +   4.0  * struct_term
        + rng.normal(0, 2.5, len(df)),
        350, 680
    )

    df["Dk_1GHz"] = np.maximum(
        3.358
        + 0.065 * (df["DielectricPolarizability"] - 25.2)
        - 8.0   * (df["FreeVolumeFraction"]        - 0.09)
        + 2.0   * (df["DegreeOfCrystallinity"]     - 0.42)
        - 1.0   * (df["FractionCSP3"]              - 0.00)
        - 0.015 * latent_backbone
        + rng.normal(0, 0.03, len(df)),
        1.5
    )

    df["YoungModulus_GPa"] = np.maximum(
        3.66
        + 1.5   * (df["ChainRigidityIndex"]    - 0.88)
        + 0.5   * (df["DegreeOfCrystallinity"] - 0.42)
        - 3.0   * (df["FreeVolumeFraction"]    - 0.09)
        + 100.0 * (df["CrosslinkingDensity"]   - 0.004)
        + 0.05  * latent_backbone
        + rng.normal(0, 0.08, len(df)),
        0.05
    )

    df["Outgassing_TML_pct"] = np.maximum(
        0.153
        + 1.2  * (df["FreeVolumeFraction"]    - 0.09)
        + 0.25 * (df["DipoleMomentRepeat"]     - 3.6) / 10.0
        - 0.30 * (df["DegreeOfCrystallinity"]  - 0.42)
        + rng.normal(0, 0.015, len(df)),
        0.001
    )

    df["RadiationDose_MGy"] = np.maximum(
        25.0
        +  8.0   * (df["NumAromaticRings"]     - 4.0)
        + 1000.0 * (df["CrosslinkingDensity"]  - 0.004)
        -  50.0  * (df["FreeVolumeFraction"]   - 0.09)
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
    r2_scores  = [r2_score(y_test[:, i],           y_pred[:, i]) for i in range(5)]
    avg_r2 = np.mean(r2_scores)

    lo, hi = np.array([0.04, 0.10, 0.50, 15.0]), np.array([0.22, 0.80, 1.00, 30.0])
    samples = qmc.scale(qmc.LatinHypercube(d=4, seed=42).random(n=5000), lo, hi)
    mean_feats = np.mean(X, axis=0)
    X_design = np.tile(mean_feats, (5000, 1))

    physics_offset = len(STRUCTURAL_FEATURE_NAMES) + len(LATENT_FEATURE_NAMES)

    idx_fvf    = physics_offset + PHYSICS_FEATURE_NAMES.index("FreeVolumeFraction")
    idx_doc    = physics_offset + PHYSICS_FEATURE_NAMES.index("DegreeOfCrystallinity")
    idx_cri    = physics_offset + PHYSICS_FEATURE_NAMES.index("ChainRigidityIndex")
    idx_dpol   = physics_offset + PHYSICS_FEATURE_NAMES.index("DielectricPolarizability")
    idx_amorph = physics_offset + PHYSICS_FEATURE_NAMES.index("AmorphousPhaseContent")
    idx_cryst  = physics_offset + PHYSICS_FEATURE_NAMES.index("CrystallinePhaseContent")

    X_design[:, idx_fvf]    = samples[:, 0]
    X_design[:, idx_doc]    = samples[:, 1]
    X_design[:, idx_cri]    = samples[:, 2]
    X_design[:, idx_dpol]   = samples[:, 3]
    X_design[:, idx_amorph] = 1.0 - samples[:, 1]
    X_design[:, idx_cryst]  = samples[:, 1]

    preds = model.predict(scaler.transform(X_design))

    feasible_mask = (preds[:,0] > 150.0) & (preds[:,1] < 3.20) & (preds[:,2] > 2.5) & (preds[:,3] < 1.0) & (preds[:,4] > 10.0)
    n_feasible = int(np.sum(feasible_mask))

    # ── Export ────────────────────────────────────────────────────────────────
    UNIQUE_10_FEATURES = [
        "fr_ether", "fr_ketone", "MolLogP",
        "DegreeOfCrystallinity", "ThermalDiffusivity", "HeatCapacity_Cp",
        "polyBERT_dim_01", "polyBERT_dim_02",
        "polyBERT_dim_31", "polyBERT_dim_32"
    ]
    TARGET_COLS = y_cols

    export_cols = UNIQUE_10_FEATURES + TARGET_COLS
    df_export = df[export_cols].head(100)
    df_export.to_csv(f"{POLYMER_NAME}_output.csv", index=False)

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
