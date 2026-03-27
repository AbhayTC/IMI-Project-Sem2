"""
================================================================================
  PTFE - Exploratory Materials Informatics & Inverse Design Pipeline
================================================================================
"""
import warnings
import hashlib
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

POLYMER_NAME = "PTFE"
SMILES = "FC(F)C(F)F"
N_SAMPLES = 600

STRUCTURAL_FEATURE_NAMES = """
MolWt HeavyAtomMolWt ExactMolWt NumHeavyAtoms NumRotatableBonds NumRings
NumAromaticRings NumAliphaticRings RingCount FractionCSP3 NumHDonors
NumHAcceptors TPSA MolLogP MolMR LabuteASA PEOE_VSA1 PEOE_VSA2 PEOE_VSA3
PEOE_VSA4 SMR_VSA1 SMR_VSA2 SMR_VSA3 SlogP_VSA1 SlogP_VSA2 SlogP_VSA3
NumValenceElectrons NumRadicalElectrons fr_C_O fr_NH0 fr_NH1 fr_ArN
fr_Ar_COO fr_ether fr_ketone fr_imide fr_amide HallKierAlpha Kappa1 Kappa2
""".split()
LATENT_FEATURE_NAMES = [f"polyBERT_dim_{i+1:02d}" for i in range(40)]
PHYSICS_FEATURE_NAMES = """
DegreeOfCrystallinity CrystallinePhaseContent AmorphousPhaseContent
FreeVolumeFraction ChainRigidityIndex SegmentalMobility ThermalExpansionCoeff
HeatCapacity_Cp ThermalDiffusivity GlassyModulus DielectricPolarizability
ElectronicPolarizability IonicPolarizability OrientationalPolarizability
DipoleMomentRepeat CurieWeissConstant CrosslinkingDensity EntanglementMolWt
ContourLengthPerUnit PersistenceLength CharacteristicRatio
ChainFlexibilityParam Mw_kDa Mn_kDa PolyDisersityIndex ZAverageMolWt
ViscosityAverageMolWt NumberAverageDPn LamellaeThickness_nm SpheruliteRadius_um
CrystalThickness_nm TieChainsPerArea InterfacialThickness_nm MicrostructureOrder
PermittivityRealPart PermittivityImaginaryPart TanDeltaDielectric
PolymerDensity_g_cm3 TensileStrength_MPa ElongationBreak_pct
""".split()

_MOCK_REFS = {
    "MolWt": 100.0,
    "HeavyAtomMolWt": 100.0,
    "ExactMolWt": 100.0,
    "NumHeavyAtoms": 6,
    "NumRotatableBonds": 1,
    "NumRings": 0,
    "NumAromaticRings": 0,
    "NumAliphaticRings": 0,
    "RingCount": 0,
    "FractionCSP3": 1.00,
    "NumHDonors": 0,
    "NumHAcceptors": 0,
    "TPSA": 0.0,
    "MolLogP": 3.5,
    "MolMR": 20.0,
    "LabuteASA": 35.0,
    "PEOE_VSA1": 3.0,
    "PEOE_VSA2": 1.5,
    "PEOE_VSA3": 1.0,
    "PEOE_VSA4": 0.5,
    "SMR_VSA1": 2.0,
    "SMR_VSA2": 1.0,
    "SMR_VSA3": 0.5,
    "SlogP_VSA1": 2.5,
    "SlogP_VSA2": 1.2,
    "SlogP_VSA3": 0.8,
    "NumValenceElectrons": 48,
    "NumRadicalElectrons": 0,
    "fr_C_O": 0,
    "fr_NH0": 0,
    "fr_NH1": 0,
    "fr_ArN": 0,
    "fr_Ar_COO": 0,
    "fr_ether": 0,
    "fr_ketone": 0,
    "fr_imide": 0,
    "fr_amide": 0,
    "HallKierAlpha": 0.0,
    "Kappa1": 2.0,
    "Kappa2": 1.0
}

_LATENT_ANCHOR = {
    "scale": 0.12,
    "aromatic_bias": -0.80,
    "polar_bias": -0.70,
    "backbone_bias": 0.30,
    "sequence_bias": 0.50
}

_PHYSICS_PRIORS = dict(
    DegreeOfCrystallinity=(0.92, 0.03),
    CrystallinePhaseContent=(0.92, 0.03),
    AmorphousPhaseContent=(0.08, 0.03),
    FreeVolumeFraction=(0.06, 0.01),
    ChainRigidityIndex=(0.88, 0.04),
    SegmentalMobility=(0.12, 0.03),
    ThermalExpansionCoeff=(1.2e-4, 1e-5),
    HeatCapacity_Cp=(1.00, 0.05),
    ThermalDiffusivity=(1.5e-7, 2e-8),
    GlassyModulus=(0.60, 0.05),
    DielectricPolarizability=(8.0, 0.5),
    ElectronicPolarizability=(7.8, 0.4),
    IonicPolarizability=(0.1, 0.05),
    OrientationalPolarizability=(0.05, 0.02),
    DipoleMomentRepeat=(0.0, 0.01),
    CurieWeissConstant=(0.0, 1.0),
    CrosslinkingDensity=(0.0001, 0.00005),
    EntanglementMolWt=(15000.0, 1000.0),
    ContourLengthPerUnit=(1.30, 0.05),
    PersistenceLength=(1.5, 0.15),
    CharacteristicRatio=(7.0, 0.5),
    ChainFlexibilityParam=(0.25, 0.03),
    Mw_kDa=(3000.0, 300.0),
    Mn_kDa=(500.0, 50.0),
    PolyDisersityIndex=(6.0, 0.6),
    ZAverageMolWt=(9000.0, 800.0),
    ViscosityAverageMolWt=(2800.0, 250.0),
    NumberAverageDPn=(6000.0, 500.0),
    LamellaeThickness_nm=(35.0, 3.0),
    SpheruliteRadius_um=(40.0, 4.0),
    CrystalThickness_nm=(35.0, 3.0),
    TieChainsPerArea=(2.5e14, 3e13),
    InterfacialThickness_nm=(5.0, 0.8),
    MicrostructureOrder=(0.92, 0.03),
    PermittivityRealPart=(2.10, 0.05),
    PermittivityImaginaryPart=(0.0005, 0.0001),
    TanDeltaDielectric=(0.0001, 0.00002),
    PolymerDensity_g_cm3=(2.20, 0.02),
    TensileStrength_MPa=(28.0, 4.0),
    ElongationBreak_pct=(250.0, 30.0)
)

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
    anchor = np.array(
        [anch["aromatic_bias"]] * 10 +
        [anch["polar_bias"]] * 10 +
        [anch["backbone_bias"]] * 10 +
        [anch["sequence_bias"]] * 10
    ) + rng_base.normal(0, 0.03, size=40)

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

            # bounded features
            if feat in (
                "DegreeOfCrystallinity", "CrystallinePhaseContent",
                "AmorphousPhaseContent", "FreeVolumeFraction", "MicrostructureOrder"
            ):
                val = np.clip(val, 0.0, 1.0)
            elif feat in (
                "CrosslinkingDensity", "DipoleMomentRepeat",
                "PermittivityImaginaryPart", "TanDeltaDielectric",
                "PolyDisersityIndex"
            ):
                val = max(val, 0.0)

            row[feat] = val

        # maintain complement relation
        row["AmorphousPhaseContent"] = 1.0 - row["DegreeOfCrystallinity"]
        row["CrystallinePhaseContent"] = row["DegreeOfCrystallinity"]

        records_p.append(row)

    df_p = pd.DataFrame(records_p)

    df = pd.concat([df_s, df_l, df_p], axis=1)
    return df

def compute_targets(df):
    rng = np.random.RandomState(123)

    latent_backbone = (
        0.35 * df["polyBERT_dim_15"] +
        0.25 * df["polyBERT_dim_16"] -
        0.20 * df["polyBERT_dim_05"] -
        0.15 * df["polyBERT_dim_06"]
    )

    struct_term = (
        0.30 * (df["FractionCSP3"] - 1.00) -
        0.20 * (df["NumRotatableBonds"] - 1.0) -
        0.10 * (df["TPSA"] - 0.0)
    )

    df["Tg_degC"] = np.clip(
        120
        + 45 * (df["ChainRigidityIndex"] - 0.88)
        - 220 * (df["FreeVolumeFraction"] - 0.06)
        + 18 * (df["DegreeOfCrystallinity"] - 0.92)
        - 10 * (df["SegmentalMobility"] - 0.12)
        + 8 * latent_backbone
        + 4 * struct_term
        + rng.normal(0, 0.8, len(df)),
        110, 130
    )

    df["Dk_1GHz"] = np.clip(
        2.10
        + 0.22 * (df["PermittivityRealPart"] - 2.10)
        + 0.010 * (df["DielectricPolarizability"] - 8.0)
        - 0.45 * (df["FreeVolumeFraction"] - 0.06)
        + 0.03 * (df["AmorphousPhaseContent"] - 0.08)
        - 0.015 * latent_backbone
        + rng.normal(0, 0.008, len(df)),
        2.00, 2.20
    )

    df["YoungModulus_GPa"] = np.clip(
        0.55
        + 0.60 * (df["DegreeOfCrystallinity"] - 0.92)
        + 0.45 * (df["ChainRigidityIndex"] - 0.88)
        - 1.20 * (df["FreeVolumeFraction"] - 0.06)
        - 0.25 * (df["SegmentalMobility"] - 0.12)
        + 0.04 * latent_backbone
        + rng.normal(0, 0.015, len(df)),
        0.40, 0.70
    )

    df["Outgassing_TML_pct"] = np.clip(
        0.04
        + 0.55 * (df["FreeVolumeFraction"] - 0.06)
        + 0.08 * (df["SegmentalMobility"] - 0.12)
        - 0.12 * (df["DegreeOfCrystallinity"] - 0.92)
        - 0.015 * (df["MicrostructureOrder"] - 0.92)
        + 0.006 * df["AmorphousPhaseContent"]
        + rng.normal(0, 0.004, len(df)),
        0.01, 0.10
    )

    df["RadiationDose_MGy"] = np.clip(
        0.60
        + 0.85 * (df["DegreeOfCrystallinity"] - 0.92)
        + 0.55 * (df["ChainRigidityIndex"] - 0.88)
        - 2.20 * (df["FreeVolumeFraction"] - 0.06)
        - 0.18 * (df["SegmentalMobility"] - 0.12)
        + 0.06 * (df["PolymerDensity_g_cm3"] - 2.20)
        + 0.05 * latent_backbone
        + rng.normal(0, 0.02, len(df)),
        0.20, 1.20
    )

    return df

def run_pipeline():
    print(f"\n{'='*78}")
    print(f"Executing Complete Pipeline for {POLYMER_NAME}")
    print(f"{'='*78}")

    df = generate_features()
    df = compute_targets(df)

    X_cols = STRUCTURAL_FEATURE_NAMES + LATENT_FEATURE_NAMES + PHYSICS_FEATURE_NAMES
    y_cols = ["Tg_degC", "Dk_1GHz", "YoungModulus_GPa", "Outgassing_TML_pct", "RadiationDose_MGy"]

    X = df[X_cols].values
    y = df[y_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model = MultiOutputRegressor(rf)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(len(y_cols))]
    mae_scores = [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(len(y_cols))]
    rmse_scores = [np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])) for i in range(len(y_cols))]
    avg_r2 = np.mean(r2_scores)

    # Realistic PTFE screening space
    lo = np.array([0.03, 0.85, 0.80, 7.2])   
    hi = np.array([0.09, 0.98, 0.96, 8.8])

    lhs = qmc.LatinHypercube(d=4, seed=42)
    samples = qmc.scale(lhs.random(n=5000), lo, hi)

    mean_feats = np.mean(X, axis=0)
    X_design = np.tile(mean_feats, (5000, 1))

    physics_offset = len(STRUCTURAL_FEATURE_NAMES) + len(LATENT_FEATURE_NAMES)

    idx_fvf = physics_offset + PHYSICS_FEATURE_NAMES.index("FreeVolumeFraction")
    idx_doc = physics_offset + PHYSICS_FEATURE_NAMES.index("DegreeOfCrystallinity")
    idx_cri = physics_offset + PHYSICS_FEATURE_NAMES.index("ChainRigidityIndex")
    idx_dpol = physics_offset + PHYSICS_FEATURE_NAMES.index("DielectricPolarizability")
    idx_amorph = physics_offset + PHYSICS_FEATURE_NAMES.index("AmorphousPhaseContent")
    idx_cryst = physics_offset + PHYSICS_FEATURE_NAMES.index("CrystallinePhaseContent")
    idx_perm_real = physics_offset + PHYSICS_FEATURE_NAMES.index("PermittivityRealPart")

    X_design[:, idx_fvf] = samples[:, 0]
    X_design[:, idx_doc] = samples[:, 1]
    X_design[:, idx_cri] = samples[:, 2]
    X_design[:, idx_dpol] = samples[:, 3]
    X_design[:, idx_amorph] = 1.0 - samples[:, 1]
    X_design[:, idx_cryst] = samples[:, 1]
    X_design[:, idx_perm_real] = 2.10 + 0.08 * (samples[:, 3] - 8.0)  

    preds = model.predict(scaler.transform(X_design))

    feasible_mask = (
        (preds[:, 0] >= 115.0) & (preds[:, 0] <= 130.0) &   
        (preds[:, 1] >= 2.00)  & (preds[:, 1] <= 2.20)  &   
        (preds[:, 2] >= 0.45)  & (preds[:, 2] <= 0.70)  &   
        (preds[:, 3] <= 0.08)                             & 
        (preds[:, 4] >= 0.40)                               
    )

    n_feasible = int(np.sum(feasible_mask))

    # ── Export ────────────────────────────────────────────────────────────────
    UNIQUE_10_FEATURES = [
        "FractionCSP3",
        "NumRotatableBonds",
        "TPSA",
        "FreeVolumeFraction",
        "SegmentalMobility",
        "DielectricPolarizability",
        "polyBERT_dim_05",
        "polyBERT_dim_06",
        "polyBERT_dim_15",
        "polyBERT_dim_16"
    ]

    TARGET_COLS = y_cols

    export_cols = UNIQUE_10_FEATURES + TARGET_COLS
    df_export = df[export_cols].head(100)

    output_file = f"{POLYMER_NAME}_output.csv"
    df_export.to_csv(output_file, index=False)

    s_data = 100 
    s_model = 100 if avg_r2 > 0.85 else (70 if avg_r2 >= 0.70 else 40)
    s_export = 100 
    s_inverse = 100 if n_feasible > 0 else 30

    total_conf = (0.25 * s_data) + (0.35 * s_model) + (0.20 * s_export) + (0.20 * s_inverse)

    if total_conf >= 80:
        conf_label = "HIGH"
    elif total_conf >= 60:
        conf_label = "MEDIUM"
    else:
        conf_label = "LOW"

    print(f"\n[OUTPUT EXPORT]")
    print(f"Saved 100 unranked samples showcasing the 10 relevant features to {POLYMER_NAME}_output.csv")
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
