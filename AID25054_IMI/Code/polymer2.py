import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================
# SETTINGS
# =========================
BASE_DIR = "/home/vishnu-sai/Polymer_Project_Submission"
RANDOM_SEED = 42
TARGET_TG = 350.0

np.random.seed(RANDOM_SEED)

# =========================
# INPUT FILES
# =========================
polymer_files = {
    "Polyimide": os.path.join(BASE_DIR, "Polyimide_feature_bank.csv"),
    "PEEK": os.path.join(BASE_DIR, "PEEK_feature_bank.csv"),
    "PTFE": os.path.join(BASE_DIR, "PTFE_feature_bank.csv")
}

# =========================
# OUTPUT FILES
# =========================
master_output = os.path.join(BASE_DIR, "master_space_polymer_dataset.csv")
report_output = os.path.join(BASE_DIR, "tg_prediction_report.csv")
inverse_output = os.path.join(BASE_DIR, "inverse_design_spacegrade.csv")

# =========================
# REQUIRED FEATURES
# =========================
required_features = [
    "Crystallinity_percent",
    "Annealing_Temp_C",
    "Molecular_Weight_kDa",
    "Thermal_Stability_Index"
]

# =========================
# POLYMER OFFSETS FOR Tg
# =========================
polymer_offsets = {
    "Polyimide": 85,
    "PEEK": 55,
    "PTFE": 35
}

# =========================
# LOAD DATASETS
# =========================
all_dfs = []

for polymer_name, file_path in polymer_files.items():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)

    if df.shape[0] != 100:
        raise ValueError(f"{polymer_name} file must have 100 rows. Found {df.shape[0]} rows.")

    if df.shape[1] != 45:
        raise ValueError(f"{polymer_name} file must have 45 columns. Found {df.shape[1]} columns.")

    for feature in required_features:
        if feature not in df.columns:
            raise ValueError(f"Missing required feature '{feature}' in {polymer_name} dataset.")

    df["Sample_ID"] = [f"{polymer_name[:3].upper()}_{i+1:03d}" for i in range(len(df))]
    df["Polymer_Type"] = polymer_name

    all_dfs.append(df)

# =========================
# COMBINE ALL DATA
# =========================
master_df = pd.concat(all_dfs, ignore_index=True)

# =========================
# GENERATE Tg TARGET
# =========================
def calculate_tg(row):
    polymer = row["Polymer_Type"]
    offset = polymer_offsets.get(polymer, 40.0)

    tg = (
        1.0 * row["Crystallinity_percent"] +
        0.55 * row["Annealing_Temp_C"] +
        0.30 * row["Molecular_Weight_kDa"] +
        1.0 * row["Thermal_Stability_Index"] +
        offset +
        np.random.normal(0, 2)
    )

    return tg

master_df["Tg"] = master_df.apply(calculate_tg, axis=1)

# Save master dataset
master_df.to_csv(master_output, index=False)

# =========================
# MACHINE LEARNING DATA
# =========================
feature_columns = [col for col in master_df.columns if col not in ["Sample_ID", "Polymer_Type", "Tg"]]

X = master_df[feature_columns]
y = master_df["Tg"]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, master_df.index, test_size=0.2, random_state=RANDOM_SEED
)

# =========================
# TRAIN MODEL
# =========================
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# EVALUATION METRICS
# =========================
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== MODEL PERFORMANCE =====")
print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# =========================
# SAVE PREDICTION REPORT
# =========================
report_df = pd.DataFrame({
    "Sample_ID": master_df.loc[idx_test, "Sample_ID"].values,
    "Polymer_Type": master_df.loc[idx_test, "Polymer_Type"].values,
    "Actual_Tg": y_test.values,
    "Predicted_Tg": y_pred,
    "Absolute_Error": np.abs(y_test.values - y_pred)
})

report_df.to_csv(report_output, index=False)

# =========================
# INVERSE DESIGN
# =========================
inverse_results = []

for polymer_name in polymer_files.keys():
    polymer_data = master_df[master_df["Polymer_Type"] == polymer_name].copy()

    # Use median values of this polymer as base
    median_features = polymer_data[feature_columns].median()

    best_error = float("inf")
    best_pred = None
    best_anneal = None
    best_cryst = None

    # Search ranges from actual polymer data
    anneal_min = polymer_data["Annealing_Temp_C"].min()
    anneal_max = polymer_data["Annealing_Temp_C"].max()

    cryst_min = polymer_data["Crystallinity_percent"].min()
    cryst_max = polymer_data["Crystallinity_percent"].max()

    anneal_values = np.linspace(anneal_min, anneal_max, 20)
    cryst_values = np.linspace(cryst_min, cryst_max, 20)

    for anneal in anneal_values:
        for cryst in cryst_values:
            test_vector = median_features.copy()
            test_vector["Annealing_Temp_C"] = anneal
            test_vector["Crystallinity_percent"] = cryst

            test_df = pd.DataFrame([test_vector], columns=feature_columns)
            pred_tg = model.predict(test_df)[0]
            error = abs(pred_tg - TARGET_TG)

            if error < best_error:
                best_error = error
                best_pred = pred_tg
                best_anneal = anneal
                best_cryst = cryst

    inverse_results.append({
        "Polymer": polymer_name,
        "Tg": TARGET_TG,
        "Optimal crystallinity": round(best_cryst, 4),
        "Optimal annealingtemp": round(best_anneal, 4),
        "predicted Tg": round(best_pred, 4)
    })

inverse_df = pd.DataFrame(inverse_results)
inverse_df.to_csv(inverse_output, index=False)

# =========================
# DISPLAY INVERSE DESIGN OUTPUT
# =========================
print("\n===== INVERSE DESIGN OUTPUT =====")
print(inverse_df.to_string(index=False))

# =========================
# FILES CREATED
# =========================
print("\n===== FILES CREATED =====")
print(master_output)
print(report_output)
print(inverse_output)

print("\n🎯 Final pipeline completed successfully.")
