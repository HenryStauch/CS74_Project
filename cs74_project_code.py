# 1. Connect to Drive, extract data from zip files
# Note: Data were downloaded from https://zenodo.org/records/15095445 to local 
# device and uploaded to Colab before extraction

import os, zipfile, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

PREPROCESS_ZIP = "../preprocess.zip"
EXTRACT_DIR = "../ee_data"
os.makedirs(EXTRACT_DIR, exist_ok=True)

print("Extracting Estonia (EE00*) files from preprocess.zip...")
with zipfile.ZipFile(PREPROCESS_ZIP, 'r') as zf:
    print("Sample paths inside zip:", zf.namelist()[:5])
    # Files are named NUTS3region_parcelID_EC_hcat_c.npz (e.g., EE001_12345_3301010100.npz)
    ee_members = [
        m for m in zf.namelist()
        if not m.endswith("/") and (
            os.path.basename(m).startswith("EE")
            or "/EE/" in m
            or "meta" in m.lower()
        )
    ]
    for member in tqdm(ee_members, desc="Unzipping EE"):
        zf.extract(member, EXTRACT_DIR)

npz_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, "**", "*.npz"), recursive=True))
print(f"Total Estonia .npz parcel files extracted: {len(npz_files):,}")



# 2. Parse filenames and data structure
# Inspect keys of the first .npz file to adapt automatically to exact schema
sample = np.load(npz_files[0], allow_pickle=True)
print("Sample filename:", os.path.basename(npz_files[0]))
print("Keys inside .npz:", list(sample.keys()))
for k in sample.keys():
    print(f"  {k}: shape={sample[k].shape}, dtype={sample[k].dtype}")

# Helper to parse parcel metadata from EuroCropsML filename: <REGION>_<ID>_<HCAT_CODE>.npz
# (Falls back to internal .npz keys if named differently)
records = []
for fp in npz_files:
    fname = os.path.splitext(os.path.basename(fp))[0]
    parts = fname.split("_")
    # Standard EuroCropsML naming: e.g. EE001_123456_3301010100.npz
    region = parts[0] if len(parts) >= 3 else "EE"
    hcat_code = parts[-1] if len(parts) >= 2 else "unknown"
    records.append({"filepath": fp, "region": region, "hcat_code": hcat_code})

meta_df = pd.DataFrame(records)

# Filter to Top 12 most frequent crop classes in Estonia (keeps classes well-populated & fast)
TOP_K_CLASSES = 12
top_classes = meta_df["hcat_code"].value_counts().head(TOP_K_CLASSES).index.tolist()
meta_df = meta_df[meta_df["hcat_code"].isin(top_classes)].reset_index(drop=True)

# Map HCAT codes to clean integer labels 0..K-1 (and readable names if lookup CSV exists)
code_to_idx = {c: i for i, c in enumerate(top_classes)}
meta_df["label"] = meta_df["hcat_code"].map(code_to_idx)

# Optional: load human-readable crop names if meta_data CSV was in the zip
name_map = {c: str(c) for c in top_classes}
meta_csvs = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True)
for csv_path in meta_csvs:
    try:
        tmp = pd.read_csv(csv_path)
        if "EC_hcat_c" in tmp.columns and "EC_hcat_n" in tmp.columns:
            lookup = dict(zip(tmp["EC_hcat_c"].astype(str), tmp["EC_hcat_n"].astype(str)))
            name_map.update({c: lookup.get(str(c), str(c)) for c in top_classes})
            break
    except Exception:
        pass

class_names = [name_map[c][:22] for c in top_classes]
print(f"\nFiltered to Top {TOP_K_CLASSES} classes ({len(meta_df):,} parcels):")
for c, name in zip(top_classes, class_names):
    print(f"  {c} ({name}): {(meta_df['hcat_code'] == c).sum():,} parcels")

# Load all filtered parcel arrays into memory once (~1-2 mins, makes all time-block loops instant)
# EuroCropsML .npz stores 12 Sentinel-2 bands (B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12) and dates
data_key = next(k for k in sample.keys() if k in ["data", "X", "ts", "bands"] or sample[k].ndim == 2)
date_key = next((k for k in sample.keys() if k in ["dates", "timestamps", "doy", "time"] and k != data_key), None)
coord_key = next((k for k in sample.keys() if k in ["coords", "center", "centroid", "latlon"]), None)

all_ts = []
all_months = []
coords_list = []

for fp in tqdm(meta_df["filepath"], desc="Loading parcel arrays into RAM"):
    arr = np.load(fp, allow_pickle=True)
    ts = arr[data_key].astype(np.float32)  # shape: (T, 12)
    all_ts.append(ts)
    
    if date_key is not None:
        raw_dates = arr[date_key]
        # Handle string dates ('2021-05-14'), datetime64, or integer Day-Of-Year
        if np.issubdtype(raw_dates.dtype, np.integer):
            months = np.clip((raw_dates - 1) // 30 + 1, 1, 12)
        else:
            months = pd.to_datetime(raw_dates.astype(str)).month.values
    else:
        # Fallback if dates key wasn't matched: evenly space across 1..12
        months = np.linspace(1, 12, len(ts)).astype(int)
    all_months.append(months)
    
    if coord_key is not None:
        coords_list.append(arr[coord_key][:2])

# Create 25km spatial grid blocks unconditionally from coordinates to ensure clean 70/15/15 spatial splits
if len(coords_list) == len(meta_df):
    print("Making custom 25km spatial blocks from parcel coordinates...")
    coords_arr = np.array(coords_list)
    grid_x = (coords_arr[:, 0] // 0.35).astype(int)
    grid_y = (coords_arr[:, 1] // 0.20).astype(int)
    meta_df["region"] = [f"BLK_{x}_{y}" for x, y in zip(grid_x, grid_y)]

print(f"Spatial split groups (NUTS-3 / Spatial Blocks): {meta_df['region'].nunique()}")




# 3. Split into train, validation, and test
# 70% train, 15% validation, 15% test grouped by spatial region
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_val_idx, test_idx = next(gss_test.split(meta_df, meta_df["label"], groups=meta_df["region"]))

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42) # 0.1765 * 0.85 ≈ 0.15
sub_train_idx, sub_val_idx = next(
    gss_val.split(meta_df.iloc[train_val_idx], meta_df["label"].iloc[train_val_idx], groups=meta_df["region"].iloc[train_val_idx])
)
train_idx = train_val_idx[sub_train_idx]
val_idx   = train_val_idx[sub_val_idx]

print(f"Spatial Split Sizes -> Train: {len(train_idx):,} ({len(train_idx)/len(meta_df)*100:.1f}%) | Val: {len(val_idx):,} ({len(val_idx)/len(meta_df)*100:.1f}%) | Test: {len(test_idx):,} ({len(test_idx)/len(meta_df)*100:.1f}%)")

# Stratified subsampling helper for Train split so RBF-SVC fits quickly while keeping rare classes
def get_balanced_train_indices(indices, labels, max_per_class=1500, seed=42):
    rng = np.random.default_rng(seed)
    selected = []
    for cls in np.unique(labels[indices]):
        cls_idx = indices[labels[indices] == cls]
        if len(cls_idx) > max_per_class:
            cls_idx = rng.choice(cls_idx, size=max_per_class, replace=False)
        selected.append(cls_idx)
    return np.concatenate(selected)

y_all = meta_df["label"].values
svc_train_idx = get_balanced_train_indices(train_idx, y_all, max_per_class=1500)
print(f"Stratified SVC Train subset size: {len(svc_train_idx):,} parcels")




# 4. Construct month-by-month feature sets
def build_features_up_to_month(cutoff_month):
    """
    Constructs fixed-length monthly median spectral + vegetation index features
    using ONLY observations from Month 1 up to `cutoff_month`.
    Bands in Sentinel-2 L1C (13): [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]
    Indices: B3=Green (idx 2), B4=Red (idx 3), B8=NIR (idx 7)
    """
    n_parcels = len(all_ts)
    n_bands = all_ts[0].shape[1]  # Automatically detects 13 bands
    feats_per_month = n_bands + 2  # 13 bands + NDVI + GNDVI = 15
    X = np.full((n_parcels, cutoff_month * feats_per_month), np.nan, dtype=np.float32)
    
    for i in range(n_parcels):
        ts = all_ts[i]
        months = all_months[i]
        valid = months <= cutoff_month
        if not np.any(valid):
            continue
        ts_sub = ts[valid]
        m_sub = months[valid]
        
        for m in range(1, cutoff_month + 1):
            m_mask = (m_sub == m)
            if np.any(m_mask):
                med = np.median(ts_sub[m_mask], axis=0)
                green, red, nir = med[2], med[3], med[7]
                ndvi = (nir - red) / (nir + red + 1e-6)
                gndvi = (nir - green) / (nir + green + 1e-6)
                start = (m - 1) * feats_per_month
                X[i, start:start + n_bands] = med
                X[i, start + n_bands] = ndvi
                X[i, start + n_bands + 1] = gndvi
                
    # Impute missing cloud-covered months using training-set column medians (no leakage!)
    col_medians = np.nanmedian(X[train_idx], axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    nan_rows, nan_cols = np.where(np.isnan(X))
    X[nan_rows, nan_cols] = col_medians[nan_cols]
    return X




# 5. Train, validate, test within each time block
TIME_BLOCKS = {
    "Apr 30 (M4)": 4,
    "May 31 (M5)": 5,
    "Jun 30 (M6)": 6,
    "Jul 31 (M7)": 7,
    "Full Year (M12)": 12,
}

results = []
saved_predictions = {}  # Store test predictions for failure analysis

for label_name, cutoff_m in TIME_BLOCKS.items():
    print(f"\n--- Running Time Block: {label_name} ---")
    X = build_features_up_to_month(cutoff_m)
    
    X_train_full, y_train_full = X[train_idx], y_all[train_idx]
    X_train_svc,  y_train_svc  = X[svc_train_idx], y_all[svc_train_idx]
    X_val,        y_val        = X[val_idx], y_all[val_idx]
    X_test,       y_test       = X[test_idx], y_all[test_idx]
    
    # 2. RBF-SVC (quick validation tuning over C)
    best_svc, best_val_f1, best_c = None, -1.0, 1.0
    for c_val in [1.0, 10.0]:
        candidate = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=c_val, gamma="scale", class_weight="balanced")
        )
        candidate.fit(X_train_svc, y_train_svc)
        val_pred = candidate.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_svc = candidate
            best_c = c_val
            
    # Evaluate the model on Held-Out Spatial Test Set
    models = {
        f"RBF-SVC (C={best_c})": best_svc,
    }
    
    for m_name, model in models.items():
        y_pred = model.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)
        results.append({
            "Time Block": label_name,
            "Month": cutoff_m,
            "Model": m_name.split(" (")[0],
            "Val Macro-F1": round(best_val_f1, 4) if "SVC" in m_name else np.nan,
            "Test Macro-F1": round(macro_f1, 4),
            "Test Weighted-F1": round(weighted_f1, 4),
            "Test Accuracy": round(acc, 4),
        })
        if "SVC" in m_name:
            saved_predictions[label_name] = (y_test, y_pred)
        print(f"  {m_name:32s} | Test Macro-F1: {macro_f1:.4f} | Test Acc: {acc:.4f}")

results_df = pd.DataFrame(results)
display(results_df)
results_df.to_csv("/content/estonia_early_season_results.csv", index=False)





# 6. Figure generation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Filter results table to only RBF-SVC
svc_df = results_df[results_df["Model"].str.contains("SVC")].copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot 1: RBF-SVC Performance Across Growing-Season Cutoffs ---
metrics_to_plot = ["Test Macro-F1", "Test Weighted-F1", "Test Accuracy"]
svc_melted = svc_df.melt(
    id_vars=["Time Block", "Month"],
    value_vars=metrics_to_plot,
    var_name="Metric",
    value_name="Score",
)

sns.lineplot(
    data=svc_melted,
    x="Time Block",
    y="Score",
    hue="Metric",
    style="Metric",
    markers=True,
    markersize=9,
    linewidth=2.5,
    ax=axes[0],
)

# Annotate Macro-F1 values directly above the curve for easy slide reading
for _, row in svc_df.iterrows():
  axes[0].annotate(
      f"{row['Test Macro-F1']:.3f}",
      (row["Time Block"], row["Test Macro-F1"]),
      textcoords="offset points",
      xytext=(0, 8),
      ha="center",
      fontsize=9.5,
      fontweight="bold",
      color="#1f77b4",
  )

axes[0].set_title(
    "RBF-SVC Early-Season Crop Forecasting Performance",
    fontsize=13,
    fontweight="bold",
)
axes[0].set_ylabel("Held-Out Spatial Test Score")
axes[0].set_xlabel("Forecasting Horizon (Observations Used)")
axes[0].set_ylim(0, 1.0)
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(title="Evaluation Metric", loc="lower right")

# --- Plot 2: Normalized Confusion Matrix at May 31 (Failure Analysis) ---
target_block = "May 31 (M5)"  # Change to "Jun 30 (M6)" if you prefer June
y_true_m5, y_pred_m5 = saved_predictions[target_block]
cm = confusion_matrix(y_true_m5, y_pred_m5, normalize="true")

sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[1],
    cbar=False,
)
axes[1].set_title(
    f"RBF-SVC Normalized Confusion Matrix ({target_block})",
    fontsize=13,
    fontweight="bold",
)
axes[1].set_xlabel("Predicted Crop Class")
axes[1].set_ylabel("True Crop Class")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("/content/svc_presentation_charts.png", dpi=200)
plt.show()

# Print clean SVC-only summary table for copy-pasting into slides
print("\n=== RBF-SVC SUMMARY TABLE ===")
display(
    svc_df[[
        "Time Block",
        "Model",
        "Val Macro-F1",
        "Test Macro-F1",
        "Test Weighted-F1",
        "Test Accuracy",
    ]]
)
