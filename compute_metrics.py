"""
Compute & cache all model-performance metrics for the dashboard.
Run once:  python compute_metrics.py
Produces: static/metrics.json
"""

import json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    log_loss, matthews_corrcoef, cohen_kappa_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings("ignore")

# ---- Load model & data ----
model = pickle.load(open("model.pkl", "rb"))
df = pd.read_csv(r"C:\Users\ADMIN\Downloads\archive (18)\loan_approval_dataset.csv")

# Clean columns
df.columns = df.columns.str.strip()

# Strip whitespace from all string columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Encode target
df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

# Feature engineering (matching notebook)
df["total_assets_value"] = (
    df["luxury_assets_value"] +
    df["residential_assets_value"] +
    df["commercial_assets_value"] +
    df["bank_asset_value"]
)
df["loan_to_income"] = df["loan_amount"] / df["income_annum"]
df["loan_to_asset"] = df["loan_amount"] / df["total_assets_value"]
df["asset_to_income"] = df["total_assets_value"] / df["income_annum"]

# Drop ID & target
feature_cols = [c for c in df.columns if c not in ("loan_id", "loan_status")]
X = df[feature_cols]
y = df["loan_status"]

# ---- Predictions ----
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

# ---- Classification metrics ----
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
mcc = matthews_corrcoef(y, y_pred)
kappa = cohen_kappa_score(y, y_pred)
logloss = log_loss(y, y_proba)
specificity = recall_score(y, y_pred, pos_label=0)

# ---- Confusion matrix ----
cm = confusion_matrix(y, y_pred).tolist()

# ---- ROC curve ----
fpr, tpr, roc_thresholds = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

# Subsample the ROC curve for JSON (limit to ~100 points)
step = max(1, len(fpr) // 100)
roc_data = {
    "fpr": fpr[::step].tolist(),
    "tpr": tpr[::step].tolist(),
    "auc": round(roc_auc, 4)
}

# ---- Precision-Recall curve ----
pr_precision, pr_recall, _ = precision_recall_curve(y, y_proba)
avg_prec = average_precision_score(y, y_proba)
step_pr = max(1, len(pr_precision) // 100)
pr_data = {
    "precision": pr_precision[::step_pr].tolist(),
    "recall": pr_recall[::step_pr].tolist(),
    "avg_precision": round(avg_prec, 4)
}

# ---- Cross-validation (5-fold) ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")
cv_precision = cross_val_score(model, X, y, cv=cv, scoring="precision")
cv_recall = cross_val_score(model, X, y, cv=cv, scoring="recall")

cv_data = {
    "accuracy": {"scores": cv_accuracy.tolist(), "mean": round(cv_accuracy.mean(), 4), "std": round(cv_accuracy.std(), 4)},
    "f1": {"scores": cv_f1.tolist(), "mean": round(cv_f1.mean(), 4), "std": round(cv_f1.std(), 4)},
    "precision": {"scores": cv_precision.tolist(), "mean": round(cv_precision.mean(), 4), "std": round(cv_precision.std(), 4)},
    "recall": {"scores": cv_recall.tolist(), "mean": round(cv_recall.mean(), 4), "std": round(cv_recall.std(), 4)},
}

# ---- Feature importance ----
# Get feature names after transformation
try:
    transformer = model.named_steps["preprocess"]
    xgb_model = model.named_steps["model"]
    importances = xgb_model.feature_importances_

    # Get transformed feature names
    ohe_features = list(transformer.transformers_[0][1].get_feature_names_out(["education", "self_employed"]))
    scaler_features = transformer.transformers_[1][2]
    all_features = ohe_features + list(scaler_features)

    feat_imp = sorted(
        zip(all_features, importances.tolist()),
        key=lambda x: x[1], reverse=True
    )
    feature_importance = {
        "names": [f[0] for f in feat_imp],
        "values": [round(f[1], 4) for f in feat_imp]
    }
except Exception:
    feature_importance = {"names": feature_cols, "values": [0] * len(feature_cols)}

# ---- Class distribution ----
class_dist = {
    "labels": ["Rejected (0)", "Approved (1)"],
    "counts": [int(y.value_counts().get(0, 0)), int(y.value_counts().get(1, 0))]
}

# ---- CIBIL score distribution by class ----
cibil_approved = df[df["loan_status"] == 1]["cibil_score"].tolist()
cibil_rejected = df[df["loan_status"] == 0]["cibil_score"].tolist()

# Histogram bins
bins = list(range(300, 920, 50))
hist_approved, _ = np.histogram(cibil_approved, bins=bins)
hist_rejected, _ = np.histogram(cibil_rejected, bins=bins)

cibil_dist = {
    "bins": [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)],
    "approved": hist_approved.tolist(),
    "rejected": hist_rejected.tolist()
}

# ---- Prediction distribution (probability histogram) ----
prob_bins = np.linspace(0, 1, 21)
prob_hist_approved, _ = np.histogram(y_proba[y == 1], bins=prob_bins)
prob_hist_rejected, _ = np.histogram(y_proba[y == 0], bins=prob_bins)

pred_dist = {
    "bins": [f"{prob_bins[i]:.2f}" for i in range(len(prob_bins) - 1)],
    "approved": prob_hist_approved.tolist(),
    "rejected": prob_hist_rejected.tolist()
}

# ---- Model info ----
model_info = {
    "name": "XGBClassifier (Pipeline)",
    "algorithm": "XGBoost",
    "n_estimators": 157,
    "max_depth": 9,
    "learning_rate": 0.1712,
    "subsample": 0.768,
    "total_samples": len(df),
    "n_features": len(feature_cols)
}

# ---- Assemble ----
metrics = {
    "scores": {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "mcc": round(mcc, 4),
        "kappa": round(kappa, 4),
        "log_loss": round(logloss, 4),
        "roc_auc": round(roc_auc, 4),
        "avg_precision": round(avg_prec, 4)
    },
    "confusion_matrix": cm,
    "roc_curve": roc_data,
    "pr_curve": pr_data,
    "cross_validation": cv_data,
    "feature_importance": feature_importance,
    "class_distribution": class_dist,
    "cibil_distribution": cibil_dist,
    "prediction_distribution": pred_dist,
    "model_info": model_info
}

with open("static/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Metrics saved to static/metrics.json")
print(f"   Accuracy:  {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall:    {rec:.4f}")
print(f"   F1 Score:  {f1:.4f}")
print(f"   ROC AUC:   {roc_auc:.4f}")
print(f"   MCC:       {mcc:.4f}")
