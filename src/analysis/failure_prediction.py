"""
Failure prediction: can we predict when GNN beats classical from graph properties?

Input: graph structural/spectral features
Target: binary — does GNN beat the best classical algorithm?
Method: logistic regression (interpretable) + random forest (accurate)

**Important**: with a heavily imbalanced target (GNN wins ~2% of the time),
raw accuracy is misleading — the majority-class baseline (always predict
"classical wins") achieves ~98%. We report balanced accuracy, F1, and the
majority baseline alongside raw accuracy so the comparison is honest.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    roc_curve, auc, classification_report,
    balanced_accuracy_score, f1_score, precision_score, recall_score,
)
from pathlib import Path


FEATURE_COLS = [
    "prop_n", "prop_m", "prop_density", "prop_avg_degree",
    "prop_max_degree", "prop_min_degree", "prop_degree_std",
    "prop_clustering_coeff", "prop_spectral_gap",
    "prop_lambda_max_laplacian", "prop_spectral_radius",
    "prop_lambda_min_adj",
]


def run_failure_prediction(df: pd.DataFrame, fig_dir: Path):
    """Build and evaluate the GNN failure prediction model.

    Reports balanced accuracy, F1, precision, and recall alongside raw
    accuracy and the majority-class baseline, so the reader can see whether
    the model beats the trivial "always predict classical wins" classifier.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Prepare features
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].copy()
    X = X.fillna(0)
    y = df["gnn_wins"].values

    # Handle any remaining issues
    X = X.replace([np.inf, -np.inf], 0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Majority-class baseline: always predict the more frequent class.
    majority_acc = max(y.mean(), 1 - y.mean())

    print(f"Feature matrix: {X_scaled.shape}")
    print(f"GNN win rate: {y.mean():.1%}")
    print(f"Majority-class baseline accuracy: {majority_acc:.3f}")

    # Logistic Regression (interpretable)
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_acc = cross_val_score(lr, X_scaled, y, cv=5, scoring="accuracy")
    lr_bal = cross_val_score(lr, X_scaled, y, cv=5, scoring="balanced_accuracy")
    lr_f1 = cross_val_score(lr, X_scaled, y, cv=5, scoring="f1")
    print(f"\nLogistic Regression (balanced class weights):")
    print(f"  CV accuracy:          {lr_acc.mean():.3f} ± {lr_acc.std():.3f}")
    print(f"  CV balanced accuracy: {lr_bal.mean():.3f} ± {lr_bal.std():.3f}")
    print(f"  CV F1:                {lr_f1.mean():.3f} ± {lr_f1.std():.3f}")
    print(f"  Majority baseline:    {majority_acc:.3f}")

    lr.fit(X_scaled, y)

    # Random Forest (accurate)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf_acc = cross_val_score(rf, X_scaled, y, cv=5, scoring="accuracy")
    rf_bal = cross_val_score(rf, X_scaled, y, cv=5, scoring="balanced_accuracy")
    rf_f1 = cross_val_score(rf, X_scaled, y, cv=5, scoring="f1")
    print(f"\nRandom Forest (balanced class weights):")
    print(f"  CV accuracy:          {rf_acc.mean():.3f} ± {rf_acc.std():.3f}")
    print(f"  CV balanced accuracy: {rf_bal.mean():.3f} ± {rf_bal.std():.3f}")
    print(f"  CV F1:                {rf_f1.mean():.3f} ± {rf_f1.std():.3f}")
    print(f"  Majority baseline:    {majority_acc:.3f}")

    rf.fit(X_scaled, y)

    # --- Figure: Feature importance ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Logistic regression coefficients
    feature_names = [c.replace("prop_", "") for c in available_cols]
    coefs = lr.coef_[0]
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    ax1.barh(range(len(coefs)), coefs[sorted_idx], color=["green" if c > 0 else "red" for c in coefs[sorted_idx]])
    ax1.set_yticks(range(len(coefs)))
    ax1.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax1.set_xlabel("Coefficient")
    ax1.set_title("Logistic Regression Coefficients\n(+) = favors GNN, (-) = favors classical")
    ax1.invert_yaxis()

    # Random forest importances
    importances = rf.feature_importances_
    sorted_idx_rf = np.argsort(importances)[::-1]
    ax2.barh(range(len(importances)), importances[sorted_idx_rf], color="steelblue")
    ax2.set_yticks(range(len(importances)))
    ax2.set_yticklabels([feature_names[i] for i in sorted_idx_rf])
    ax2.set_xlabel("Importance")
    ax2.set_title("Random Forest Feature Importance")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance.png", dpi=300)
    plt.close()
    print("Saved feature_importance.png")

    # --- Figure: ROC curve (held-out, not in-sample) ---
    from sklearn.model_selection import StratifiedKFold
    fig, ax = plt.subplots(figsize=(6, 6))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model_obj, color in [
        ("Logistic Regression", lr, "blue"),
        ("Random Forest", rf, "green"),
    ]:
        # Out-of-fold predictions for an honest ROC
        from sklearn.model_selection import cross_val_predict
        y_prob = cross_val_predict(model_obj, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC (out-of-fold): Predicting When GNN Beats Classical")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png", dpi=300)
    plt.close()
    print("Saved roc_curve.png")

    # --- Figure: GNN performance vs spectral gap ---
    fig, ax = plt.subplots(figsize=(8, 5))
    if "prop_spectral_gap" in df.columns:
        for fam in df["family"].unique():
            sub = df[df["family"] == fam]
            ax.scatter(sub["prop_spectral_gap"], sub["gnn_relative"],
                       label=fam, alpha=0.5, s=20)
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Spectral Gap (λ₂)")
        ax.set_ylabel("GNN / Best Classical")
        ax.set_title("GNN Performance vs Graph Spectral Gap")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "performance_vs_spectral_gap.png", dpi=300)
    plt.close()
    print("Saved performance_vs_spectral_gap.png")

    return {
        "lr_accuracy": lr_acc.mean(),
        "lr_balanced_accuracy": lr_bal.mean(),
        "lr_f1": lr_f1.mean(),
        "rf_accuracy": rf_acc.mean(),
        "rf_balanced_accuracy": rf_bal.mean(),
        "rf_f1": rf_f1.mean(),
        "majority_baseline": majority_acc,
        "top_lr_features": [(feature_names[i], float(coefs[i]))
                           for i in sorted_idx[:5]],
        "top_rf_features": [(feature_names[i], float(importances[i]))
                           for i in sorted_idx_rf[:5]],
    }
