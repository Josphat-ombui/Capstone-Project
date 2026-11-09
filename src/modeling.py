"""Batch quality classifier (Module A): time split, model suite, thresholds, explanations."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (ADOPTION_MARGIN, MIN_RECALL, RANDOM_STATE, TARGET_COL, TEST_YEARS, TRAIN_YEARS,
                        VALIDATION_YEARS)
from src.features import FEATURE_LABELS, build_features, feature_columns

BASELINE = "Logistic regression (baseline)"


# ----------------------------------------------------------------------------- split
def split_by_time(batches):
    """Train 2007-2021, validate 2022-2023, test 2024-2026."""
    parts = {}
    for name, (start, end) in (("train", TRAIN_YEARS), ("validation", VALIDATION_YEARS), ("test", TEST_YEARS)):
        parts[name] = batches[batches["year"].between(start, end)]

    def describe():
        rows = []
        for name, label, years in (("train", "Train", TRAIN_YEARS), ("validation", "Validation", VALIDATION_YEARS),
                                   ("test", "Test", TEST_YEARS)):
            part = parts[name]
            rows.append({"partition": label, "period": f"{years[0]}-{years[1]}", "batches": len(part),
                         "failed": int(part[TARGET_COL].sum()), "failure_rate": part[TARGET_COL].mean()})
        return pd.DataFrame(rows)

    return SimpleNamespace(**parts, describe=describe)


# ----------------------------------------------------------------------------- models
def _models():
    models = {
        BASELINE: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(class_weight="balanced", max_iter=2000)),
        "Random forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE)),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost (gradient boosting)"] = make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=30, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1))
    except Exception:   # xgboost missing or unable to load (e.g. OpenMP library mismatch on Mac): use scikit-learn's gradient boosting
        from sklearn.ensemble import HistGradientBoostingClassifier
        models["Gradient boosting (scikit-learn)"] = make_pipeline(SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05,
                                           class_weight="balanced", random_state=RANDOM_STATE))
    return models


def choose_threshold(y, proba, min_recall=MIN_RECALL):
    """Highest-precision threshold that still reaches the recall target (set on validation only)."""
    order = np.sort(np.unique(proba))[::-1]
    best_t, best_prec = order[-1], -1.0
    for t in order:
        pred = proba >= t
        if recall_score(y, pred, zero_division=0) >= min_recall:
            prec = precision_score(y, pred, zero_division=0)
            if prec > best_prec:
                best_t, best_prec = t, prec
    return float(best_t)


def _metrics(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0), "flag_rate": pred.mean(),
            "threshold": threshold, "true_positives": tp, "false_positives": fp, "false_negatives": fn}


def fit_classifier_suite(batches):
    """Fit every model on the training period, set thresholds on validation, report test once."""
    split = split_by_time(batches)
    X = {k: build_features(getattr(split, k)) for k in ("train", "validation", "test")}
    y = {k: getattr(split, k)[TARGET_COL].to_numpy() for k in ("train", "validation", "test")}

    models, val_rows, test_rows, thresholds, val_proba, test_proba = _models(), {}, {}, {}, {}, {}
    for name, model in models.items():
        model.fit(X["train"], y["train"])
        val_proba[name] = model.predict_proba(X["validation"])[:, 1]
        test_proba[name] = model.predict_proba(X["test"])[:, 1]
        thresholds[name] = choose_threshold(y["validation"], val_proba[name])
        val_rows[name] = _metrics(y["validation"], val_proba[name], thresholds[name])
        test_rows[name] = _metrics(y["test"], test_proba[name], thresholds[name])

    validation_table = pd.DataFrame.from_dict(val_rows, orient="index").rename_axis("model")
    test_table = pd.DataFrame.from_dict(test_rows, orient="index").rename_axis("model")

    # Adoption rule: a challenger must beat the baseline's validation PR-AUC by the margin.
    chosen = BASELINE
    base_pr = validation_table.loc[BASELINE, "pr_auc"]
    challengers = validation_table.drop(index=BASELINE)
    best = challengers["pr_auc"].idxmax()
    if challengers.loc[best, "pr_auc"] >= base_pr + ADOPTION_MARGIN:
        chosen = best

    imp = permutation_importance(models[chosen], X["test"], y["test"], scoring="roc_auc", n_repeats=5,
                                 random_state=RANDOM_STATE, n_jobs=1)
    importance = pd.DataFrame({"feature": feature_columns(), "importance_mean": imp.importances_mean,
                               "importance_std": imp.importances_std})
    importance["label"] = importance["feature"].map(FEATURE_LABELS)
    importance = importance.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    importance = importance[["label", "feature", "importance_mean", "importance_std"]]

    return {"models": models, "chosen_model": chosen, "thresholds": thresholds,
            "validation_table": validation_table, "test_table": test_table,
            "validation_proba": val_proba, "test_proba": test_proba,
            "X_train": X["train"], "X_validation": X["validation"], "X_test": X["test"],
            "y_train": y["train"], "y_validation": y["validation"], "y_test": y["test"],
            "importance": importance, "split": split}


# ----------------------------------------------------------------------------- explanations
def driver_contributions(model, X):
    """Each feature's push on the log-odds of failure for every batch.

    For the logistic model this is coefficient x standardised value, so the explanation matches the
    prediction exactly. For tree models, a permutation-free approximation is used: the standardised
    value weighted by the model's feature importance.
    """
    steps = model.named_steps
    imputer = steps["simpleimputer"]
    Z = imputer.transform(X)
    if "logisticregression" in steps:
        Z = steps["standardscaler"].transform(Z)
        weights = steps["logisticregression"].coef_.ravel()
    else:
        Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) == 0, 1, Z.std(axis=0))
        last = list(steps.values())[-1]
        weights = getattr(last, "feature_importances_", np.ones(Z.shape[1]))
    return pd.DataFrame(Z * weights, index=X.index, columns=X.columns)


def top_drivers(contribs, n=3):
    """The n features that push each batch's risk up the most (labels and column names)."""
    values = contribs.to_numpy()
    order = np.argsort(-values, axis=1)[:, :n]
    cols = np.array(contribs.columns)
    out = pd.DataFrame(index=contribs.index)
    for k in range(n):
        feats = cols[order[:, k]]
        out[f"driver_{k + 1}"] = [FEATURE_LABELS.get(f, f) for f in feats]
        out[f"driver_{k + 1}_feature"] = feats
    return out


_ACTIONS = {
    "maintenance": "Complete overdue maintenance on this equipment and confirm calibration before release.",
    "process": "Review the process trend for this run and verify environmental controls before release.",
    "operator": "Have a senior operator or supervisor review the batch record before release.",
    "material": "Check the raw material certificate and retest the lot before release.",
    "product": "Apply the standard release checks for this product type and review its recent failures.",
    "run": "Review the batch record and in-process checks for this run before release.",
}


def recommended_action(feature):
    """Recommended QA step tied to a batch's leading risk driver."""
    if feature.startswith("maintenance") or feature == "equipment_age_years":
        return _ACTIONS["maintenance"]
    if feature in ("temp_avg", "temp_variance", "pressure_avg", "pressure_variance", "humidity_avg"):
        return _ACTIONS["process"]
    if feature == "operator_experience_years":
        return _ACTIONS["operator"]
    if feature == "raw_material_quality":
        return _ACTIONS["material"]
    if feature.startswith(("dosage_", "section_")):
        return _ACTIONS["product"]
    return _ACTIONS["run"]


def train_failure_type_model(X, failure_mode):
    """Random forest that proposes the likely failure mode, trained on failed batches only."""
    model = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
    return model.fit(X, failure_mode)
