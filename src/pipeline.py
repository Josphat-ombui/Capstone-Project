"""Risk score, alert table and back-test (Module D), plus a one-call runner for the whole project.

Run from the project folder with:  python -m src.pipeline
"""
import numpy as np
import pandas as pd

from src.config import HIGH_RISK_TIER_SHARE, REPORTS_TABLES, RISK_WEIGHTS, TARGET_COL
from src.data_io import load_processed, save_processed
from src.modeling import driver_contributions, fit_classifier_suite, recommended_action, top_drivers
from src.process_control import control_limits, flag_out_of_control

_STATUS_SCORE = {"Current": 0.0, "Due Soon": 0.5, "Critical": 1.0}


def _rule_summary(name, flagged, test):
    y = test[TARGET_COL].to_numpy()
    f = np.asarray(flagged, bool)
    units = test["batch_size_unit"]
    return {
        "rule": name,
        "batches_flagged": int(f.sum()),
        "share_of_batches_flagged": f.mean(),
        "failures_in_period": int(y.sum()),
        "failures_caught": int(y[f].sum()),
        "share_of_failures_caught": y[f].sum() / y.sum(),
        "failure_rate_flagged": y[f].mean() if f.any() else np.nan,
        "failure_rate_not_flagged": y[~f].mean() if (~f).any() else np.nan,
        "lift_flagged_vs_base": (y[f].mean() / y.mean()) if f.any() else np.nan,
        "volume_flagged_kg": float(test.loc[f & (units == "kg").to_numpy(), "batch_size"].sum()),
        "volume_flagged_litres": float(test.loc[f & (units == "L").to_numpy(), "batch_size"].sum()),
    }


def build_risk_outputs(batches=None, suite=None):
    """Score every test-period batch, save the alert table and the back-test CSV files."""
    batches = load_processed("batches") if batches is None else batches
    suite = fit_classifier_suite(batches) if suite is None else suite
    chosen = suite["chosen_model"]
    X_test = suite["X_test"]
    test = batches.loc[X_test.index]
    proba = suite["test_proba"][chosen]
    threshold = suite["thresholds"][chosen]

    flags = flag_out_of_control(batches, control_limits(batches)).loc[X_test.index]
    alerts = test[["batch_id", "production_start", "drug_name", "equipment_id", "maintenance_status",
                   "operator_pseudo_id", "batch_passed", "failure_mode"]].copy()
    alerts["classifier_probability"] = proba
    alerts["classifier_flag"] = (proba >= threshold).astype(int)
    # Components on a 0-1 scale: quality = rank of the classifier probability; equipment = half age
    # (scaled 0-1 within the test period) and half maintenance status; process = share of the five
    # process readings that are out of control.
    alerts["quality_component"] = pd.Series(proba, index=X_test.index).rank(pct=True, method="average") - 1 / len(proba)
    alerts["quality_component"] = alerts["quality_component"].clip(lower=0) / alerts["quality_component"].max()
    age = test["equipment_age_years"]
    age_scale = (age - age.min()) / (age.max() - age.min())
    alerts["equipment_component"] = 0.5 * age_scale + 0.5 * test["maintenance_status"].map(_STATUS_SCORE)
    alerts["process_component"] = flags["n_ooc"] / 5
    alerts["risk_score"] = (RISK_WEIGHTS["quality"] * alerts["quality_component"]
                            + RISK_WEIGHTS["equipment"] * alerts["equipment_component"]
                            + RISK_WEIGHTS["process"] * alerts["process_component"])
    n_high = int(np.ceil(HIGH_RISK_TIER_SHARE * len(alerts)))
    high_index = alerts["risk_score"].nlargest(n_high).index
    alerts["risk_tier"] = np.where(alerts.index.isin(high_index), "High", "Standard")

    drivers = top_drivers(driver_contributions(suite["models"][chosen], X_test), n=3)
    alerts["primary_driver"] = drivers["driver_1"]
    alerts["recommended_action"] = drivers["driver_1_feature"].map(recommended_action)

    save_processed(alerts.reset_index(), "alerts_test_period")

    # Back-test tables ------------------------------------------------------------------
    REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([
        _rule_summary("Classifier alert at 80% recall target", alerts["classifier_flag"] == 1, test),
        _rule_summary(f"High risk tier (top {HIGH_RISK_TIER_SHARE:.0%} by composite score)",
                      alerts["risk_tier"] == "High", test),
    ])
    summary.to_csv(REPORTS_TABLES / "backtest_summary.csv", index=False)

    by_year = []
    for year, idx in test.groupby("year").groups.items():
        row = _rule_summary("Classifier alert at 80% recall target", alerts.loc[idx, "classifier_flag"] == 1,
                            test.loc[idx])
        by_year.append({"year": year, **{k: v for k, v in row.items() if k != "rule"}})
    pd.DataFrame(by_year).to_csv(REPORTS_TABLES / "backtest_by_year.csv", index=False)

    crit = test["maintenance_status"] == "Critical"
    pd.DataFrame([
        {"group": "Critical maintenance status", "batches": int(crit.sum()),
         "failures": int(test.loc[crit, TARGET_COL].sum()), "failure_rate": test.loc[crit, TARGET_COL].mean()},
        {"group": "Current or Due Soon", "batches": int((~crit).sum()),
         "failures": int(test.loc[~crit, TARGET_COL].sum()), "failure_rate": test.loc[~crit, TARGET_COL].mean()},
    ]).to_csv(REPORTS_TABLES / "backtest_critical_equipment.csv", index=False)

    return alerts.reset_index()


def run_all():
    """Clean the raw files, then build the risk outputs and back-test."""
    from src.cleaning import run_cleaning
    tables = run_cleaning(save=True)
    build_risk_outputs(tables["batches"])
    print("Processed tables and back-test files written.")


if __name__ == "__main__":
    run_all()
