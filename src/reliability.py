"""Failure rates, maintenance effect and equipment ranking (Module B)."""
import numpy as np
import pandas as pd

from src.config import MTTR_SCENARIOS, TARGET_COL
from src.stats_utils import fit_logit, poisson_ratio_interval, wilson_interval

AGE_BANDS = ([0, 2, 5, 10, np.inf], ["0-2 yrs", "2-5 yrs", "5-10 yrs", "10+ yrs"])
OPERATOR_BANDS = ([-np.inf, 2, 5, 10, np.inf], ["0-2 yrs", "3-5 yrs", "6-10 yrs", "11+ yrs"])
STATUS_ORDER = ["Current", "Due Soon", "Critical"]


def add_bands(batches):
    """Add equipment-age and operator-experience bands."""
    out = batches.copy()
    out["age_band"] = pd.cut(out["equipment_age_years"], AGE_BANDS[0], labels=AGE_BANDS[1], right=False)
    out["operator_band"] = pd.cut(out["operator_experience_years"], OPERATOR_BANDS[0], labels=OPERATOR_BANDS[1])
    return out


def rate_table(df, columns):
    """Batches, failures, failure rate and 95% Wilson interval for each group."""
    table = df.groupby(columns, observed=True)[TARGET_COL].agg(batches="size", failures="sum").reset_index()
    table["failure_rate"] = table["failures"] / table["batches"]
    table["ci_low"], table["ci_high"] = wilson_interval(table["failures"], table["batches"])
    return table


def status_summary(batches):
    """Failure rate by maintenance status in the order Current, Due Soon, Critical."""
    table = rate_table(batches, ["maintenance_status"]).set_index("maintenance_status").loc[STATUS_ORDER].reset_index()
    table["share_of_batches"] = table["batches"] / table["batches"].sum()
    return table


def design_matrix(batches, with_equipment_fixed_effects=False, extra=None):
    """Predictors for the adjusted model: maintenance status, age, experience, dosage form, section."""
    X = pd.DataFrame(index=batches.index)
    X["const"] = 1.0
    X["maintenance_critical"] = (batches["maintenance_status"] == "Critical").astype(float)
    X["maintenance_due_soon"] = (batches["maintenance_status"] == "Due Soon").astype(float)
    X["equipment_age_years"] = batches["equipment_age_years"]
    X["operator_experience_years"] = batches["operator_experience_years"]
    for col in extra or []:
        X[col] = batches[col]
    if with_equipment_fixed_effects:
        # Within-machine comparison: one dummy per machine (first machine is the reference).
        # Each machine makes one section's products, so the machine dummies replace dosage and section.
        eq = pd.get_dummies(batches["equipment_id"], prefix="equipment").astype(float)
        return X.join(eq.iloc[:, 1:])
    X = X.join(pd.get_dummies(batches["dosage_form"], prefix="dosage").drop(columns="dosage_Tablet").astype(float))
    X = X.join(pd.get_dummies(batches["manufacturing_section"], prefix="section")
               .drop(columns="section_Oral Solids").astype(float))
    return X


def fit_maintenance_model(batches, with_equipment_fixed_effects=False):
    """Logistic model of failure on maintenance status, adjusted for age, experience and product."""
    X = design_matrix(batches, with_equipment_fixed_effects)
    return fit_logit(X, batches[TARGET_COL])


_TERM_LABELS = {
    "maintenance_critical": "Maintenance: Critical",
    "maintenance_due_soon": "Maintenance: Due Soon",
    "equipment_age_years": "Equipment age (per year)",
    "operator_experience_years": "Operator experience (per year)",
}


def odds_ratio_table(result):
    """Odds ratios with 95% intervals for the maintenance, age and experience terms."""
    rows = []
    for term, label in _TERM_LABELS.items():
        if term not in result.params:
            continue
        b, se = result.params[term], result.bse[term]
        rows.append({"term": label, "coefficient": b, "odds_ratio": np.exp(b),
                     "or_ci_low": np.exp(b - 1.959964 * se), "or_ci_high": np.exp(b + 1.959964 * se),
                     "p_value": result.pvalues[term]})
    return pd.DataFrame(rows)


def equipment_table(batches):
    """Observed vs expected failures per machine (indirect standardisation), reliability proxies."""
    rates = batches.groupby(["manufacturing_section", "dosage_form"])[TARGET_COL].transform("mean")
    df = batches.assign(expected=rates)
    table = df.groupby("equipment_id").agg(
        equipment_name=("equipment_name", "first"),
        manufacturing_section=("manufacturing_section", "first"),
        batches=(TARGET_COL, "size"),
        failures=(TARGET_COL, "sum"),
        expected_failures=("expected", "sum"),
        production_hours=("production_hours", "sum"),
    ).reset_index()
    table["failure_rate"] = table["failures"] / table["batches"]
    table["standardised_ratio"] = table["failures"] / table["expected_failures"]
    table["ratio_ci_low"], table["ratio_ci_high"] = poisson_ratio_interval(table["failures"], table["expected_failures"])
    above = table["standardised_ratio"] > 1 + 1e-9
    table["priority"] = np.select(
        [above & (table["ratio_ci_low"] > 1), above],
        ["High: above expected, interval excludes 1", "Watch: above expected, interval includes 1"],
        default="Standard")
    failures = table["failures"].replace(0, np.nan)
    table["proxy_mtbf_production_hours"] = table["production_hours"] / failures
    table["proxy_batches_between_failures"] = table["batches"] / failures
    for mttr in MTTR_SCENARIOS:
        mtbf = table["proxy_mtbf_production_hours"]
        table[f"scenario_availability_mttr_{mttr}h"] = mtbf / (mtbf + mttr)
    return table.sort_values("standardised_ratio", ascending=False).reset_index(drop=True)
