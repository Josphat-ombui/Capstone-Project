"""Failure rate over time and the rise since 2019 (Module C)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.config import POST_PERIOD, PRE_PERIOD, TARGET_COL
from src.reliability import AGE_BANDS, OPERATOR_BANDS, add_bands, design_matrix, rate_table
from src.stats_utils import fit_logit

PRE_LABEL = f"Pre ({PRE_PERIOD[0]}-{PRE_PERIOD[1]})"
POST_LABEL = f"Post ({POST_PERIOD[0]}-{POST_PERIOD[1]})"


def yearly_table(batches):
    """Failure rate with 95% interval for each production year."""
    return rate_table(batches, ["year"])


def _period(batches):
    year = batches["year"]
    return np.select([year.between(*PRE_PERIOD), year.between(*POST_PERIOD)], [PRE_LABEL, POST_LABEL], default="")


def run_trend_analysis(batches):
    """Compare 2014-2018 with 2020-2025 and measure how much of the rise the measured factors explain."""
    df = batches.assign(period=_period(batches))
    df = df[df["period"] != ""]

    # 1) What changed between the two periods.
    def measures(g):
        return pd.Series({
            "batches": len(g),
            "failures": g[TARGET_COL].sum(),
            "failure_rate": g[TARGET_COL].mean(),
            "mean_equipment_age_years": g["equipment_age_years"].mean(),
            "share_critical": (g["maintenance_status"] == "Critical").mean(),
            "share_operators_0_2_years": (g["operator_experience_years"] <= 2).mean(),
            "share_injectables": (g["manufacturing_section"] == "Injectables").mean(),
            "share_liquid_batches": (g["batch_size_unit"] == "L").mean(),
            "mean_operator_experience_years": g["operator_experience_years"].mean(),
        })
    periods = pd.DataFrame({p: measures(df[df["period"] == p]) for p in (PRE_LABEL, POST_LABEL)})
    periods.index.name = "measure"
    periods.columns.name = "period"
    periods = periods.reset_index()

    # 2) Change predicted by the measured factors: one model fitted on all batches, applied to each period's mix.
    model = fit_logit(design_matrix(batches), batches[TARGET_COL])
    pred = pd.Series(model.predict(design_matrix(df)), index=df.index)
    actual_pre = df.loc[df["period"] == PRE_LABEL, TARGET_COL].mean()
    actual_post = df.loc[df["period"] == POST_LABEL, TARGET_COL].mean()
    pred_change = pred[df["period"] == POST_LABEL].mean() - pred[df["period"] == PRE_LABEL].mean()
    actual_change = actual_post - actual_pre
    decomposition = pd.DataFrame({
        "quantity": [f"Actual failure rate, {PRE_LABEL.lower()}", f"Actual failure rate, {POST_LABEL.lower()}",
                     "Actual change (percentage points)", "Change predicted by measured factors (percentage points)",
                     "Share of change explained by measured factors", "Unexplained change (percentage points)"],
        "value": [actual_pre, actual_post, actual_change * 100, pred_change * 100,
                  pred_change / actual_change, (actual_change - pred_change) * 100],
    })

    # 3) Year trend before and after adjustment (all years).
    allb = batches.copy()
    year = (allb["year"] - allb["year"].min()).astype(float)
    raw = fit_logit(pd.DataFrame({"const": 1.0, "year": year}, index=allb.index), allb[TARGET_COL])
    adj = fit_logit(design_matrix(allb).assign(year=year), allb[TARGET_COL])
    b_raw, b_adj = raw.params["year"], adj.params["year"]
    year_coefficients = pd.DataFrame({
        "model": ["Year trend only", "Year trend, adjusted for measured factors"],
        "year_coefficient": [b_raw, b_adj],
        "odds_ratio_per_year": [np.exp(b_raw), np.exp(b_adj)],
        "p_value": [raw.pvalues["year"], adj.pvalues["year"]],
        "share_of_trend_absorbed": [np.nan, 1 - b_adj / b_raw],
    })

    # 4) Composition of each period by band.
    banded = add_bands(df)
    parts = []
    for band_type, column in (("Equipment age", "age_band"), ("Operator experience", "operator_band"),
                              ("Maintenance status", "maintenance_status")):
        share = banded.groupby("period")[column].value_counts(normalize=True).rename("share").reset_index()
        parts.append(share.rename(columns={column: "band"}).assign(band_type=band_type))
    composition = pd.concat(parts, ignore_index=True)[["band_type", "band", "period", "share"]]
    composition["band"] = composition["band"].astype(str)

    return SimpleNamespace(periods=periods, decomposition=decomposition,
                           year_coefficients=year_coefficients, composition=composition)
