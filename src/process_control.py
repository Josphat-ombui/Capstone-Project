"""Control limits, out-of-control flags, lead-time test and assay capability (Module C)."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.config import (ASSAY_LSL, ASSAY_USL, CONTROL_BASELINE_LAST_YEAR, CONTROL_VARIABLES, LEAD_WINDOW,
                        TARGET_COL, VARIANCE_VARIABLES)


def control_limits(batches):
    """Mean +/- 3 SD per section and variable, from passed batches up to 2021 (variances: upper limit only)."""
    base = batches[(batches[TARGET_COL] == 0) & (batches["year"] <= CONTROL_BASELINE_LAST_YEAR)]
    rows = []
    for section, g in base.groupby("manufacturing_section"):
        for var in CONTROL_VARIABLES:
            values = g[var].dropna()
            mean, sd = values.mean(), values.std()
            lcl = max(mean - 3 * sd, 0.0) if var in VARIANCE_VARIABLES else mean - 3 * sd
            rows.append({"manufacturing_section": section, "variable": var, "baseline_n": len(values),
                         "mean": mean, "sd": sd, "lcl": lcl, "ucl": mean + 3 * sd})
    return pd.DataFrame(rows)


def flag_out_of_control(batches, limits):
    """One 0/1 column per variable (ooc_<variable>) and the count per batch (n_ooc)."""
    flags = pd.DataFrame(index=batches.index)
    for var in CONTROL_VARIABLES:
        lim = limits[limits["variable"] == var].set_index("manufacturing_section")
        lcl = batches["manufacturing_section"].map(lim["lcl"])
        ucl = batches["manufacturing_section"].map(lim["ucl"])
        value = batches[var]
        if var in VARIANCE_VARIABLES:
            out = value > ucl
        else:
            out = (value < lcl) | (value > ucl)
        flags[f"ooc_{var}"] = out.fillna(False).astype(int)
    flags["n_ooc"] = flags[[f"ooc_{v}" for v in CONTROL_VARIABLES]].sum(axis=1)
    return flags


def within_limits_share(flags):
    """Share of batches within limits for each variable, and the count out of control."""
    rows = [{"variable": var, "share_within_limits": 1 - flags[f"ooc_{var}"].mean(),
             "readings_out_of_control": int(flags[f"ooc_{var}"].sum())} for var in CONTROL_VARIABLES]
    return pd.DataFrame(rows)


def lead_time_analysis(batches, flags, window=LEAD_WINDOW):
    """Did any of the previous `window` batches on the same machine have an out-of-control reading?"""
    df = batches[["equipment_id", "production_start", "batch_id", TARGET_COL]].join(flags["n_ooc"])
    df = df.sort_values(["equipment_id", "production_start", "batch_id"])
    signal = (df["n_ooc"] > 0).astype(int)
    prior = pd.Series(0, index=df.index)
    for lag in range(1, window + 1):
        prior = prior | signal.groupby(df["equipment_id"]).shift(lag).fillna(0).astype(int)
    df["prior_signal"] = prior
    table = df.groupby(TARGET_COL)["prior_signal"].agg(batches="size", share_with_prior_signal="mean")
    table.index = table.index.map({0: "Passed", 1: "Failed"})
    rr = table.loc["Failed", "share_with_prior_signal"] / table.loc["Passed", "share_with_prior_signal"]
    p_value = chi2_contingency(pd.crosstab(df[TARGET_COL], df["prior_signal"]))[1]
    table["relative_risk_of_prior_signal"] = rr
    table["chi_square_p_value"] = p_value
    return table.loc[["Passed", "Failed"]].rename_axis("outcome").reset_index()


def capability(batches, lsl=ASSAY_LSL, usl=ASSAY_USL):
    """Cp and Cpk of the assay result per section against assumed limits of 95 to 105%."""
    rows = []
    for section, g in batches.groupby("manufacturing_section"):
        values = g["assay_result_purity"].dropna()
        mean, sd = values.mean(), values.std()
        cpk = min(usl - mean, mean - lsl) / (3 * sd)
        rows.append({"manufacturing_section": section, "n": len(values), "mean": mean, "sd": sd,
                     "cp": (usl - lsl) / (6 * sd), "cpk": cpk,
                     "share_out_of_spec": ((values < lsl) | (values > usl)).mean(),
                     "capable_at_1_33": bool(cpk >= 1.33)})
    return pd.DataFrame(rows)
