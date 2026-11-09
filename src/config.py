"""Project settings: folders, file names, targets and modelling constants.

Every other module imports its paths and constants from here, so a change made here applies
to the whole analysis.
"""
from pathlib import Path

# Folders -------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # the project folder (the one holding src/)

# Raw files are read from data/raw/ when that folder holds them, otherwise from the project folder.
_RAW_CANDIDATE = ROOT / "data" / "raw"
DATA_RAW = _RAW_CANDIDATE if (_RAW_CANDIDATE / "LAL_Batch_Manufacturing_Production.csv").exists() else ROOT
DATA_PROCESSED = ROOT / "data" / "processed"        # cleaned tables written by the cleaning step
REPORTS_TABLES = ROOT / "reports" / "tables"        # CSV outputs (quality register, back-test)
REPORTS_FIGURES = ROOT / "reports" / "figures"      # PNG charts saved by save_figure()

RAW_FILES = {
    "batch": "LAL_Batch_Manufacturing_Production.csv",
    "maintenance": "LAL_Equipment_Maintenance_Records.csv",
    "sales": "sales_records.csv",
    "inventory": "inventory_records.csv",
}

# Outcome and targets (proposal section 8) ------------------------------------------------
TARGET_COL = "batch_failed"        # 1 = failed QC release, 0 = passed
MIN_RECALL = 0.80                  # alert must catch at least 80% of failures
MIN_PRECISION = 0.19               # with at least 19% of alerts being real failures
ROC_AUC_TARGET = 0.80              # ranking quality target

# Time-separated split ----------------------------------------------------------------
TRAIN_YEARS = (2007, 2021)
VALIDATION_YEARS = (2022, 2023)
TEST_YEARS = (2024, 2026)

# A challenger replaces the logistic baseline only if its validation PR-AUC is higher by this margin.
ADOPTION_MARGIN = 0.01

RANDOM_STATE = 42

# Risk score (Module D): weights are an assumption pending stakeholder sign-off -------------
RISK_WEIGHTS = {"quality": 0.6, "equipment": 0.2, "process": 0.2}
HIGH_RISK_TIER_SHARE = 0.10        # High tier = top 10% of batches by risk score

# Process control and capability -------------------------------------------------------
CONTROL_VARIABLES = ["temp_avg", "pressure_avg", "humidity_avg", "temp_variance", "pressure_variance"]
VARIANCE_VARIABLES = ["temp_variance", "pressure_variance"]   # upper limit only
CONTROL_BASELINE_LAST_YEAR = 2021                              # limits from passed batches up to 2021
LEAD_WINDOW = 5                                                # previous batches checked for a signal
ASSAY_LSL, ASSAY_USL = 95.0, 105.0                             # assumed assay specification (%)

# Trend periods (proposal) -------------------------------------------------------------
PRE_PERIOD = (2014, 2018)
POST_PERIOD = (2020, 2025)

# Availability scenarios: assumed repair times in hours (not measured) ---------------------
MTTR_SCENARIOS = [2, 8, 24]

# Forecasting --------------------------------------------------------------------------
FORECAST_HORIZON = 12              # months held out for testing
WAPE_IMPROVEMENT_TARGET = 0.15

# Salt for pseudonymising operator IDs (keep private in a real deployment).
PSEUDONYM_SALT = "lal-capstone-2026"
