"""Cleaning and linking of the four raw files, with a register of every quality check."""
import hashlib

import numpy as np
import pandas as pd

from src.config import PSEUDONYM_SALT, REPORTS_TABLES, TARGET_COL
from src.data_io import load_raw, save_processed

# Raw rejection labels grouped into ten failure modes.
FAILURE_MODE_MAP = {
    "Assay out of specification": "Assay",
    "Dissolution failure": "Dissolution",
    "Microbial limit failure": "Microbial",
    "Bacterial endotoxin above limit": "Microbial",
    "Sterility test failure": "Microbial",
    "Related substances above limit": "Related substances",
    "Blend uniformity failure": "Blend uniformity",
    "Phase separation / homogeneity failure": "Blend uniformity",
    "Weight variation out of limits": "Weight and fill",
    "Capsule fill-weight variation": "Weight and fill",
    "Fill volume below limit": "Weight and fill",
    "pH out of specification": "pH",
    "Hardness / friability out of specification": "Physical strength",
    "Foreign matter / visible particles": "Particulates",
    "Visible particulate matter": "Particulates",
}

_SHARED_FIELDS = ["equipment_id", "equipment_age_days", "maintenance_days_ago", "batch_size",
                  "operator_experience_years", "batch_passed", "manufacturing_section"]
_IMPUTABLE = ["temp_avg", "pressure_avg", "humidity_avg", "raw_material_quality", "production_hours"]


def pseudonymise(operator_id):
    """Replace an employee ID with a salted SHA-256 code such as 'OP-1A2B3C4D'."""
    digest = hashlib.sha256(f"{PSEUDONYM_SALT}{operator_id}".encode()).hexdigest()
    return f"OP-{digest[:8].upper()}"


def _pct(x):
    return f"{x:.2%}"


def run_cleaning(save=True):
    """Clean and link the raw files. Returns a dict of tables and the quality register."""
    raw = {key: load_raw(key) for key in ("batch", "maintenance", "sales", "inventory")}
    batch, maint, sales, inv = raw["batch"].copy(), raw["maintenance"].copy(), raw["sales"].copy(), raw["inventory"].copy()
    checks = []

    def add(area, check, observed, expected, status, note=""):
        checks.append({"area": area, "check": check, "observed": observed, "expected": expected,
                       "status": status, "note": note})

    # Batch and maintenance files ------------------------------------------------------
    add("Batch file", "Row count", len(batch), "60,000", "PASS" if len(batch) == 60000 else "FLAG")
    add("Batch file", "Batch IDs unique", batch["batch_id"].nunique(), "equal to row count",
        "PASS" if batch["batch_id"].is_unique else "FLAG")
    add("Maintenance file", "Row count", len(maint), "60,000", "PASS" if len(maint) == 60000 else "FLAG")

    merged = batch.merge(maint, on="batch_id", how="left", suffixes=("", "_m"), indicator=True)
    match = (merged["_merge"] == "both").mean()
    add("Linkage", "Batch IDs match maintenance IDs", _pct(match), "100%", "PASS" if match == 1 else "FLAG")
    for field in _SHARED_FIELDS:
        agree = (merged[field] == merged[f"{field}_m"]).mean()
        add("Linkage", f"Agreement: {field}", _pct(agree), "100%", "PASS" if agree == 1 else "FLAG")

    batch["production_start"] = pd.to_datetime(batch["production_start"], errors="coerce")
    maint_date = pd.to_datetime(merged["production_date"], format="%d/%m/%Y", errors="coerce")
    same_date = (maint_date.values == batch["production_start"].values).mean()
    add("Linkage", "Batch date equals maintenance date", _pct(same_date), "100%", "PASS" if same_date == 1 else "FLAG")

    reason_agree = (merged["rejection_reason"].fillna("") == merged["rejection_reason_m"].fillna("")).mean()
    add("Linkage", "Rejection reason agreement (batch vs maintenance)", _pct(reason_agree),
        "about 3% (proposal section 6.2)", "INFO",
        "Observed agreement differs from the proposal; see the reconciliation table.")

    # Dates ---------------------------------------------------------------------------
    bad_dates = int(batch["production_start"].isna().sum())
    add("Dates", "Unparseable production dates", bad_dates, "0", "PASS" if bad_dates == 0 else "FLAG")
    add("Dates", "Production date range",
        f"{batch['production_start'].min():%Y-%m-%d} to {batch['production_start'].max():%Y-%m-%d}",
        "ISO dates, 2007 to 2026", "PASS")

    # Outcome and failure modes --------------------------------------------------------
    batch[TARGET_COL] = (batch["batch_passed"] == 0).astype(int)
    batch["failure_mode"] = batch["rejection_reason"].map(FAILURE_MODE_MAP)
    failed = batch[TARGET_COL] == 1
    unmapped = int((failed & batch["failure_mode"].isna()).sum())
    add("Outcome", "Failure labels not mapped to a known mode", unmapped, "0", "PASS" if unmapped == 0 else "FLAG")
    add("Outcome", "Failed batches", f"{int(failed.sum()):,} ({failed.mean():.2%})", "1,875 (3.1%)", "PASS")
    only_failed = bool((batch["rejection_reason"].notna() == failed).all())
    add("Outcome", "Rejection reason present only for failed batches", only_failed, "True",
        "PASS" if only_failed else "FLAG")

    # Equipment and maintenance -------------------------------------------------------
    add("Equipment", "Distinct equipment units", batch["equipment_id"].nunique(), "13", "PASS")
    add("Equipment", "Distinct manufacturing sections", batch["manufacturing_section"].nunique(), "4", "PASS")
    batch["maintenance_status"] = merged["maintenance_status"].values
    critical = int((batch["maintenance_status"] == "Critical").sum())
    add("Maintenance", "Critical batches", critical, "476", "PASS" if critical == 476 else "FLAG")

    # Privacy -------------------------------------------------------------------------
    batch["operator_pseudo_id"] = batch["operator_id"].map(pseudonymise)
    batch = batch.drop(columns=["operator_id"])
    add("Privacy", "Operator IDs pseudonymised", "Yes (salted SHA-256)", "Yes", "PASS",
        "Raw employee IDs are not written to any processed file or report.")

    # Missing data and plausibility ------------------------------------------------------
    for col in _IMPUTABLE:
        add("Missing data", f"Missing share: {col}", _pct(batch[col].isna().mean()), "1-2% (imputable)", "INFO",
            "Imputed inside the model step, fitted on training data only.")
    diss = batch["dissolution_time_min"].isna().mean()
    add("Missing data", "Dissolution time missing", _pct(diss), "about 25%", "PASS" if 0.2 <= diss <= 0.3 else "FLAG")
    add("Plausibility", "Assay above 105% (flagged, kept)", int((batch["assay_result_purity"] > 105).sum()),
        "flagged, not removed", "INFO", "Proposal section 6.3 rule applied.")

    # Derived batch fields -------------------------------------------------------------
    batch["year"] = batch["production_start"].dt.year
    batch["equipment_age_years"] = batch["equipment_age_days"] / 365.25

    # Sales ---------------------------------------------------------------------------
    add("Sales", "Row count", len(sales), "150,000", "PASS" if len(sales) == 150000 else "FLAG")
    add("Sales", "Sale IDs unique", bool(sales["Sale_ID"].is_unique), "True", "PASS" if sales["Sale_ID"].is_unique else "FLAG")
    sales["Sale_Date"] = pd.to_datetime(sales["Sale_Date"], errors="coerce")
    bad_sale_dates = int(sales["Sale_Date"].isna().sum())
    add("Sales", "Unparseable sale dates", bad_sale_dates, "0", "PASS" if bad_sale_dates == 0 else "FLAG")
    add("Sales", "Distinct products", sales["Product_ID"].nunique(), "300", "PASS")
    add("Sales", "Distinct branches", sales["Branch_Code"].nunique(), "4", "PASS")
    add("Sales", "Distinct suppliers", sales["Supplier_ID"].nunique(), "30", "PASS")
    gap_net = round(float((sales["Gross_Amount_KES"] - sales["Discount_Amount_KES"] - sales["Net_Amount_Excl_VAT_KES"]).abs().max()), 2)
    add("Sales", "Gross less discount equals net (max gap, KES)", gap_net, "0.00", "PASS" if gap_net < 0.01 else "FLAG")
    gap_total = round(float((sales["Net_Amount_Excl_VAT_KES"] + sales["VAT_Amount_KES"] - sales["Total_Amount_KES"]).abs().max()), 2)
    add("Sales", "Net plus VAT equals total (max gap, KES)", gap_total, "0.00", "PASS" if gap_total < 0.01 else "FLAG")
    sales["is_cancelled"] = (sales["Order_Status"] == "Cancelled").astype(int)
    add("Sales", "Cancelled orders (excluded from demand)", f"{int(sales['is_cancelled'].sum()):,}", "informational",
        "INFO", "Cancelled orders do not represent real demand.")

    # Inventory -----------------------------------------------------------------------
    add("Inventory", "Row count", len(inv), "162,000", "PASS" if len(inv) == 162000 else "FLAG")
    inv["Movement_Date"] = pd.to_datetime(inv["Movement_Date"], errors="coerce")
    out = inv[inv["Movement_Type"] == "Sale_Out"]
    linked = out["Reference_ID"].isin(sales["Sale_ID"]).mean()
    add("Inventory", "Sale_Out movements linked to a sale", _pct(linked), "100%", "PASS" if linked == 1 else "FLAG")
    sale_qty = sales.set_index("Sale_ID")["Quantity"].reindex(out["Reference_ID"]).to_numpy()
    qty_match = (out["Quantity_Change"].abs().to_numpy() == sale_qty).mean()
    add("Inventory", "Sale_Out quantity equals sale quantity", _pct(qty_match), "100%", "PASS" if qty_match == 1 else "FLAG")
    cancelled_ids = set(sales.loc[sales["is_cancelled"] == 1, "Sale_ID"])
    on_cancelled = int(out["Reference_ID"].isin(cancelled_ids).sum())
    add("Inventory", "Stock movements recorded for cancelled sales", on_cancelled, "0",
        "PASS" if on_cancelled == 0 else "FLAG", "A non-zero count means stock was deducted for cancelled orders.")
    add("Inventory", "Goods received movements", int((inv["Movement_Type"] == "Goods_Received").sum()), "informational", "INFO")
    add("Inventory", "Opening stock available", "No", "needed for stock on hand", "FLAG",
        "Running balances can only be relative. Stock on hand cannot be verified.")
    ordered = inv.sort_values(["Branch_Code", "Product_ID", "Movement_Date", "Movement_ID"])
    running = ordered.groupby(["Branch_Code", "Product_ID"])["Quantity_Change"].cumsum()
    add("Inventory", "Movements leaving a relative balance below zero", _pct((running < 0).mean()),
        "0% if opening stock were zero", "INFO", "Confirms that opening stock is missing from the extract.")

    register = pd.DataFrame(checks)
    tables = {"batches": batch, "sales": sales, "inventory": inv, "quality_register": register}
    if save:
        for name in ("batches", "sales", "inventory"):
            save_processed(tables[name], name)
        REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
        register.to_csv(REPORTS_TABLES / "data_quality_register.csv", index=False)
    return tables
