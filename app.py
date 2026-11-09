"""Pharmaceutical Operations Intelligence System — Streamlit dashboard (Group 12, Moringa capstone).

Run locally:   streamlit run app.py
"""
import warnings                                                             # silence harmless notices
import numpy as np                                                          # numbers
import pandas as pd                                                         # tables
import streamlit as st                                                      # dashboard
from sklearn.impute import SimpleImputer                                    # fill missing inputs
from sklearn.linear_model import LogisticRegression                         # chosen batch-failure model
from sklearn.pipeline import make_pipeline                                  # chain the steps
from sklearn.preprocessing import StandardScaler                            # put inputs on one scale

from src.cleaning import run_cleaning                                       # project cleaning step
from src.features import DOSAGE_LEVELS, SECTION_LEVELS, build_features      # model inputs
from src.forecasting import _holt_winters, evaluate_series, monthly_demand  # demand forecasting
from src.modeling import choose_threshold                                   # threshold at 80% recall
from src.reliability import equipment_table                                 # observed vs expected per machine

warnings.filterwarnings("ignore", category=UserWarning)                     # e.g. statsmodels convergence notes
warnings.filterwarnings("ignore", module="statsmodels")                     # Holt-Winters fitting messages
st.set_page_config(page_title="Pharma Operations Intelligence", page_icon="💊", layout="wide")  # page settings


# ------------------------------------------------------------------ data (cached: runs once per session)
@st.cache_data(show_spinner="Cleaning the four data files...")
def load_data():
    t = run_cleaning(save=False)                                            # cleaned batches, sales, inventory
    sales, inv = t["sales"], t["inventory"]                                 # shortcuts
    cancelled = sales.loc[sales["is_cancelled"] == 1, "Sale_ID"]            # cancelled orders
    moves = inv[~inv["Reference_ID"].isin(cancelled)].copy()                # valid stock movements only
    moves["Product_Category"] = moves["Product_ID"].map(sales.groupby("Product_ID")["Product_Category"].first())  # category
    return t["batches"], sales, moves


@st.cache_data
def reorder_policy(moves, lead_days=14, z=1.65, over_days=180):
    week = moves["Movement_Date"].dt.to_period("W-SUN").dt.start_time      # week of each movement
    last52 = pd.date_range(end=week.max(), periods=52, freq="7D")           # the last 52 weeks
    sold = (-moves["Quantity_Change"]).clip(lower=0)                        # units out
    weekly = sold.groupby([moves["Branch_Code"], moves["Product_ID"], week]).sum()  # units out per week
    pairs = moves[["Branch_Code", "Product_ID"]].drop_duplicates()          # every branch-product item
    grid = pd.MultiIndex.from_tuples([(b, p, w) for b, p in pairs.itertuples(index=False) for w in last52])
    recent = weekly.reindex(grid, fill_value=0).groupby(level=[0, 1])       # weekly sales, zero weeks included
    pol = pd.DataFrame({"avg_weekly": recent.mean(), "sd_weekly": recent.std(),  # demand level and spread
                        "stock_now": moves.groupby(["Branch_Code", "Product_ID"])["Quantity_Change"].sum()})  # stock rebuilt from zero
    pol["category"] = moves.groupby(["Branch_Code", "Product_ID"])["Product_Category"].first()  # product category
    pol["safety_stock"] = z * pol["sd_weekly"] / np.sqrt(7) * np.sqrt(lead_days)  # buffer for the lead time
    pol["reorder_point"] = pol["avg_weekly"] / 7 * lead_days + pol["safety_stock"]  # lead-time demand + buffer
    pol["days_of_cover"] = pol["stock_now"] / (pol["avg_weekly"] / 7).replace(0, np.nan)  # how long stock lasts
    pol["status"] = np.select([pol["avg_weekly"] == 0, pol["stock_now"] <= pol["reorder_point"], pol["days_of_cover"] > over_days],
                              ["No recent demand", "Reorder now", "Overstock"], default="OK")  # action per item
    return pol.round(1).reset_index().rename(columns={"level_0": "Branch_Code", "level_1": "Product_ID"})


@st.cache_resource(show_spinner="Training the batch-failure model...")
def risk_model(batches):
    train = batches[batches["year"].between(2007, 2021)]                    # training years
    val = batches[batches["year"].between(2022, 2023)]                      # validation years
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),  # chosen model: logistic regression
                          LogisticRegression(class_weight="balanced", max_iter=2000))
    model.fit(build_features(train), train["batch_failed"])                 # fit on training years
    thr = choose_threshold(val["batch_failed"].to_numpy(), model.predict_proba(build_features(val))[:, 1])  # 80% recall
    return model, thr, train


batches, sales, moves = load_data()                                         # load once
delivered = sales[sales["is_cancelled"] == 0]                               # delivered orders
policy = reorder_policy(moves)                                              # reorder status per item

# ------------------------------------------------------------------ header and sidebar
st.title("💊 Pharmaceutical Operations Intelligence System")                # page title
st.caption("Batch quality, equipment maintenance, demand and inventory — LAL data, 2007–2026")
yr = st.sidebar.slider("Production years", 2007, 2026, (2007, 2026))       # year filter for quality tabs
sections = st.sidebar.multiselect("Manufacturing section", sorted(batches["manufacturing_section"].unique()),
                                  default=sorted(batches["manufacturing_section"].unique()))  # section filter
b = batches[batches["year"].between(*yr) & batches["manufacturing_section"].isin(sections)]  # filtered batches
st.sidebar.markdown(f"**{len(b):,} batches selected**")                    # filter summary

tab = st.tabs(["Overview", "Batch quality", "Equipment", "Risk checker", "Demand forecast", "Inventory"])

# ------------------------------------------------------------------ 1. overview
with tab[0]:
    c = st.columns(5)                                                       # five headline numbers
    c[0].metric("Batches", f"{len(b):,}")                                  # batches in the filter
    c[1].metric("Failure rate", f"{b['batch_failed'].mean():.2%}")          # share failed
    c[2].metric("Net sales (KES bn)", f"{delivered['Net_Amount_Excl_VAT_KES'].sum() / 1e9:.1f}")  # revenue
    c[3].metric("Unpaid (KES m)", f"{delivered['Balance_Due_KES'].sum() / 1e6:,.0f}")  # outstanding balances
    c[4].metric("Items overstocked", f"{(policy['status'] == 'Overstock').sum():,} / {len(policy):,}")  # stock problem
    st.subheader("Bottom line")
    st.markdown("""
- **Quality is high:** about 97% of batches pass; assay and dissolution make up 47% of failures.
- **Overdue maintenance doubles the risk:** Critical equipment fails at 6.9% vs 3.1% (odds ratio 2.3 after adjustment).
- **The rise since 2019 is mostly explained (60%)**, mainly by ageing equipment.
- **Failure prediction is weak** (ROC-AUC 0.58): keep laboratory release testing.
- **Stock is bought without regard to sales:** about 2 units received per unit sold; Equipment 22×.
- **Demand can be planned:** Holt-Winters is 34% better than last year's pattern.""")

# ------------------------------------------------------------------ 2. batch quality
with tab[1]:
    factor = st.selectbox("Failure rate by", ["maintenance_status", "dosage_form", "manufacturing_section",
                                              "equipment_id", "drug_name"])  # grouping column
    by = b.groupby(factor)["batch_failed"].agg(["mean", "size"]).sort_values("mean", ascending=False)  # rate and count
    st.bar_chart((by["mean"] * 100).rename("Failure rate (%)"))             # chart of rates
    st.dataframe(by.assign(mean=(by["mean"] * 100).round(2)).rename(columns={"mean": "failure_rate_%", "size": "batches"}))
    left, right = st.columns(2)                                             # two charts side by side
    left.subheader("Failure modes")
    left.bar_chart(b["failure_mode"].value_counts().rename("Failed batches"))  # Pareto counts
    right.subheader("Failure rate by year")
    right.line_chart((b.groupby("year")["batch_failed"].mean() * 100).rename("Failure rate (%)"))  # trend

# ------------------------------------------------------------------ 3. equipment
with tab[2]:
    st.subheader("Failure rate by maintenance status")
    st.bar_chart((b.groupby("maintenance_status")["batch_failed"].mean() * 100).rename("Failure rate (%)"))
    st.subheader("Machines: observed vs expected failures")
    st.caption("Ratio above 1 = more failures than the product mix predicts. Intervals crossing 1 are not conclusive.")
    st.dataframe(equipment_table(b), width="stretch")              # ranking table

# ------------------------------------------------------------------ 4. risk checker
with tab[3]:
    model, thr, train = risk_model(batches)                                 # cached model
    st.caption("Chosen model: logistic regression (test ROC-AUC 0.58). Use it to target extra checks, not to release batches.")
    with st.form("risk"):                                                   # input form
        c = st.columns(3)                                                   # three input columns
        status = c[0].selectbox("Maintenance status", ["Current", "Due Soon", "Critical"])
        age = c[0].number_input("Equipment age (years)", 0.0, 40.0, 10.0)
        days = c[0].number_input("Days since maintenance", 0, 400, 60)
        exp = c[1].number_input("Operator experience (years)", 0.0, 40.0, 5.0)
        rmq = c[1].number_input("Raw material quality", 0.0, 100.0, float(train["raw_material_quality"].median()))
        tvar = c[1].number_input("Temperature variance", 0.0, 10.0, float(train["temp_variance"].median()))
        dosage = c[2].selectbox("Dosage form", ["Tablet"] + DOSAGE_LEVELS)
        section = c[2].selectbox("Section", ["Oral Solids"] + SECTION_LEVELS)
        go = st.form_submit_button("Score this batch")                      # run button
    if go:
        row = train.median(numeric_only=True).to_frame().T                  # typical values for the other inputs
        row = row.assign(maintenance_status=status, equipment_age_years=age, maintenance_days_ago=days,
                         operator_experience_years=exp, raw_material_quality=rmq, temp_variance=tvar,
                         dosage_form=dosage, manufacturing_section=section)  # the user's inputs
        p = model.predict_proba(build_features(row))[0, 1]                  # probability of failure
        past = model.predict_proba(build_features(train))[:, 1]             # scores of past batches
        c = st.columns(4)
        c[0].metric("Risk score", f"{p:.2f}", f"{p / past.mean():.1f}× average")   # result vs average
        c[3].metric("Riskier than", f"{(past < p).mean():.0%} of past batches")  # position among past batches
        c[1].metric("Alert threshold", f"{thr:.2f}")                        # 80%-recall threshold
        c[2].metric("Flag", "⚠️ Extra checks" if p >= thr else "✅ Standard")  # decision

# ------------------------------------------------------------------ 5. demand forecast
with tab[4]:
    md = monthly_demand(sales)                                              # units per month by category
    cat = st.selectbox("Category", ["Total"] + [c for c in md.columns if c != "Total"])
    series = md[cat]                                                        # chosen series
    acc, _ = evaluate_series(series)                                        # 12-month holdout comparison
    future = pd.Series(_holt_winters(series.to_numpy(), 12),                # Holt-Winters, next 12 months
                       index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS"))
    chart = pd.DataFrame({"Actual": series, "Forecast (Holt-Winters)": future})  # history + forecast
    st.line_chart(chart)                                                    # forecast chart
    st.subheader("Accuracy on the last 12 months")
    st.dataframe((acc.drop(columns="RMSE") * 100).round(1).add_suffix(" (%)"))  # WAPE, bias, gain vs baseline
    st.download_button("Download forecast (CSV)", future.rename("units").to_csv(), "forecast.csv")

# ------------------------------------------------------------------ 6. inventory
with tab[5]:
    c = st.columns(2)                                                       # two filters
    br = c[0].multiselect("Branch", sorted(policy["Branch_Code"].unique()), default=sorted(policy["Branch_Code"].unique()))
    ct = c[1].multiselect("Category", sorted(policy["category"].dropna().unique()),
                          default=sorted(policy["category"].dropna().unique()))
    p = policy[policy["Branch_Code"].isin(br) & policy["category"].isin(ct)]  # filtered items
    st.bar_chart(p["status"].value_counts().rename("Items"))                # items per action
    st.caption("Assumptions: 14-day lead time, 95% service level, overstock above 180 days of cover. Stock rebuilt from zero.")
    show = st.radio("Show", ["Reorder now", "Overstock", "All"], horizontal=True)  # list filter
    st.dataframe(p if show == "All" else p[p["status"] == show], width="stretch")  # item list
    st.download_button("Download list (CSV)", p.to_csv(index=False), "reorder_policy.csv")
