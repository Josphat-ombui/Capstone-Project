"""Pharmaceutical Operations Intelligence System — Streamlit dashboard (Group 12, Moringa capstone).

Run locally:   streamlit run app.py
"""
import warnings                                                             # silence harmless notices
import altair as alt                                                        # interactive charts (ships with Streamlit)
import numpy as np                                                          # numbers
import pandas as pd                                                         # tables
import streamlit as st                                                      # dashboard
from sklearn.impute import SimpleImputer                                    # fill missing inputs
from sklearn.linear_model import LogisticRegression                         # chosen batch-failure model
from sklearn.pipeline import make_pipeline                                  # chain the steps
from sklearn.preprocessing import StandardScaler                            # put inputs on one scale

from src.cleaning import run_cleaning                                       # project cleaning step
from src.features import DOSAGE_LEVELS, FEATURE_LABELS, SECTION_LEVELS, build_features  # model inputs
from src.forecasting import _holt_winters, evaluate_series, monthly_demand  # demand forecasting
from src.modeling import choose_threshold                                   # threshold at 80% recall
from src.process_control import capability, control_limits, flag_out_of_control, within_limits_share  # process control
from src.reliability import add_bands, equipment_table, fit_maintenance_model, odds_ratio_table  # maintenance analysis

warnings.filterwarnings("ignore", category=UserWarning)                     # e.g. statsmodels convergence notes
warnings.filterwarnings("ignore", module="statsmodels")                     # Holt-Winters fitting messages
st.set_page_config(page_title="Pharma Operations Intelligence", page_icon="💊", layout="wide")  # page settings
alt.data_transformers.disable_max_rows()                                    # allow charts with many points
BLUE, ORANGE, RED, GREY = "#1f4e79", "#f28e2b", "#c0392b", "#7f7f7f"        # chart colours


# ------------------------------------------------------------------ data (cached: runs once per session)
@st.cache_data(show_spinner="Cleaning the four data files...")
def load_data():
    t = run_cleaning(save=False)                                            # cleaned batches, sales, inventory
    sales, inv = t["sales"], t["inventory"]                                 # shortcuts
    cancelled = sales.loc[sales["is_cancelled"] == 1, "Sale_ID"]            # cancelled orders
    moves = inv[~inv["Reference_ID"].isin(cancelled)].copy()                # valid stock movements only
    moves["Product_Category"] = moves["Product_ID"].map(sales.groupby("Product_ID")["Product_Category"].first())  # category
    return add_bands(t["batches"]), sales, moves                            # batches with age/experience bands


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


@st.cache_data(show_spinner="Fitting the adjusted maintenance model...")
def odds_ratios(batches):
    return odds_ratio_table(fit_maintenance_model(batches))                 # adjusted odds ratios with 95% CI


def rate_chart(df, col, title, sort="-y"):                                  # bar chart of failure rate by a column
    t = df.groupby(col, observed=True)["batch_failed"].agg(rate="mean", batches="size").reset_index()
    t["rate"] = t["rate"] * 100                                             # percent
    bars = alt.Chart(t).mark_bar(color=BLUE).encode(
        x=alt.X(f"{col}:N", sort=sort, title=None), y=alt.Y("rate:Q", title="Failure rate (%)"),
        tooltip=[col, alt.Tooltip("rate:Q", format=".2f"), "batches"])      # hover details
    avg = alt.Chart(pd.DataFrame({"y": [df["batch_failed"].mean() * 100]})).mark_rule(color=RED, strokeDash=[5, 3]).encode(y="y:Q")
    return (bars + avg).properties(title=title, height=280)                 # bars + plant average line


batches, sales, moves = load_data()                                         # load once
delivered = sales[sales["is_cancelled"] == 0]                               # delivered orders
policy = reorder_policy(moves)                                              # reorder status per item

# ------------------------------------------------------------------ header and sidebar
st.title("💊 Pharmaceutical Operations Intelligence System")                # page title
st.caption("Batch quality, equipment maintenance, process control, sales, demand and inventory — LAL data, 2007–2026")
yr = st.sidebar.slider("Production years", 2007, 2026, (2007, 2026))       # year filter for quality tabs
sections = st.sidebar.multiselect("Manufacturing section", sorted(batches["manufacturing_section"].unique()),
                                  default=sorted(batches["manufacturing_section"].unique()))  # section filter
b = batches[batches["year"].between(*yr) & batches["manufacturing_section"].isin(sections)]  # filtered batches
st.sidebar.markdown(f"**{len(b):,} batches selected**")                    # filter summary
st.sidebar.caption("The filters apply to the Batch quality, Equipment and Process control tabs.")

tab = st.tabs(["Overview", "Sales", "Batch quality", "Equipment", "Process control", "Risk checker",
               "Demand forecast", "Inventory", "Model summary"])

# ------------------------------------------------------------------ 1. overview
with tab[0]:
    c = st.columns(5)                                                       # five headline numbers
    c[0].metric("Batches", f"{len(b):,}")                                  # batches in the filter
    c[1].metric("Failure rate", f"{b['batch_failed'].mean():.2%}")          # share failed
    c[2].metric("Net sales (KES bn)", f"{delivered['Net_Amount_Excl_VAT_KES'].sum() / 1e9:.1f}")  # revenue
    c[3].metric("Unpaid (KES m)", f"{delivered['Balance_Due_KES'].sum() / 1e6:,.0f}")  # outstanding balances
    c[4].metric("Items overstocked", f"{(policy['status'] == 'Overstock').sum():,} / {len(policy):,}")  # stock problem
    left, right = st.columns([3, 2])                                        # text and chart
    left.subheader("Bottom line")
    left.markdown("""
- **Quality is high:** about 97% of batches pass; assay and dissolution make up 47% of failures.
- **Overdue maintenance doubles the risk:** Critical equipment fails at 6.9% vs 3.1% (odds ratio 2.3 after adjustment).
- **The rise since 2019 is mostly explained (60%)**, mainly by ageing equipment.
- **Failure prediction is weak** (ROC-AUC 0.58): keep laboratory release testing.
- **Stock is bought without regard to sales:** about 2 units received per unit sold; Equipment 22×.
- **Demand can be planned:** Holt-Winters is 34% better than last year's pattern.""")
    yearly = pd.DataFrame({"Batches": batches.groupby("year").size(), "Failure rate (%)": batches.groupby("year")["batch_failed"].mean() * 100})
    right.subheader("Failure rate by year")
    right.line_chart(yearly["Failure rate (%)"], color=RED)                 # trend line
    st.subheader("Key figures")
    received = moves.loc[moves["Quantity_Change"] > 0, "Quantity_Change"].sum()  # units received
    st.dataframe(pd.DataFrame({"Indicator": ["Batches / failed", "Failure rate: Critical vs Current", "Delivered orders / cancelled",
                                             "Units received / sold", "Items to reorder now / overstocked"],
                               "Value": [f"{len(batches):,} / {batches['batch_failed'].sum():,}",
                                         f"{batches.loc[batches['maintenance_status'] == 'Critical', 'batch_failed'].mean():.1%} vs "
                                         f"{batches.loc[batches['maintenance_status'] == 'Current', 'batch_failed'].mean():.1%}",
                                         f"{len(delivered):,} / {sales['is_cancelled'].sum():,}",
                                         f"{received:,.0f} / {delivered['Quantity'].sum():,.0f}",
                                         f"{(policy['status'] == 'Reorder now').sum()} / {(policy['status'] == 'Overstock').sum()}"]}),
                 hide_index=True, width="stretch")

# ------------------------------------------------------------------ 2. sales
with tab[1]:
    m = delivered.groupby(delivered["Sale_Date"].dt.to_period("M").dt.to_timestamp())["Net_Amount_Excl_VAT_KES"].sum() / 1e6
    st.subheader("Net sales per month (KES million)")
    st.area_chart(m.rename("Net sales (KES m)"), color=BLUE)               # monthly revenue
    c = st.columns(2)                                                       # two charts
    br = delivered.groupby("Branch_Code")["Net_Amount_Excl_VAT_KES"].sum().div(1e6).reset_index(name="KES_m")
    c[0].altair_chart(alt.Chart(br).mark_arc(innerRadius=60).encode(        # branch share donut
        theta="KES_m:Q", color=alt.Color("Branch_Code:N", title="Branch"), tooltip=["Branch_Code", alt.Tooltip("KES_m:Q", format=",.0f")])
        .properties(title="Sales by branch", height=300), width="stretch")
    cat = delivered.groupby("Product_Category")["Net_Amount_Excl_VAT_KES"].sum().div(1e6).sort_values().reset_index(name="KES_m")
    c[1].altair_chart(alt.Chart(cat).mark_bar(color=ORANGE).encode(        # category bars
        y=alt.Y("Product_Category:N", sort="-x", title=None), x=alt.X("KES_m:Q", title="Net sales (KES m)"),
        tooltip=["Product_Category", alt.Tooltip("KES_m:Q", format=",.0f")]).properties(title="Sales by category", height=300),
        width="stretch")
    c = st.columns(2)                                                       # two tables
    c[0].subheader("Discount level vs order size")
    c[0].dataframe(delivered.groupby("Discount_Pct")["Quantity"].agg(orders="size", median_units="median", mean_units="mean").round(1),
                   width="stretch")
    c[1].subheader("Payment status by method (% of orders)")
    c[1].dataframe((pd.crosstab(delivered["Payment_Method"], delivered["Payment_Status"], normalize="index") * 100).round(1),
                   width="stretch")
    st.subheader("Top 10 customers by net sales")
    top = delivered.groupby("Customer_ID").agg(orders=("Sale_ID", "size"), net_sales_kes_m=("Net_Amount_Excl_VAT_KES", lambda s: s.sum() / 1e6),
                                               unpaid_kes_m=("Balance_Due_KES", lambda s: s.sum() / 1e6))
    st.dataframe(top.sort_values("net_sales_kes_m", ascending=False).head(10).round(2), width="stretch")

# ------------------------------------------------------------------ 3. batch quality
with tab[2]:
    c = st.columns(3)                                                       # three headline numbers
    c[0].metric("Batches", f"{len(b):,}")
    c[1].metric("Failed", f"{b['batch_failed'].sum():,}")
    c[2].metric("Failure rate", f"{b['batch_failed'].mean():.2%}")
    factor = st.selectbox("Failure rate by", ["maintenance_status", "age_band", "operator_band", "dosage_form",
                                              "manufacturing_section", "equipment_id", "drug_name"])  # grouping column
    st.altair_chart(rate_chart(b, factor, f"Failure rate by {factor} (dashed: average)"), width="stretch")
    st.subheader("Failure modes (Pareto)")
    fm = b["failure_mode"].value_counts().rename_axis("mode").reset_index(name="failed")  # failures per mode
    fm["cumulative_%"] = (fm["failed"].cumsum() / fm["failed"].sum() * 100).round(1)  # running share
    base = alt.Chart(fm).encode(x=alt.X("mode:N", sort=None, title=None))
    pareto = alt.layer(base.mark_bar(color=BLUE).encode(y=alt.Y("failed:Q", title="Failed batches")),
                       base.mark_line(color=RED, point=True).encode(y=alt.Y("cumulative_%:Q", title="Cumulative %", scale=alt.Scale(domain=[0, 100]))))
    st.altair_chart(pareto.resolve_scale(y="independent").properties(height=300), width="stretch")
    c = st.columns(2)                                                       # two distributions
    for col, (name, lims) in zip(c, [("actual_yield", None), ("assay_result_purity", (95, 105))]):
        h = alt.Chart(b[[name]].dropna()).mark_bar(color=BLUE).encode(
            x=alt.X(f"{name}:Q", bin=alt.Bin(maxbins=50), title=name.replace("_", " ")), y=alt.Y("count():Q", title="Batches"))
        if lims:                                                            # specification limits
            h = h + alt.Chart(pd.DataFrame({"x": lims})).mark_rule(color=RED, strokeDash=[5, 3]).encode(x="x:Q")
        col.altair_chart(h.properties(height=250, title=f"Distribution of {name.replace('_', ' ')}"), width="stretch")
    st.subheader("Failure rate by assay purity band")
    band = pd.cut(b["assay_result_purity"], [0, 95, 105, 200], labels=["below 95", "95-105", "above 105"])
    st.dataframe(b.groupby(band, observed=True)["batch_failed"].agg(batches="size", failure_rate_pct=lambda s: s.mean() * 100).round(1),
                 width="stretch")

# ------------------------------------------------------------------ 4. equipment
with tab[3]:
    c = st.columns(2)                                                       # two charts
    c[0].altair_chart(rate_chart(b, "maintenance_status", "Failure rate by maintenance status"), width="stretch")
    c[1].altair_chart(rate_chart(b, "age_band", "Failure rate by equipment age", sort=None), width="stretch")
    st.subheader("Adjusted odds ratios (all batches)")
    orr = odds_ratios(batches)                                              # cached model
    forest = alt.Chart(orr).encode(y=alt.Y("term:N", title=None))           # forest plot
    st.altair_chart((forest.mark_rule(color=BLUE, strokeWidth=3).encode(x=alt.X("or_ci_low:Q", title="Odds ratio (95% CI)"), x2="or_ci_high:Q")
                     + forest.mark_point(color=BLUE, filled=True, size=80).encode(x="odds_ratio:Q")
                     + alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color=GREY, strokeDash=[4, 3]).encode(x="x:Q")).properties(height=200),
                    width="stretch")
    st.dataframe(orr.round(3), hide_index=True, width="stretch")
    st.caption("Critical status more than doubles the odds of failure after adjusting for age, experience and product. Due Soon has no effect.")
    st.subheader("Machines: observed vs expected failures")
    eq = equipment_table(b)                                                 # ranking table
    st.altair_chart(alt.Chart(eq).mark_bar(color=ORANGE).encode(
        y=alt.Y("equipment_id:N", sort="-x", title=None), x=alt.X("standardised_ratio:Q", title="Observed / expected failures"),
        tooltip=list(eq.columns[:6])) + alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color=RED, strokeDash=[4, 3]).encode(x="x:Q"),
        width="stretch")
    st.dataframe(eq, width="stretch")

# ------------------------------------------------------------------ 5. process control
with tab[4]:
    limits = control_limits(batches)                                        # limits from passed batches up to 2021
    flags = flag_out_of_control(b, limits)                                  # out-of-control flags for the filter
    st.subheader("Share of readings within control limits")
    share = within_limits_share(flags)                                      # one row per variable
    st.altair_chart(alt.Chart(share.assign(pct=share["share_within_limits"] * 100)).mark_bar(color=BLUE, clip=True).encode(
        x=alt.X("variable:N", title=None), y=alt.Y("pct:Q", title="Within limits (%)", scale=alt.Scale(domain=[90, 100])),
        tooltip=["variable", alt.Tooltip("pct:Q", format=".2f"), "readings_out_of_control"]).properties(height=260)
        + alt.Chart(pd.DataFrame({"y": [90]})).mark_rule(color=RED).encode(y="y:Q"), width="stretch")
    st.subheader("Control chart")
    c = st.columns(2)                                                       # two pickers
    eq_id = c[0].selectbox("Machine", b["equipment_id"].value_counts().index.tolist())  # busiest first
    var = c[1].selectbox("Reading", ["temp_avg", "pressure_avg", "humidity_avg", "temp_variance", "pressure_variance"])
    d = b[b["equipment_id"] == eq_id].sort_values("production_start")      # one machine in time order
    lim = limits[(limits["manufacturing_section"] == d["manufacturing_section"].iloc[0]) & (limits["variable"] == var)].iloc[0]
    pts = alt.Chart(d.assign(outcome=np.where(d["batch_failed"] == 1, "Failed", "Passed"))).mark_circle(size=12, opacity=0.6).encode(
        x=alt.X("production_start:T", title=None), y=alt.Y(f"{var}:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["Passed", "Failed"], range=[GREY, RED])), tooltip=["batch_id", var, "outcome"])
    rules = alt.Chart(pd.DataFrame({"y": [lim["lcl"], lim["mean"], lim["ucl"]]})).mark_rule(color=BLUE, strokeDash=[5, 3]).encode(y="y:Q")
    st.altair_chart((pts + rules).properties(height=320), width="stretch")
    st.caption(f"{len(d):,} batches; dashed lines = lower limit, mean and upper limit for {d['manufacturing_section'].iloc[0]}.")
    st.subheader("Assay capability by section (target Cpk ≥ 1.33)")
    st.dataframe(capability(b).round(3), hide_index=True, width="stretch")

# ------------------------------------------------------------------ 6. risk checker
with tab[5]:
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
        c[1].metric("Alert threshold", f"{thr:.2f}")                        # 80%-recall threshold
        c[2].metric("Flag", "⚠️ Extra checks" if p >= thr else "✅ Standard")  # decision
        c[3].metric("Riskier than", f"{(past < p).mean():.0%} of past batches")  # position among past batches
    st.subheader("What the model weighs (standardised coefficients)")
    coef = pd.Series(model[-1].coef_.ravel(), index=build_features(train.head(1)).columns).rename(FEATURE_LABELS)  # push on log-odds
    cf = coef.reindex(coef.abs().sort_values(ascending=False).index).head(12).reset_index()
    cf.columns = ["input", "coefficient"]
    st.altair_chart(alt.Chart(cf).mark_bar().encode(
        y=alt.Y("input:N", sort=None, title=None), x=alt.X("coefficient:Q", title="Effect on log-odds of failure (per SD)"),
        color=alt.condition(alt.datum.coefficient > 0, alt.value(RED), alt.value(BLUE))).properties(height=320), width="stretch")
    st.caption("Red raises the risk, blue lowers it.")
    st.subheader("Model comparison (from the notebook)")
    st.dataframe(pd.DataFrame({"Model": ["Logistic regression (chosen)", "Random forest", "Gradient boosting"],
                               "Validation PR-AUC": [0.053, 0.043, 0.042], "Validation ROC-AUC": [0.594, 0.550, 0.539],
                               "Test ROC-AUC": [0.581, 0.564, 0.554], "Test recall": [0.82, 0.88, 0.85],
                               "Test precision": [0.040, 0.039, 0.038]}), hide_index=True, width="stretch")

# ------------------------------------------------------------------ 7. demand forecast
with tab[6]:
    md = monthly_demand(sales)                                              # units per month by category
    cat = st.selectbox("Category", ["Total"] + [c for c in md.columns if c != "Total"])
    series = md[cat]                                                        # chosen series
    with warnings.catch_warnings():                                         # hide statsmodels fitting notes
        warnings.simplefilter("ignore")
        acc, back = evaluate_series(series)                                 # 12-month holdout comparison
        hw = _holt_winters(series.to_numpy(), 12)                           # Holt-Winters, next 12 months
    future = pd.Series(hw,                                                  # forecast with month labels
                       index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS"))
    c = st.columns(3)                                                       # headline numbers
    c[0].metric("Next 12 months (units)", f"{future.sum():,.0f}")
    c[1].metric("Last 12 months (units)", f"{series.iloc[-12:].sum():,.0f}")
    c[2].metric("Holt-Winters gain vs baseline", f"{acc['WAPE_improvement_vs_baseline'].iloc[-1]:.1%}")
    st.subheader("History and 12-month forecast")
    st.line_chart(pd.DataFrame({"Actual": series, "Forecast (Holt-Winters)": future}), color=[BLUE, ORANGE])
    st.subheader("Back-test on the last 12 months")
    st.line_chart(back.set_index("month"))                                  # actual vs each method
    st.dataframe((acc.drop(columns="RMSE") * 100).round(1).add_suffix(" (%)"), width="stretch")  # WAPE, bias, gain
    st.subheader("Seasonality: units sold by year and month")
    heat = series.rename("units").reset_index()                             # long table for the heat map
    heat["year"], heat["month"] = heat["year_month"].dt.year, heat["year_month"].dt.strftime("%b")
    st.altair_chart(alt.Chart(heat).mark_rect().encode(
        x=alt.X("month:O", sort=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], title=None),
        y=alt.Y("year:O", title=None), color=alt.Color("units:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["year", "month", alt.Tooltip("units:Q", format=",.0f")]).properties(height=360), width="stretch")
    st.download_button("Download forecast (CSV)", future.rename("units").to_csv(), "forecast.csv")

# ------------------------------------------------------------------ 8. inventory
with tab[7]:
    c = st.columns(2)                                                       # two filters
    br = c[0].multiselect("Branch", sorted(policy["Branch_Code"].unique()), default=sorted(policy["Branch_Code"].unique()))
    ct = c[1].multiselect("Category", sorted(policy["category"].dropna().unique()),
                          default=sorted(policy["category"].dropna().unique()))
    p = policy[policy["Branch_Code"].isin(br) & policy["category"].isin(ct)]  # filtered items
    k = st.columns(4)                                                       # counts per action
    for col, s in zip(k, ["Reorder now", "Overstock", "No recent demand", "OK"]):
        col.metric(s, f"{(p['status'] == s).sum():,}")
    c = st.columns(2)                                                       # two charts
    st_counts = p.groupby(["category", "status"]).size().reset_index(name="items")  # items per category and action
    c[0].altair_chart(alt.Chart(st_counts).mark_bar().encode(
        y=alt.Y("category:N", title=None), x=alt.X("items:Q", stack="normalize", title="Share of items"),
        color=alt.Color("status:N", scale=alt.Scale(domain=["Reorder now", "Overstock", "No recent demand", "OK"],
                                                    range=[RED, ORANGE, GREY, BLUE])), tooltip=["category", "status", "items"])
        .properties(title="Stock status by category", height=280), width="stretch")
    flow = pd.DataFrame({"received": moves[moves["Quantity_Change"] > 0].groupby("Product_Category")["Quantity_Change"].sum(),
                         "sold": delivered.groupby("Product_Category")["Quantity"].sum()}).reset_index(names="category")
    c[1].altair_chart(alt.Chart(flow.melt("category", var_name="flow", value_name="units")).mark_bar().encode(
        y=alt.Y("category:N", title=None), x=alt.X("units:Q"), yOffset="flow:N",
        color=alt.Color("flow:N", scale=alt.Scale(range=[BLUE, ORANGE])), tooltip=["category", "flow", alt.Tooltip("units:Q", format=",")])
        .properties(title="Units received vs sold", height=280), width="stretch")
    flow["received per unit sold"] = (flow["received"] / flow["sold"]).round(2)
    st.dataframe(flow, hide_index=True, width="stretch")
    st.subheader("Top suppliers by purchase volume")
    rec = moves[moves["Quantity_Change"] > 0]                               # goods received
    sup = (rec.groupby("Supplier_ID")["Quantity_Change"].sum() / rec["Quantity_Change"].sum() * 100).sort_values(ascending=False)
    st.bar_chart(sup.head(10).rename("% of units received"), color=BLUE)    # supplier concentration
    st.subheader("Item list")
    st.caption("Assumptions: 14-day lead time, 95% service level, overstock above 180 days of cover. Stock rebuilt from zero.")
    show = st.radio("Show", ["Reorder now", "Overstock", "All"], horizontal=True)  # list filter
    st.dataframe(p if show == "All" else p[p["status"] == show], width="stretch")  # item list
    st.download_button("Download list (CSV)", p.to_csv(index=False), "reorder_policy.csv")

# ------------------------------------------------------------------ 9. model summary
with tab[8]:
    st.subheader("Model comparison, top features and choice for each problem")
    st.dataframe(pd.DataFrame([
        ["Batch failure", "Logistic, Random Forest, Gradient Boosting", "Logistic regression", "Test ROC-AUC 0.581", "Operator experience, equipment age, raw material"],
        ["Failure mode", "Majority guess, Random Forest", "None adopted (RF as a hint)", "Macro F1 0.141", "Pressure, production hours, humidity"],
        ["Maintenance effect", "Plant-wide vs within-machine logistic", "Plant-wide adjusted model", "Critical OR 2.33", "Critical status, experience, age"],
        ["Maintenance status", "Baseline, Logistic, Random Forest", "Random Forest", "Macro F1 0.501", "Month, equipment age, year"],
        ["Rise since 2019", "Trend only vs adjusted trend", "Adjusted model", "60% explained", "Equipment age"],
        ["Risk view", "Classifier alert vs High tier", "High risk tier", "Lift 1.71", "Classifier probability, equipment"],
        ["Monthly demand", "Seasonal naive, Moving average, Holt-Winters", "Holt-Winters", "33.8% WAPE gain", "– (time series)"],
        ["Weekly demand", "Seasonal naive, MA, Croston-SBA, Linear Regression", "Croston-SBA", "19.7% WAPE gain", "– (own history)"],
        ["Stock-out risk", "Baseline, Logistic, Random Forest, Gradient Boosting", "Gradient Boosting", "PR-AUC 0.278 (205×)", "Stock, weeks of cover, branch BR01"],
        ["Product groups", "K-Means, k = 2 to 6", "K-Means, k = 2", "Silhouette 0.540", "Weekly units, variability, price"],
        ["Reorder policy", "Rule, not a model", "Reorder-point rule", "128 reorder / 1,033 overstock", "–"],
    ], columns=["Problem", "Models compared", "Chosen", "Score", "Top features"]), hide_index=True, width="stretch")
    st.subheader("Recommendations")
    st.dataframe(pd.DataFrame([
        ["High", "QA / CAPA", "Corrective action on assay and dissolution failures", "47% of failures"],
        ["High", "Engineering", "Service equipment before it becomes Critical", "Odds of failure 2.3× on overdue equipment"],
        ["High", "Procurement", "Buy to the reorder-point policy; stop receipts for overstocked items", "Receipts about 2× sales"],
        ["Medium", "Branch stores", "Weekly reorder list and stock-out watch-list", "Catches 80% of stock-outs"],
        ["Medium", "Planning", "Holt-Winters for monthly volumes, Croston for weekly product demand", "34% / ~20% better than baseline"],
        ["Medium", "Finance", "Review credit-account terms and collections", "All unpaid balances are on credit accounts"],
        ["Low", "QA", "Use the risk tier only to target extra in-process checks", "Top 10% fail at 1.7× the average"],
    ], columns=["Priority", "Owner", "Action", "Evidence"]), hide_index=True, width="stretch")
