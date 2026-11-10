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
TONES = {"blue": ("#1f4e79", "#2e86c1"), "green": ("#1e8449", "#52be80"), "orange": ("#d35400", "#f5b041"),  # card gradients
         "red": ("#922b21", "#e74c3c"), "purple": ("#5b2c6f", "#a569bd"), "teal": ("#0e6655", "#48c9b0")}
st.markdown("""
<style>
.banner {background: linear-gradient(120deg, #0b2e4f 0%, #1f4e79 55%, #2e86c1 100%); color: #fff;
         padding: 22px 28px; border-radius: 16px; margin-bottom: 18px; box-shadow: 0 6px 18px rgba(31,78,121,.25);}
.banner h1 {color: #fff; font-size: 1.9rem; margin: 0; padding: 0;}
.banner p {color: #dbe9f6; margin: 6px 0 0 0; font-size: .95rem;}
.kpi {border-radius: 16px; padding: 16px 18px 14px 18px; color: #fff; min-height: 132px; margin-bottom: 12px;
      box-shadow: 0 6px 16px rgba(0,0,0,.14); transition: transform .15s ease, box-shadow .15s ease;}
.kpi:hover {transform: translateY(-3px); box-shadow: 0 10px 22px rgba(0,0,0,.20);}
.kpi .top {display: flex; justify-content: space-between; align-items: center;}
.kpi .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .92; font-weight: 600;}
.kpi .icon {font-size: 1.5rem; background: rgba(255,255,255,.18); border-radius: 50%; width: 42px; height: 42px;
            display: flex; align-items: center; justify-content: center;}
.kpi .value {font-size: clamp(1.25rem, 2vw, 1.95rem); font-weight: 800; line-height: 1.15; margin-top: 6px;}
.kpi .sub {font-size: .8rem; opacity: .92; margin-top: 2px;}
.kpi .bar {height: 6px; background: rgba(255,255,255,.25); border-radius: 4px; margin-top: 10px; overflow: hidden;}
.kpi .bar > div {height: 100%; background: #fff; border-radius: 4px;}
div[data-testid="stTabs"] button p {font-weight: 600;}
</style>""", unsafe_allow_html=True)                                         # page styles: banner and KPI cards


def kpi(col, icon, label, value, sub="", tone="blue", bar=None):            # one coloured KPI card
    a, z = TONES[tone]                                                      # gradient colours
    meter = f'<div class="bar"><div style="width:{min(max(bar, 0), 100):.0f}%"></div></div>' if bar is not None else ""  # optional progress bar
    col.markdown(f'''<div class="kpi" style="background: linear-gradient(135deg, {a}, {z});">
      <div class="top"><div class="label">{label}</div><div class="icon">{icon}</div></div>
      <div class="value">{value}</div><div class="sub">{sub}</div>{meter}</div>''', unsafe_allow_html=True)


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


# ------------------------------------------------------------------ commercial, finance and forecasting helpers
@st.cache_data(show_spinner=False)
def hw_forecast(y, h=12):                                                   # Holt-Winters: damped trend + monthly season
    y = y.astype(float)                                                     # numbers as floats
    try:
        import warnings                                                     # hide fitting notes
        from statsmodels.tsa.holtwinters import ExponentialSmoothing        # the forecasting model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = ExponentialSmoothing(y.to_numpy(), trend="add", damped_trend=True, seasonal="add",
                                     seasonal_periods=12).fit().forecast(h)  # fit and forecast h months
    except Exception:                                                       # statsmodels missing: last year x growth
        growth = y.iloc[-12:].sum() / y.iloc[-24:-12].sum()
        f = np.tile(y.iloc[-12:].to_numpy(), int(np.ceil(h / 12)))[:h] * growth
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")  # the next h months
    return pd.Series(np.asarray(f), index=idx)


def wape(actual, forecast):                                                 # weighted absolute percentage error
    return float(np.abs(np.asarray(actual) - np.asarray(forecast)).sum() / np.abs(np.asarray(actual)).sum())


def backtest(y):                                                            # hold out the last 12 months
    hold = y.iloc[-12:]                                                     # months to predict
    fc = pd.DataFrame({"Actual": hold.to_numpy(),
                       "Seasonal naive": y.iloc[-24:-12].to_numpy(),        # same month last year
                       "Moving average (12 m)": np.repeat(y.iloc[-24:-12].mean(), 12),  # average of the year before
                       "Holt-Winters": hw_forecast(y.iloc[:-12], 12).to_numpy()}, index=hold.index)
    acc = pd.DataFrame({"WAPE (%)": [wape(fc["Actual"], fc[m]) * 100 for m in fc.columns[1:]],
                        "Bias (%)": [(fc[m].sum() / fc["Actual"].sum() - 1) * 100 for m in fc.columns[1:]]}, index=fc.columns[1:])
    acc["Gain vs seasonal naive (%)"] = (1 - acc["WAPE (%)"] / acc.loc["Seasonal naive", "WAPE (%)"]) * 100
    return fc, acc.round(1)


@st.cache_data(show_spinner="Building the commercial tables...")
def finance_tables(sales, moves):
    s, m = sales.rename(columns=str.lower), moves.rename(columns=str.lower)  # one naming style
    cancelled = s["order_status"].eq("Cancelled")                           # void orders
    d = s[~cancelled].copy()                                                # delivered orders only
    unit_cost = m[m["movement_type"].eq("Sale_Out")].drop_duplicates("reference_id").set_index("reference_id")["unit_cost_kes"]
    d["cogs_kes"] = d["quantity"] * d["sale_id"].map(unit_cost)             # cost of goods sold per order
    d["gross_profit_kes"] = d["net_amount_excl_vat_kes"] - d["cogs_kes"]    # revenue less cost
    d["month"] = d["sale_date"].dt.to_period("M").dt.to_timestamp()        # month of sale
    d["age_days"] = (s["sale_date"].max() - d["sale_date"]).dt.days         # invoice age at the data end date
    stock = m.groupby(["branch_code", "product_id"])["quantity_change"].sum().clip(lower=0)  # units on hand (rebuilt from 0)
    last_cost = m.groupby("product_id")["unit_cost_kes"].last()             # latest unit cost
    stock_value = float((stock * stock.index.get_level_values("product_id").map(last_cost)).sum())  # stock at cost
    return d, s[cancelled], stock_value


def finance_page(sales, moves, pal, tn):                                    # the Commercial & finance view
    c1, c2, c3, grey = pal                                                  # chart colours for this app
    d, canc, stock_value = finance_tables(sales, moves)
    rev, gp, cogs = d["net_amount_excl_vat_kes"].sum(), d["gross_profit_kes"].sum(), d["cogs_kes"].sum()
    disc, gross = d["discount_amount_kes"].sum(), d["gross_amount_kes"].sum()
    due, billed = d["balance_due_kes"].sum(), d["total_amount_kes"].sum()
    cogs12 = d.loc[d["month"] > d["month"].max() - pd.DateOffset(months=12), "cogs_kes"].sum()  # last 12 months cost of sales
    k = st.columns(4)
    kpi(k[0], "💰", "Net revenue", f"KES {rev / 1e9:.2f} bn", f"{len(d):,} delivered orders, 2014–2025", tn[0])
    kpi(k[1], "📈", "Gross profit", f"KES {gp / 1e9:.2f} bn", f"{gp / rev:.1%} gross margin", tn[1], bar=gp / rev * 100)
    kpi(k[2], "🏷️", "Discounts given", f"KES {disc / 1e6:,.0f} m", f"{disc / gross:.1%} of gross sales", tn[2], bar=disc / gross * 100)
    kpi(k[3], "🧾", "Receivables", f"KES {due / 1e6:,.0f} m", f"{1 - due / billed:.1%} of billed value collected", tn[3],
        bar=(1 - due / billed) * 100)
    k = st.columns(4)
    kpi(k[0], "🏛️", "VAT billed", f"KES {d['vat_amount_kes'].sum() / 1e9:.2f} bn", "at 16% on net sales", tn[4])
    kpi(k[1], "❌", "Cancelled orders", f"KES {canc['net_amount_excl_vat_kes'].sum() / 1e6:,.0f} m", f"{len(canc):,} orders not delivered", tn[2])
    kpi(k[2], "📦", "Stock at cost", f"KES {stock_value / 1e9:.1f} bn", f"{stock_value / (cogs12 / 12):,.0f} months of cost of sales", tn[3])
    kpi(k[3], "🛍️", "Average order", f"KES {rev / len(d):,.0f}", "net, excluding VAT", tn[5])

    st.subheader("Revenue, gross profit and margin by year")
    yr = d.groupby(d["sale_date"].dt.year.rename("year")).agg(revenue=("net_amount_excl_vat_kes", "sum"),
                                                              gross_profit=("gross_profit_kes", "sum")).reset_index()
    yr["margin_pct"] = yr["gross_profit"] / yr["revenue"] * 100
    yl = yr.melt(["year", "margin_pct"], ["revenue", "gross_profit"], "measure", "kes")
    yl["kes_m"] = yl["kes"] / 1e6
    bars = alt.Chart(yl).mark_bar().encode(x=alt.X("year:O", title=None), xOffset="measure:N",
                                           y=alt.Y("kes_m:Q", title="KES million"),
                                           color=alt.Color("measure:N", scale=alt.Scale(range=[c2, c1]), title=None),
                                           tooltip=["year", "measure", alt.Tooltip("kes_m:Q", format=",.0f")])
    line = alt.Chart(yr).mark_line(color=c3, point=True).encode(x="year:O", y=alt.Y("margin_pct:Q", title="Gross margin (%)"),
                                                                tooltip=["year", alt.Tooltip("margin_pct:Q", format=".1f")])
    st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=320), width="stretch")

    c = st.columns(2)
    cat = d.groupby("product_category").agg(revenue_kes_m=("net_amount_excl_vat_kes", lambda x: x.sum() / 1e6),
                                            gross_profit_kes_m=("gross_profit_kes", lambda x: x.sum() / 1e6),
                                            discount_kes_m=("discount_amount_kes", lambda x: x.sum() / 1e6),
                                            orders=("sale_id", "size"))
    cat["margin_pct"] = cat["gross_profit_kes_m"] / cat["revenue_kes_m"] * 100
    cat = cat.sort_values("gross_profit_kes_m", ascending=False).round(1)
    c[0].altair_chart(alt.Chart(cat.reset_index()).mark_bar().encode(
        y=alt.Y("product_category:N", sort="-x", title=None), x=alt.X("gross_profit_kes_m:Q", title="Gross profit (KES m)"),
        color=alt.Color("margin_pct:Q", scale=alt.Scale(scheme="greens"), title="Margin %"),
        tooltip=list(cat.reset_index().columns)).properties(height=260, title="Gross profit by category (colour = margin)"), width="stretch")
    c[0].dataframe(cat, width="stretch")
    br = d.groupby("branch_code").agg(revenue_kes_m=("net_amount_excl_vat_kes", lambda x: x.sum() / 1e6),
                                      gross_profit_kes_m=("gross_profit_kes", lambda x: x.sum() / 1e6),
                                      orders=("sale_id", "size"), avg_order_kes=("net_amount_excl_vat_kes", "mean"),
                                      unpaid_kes_m=("balance_due_kes", lambda x: x.sum() / 1e6))
    br["margin_pct"] = br["gross_profit_kes_m"] / br["revenue_kes_m"] * 100
    br["share_of_revenue_pct"] = br["revenue_kes_m"] / br["revenue_kes_m"].sum() * 100
    c[1].altair_chart(alt.Chart(br.reset_index()).mark_arc(innerRadius=60).encode(
        theta="revenue_kes_m:Q", color=alt.Color("branch_code:N", title="Branch"),
        tooltip=["branch_code", alt.Tooltip("revenue_kes_m:Q", format=",.0f"), alt.Tooltip("margin_pct:Q", format=".1f")])
        .properties(height=260, title="Revenue share by branch"), width="stretch")
    c[1].dataframe(br.round(1), width="stretch")

    st.subheader("Profitability")
    last = d["month"] > d["month"].max() - pd.DateOffset(months=12)         # last 12 months of sales
    rev12, gp12, cogs12b = (d.loc[last, c].sum() for c in ["net_amount_excl_vat_kes", "gross_profit_kes", "cogs_kes"])
    k = st.columns(4)
    kpi(k[0], "📊", "Gross profit margin", f"{gp / rev:.1%}", f"last 12 months {gp12 / rev12:.1%}", tn[1], bar=gp / rev * 100)
    kpi(k[1], "➕", "Markup on cost", f"{gp / cogs:.1%}", "gross profit ÷ cost of goods sold", tn[4])
    kpi(k[2], "🧮", "Gross profit per order", f"KES {gp / len(d):,.0f}", f"on an average order of KES {rev / len(d):,.0f}", tn[5])
    kpi(k[3], "💸", "Discount share of profit", f"{disc / gp:.1%}", "discounts given ÷ gross profit", tn[2], bar=disc / gp * 100)
    mm = d.groupby("month").agg(revenue=("net_amount_excl_vat_kes", "sum"), gp=("gross_profit_kes", "sum"))
    mm["Monthly margin (%)"] = mm["gp"] / mm["revenue"] * 100               # gross margin each month
    mm["12-month rolling margin (%)"] = mm["gp"].rolling(12).sum() / mm["revenue"].rolling(12).sum() * 100
    st.markdown("**Gross profit margin over time**")
    st.line_chart(mm[["Monthly margin (%)", "12-month rolling margin (%)"]], color=[grey, c1])
    c = st.columns(2)
    dl = d.groupby("discount_pct").agg(orders=("sale_id", "size"), revenue_kes_m=("net_amount_excl_vat_kes", lambda x: x.sum() / 1e6),
                                       gross_profit_kes_m=("gross_profit_kes", lambda x: x.sum() / 1e6))
    dl["margin_pct"] = dl["gross_profit_kes_m"] / dl["revenue_kes_m"] * 100
    c[0].altair_chart(alt.Chart(dl.reset_index()).mark_bar(color=c3).encode(
        x=alt.X("discount_pct:O", title="Discount (%)"), y=alt.Y("margin_pct:Q", title="Gross margin (%)"),
        tooltip=["discount_pct", "orders", alt.Tooltip("revenue_kes_m:Q", format=",.0f"), alt.Tooltip("margin_pct:Q", format=".1f")])
        .properties(height=260, title="Margin falls as the discount rises"), width="stretch")
    c[0].dataframe(dl.round(1), width="stretch")
    rep = d.groupby("sales_rep_id").agg(orders=("sale_id", "size"), revenue_kes_m=("net_amount_excl_vat_kes", lambda x: x.sum() / 1e6),
                                        gross_profit_kes_m=("gross_profit_kes", lambda x: x.sum() / 1e6),
                                        discount_pct=("discount_pct", "mean"), unpaid_kes_m=("balance_due_kes", lambda x: x.sum() / 1e6))
    rep["margin_pct"] = rep["gross_profit_kes_m"] / rep["revenue_kes_m"] * 100
    c[1].altair_chart(alt.Chart(rep.reset_index()).mark_bar().encode(
        y=alt.Y("sales_rep_id:N", sort="-x", title=None), x=alt.X("gross_profit_kes_m:Q", title="Gross profit (KES m)"),
        color=alt.Color("margin_pct:Q", scale=alt.Scale(scheme="greens"), title="Margin %"),
        tooltip=["sales_rep_id", "orders", alt.Tooltip("gross_profit_kes_m:Q", format=",.1f"), alt.Tooltip("margin_pct:Q", format=".1f")])
        .properties(height=260, title="Gross profit by sales rep"), width="stretch")
    c[1].dataframe(rep.sort_values("gross_profit_kes_m", ascending=False).round(1), width="stretch")
    prod = d.groupby(["product_id", "product_category"]).agg(revenue_kes_m=("net_amount_excl_vat_kes", lambda x: x.sum() / 1e6),
                                                             gross_profit_kes_m=("gross_profit_kes", lambda x: x.sum() / 1e6),
                                                             units=("quantity", "sum")).reset_index()
    prod["margin_pct"] = prod["gross_profit_kes_m"] / prod["revenue_kes_m"] * 100
    c = st.columns(2)
    c[0].markdown("**Top 10 products by gross profit**")
    c[0].dataframe(prod.nlargest(10, "gross_profit_kes_m").round(2), hide_index=True, width="stretch")
    c[1].markdown("**Lowest-margin 10 products**")
    c[1].dataframe(prod.nsmallest(10, "margin_pct").round(2), hide_index=True, width="stretch")

    st.subheader("Working capital and customers")
    billed12 = d.loc[last, "total_amount_kes"].sum()                         # value invoiced in the last 12 months
    dso, turns = due / billed12 * 365, cogs12b / stock_value                # days sales outstanding, stock turns a year
    top10 = d.groupby("customer_id")["net_amount_excl_vat_kes"].sum().nlargest(10).sum() / rev  # customer concentration
    k = st.columns(4)
    kpi(k[0], "⏳", "Days sales outstanding", f"{dso:,.0f} days", "unpaid ÷ last 12 months billed × 365", tn[3])
    kpi(k[1], "🔄", "Inventory turnover", f"{turns:.2f}× a year", f"about {365 / turns:,.0f} days of stock on hand", tn[2])
    kpi(k[2], "💹", "GMROI", f"{gp12 / stock_value:.3f}", "gross profit per KES 1 of stock at cost", tn[5])
    kpi(k[3], "👥", "Top 10 customers", f"{top10:.1%}", f"of revenue · {d['customer_id'].nunique():,} customers", tn[4], bar=top10 * 100)
    st.caption("Stock is rebuilt from zero (no opening stock in the data) and is mostly Equipment, so turnover is very low and GMROI tiny: "
               "the clearest sign that purchasing runs far ahead of sales. Net profit cannot be calculated because the data has no operating costs.")

    st.subheader("Receivables (unpaid balances)")
    ag = d[d["balance_due_kes"] > 0].copy()                                 # invoices with money owed
    ag["age"] = pd.cut(ag["age_days"], [-1, 30, 90, 180, 365, 100000], labels=["0–30 days", "31–90", "91–180", "181–365", "over 1 year"])
    aging = ag.groupby("age", observed=False)["balance_due_kes"].sum().div(1e6).reset_index(name="kes_m")
    c = st.columns(2)
    c[0].altair_chart(alt.Chart(aging).mark_bar(color=c3).encode(
        x=alt.X("age:N", sort=None, title="Invoice age at the data end date"), y=alt.Y("kes_m:Q", title="Unpaid (KES m)"),
        tooltip=["age", alt.Tooltip("kes_m:Q", format=",.1f")]).properties(height=260, title="Unpaid balance by invoice age"), width="stretch")
    pm = d.groupby("payment_method").agg(billed_kes_m=("total_amount_kes", lambda x: x.sum() / 1e6),
                                         unpaid_kes_m=("balance_due_kes", lambda x: x.sum() / 1e6))
    pm["unpaid_pct"] = pm["unpaid_kes_m"] / pm["billed_kes_m"] * 100
    c[1].markdown("**Billed and unpaid by payment method**")
    c[1].dataframe(pm.round(1), width="stretch")
    c[1].markdown("**Top 10 customers by unpaid balance**")
    c[1].dataframe(d.groupby("customer_id").agg(unpaid_kes_m=("balance_due_kes", lambda x: x.sum() / 1e6), orders=("sale_id", "size"))
                   .sort_values("unpaid_kes_m", ascending=False).head(10).round(2), width="stretch")

    st.subheader("Revenue forecast (next 12 months)")
    choice = st.selectbox("Revenue series", ["Total"] + sorted(d["product_category"].unique()), key="fin_series")
    dd = d if choice == "Total" else d[d["product_category"] == choice]     # chosen category
    y = dd.groupby("month")["net_amount_excl_vat_kes"].sum().asfreq("MS", fill_value=0) / 1e6  # monthly revenue, KES m
    fc, acc = backtest(y)                                                   # accuracy on the last 12 months
    future = hw_forecast(y, 12)                                             # next 12 months
    m12 = dd.loc[dd["month"] > dd["month"].max() - pd.DateOffset(months=12)]
    margin12 = m12["gross_profit_kes"].sum() / m12["net_amount_excl_vat_kes"].sum()  # trailing 12-month margin
    k = st.columns(4)
    growth = future.sum() / y.iloc[-12:].sum() - 1
    kpi(k[0], "🔮", "Forecast revenue", f"KES {future.sum():,.0f} m", "next 12 months", tn[0])
    kpi(k[1], "📈" if growth >= 0 else "📉", "Change vs last 12 months", f"{growth:+.1%}", f"last 12 months KES {y.iloc[-12:].sum():,.0f} m",
        tn[1] if growth >= 0 else tn[2])
    kpi(k[2], "💵", "Projected gross profit", f"KES {future.sum() * margin12:,.0f} m", f"at the last 12 months' margin ({margin12:.1%})", tn[4])
    kpi(k[3], "🎯", "Forecast error", f"{acc.loc['Holt-Winters', 'WAPE (%)']:.1f}%",
        f"WAPE on last 12 months vs {acc.loc['Seasonal naive', 'WAPE (%)']:.1f}% seasonal naive", tn[5])
    st.line_chart(pd.DataFrame({"Actual (KES m)": y, "Forecast (KES m)": future}), color=[c1, c3])
    c = st.columns(2)
    c[0].markdown("**Back-test on the last 12 months (KES m)**")
    c[0].line_chart(fc)
    c[1].markdown("**Accuracy of each method**")
    c[1].dataframe(acc, width="stretch")
    out = pd.DataFrame({"forecast_revenue_kes_m": future.round(1), "projected_gross_profit_kes_m": (future * margin12).round(1)})
    c[1].markdown("**Monthly forecast**")
    c[1].dataframe(out, width="stretch")
    st.download_button("Download revenue forecast (CSV)", out.to_csv(), "revenue_forecast.csv", key="fin_dl")
    st.caption("Gross profit uses the unit cost recorded on each sale's stock movement. Receivables age is measured at the last sale date "
               "in the data. Forecasts use Holt-Winters (damped trend, monthly seasonality).")


def finance_headline(sales, moves):                                         # three numbers for the executive summary
    d, canc, stock_value = finance_tables(sales, moves)
    rev, gp = d["net_amount_excl_vat_kes"].sum(), d["gross_profit_kes"].sum()
    y = d.groupby("month")["net_amount_excl_vat_kes"].sum().asfreq("MS", fill_value=0) / 1e6
    return dict(revenue=rev, margin=gp / rev, unpaid=d["balance_due_kes"].sum(), stock_value=stock_value,
                forecast=hw_forecast(y, 12).sum(), last12=y.iloc[-12:].sum(), cancelled=canc["net_amount_excl_vat_kes"].sum())


def exec_commercial(sales, moves, tn):                                      # commercial picture for the executive summary
    h = finance_headline(sales, moves)
    st.subheader("Commercial picture")
    k = st.columns(4)
    kpi(k[0], "💰", "Revenue & margin", f"KES {h['revenue'] / 1e9:.1f} bn", f"{h['margin']:.1%} gross margin, 2014–2025", tn[0], bar=h["margin"] * 100)
    kpi(k[1], "🔮", "Next 12 months", f"KES {h['forecast'] / 1e3:.2f} bn", f"{h['forecast'] / h['last12'] - 1:+.1%} vs last 12 months", tn[1])
    kpi(k[2], "🧾", "Receivables", f"KES {h['unpaid'] / 1e6:,.0f} m", "unpaid, all on credit accounts", tn[3])
    kpi(k[3], "📦", "Stock at cost", f"KES {h['stock_value'] / 1e9:.1f} bn", "cash tied up in inventory", tn[2])
    st.caption(f"Cancelled orders took KES {h['cancelled'] / 1e6:,.0f} m of revenue off the books. Freeing over-bought stock and collecting "
               "credit-account balances are the two biggest cash levers; see the Commercial & finance view for detail.")


# ------------------------------------------------------------------ what-if predictor (enter values, see the predicted outcome)
from sklearn.compose import make_column_transformer                         # treat text and number inputs differently
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # tree models
from sklearn.impute import SimpleImputer as _Imputer                        # fill gaps
from sklearn.linear_model import LogisticRegression as _Logit               # failure-risk model
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score    # how good each model is
from sklearn.model_selection import train_test_split                        # hold-out test set
from sklearn.pipeline import make_pipeline as _pipe                         # chain the steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler as _Scaler  # encode text, scale numbers

P_NUM = ["batch_size", "production_hours", "target_yield", "temp_avg", "temp_variance", "pressure_avg", "pressure_variance",
         "humidity_avg", "raw_material_quality", "equipment_age_years", "maintenance_days_ago", "operator_experience_years"]
P_CAT = ["manufacturing_section", "dosage_form", "maintenance_status"]
S_NUM = ["equipment_age_years", "maintenance_days_ago", "operator_experience_years", "batch_size", "production_hours"]
S_CAT = ["manufacturing_section"]
LABELS = {"batch_size": "Batch size", "production_hours": "Production hours", "target_yield": "Target yield (%)",
          "temp_avg": "Average temperature (°C)", "temp_variance": "Temperature variance", "pressure_avg": "Average pressure",
          "pressure_variance": "Pressure variance", "humidity_avg": "Average humidity (%)", "raw_material_quality": "Raw material quality",
          "equipment_age_years": "Equipment age (years)", "maintenance_days_ago": "Days since maintenance",
          "operator_experience_years": "Operator experience (years)"}


def _prep(num, cat, scale=False):                                           # imputing, encoding (and scaling) step
    num_steps = [_Imputer(strategy="median")] + ([_Scaler()] if scale else [])
    return make_column_transformer((_pipe(*num_steps), num),
                                   (_pipe(_Imputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat))


@st.cache_resource(show_spinner="Training the what-if models (about 30 seconds, once)...")
def whatif_models(_df, target, key):                                        # key names the dataset for the cache
    df = _df.dropna(subset=["maintenance_status"])
    X, y = df[P_NUM + P_CAT], df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)  # 80/20 split
    risk = _pipe(_prep(P_NUM, P_CAT, scale=True), _Logit(class_weight="balanced", max_iter=2000)).fit(Xtr, ytr)  # failure risk
    yv = df["actual_yield"]
    ytr_y, yte_y = yv.loc[Xtr.index], yv.loc[Xte.index]
    ymodel = _pipe(_prep(P_NUM, P_CAT), RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=5,
                                                               n_jobs=-1, random_state=42)).fit(Xtr[ytr_y.notna()], ytr_y.dropna())
    ok = yte_y.notna()
    scores = {"risk_auc": roc_auc_score(yte, risk.predict_proba(Xte)[:, 1]),                  # failure model ranking quality
              "yield_mae": mean_absolute_error(yte_y[ok], ymodel.predict(Xte[ok])),           # average yield error
              "yield_base": mean_absolute_error(yte_y[ok], np.repeat(ytr_y.mean(), ok.sum()))}
    train_risk = risk.predict_proba(Xtr)[:, 1]                              # past scores, for the risk bands
    return risk, ymodel, scores, np.percentile(train_risk, [50, 90]), train_risk.mean()


def predictor_tab(df, target, key, pal, tn, sales, moves, policy):          # the What-if predictor view
    c1, c2, c3, grey = pal
    risk, ymodel, scores, (p50, p90), avg_risk = whatif_models(df, target, key)
    part = st.radio("What do you want to predict?", ["Batch outcome", "Equipment maintenance status", "Stock and reorder",
                                                     "Price and discount scenario"], horizontal=True, key=f"wi_{key}")
    q = lambda col, p: float(np.nanpercentile(df[col], p))                  # a percentile of a column, for input limits
    med = lambda col: float(df[col].median())

    if part == "Batch outcome":
        st.caption(f"Enter a planned batch. Failure risk: logistic regression (test ROC-AUC {scores['risk_auc']:.2f}, a weak signal). "
                   f"Yield: random forest (test MAE {scores['yield_mae']:.2f} points vs {scores['yield_base']:.2f} for the average).")
        c = st.columns(3)
        v = {"manufacturing_section": c[0].selectbox("Manufacturing section", sorted(df["manufacturing_section"].unique()), key=f"s_{key}"),
             "dosage_form": c[0].selectbox("Dosage form", sorted(df["dosage_form"].unique()), key=f"d_{key}"),
             "maintenance_status": c[0].selectbox("Maintenance status", ["Current", "Due Soon", "Critical"], key=f"m_{key}")}
        for i, col in enumerate(P_NUM):                                      # one number box per input, prefilled with the median
            v[col] = c[(i % 2) + 1].number_input(LABELS[col], q(col, 0.5), q(col, 99.5), round(med(col), 2), key=f"n_{col}_{key}")
        row = pd.DataFrame([v])
        p, yl = risk.predict_proba(row)[0, 1], ymodel.predict(row)[0]
        band = "High" if p >= p90 else ("Medium" if p >= p50 else "Low")
        k = st.columns(4)
        kpi(k[0], "🎯", "Failure risk score", f"{p:.2f}", f"{p / avg_risk:.1f}× the average batch", {"High": tn[2], "Medium": tn[5], "Low": tn[1]}[band], bar=p * 100)
        kpi(k[1], "🚦", "Risk band", band, "High = top 10% of past batches", {"High": tn[2], "Medium": tn[5], "Low": tn[1]}[band])
        kpi(k[2], "⚗️", "Predicted yield", f"{yl:.2f}%", f"target {v['target_yield']:.2f}%", tn[0])
        gap = yl - v["target_yield"]
        kpi(k[3], "📏", "Yield vs target", f"{gap:+.2f} pts", "above target" if gap >= 0 else "below target", tn[1] if gap >= 0 else tn[2])
        st.markdown("**See how the prediction moves when you change one input** (all other inputs stay as entered)")
        vary = st.selectbox("Input to vary", [LABELS[c] for c in P_NUM] + ["Maintenance status"], key=f"vary_{key}")
        if vary == "Maintenance status":
            grid = pd.concat([row.assign(maintenance_status=s) for s in ["Current", "Due Soon", "Critical"]], ignore_index=True)
            xs = ["Current", "Due Soon", "Critical"]
        else:
            col = next(c for c in P_NUM if LABELS[c] == vary)
            xs = np.linspace(q(col, 2), q(col, 98), 25)
            grid = pd.concat([row] * len(xs), ignore_index=True)
            grid[col] = xs
        curve = pd.DataFrame({"value": xs, "Failure risk score": risk.predict_proba(grid)[:, 1], "Predicted yield (%)": ymodel.predict(grid)})
        cc = st.columns(2)
        for col_, measure, colour in [(cc[0], "Failure risk score", c3), (cc[1], "Predicted yield (%)", c1)]:
            ch = alt.Chart(curve).mark_line(point=True, color=colour).encode(
                x=alt.X("value:" + ("N" if vary == "Maintenance status" else "Q"), title=vary, sort=None),
                y=alt.Y(f"{measure}:Q", scale=alt.Scale(zero=False)), tooltip=["value", alt.Tooltip(f"{measure}:Q", format=".3f")])
            col_.altair_chart(ch.properties(height=260, title=measure), width="stretch")

    elif part == "Equipment maintenance status":
        lim = df.groupby("maintenance_status")["maintenance_days_ago"].agg(["min", "max"])  # the service rule seen in the data
        due, crit = int(lim.loc["Due Soon", "min"]), int(lim.loc["Critical", "min"])
        st.caption(f"In this data the status follows the service schedule exactly: Current up to {due - 1} days since the last service, "
                   f"Due Soon from {due} days, Critical from {crit} days. Enter a machine's details to see its status and risk.")
        c = st.columns(3)
        days = c[0].number_input("Days since maintenance", 0, 400, 60, key=f"ed_{key}")
        age = c[1].number_input("Equipment age (years)", 0.0, 40.0, round(med("equipment_age_years"), 1), key=f"ea_{key}")
        sec = c[2].selectbox("Manufacturing section", sorted(df["manufacturing_section"].unique()), key=f"es_{key}")
        pred = "Critical" if days >= crit else ("Due Soon" if days >= due else "Current")
        nxt = (due - days) if pred == "Current" else ((crit - days) if pred == "Due Soon" else 0)
        hist = df.groupby("maintenance_status")[target].mean()              # failure rate seen for each status
        near = df[(df["manufacturing_section"] == sec) & (df["equipment_age_years"].sub(age).abs() <= 2)]  # similar machines
        tone = {"Current": tn[1], "Due Soon": tn[5], "Critical": tn[2]}
        k = st.columns(4)
        kpi(k[0], "🔧", "Status", pred, f"{days} days since the last service", tone[pred])
        kpi(k[1], "⏱️", "Days to next stage", "now" if pred == "Critical" else f"{nxt}",
            "Critical: service immediately" if pred == "Critical" else ("until Due Soon" if pred == "Current" else "until Critical"), tone[pred])
        kpi(k[2], "⚠️", "Failure rate at this status", f"{hist[pred]:.1%}", f"vs {hist['Current']:.1%} when Current", tone[pred])
        kpi(k[3], "🏭", "Similar machines", f"{near[target].mean():.1%}" if len(near) else "–",
            f"failure rate, {sec}, age ±2 years ({len(near):,} batches)", tn[4])
        bins = pd.cut(df["maintenance_days_ago"], range(0, 200, 10))       # 10-day bands
        fr = df.groupby(bins, observed=True)[target].agg(rate="mean", batches="size").reset_index()
        fr["days"] = fr["maintenance_days_ago"].apply(lambda b: b.left + 5)
        fr["rate"] *= 100
        bars = alt.Chart(fr).mark_bar(color=c1, opacity=0.8).encode(
            x=alt.X("days:Q", title="Days since maintenance"), y=alt.Y("rate:Q", title="Failure rate (%)"),
            tooltip=["days", alt.Tooltip("rate:Q", format=".2f"), "batches"])
        marks = alt.Chart(pd.DataFrame({"x": [due, crit], "label": ["Due Soon", "Critical"]})).mark_rule(color=c3, strokeDash=[5, 3]).encode(x="x:Q")
        you = alt.Chart(pd.DataFrame({"x": [days]})).mark_rule(color="black", strokeWidth=3).encode(x="x:Q")
        st.altair_chart((bars + marks + you).properties(height=280, title="Failure rate by days since maintenance (black line = your machine)"),
                        width="stretch")

    elif part == "Stock and reorder":
        pol = policy.rename(columns=str.lower)                              # one naming style for both dashboards
        c = st.columns(2)
        br = c[0].selectbox("Branch", sorted(pol["branch_code"].unique()), key=f"rb_{key}")
        pid = c[1].selectbox("Product", sorted(pol.loc[pol["branch_code"] == br, "product_id"]), key=f"rp_{key}")
        it = pol[(pol["branch_code"] == br) & (pol["product_id"] == pid)].iloc[0]
        st.caption(f"Prefilled from the data for {pid} ({it['category']}) at {br}: change any value to test a scenario.")
        c = st.columns(5)
        stock = c[0].number_input("Stock on hand (units)", 0.0, 1e7, float(max(it["stock_now"], 0)), key=f"rs_{key}")
        wk = c[1].number_input("Average weekly demand", 0.0, 1e5, float(it["avg_weekly"]), key=f"rw_{key}")
        sd = c[2].number_input("Weekly demand spread (SD)", 0.0, 1e5, float(it["sd_weekly"]), key=f"rd_{key}")
        lead = c[3].number_input("Lead time (days)", 1, 120, 14, key=f"rl_{key}")
        cover = c[4].number_input("Cover to order up to (weeks)", 1, 52, 8, key=f"rc_{key}")
        z = 1.65                                                            # 95% service level
        daily = wk / 7
        safety = z * sd / np.sqrt(7) * np.sqrt(lead)                         # buffer for the lead time
        rop = daily * lead + safety                                         # reorder point
        days = stock / daily if daily > 0 else np.inf                       # days until stock runs out
        order = max(0.0, rop + cover * wk - stock)                          # order up to reorder point + cover
        status = "No demand" if daily == 0 else ("Reorder now" if stock <= rop else ("Overstock" if days > 180 else "OK"))
        k = st.columns(4)
        kpi(k[0], "📅", "Days of cover", "∞" if np.isinf(days) else f"{days:,.0f}", "at the entered demand", tn[0])
        kpi(k[1], "📍", "Reorder point", f"{rop:,.0f}", f"incl. safety stock {safety:,.0f}", tn[4])
        kpi(k[2], "🛒", "Suggested order", f"{order:,.0f} units", f"to cover lead time + {cover} weeks", tn[5])
        kpi(k[3], "🚦", "Status", status, "95% service level", {"Reorder now": tn[2], "Overstock": tn[5], "OK": tn[1]}.get(status, tn[3]))
        weeks = np.arange(0, 27)
        proj = pd.DataFrame({"week": weeks, "Stock if no order": np.maximum(stock - wk * weeks, 0),
                             "Reorder point": rop})
        st.line_chart(proj.set_index("week"), color=[c1, c3])
        st.caption("Projected stock over the next 26 weeks at the entered demand, against the reorder point.")

    else:
        d, _, _ = finance_tables(sales, moves)
        cat = st.selectbox("Category", ["All"] + sorted(d["product_category"].unique()), key=f"pc_{key}")
        base = d[d["month"] > d["month"].max() - pd.DateOffset(months=12)]   # last 12 months as the base
        base = base if cat == "All" else base[base["product_category"] == cat]
        R, C = base["net_amount_excl_vat_kes"].sum(), base["cogs_kes"].sum()  # base revenue and cost
        c = st.columns(4)
        price = c[0].slider("Price change (%)", -30, 30, 0, key=f"pp_{key}")
        disc = c[1].slider("Extra discount (points)", -5, 15, 0, key=f"pd_{key}")
        vol = c[2].slider("Volume change (%)", -50, 50, 0, key=f"pv_{key}")
        cost = c[3].slider("Unit cost change (%)", -20, 30, 0, key=f"pk_{key}")
        R2 = R * (1 + price / 100) * (1 - disc / 100) * (1 + vol / 100)     # scenario revenue
        C2 = C * (1 + cost / 100) * (1 + vol / 100)                         # scenario cost
        G, G2 = R - C, R2 - C2
        per_unit = R * (1 + price / 100) * (1 - disc / 100) - C * (1 + cost / 100)  # margin per base volume
        be = (G / per_unit - 1) * 100 if per_unit > 0 else np.nan            # volume change that keeps gross profit flat
        k = st.columns(4)
        kpi(k[0], "💰", "Revenue (12 months)", f"KES {R2 / 1e6:,.0f} m", f"{R2 / R - 1:+.1%} vs KES {R / 1e6:,.0f} m", tn[0])
        kpi(k[1], "📈", "Gross profit", f"KES {G2 / 1e6:,.0f} m", f"{G2 / G - 1:+.1%} vs KES {G / 1e6:,.0f} m", tn[1] if G2 >= G else tn[2])
        kpi(k[2], "📊", "Gross margin", f"{G2 / R2:.1%}" if R2 else "–", f"base {G / R:.1%}", tn[4])
        kpi(k[3], "⚖️", "Break-even volume change", "n/a" if np.isnan(be) else f"{be:+.1f}%",
            "needed to keep gross profit flat", tn[5])
        cmp = pd.DataFrame({"case": ["Base", "Base", "Scenario", "Scenario"], "measure": ["Revenue", "Gross profit"] * 2,
                            "kes_m": [R / 1e6, G / 1e6, R2 / 1e6, G2 / 1e6]})
        st.altair_chart(alt.Chart(cmp).mark_bar().encode(
            x=alt.X("measure:N", title=None), xOffset="case:N", y=alt.Y("kes_m:Q", title="KES million"),
            color=alt.Color("case:N", scale=alt.Scale(range=[grey, c1])), tooltip=["case", "measure", alt.Tooltip("kes_m:Q", format=",.0f")])
            .properties(height=280, title="Last 12 months: base vs your scenario"), width="stretch")
        st.caption("Simple arithmetic on the last 12 months, with no price elasticity: set the volume change you expect from the price "
                   "or discount change yourself.")


# ------------------------------------------------------------------ executive summary: how the features drove the findings
def _clean(name):                                                           # readable feature names from the model pipeline
    n = name.split("__", 1)[-1]
    for cat in P_CAT:
        if n.startswith(cat + "_"):
            return f"{cat.replace('_', ' ').capitalize()}: {n[len(cat) + 1:]}"
    return LABELS.get(n, n)


def _band_effect(df, target, col, label):                                   # failure rate in the lowest vs highest quarter
    lo, hi = df[col].quantile([0.25, 0.75])
    a, b = df.loc[df[col] <= lo, target].mean(), df.loc[df[col] >= hi, target].mean()
    return [label, f"≤ {lo:,.1f}", f"{a:.2%}", f"≥ {hi:,.1f}", f"{b:.2%}", (b - a) * 100]


def exec_drivers(df, target, key, pal, tn):
    c1, c2, c3, grey = pal
    risk, ymodel, scores, _, _ = whatif_models(df, target, key)             # the same models as the What-if tab
    st.subheader("How the features drove each finding")
    st.caption("Every number below is computed from the 60,000 batches; model weights come from the models in the What-if predictor tab.")

    # ---- A. batch failure
    st.markdown("#### A. Why do batches fail? — the drivers of failure risk")
    names = [_clean(n) for n in risk[0].get_feature_names_out()]
    coef = pd.Series(risk[-1].coef_.ravel(), index=names)                   # push on the log-odds per standard deviation / category
    top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(12).reset_index()
    top.columns = ["feature", "weight"]
    top["direction"] = np.where(top["weight"] > 0, "raises risk", "lowers risk")
    c = st.columns([3, 2])
    c[0].altair_chart(alt.Chart(top).mark_bar().encode(
        y=alt.Y("feature:N", sort=None, title=None), x=alt.X("weight:Q", title="Effect on the odds of failure (model weight)"),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"], range=[c3, c1]), title=None),
        tooltip=["feature", alt.Tooltip("weight:Q", format=".3f"), "direction"]).properties(height=340), width="stretch")
    rate = df.groupby("maintenance_status")[target].mean()
    age_band = pd.cut(df["equipment_age_years"], [-1, 5, 10, 100], labels=["0–5 years", "5–10 years", "over 10 years"])
    exp_band = pd.cut(df["operator_experience_years"], [-1, 2, 5, 10, 100], labels=["0–2", "3–5", "6–10", "over 10"])
    ar, er = df.groupby(age_band, observed=True)[target].mean(), df.groupby(exp_band, observed=True)[target].mean()
    c[1].markdown(f"""
- **Maintenance status** is the clearest single driver: Critical batches fail at **{rate['Critical']:.1%}**, more than double
  Current (**{rate['Current']:.1%}**). Due Soon ({rate['Due Soon']:.1%}) is no worse than Current.
- **Equipment age** raises risk steadily: **{ar.iloc[0]:.1%}** on machines up to 5 years old vs **{ar.iloc[-1]:.1%}** over 10 years.
- **Operator experience** protects: operators with 0–2 years fail **{er.iloc[0]:.1%}** of batches, those with over 10 years **{er.iloc[-1]:.1%}**.
- **Dosage form** matters through product difficulty: {", ".join(f"{k} ({v:.1%})" for k, v in df.groupby("dosage_form")[target].mean().nlargest(3).items())}
  have the highest failure rates.
- Process readings (temperature, pressure, humidity) carry **small weights**: they stay inside limits on almost every batch, which is
  why failure cannot be predicted well (test ROC-AUC {scores['risk_auc']:.2f}).""")
    eff = pd.DataFrame([_band_effect(df, target, col, LABELS[col]) for col in
                        ["equipment_age_years", "operator_experience_years", "maintenance_days_ago", "raw_material_quality",
                         "temp_variance", "production_hours", "batch_size"]],
                       columns=["Feature", "Lowest 25%", "Failure rate", "Highest 25%", "Failure rate ", "Difference (points)"])
    st.markdown("**Failure rate in the lowest vs highest quarter of each feature**")
    st.dataframe(eff.sort_values("Difference (points)", key=abs, ascending=False).round(2), hide_index=True, width="stretch")

    # ---- B. yield
    st.markdown("#### B. What sets the yield? — drivers of the yield model")
    imp = pd.Series(ymodel[-1].feature_importances_, index=[_clean(n) for n in ymodel[0].get_feature_names_out()])
    imp = imp.groupby(lambda n: n.split(":")[0]).sum().sort_values(ascending=False).head(10)  # one bar per feature (categories summed)
    yi = (imp / imp.sum() * 100).reset_index()
    yi.columns = ["feature", "share"]
    c = st.columns([3, 2])
    c[0].altair_chart(alt.Chart(yi).mark_bar(color=c1).encode(
        y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("share:Q", title="Share of the model's importance (%)"),
        tooltip=["feature", alt.Tooltip("share:Q", format=".1f")]).properties(height=300), width="stretch")
    lead = yi["feature"].iloc[0]
    c[1].markdown(f"""
- **{lead}** carries **{yi['share'].iloc[0]:.0f}%** of the yield model's importance, followed by **{yi['feature'].iloc[1]}** and
  **{yi['feature'].iloc[2]}**.
- The model predicts yield within **{scores['yield_mae']:.2f} points** on average, against {scores['yield_base']:.2f} points
  for simply using the plant average: **{(1 - scores['yield_mae'] / scores['yield_base']):.0%} better**.
- Planning can therefore estimate output per batch before production starts, even though pass/fail cannot be predicted.""")

    # ---- C. rise since 2019
    st.markdown("#### C. Why did failures rise after 2019? — ageing equipment")
    yr = df.groupby("year").agg(failure_rate=(target, "mean"),
                                old_share=("equipment_age_years", lambda s: (s > 10).mean())).reset_index()
    yr[["failure_rate", "old_share"]] *= 100
    base = alt.Chart(yr).encode(x=alt.X("year:O", title=None))
    c = st.columns([3, 2])
    c[0].altair_chart(alt.layer(
        base.mark_area(color=grey, opacity=0.35).encode(y=alt.Y("old_share:Q", title="Batches on equipment over 10 years old (%)")),
        base.mark_line(color=c3, point=True).encode(y=alt.Y("failure_rate:Q", title="Failure rate (%)"),
                                                    tooltip=["year", alt.Tooltip("failure_rate:Q", format=".2f"), alt.Tooltip("old_share:Q", format=".0f")]))
        .resolve_scale(y="independent").properties(height=300), width="stretch")
    early, late = yr[yr["year"].between(2014, 2018)], yr[yr["year"].between(2020, 2025)]
    c[1].markdown(f"""
- The share of batches made on equipment **over 10 years old** rose from **{early['old_share'].mean():.0f}%** (2014–18) to
  **{late['old_share'].mean():.0f}%** (2020–25).
- Over the same periods the failure rate rose from **{early['failure_rate'].mean():.2f}%** to **{late['failure_rate'].mean():.2f}%**.
- The two lines move together: the rise is mainly an **ageing-fleet effect**, not a sudden process problem. Replacing or overhauling
  the oldest machines is the lever.""")

    # ---- D. what fixing the drivers would be worth
    st.markdown("#### D. What acting on the drivers would be worth (estimate)")
    years = df["year"].nunique()
    crit = df[df["maintenance_status"] == "Critical"]
    old = df[df["equipment_age_years"] > 10]
    young_rate = df.loc[df["equipment_age_years"] <= 10, target].mean()
    novice = df[df["operator_experience_years"] <= 2]
    exp_rate = df.loc[df["operator_experience_years"] > 2, target].mean()
    lever = pd.DataFrame([
        ["Never run a batch on Critical equipment", f"{len(crit):,}", f"{crit[target].mean():.1%} → {rate['Current']:.1%}",
         len(crit) * (crit[target].mean() - rate["Current"])],
        ["Bring machines over 10 years to the younger fleet's failure rate", f"{len(old):,}", f"{old[target].mean():.1%} → {young_rate:.1%}",
         len(old) * (old[target].mean() - young_rate)],
        ["Pair operators with 0–2 years' experience with seniors", f"{len(novice):,}", f"{novice[target].mean():.1%} → {exp_rate:.1%}",
         len(novice) * (novice[target].mean() - exp_rate)],
    ], columns=["Action", "Batches affected (all years)", "Failure rate now → after", "Failed batches avoided (all years)"])
    lever["Per year (average)"] = lever["Failed batches avoided (all years)"] / years
    c = st.columns([2, 3])
    c[0].altair_chart(alt.Chart(lever).mark_bar(color=c1).encode(
        y=alt.Y("Action:N", sort="-x", title=None), x=alt.X("Per year (average):Q", title="Failed batches avoided per year"),
        tooltip=["Action", alt.Tooltip("Per year (average):Q", format=".1f")]).properties(height=200), width="stretch")
    c[1].dataframe(lever.round(1), hide_index=True, width="stretch")
    st.caption("Estimates assume the affected batches would fail at the better group's rate; the actions overlap, so the savings should "
               "not be added together.")


batches, sales, moves = load_data()                                         # load once
delivered = sales[sales["is_cancelled"] == 0]                               # delivered orders
policy = reorder_policy(moves)                                              # reorder status per item

# ------------------------------------------------------------------ header and sidebar
st.markdown('''<div class="banner"><h1>💊 Pharmaceutical Operations Intelligence System</h1>
<p>Batch quality · equipment maintenance · process control · sales · demand · inventory — LAL data, 2007–2026</p></div>''',
            unsafe_allow_html=True)                                         # page banner
yr = st.sidebar.slider("Production years", 2007, 2026, (2007, 2026))       # year filter for quality tabs
sections = st.sidebar.multiselect("Manufacturing section", sorted(batches["manufacturing_section"].unique()),
                                  default=sorted(batches["manufacturing_section"].unique()))  # section filter
b = batches[batches["year"].between(*yr) & batches["manufacturing_section"].isin(sections)]  # filtered batches
st.sidebar.markdown(f"**{len(b):,} batches selected**")                    # filter summary
st.sidebar.caption("The filters apply to the Batch quality, Equipment and Process control tabs.")

NAMES = ["Overview", "Sales", "Batch quality", "Equipment", "Process control", "Risk checker",
         "What-if predictor", "Demand forecast", "Inventory", "Commercial & finance", "Executive summary"]
T = dict(zip(NAMES, st.tabs(NAMES)))                                    # tabs addressed by name

# ------------------------------------------------------------------ 1. overview
with T["Overview"]:
    fr, over = b["batch_failed"].mean(), (policy["status"] == "Overstock").mean()  # failure rate, overstock share
    paid = 1 - delivered["Balance_Due_KES"].sum() / delivered["Total_Amount_KES"].sum()  # share of invoiced value collected
    c = st.columns(5)                                                       # five headline cards
    kpi(c[0], "🧪", "Batches", f"{len(b):,}", f"{1 - fr:.1%} passed QC", "blue", bar=(1 - fr) * 100)
    kpi(c[1], "⚠️", "Failure rate", f"{fr:.2%}", f"{b['batch_failed'].sum():,} failed batches", "red" if fr > 0.035 else "orange")
    kpi(c[2], "💰", "Net sales", f"KES {delivered['Net_Amount_Excl_VAT_KES'].sum() / 1e9:.1f} bn", f"{len(delivered):,} delivered orders", "green")
    kpi(c[3], "🧾", "Unpaid balances", f"KES {delivered['Balance_Due_KES'].sum() / 1e6:,.0f} m", f"{paid:.1%} of invoiced value collected", "purple", bar=paid * 100)
    kpi(c[4], "📦", "Overstocked items", f"{(policy['status'] == 'Overstock').sum():,}", f"{over:.0%} of {len(policy):,} items", "teal", bar=over * 100)
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
with T["Sales"]:
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
with T["Batch quality"]:
    crit = b.loc[b["maintenance_status"] == "Critical", "batch_failed"].mean()  # failure rate on Critical equipment
    top2 = b["failure_mode"].value_counts(normalize=True).head(2).sum()     # share of the two largest modes
    c = st.columns(4)                                                       # four headline cards
    kpi(c[0], "🧪", "Batches", f"{len(b):,}", f"{yr[0]}–{yr[1]}", "blue")
    kpi(c[1], "❌", "Failed", f"{b['batch_failed'].sum():,}", f"{b['batch_failed'].mean():.2%} of batches", "red")
    kpi(c[2], "🔧", "Critical equipment", f"{crit:.1%}" if pd.notna(crit) else "–", "failure rate when Critical", "orange")
    kpi(c[3], "📊", "Top 2 failure modes", f"{top2:.0%}", "of all failures", "purple", bar=top2 * 100)
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
with T["Equipment"]:
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
with T["Process control"]:
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
with T["Risk checker"]:
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
        c = st.columns(4)                                                   # four result cards
        kpi(c[0], "🎯", "Risk score", f"{p:.2f}", f"{p / past.mean():.1f}× the average batch", "red" if p >= thr else "green", bar=p * 100)
        kpi(c[1], "📏", "Alert threshold", f"{thr:.2f}", "set for 80% recall", "blue", bar=thr * 100)
        kpi(c[2], "🚦", "Decision", "Extra checks" if p >= thr else "Standard", "above threshold" if p >= thr else "below threshold",
            "orange" if p >= thr else "teal")
        kpi(c[3], "📈", "Riskier than", f"{(past < p).mean():.0%}", "of past batches", "purple", bar=(past < p).mean() * 100)
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
with T["Demand forecast"]:
    md = monthly_demand(sales)                                              # units per month by category
    cat = st.selectbox("Category", ["Total"] + [c for c in md.columns if c != "Total"])
    series = md[cat]                                                        # chosen series
    with warnings.catch_warnings():                                         # hide statsmodels fitting notes
        warnings.simplefilter("ignore")
        acc, back = evaluate_series(series)                                 # 12-month holdout comparison
        hw = _holt_winters(series.to_numpy(), 12)                           # Holt-Winters, next 12 months
    future = pd.Series(hw,                                                  # forecast with month labels
                       index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS"))
    growth = future.sum() / series.iloc[-12:].sum() - 1                     # expected change vs last year
    gain = acc["WAPE_improvement_vs_baseline"].iloc[-1]                     # Holt-Winters vs seasonal naive
    c = st.columns(4)                                                       # four headline cards
    kpi(c[0], "🔮", "Next 12 months", f"{future.sum():,.0f}", "units forecast", "blue")
    kpi(c[1], "🗓️", "Last 12 months", f"{series.iloc[-12:].sum():,.0f}", "units sold", "teal")
    kpi(c[2], "📈" if growth >= 0 else "📉", "Expected change", f"{growth:+.1%}", "next year vs last year", "green" if growth >= 0 else "red")
    kpi(c[3], "🏆", "Forecast gain", f"{gain:.1%}", f"WAPE {acc['WAPE'].iloc[-1]:.1%} vs {acc['WAPE'].iloc[0]:.1%} baseline", "purple", bar=gain * 100)
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
with T["Inventory"]:
    c = st.columns(2)                                                       # two filters
    br = c[0].multiselect("Branch", sorted(policy["Branch_Code"].unique()), default=sorted(policy["Branch_Code"].unique()))
    ct = c[1].multiselect("Category", sorted(policy["category"].dropna().unique()),
                          default=sorted(policy["category"].dropna().unique()))
    p = policy[policy["Branch_Code"].isin(br) & policy["category"].isin(ct)]  # filtered items
    k = st.columns(4)                                                       # one card per action
    for col, (s, icon, tone) in zip(k, [("Reorder now", "🛒", "red"), ("Overstock", "📦", "orange"),
                                        ("No recent demand", "💤", "purple"), ("OK", "✅", "green")]):
        n = (p["status"] == s).sum()                                        # items with this action
        kpi(col, icon, s, f"{n:,}", f"{n / max(len(p), 1):.0%} of {len(p):,} items", tone, bar=n / max(len(p), 1) * 100)
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

# ------------------------------------------------------------------ 9. commercial and finance
with T["Commercial & finance"]:
    finance_page(sales, moves, (BLUE, "#52be80", ORANGE, GREY), ["green", "teal", "red", "purple", "blue", "orange"])

# ------------------------------------------------------------------ what-if predictor
with T["What-if predictor"]:
    predictor_tab(batches, "batch_failed", "team", (BLUE, ORANGE, RED, GREY), ["blue", "green", "red", "purple", "teal", "orange"], sales, moves, policy)

# ------------------------------------------------------------------ 10. executive summary
with T["Executive summary"]:
    st.markdown('''<div style="background:#eef5fb;border-left:6px solid #1f4e79;padding:14px 18px;border-radius:10px;margin-bottom:14px">
    <b>Bottom line.</b> Quality is high (96.9% of 60,000 batches pass) and failures are concentrated in assay and dissolution (47%).
    Overdue (Critical) maintenance more than doubles the odds of failure, and ageing equipment explains most of the rise since 2019.
    Pre-release data cannot predict individual failures well, so laboratory testing stays; the value lies in maintenance timing,
    targeted checks, and buying stock to demand instead of far ahead of it.</div>''', unsafe_allow_html=True)  # headline message
    crit = batches.loc[batches["maintenance_status"] == "Critical", "batch_failed"].mean()  # failure rate when Critical
    cur = batches.loc[batches["maintenance_status"] == "Current", "batch_failed"].mean()   # failure rate when Current
    c = st.columns(5)                                                       # headline cards
    kpi(c[0], "🧪", "Batches passed", f"{1 - batches['batch_failed'].mean():.1%}", f"{len(batches):,} batches, {batches['batch_failed'].sum():,} failed", "blue",
        bar=(1 - batches["batch_failed"].mean()) * 100)
    kpi(c[1], "🔧", "Critical vs Current", f"{crit:.1%} vs {cur:.1%}", "failure rate · odds ratio 2.33", "red")
    kpi(c[2], "📈", "Rise since 2019 explained", "60%", "2.78% → 3.73%, mainly equipment age", "orange", bar=60)
    kpi(c[3], "🔮", "Forecast gain", "33.8%", "Holt-Winters vs last year's pattern", "green", bar=33.8)
    kpi(c[4], "📦", "Overstocked items", f"{(policy['status'] == 'Overstock').sum():,}", f"of {len(policy):,} · {(policy['status'] == 'Reorder now').sum()} to reorder", "purple",
        bar=(policy["status"] == "Overstock").mean() * 100)

    exec_commercial(sales, moves, ["green", "teal", "orange", "purple"])   # revenue, forecast, receivables, stock value
    st.subheader("1. What we asked and what the models found")
    qa = pd.DataFrame([
        ["Can we predict which batch will fail QC before release?", "Logistic regression vs random forest vs gradient boosting (time split)",
         "Best test ROC-AUC 0.581; recall 0.82 needs 75% of batches flagged; precision 0.040 (target 0.19)", "Weakly — not good enough to replace lab testing"],
        ["Does overdue maintenance raise failure risk?", "Adjusted logistic model (+ within-machine check)",
         "Critical 6.9% vs Current 3.1%; odds ratio 2.33 (1.63–3.34), 2.35 within the same machine; Due Soon no effect", "Yes — Critical more than doubles the odds"],
        ["Can maintenance status be predicted from the production record?", "Baseline vs logistic regression vs random forest",
         "Random forest macro F1 0.501 (baseline 0.317); logistic finds 63% of Critical batches", "Moderately — useful as an early flag"],
        ["Why did failures rise after 2019?", "Trend model with and without measured factors",
         "Rate rose 2.78% → 3.73%; measured factors explain 60%; year effect falls to OR 1.002 (p = 0.72)", "Mostly explained — equipment over 10 years old went 35% → 77% of batches"],
        ["Do process readings warn before a failure?", "Control limits, capability (Cpk), lead-time test",
         "23.7% of failures vs 21.9% of passes had a prior warning (p = 0.074); no section reaches Cpk 1.33", "No useful early warning; assay capability is the gap"],
        ["Which batches deserve extra checks?", "Composite risk score (0.6 quality, 0.2 equipment, 0.2 process)",
         "High tier = 10% of batches, fails at 6.3% (lift 1.71), holds 17% of failures", "Use the High tier to target checks"],
        ["How much will we sell next month and next week?", "Holt-Winters (monthly); Croston-SBA (weekly per product)",
         "Monthly WAPE 5.5% (33.8% better than baseline); weekly WAPE 115% but 19.7% better, bias −5.9%", "Yes for planning totals; product-weeks stay noisy"],
        ["Which items will run out, and is stock right-sized?", "Gradient boosting (stock-out); reorder-point rule",
         "PR-AUC 0.278 (205× random), 80% of stock-outs caught; 1,033 of 1,200 items overstocked, 128 to reorder", "Stock-outs can be flagged; over-buying is the bigger problem"],
    ], columns=["Question", "Model / method", "What we found", "Answer"])
    st.dataframe(qa, hide_index=True, width="stretch")

    exec_drivers(batches, "batch_failed", "team", (BLUE, ORANGE, RED, GREY), ["blue", "green", "red", "purple", "teal", "orange"])

    st.subheader("Model scorecard: comparison, top features and choice for each problem")
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

    st.subheader("2. The evidence in pictures")
    c = st.columns(2)
    gains = pd.DataFrame({"Problem": ["Monthly demand (Holt-Winters)", "Weekly demand (Croston-SBA)", "Maintenance status (RF, macro F1)",
                                      "High risk tier (lift)", "Batch failure (ROC-AUC vs 0.5)"],
                          "Gain over baseline (%)": [33.8, 19.7, 58.0, 71.0, 16.2]})       # chosen model vs its baseline
    c[0].altair_chart(alt.Chart(gains).mark_bar(color=BLUE).encode(
        y=alt.Y("Problem:N", sort="-x", title=None), x=alt.X("Gain over baseline (%):Q"),
        tooltip=["Problem", "Gain over baseline (%)"]).properties(height=260, title="How much better than the simple baseline"), width="stretch")
    c[0].caption("Stock-out model left out of the chart: 205× better than random would flatten the other bars.")
    rates = batches.groupby("maintenance_status")["batch_failed"].mean().mul(100).reindex(["Current", "Due Soon", "Critical"]).reset_index(name="rate")
    c[1].altair_chart(alt.Chart(rates).mark_bar().encode(
        x=alt.X("maintenance_status:N", sort=None, title=None), y=alt.Y("rate:Q", title="Failure rate (%)"),
        color=alt.Color("maintenance_status:N", scale=alt.Scale(domain=["Current", "Due Soon", "Critical"], range=[BLUE, ORANGE, RED]), legend=None),
        tooltip=["maintenance_status", alt.Tooltip("rate:Q", format=".2f")]).properties(height=260, title="Failure rate by maintenance status"), width="stretch")
    c = st.columns(2)
    rise = pd.DataFrame({"part": ["2014–18 rate", "Explained by measured factors", "Unexplained"], "points": [2.78, 0.57, 0.38]})
    rise["start"] = [0, 2.78, 3.35]
    rise["end"] = rise["start"] + rise["points"]
    c[0].altair_chart(alt.Chart(rise).mark_bar(size=46).encode(
        x=alt.X("part:N", sort=None, title=None), y=alt.Y("start:Q", title="Failure rate (%)"), y2="end:Q",
        color=alt.Color("part:N", scale=alt.Scale(domain=list(rise["part"]), range=[GREY, ORANGE, RED]), legend=None),
        tooltip=["part", "points"]).properties(height=260, title="Rise to 3.73% in 2020–25: 60% explained"), width="stretch")
    rules = pd.DataFrame({"Rule": ["Classifier alert", "Classifier alert", "High risk tier", "High risk tier"],
                          "Measure": ["Batches flagged (%)", "Failures caught (%)"] * 2, "value": [75.5, 81.9, 10.0, 17.1]})
    c[1].altair_chart(alt.Chart(rules).mark_bar().encode(
        x=alt.X("Rule:N", title=None), y=alt.Y("value:Q", title="%"), xOffset="Measure:N",
        color=alt.Color("Measure:N", scale=alt.Scale(range=[GREY, BLUE])), tooltip=["Rule", "Measure", "value"])
        .properties(height=260, title="Back-test 2024–26: effort vs failures caught"), width="stretch")
    st.altair_chart(alt.Chart(policy["status"].value_counts().reset_index(name="items")).mark_arc(innerRadius=70).encode(
        theta="items:Q", color=alt.Color("status:N", scale=alt.Scale(domain=["Overstock", "Reorder now", "No recent demand", "OK"],
                                                                     range=[ORANGE, RED, GREY, BLUE])),
        tooltip=["status", "items"]).properties(height=260, title="Stock status of 1,200 branch-product items"), width="stretch")

    st.subheader("3. How we handled what we found")
    st.dataframe(pd.DataFrame([
        ["Only 3% of batches fail (imbalanced)", "Class weights in every model; PR-AUC and recall used to choose, not accuracy"],
        ["Risk of the model 'seeing the future'", "Train 2007–21, tune 2022–23, test 2024–26 once; results measured at or after release (assay, yield) excluded"],
        ["Classifier is weak", "Kept the simplest model (logistic); threshold set on validation for 80% recall; used only to rank batches, never to release them"],
        ["Critical effect could be 'bad machines'", "Re-checked within the same machine (fixed effects): odds ratio 2.35 vs 2.33 — effect is real"],
        ["Rise since 2019", "Decomposed the trend: measured factors added to the model; 60% explained, year effect no longer significant"],
        ["Process drift as an early warning", "Control limits from passed batches up to 2021; lead-time test showed no useful signal, so not used for alerts"],
        ["Intermittent weekly demand (48% zero weeks)", "Croston-SBA chosen; models with bias beyond ±10% rejected (linear regression −22%)"],
        ["No opening stock in the data", "Stock rebuilt from zero; 6,045 cancelled-order movements removed; lead time assumed 14 days"],
    ], columns=["Finding or risk", "How it was handled"]), hide_index=True, width="stretch")

    st.subheader("4. Recommendations")
    st.dataframe(pd.DataFrame([
        ["High", "QA / CAPA", "Corrective action on assay and dissolution failures", "47% of failures", "Failure rate back below 3%"],
        ["High", "Engineering", "Service equipment before it becomes Critical; plan replacement of machines over 10 years old", "Odds of failure 2.3× when Critical; age drives the rise", "No batch run on Critical equipment"],
        ["High", "Procurement", "Buy to the reorder-point policy; stop receipts for overstocked items, especially Equipment", "1,033 items overstocked; Equipment receives 22× its sales", "Days of cover under 180"],
        ["Medium", "Branch stores", "Weekly reorder list and stock-out watch-list", "Catches 80% of stock-outs", "Stock-out weeks down"],
        ["Medium", "Planning", "Holt-Winters for monthly volumes, Croston-SBA for weekly product demand", "34% / 20% better than baseline", "Forecast error tracked monthly"],
        ["Medium", "Finance", "Review credit-account terms and collections", f"KES {delivered['Balance_Due_KES'].sum() / 1e6:,.0f} m unpaid, all on credit accounts", "Unpaid share falling"],
        ["Low", "QA", "Use the High risk tier only to target extra in-process checks", "Top 10% fail at 1.7× the average", "Failures found per check"],
    ], columns=["Priority", "Owner", "Action", "Evidence", "Measure of success"]), hide_index=True, width="stretch")
    st.caption("Limitations: no opening stock, expiry dates or supplier lead times; batches cannot be linked to sales; the maintenance-status model "
               "uses a random split and may score slightly high; no savings are claimed until stock values are confirmed with stores.")
