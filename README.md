# Batch Quality and Machine Reliability Platform

A reproducible analysis platform for a pharmaceutical manufacturing business. It turns four raw files (batch records, maintenance records, sales and inventory) into one set of results, and presents them in an interactive dashboard, an analysis notebook and a written report.

It answers four questions:

1. Which batches are likely to fail, and what drives failure?
2. Does equipment maintenance affect batch quality, once other factors are accounted for?
3. Are the process readings in control, and are the manufacturing sections capable?
4. What do pricing, credit, stock and demand look like, and what decisions do they support?

Author: Josphat Ondieki Ombui, Data Science capstone, Moringa School.

---

## Headline results

Test period 2024 to 2026, with models trained on earlier years only.

| Result | Value |
|---|---|
| Batches analysed | 60,000 (1,875 failures, 3.12%) |
| Failures caught by the batch alerts | 82% of failures in the test period |
| Precision of the batch alerts | 4.0% (base rate 3.7%) |
| ROC-AUC of the adopted classifier | 0.581 |
| Forecast gain over seasonal naive | 23.0% (target 15%) |
| Critical maintenance, adjusted odds of failure | about 2.3 times Current (range 1.6 to 3.3) |

The classifier is a triage tool. It orders batches for review and is not a release gate. The full discussion is in the report.

---

## What the platform produces

- **Pipeline outputs** in `reports/`: tables (CSV), `summary.json`, and the predictor and explore data.
- **Dashboard**: a React application that reads one JSON file and never retrains a model.
- **Analysis notebook**: `notebooks/Batch_Quality_Final_Notebook.ipynb`, with about 66 charts.
- **Report**: Word and PDF versions in the capstone package.

The notebook, the dashboard and the tests all read the same pipeline outputs, so the figures agree everywhere.

---

## Project structure

```
batch_quality_platform/
├── data/raw/                  the four input CSV files (not included, see Quick start)
├── notebooks/                 the analysis notebook
├── reports/                   tables/, summary.json, predictor.json, explore.json
├── scripts/
│   ├── run_pipeline.py        runs the full analysis
│   ├── export_dashboard_data.py
│   └── export_explore_data.py
├── src/bqp/                   the Python package
│   ├── cleaning.py            data loading, linkage and quality checks
│   ├── features.py            model inputs and leakage control
│   ├── modeling.py            classifiers, calibration, thresholds
│   ├── predictor.py           the single-batch predictor
│   ├── reliability.py         machine comparison and maintenance effect
│   ├── process_control.py     control limits and capability
│   ├── commercial.py          pricing, credit, stock, cost
│   ├── forecasting.py         demand forecasts
│   ├── risk.py                composite batch risk score
│   ├── reconciliation.py      consistency checks
│   ├── config.py              settings, assumptions and secrets handling
│   └── pipeline.py            the end-to-end run
├── dashboard/                 React dashboard (Vite, TypeScript, Recharts)
│   ├── src/pages/             one file per dashboard page
│   ├── src/components/        shared cards, tables, charts and the predictor
│   ├── src/lib/               data loading, questions, filters, formatting
│   └── public/data/           dashboard.json (written by the pipeline)
├── tests/                     Python tests, including browser tests of the dashboard
└── archive/                   earlier Streamlit prototype, kept for reference
```

---

## Requirements

- Python 3.13 or newer (the tested version is 3.13)
- Node.js 22 or newer, for the dashboard
- Jupyter, if you want to run the notebook
- Playwright with Chromium, if you want to run the browser tests

---

## Quick start

### 1. Set up Python

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the data

Place the four files in `data/raw/` with these names:

| File | Name |
|---|---|
| Batch manufacturing records | `batch_manufacturing_records.csv` |
| Maintenance records | `maintenance_records.csv` |
| Sales records | `sales_records.csv` |
| Inventory records | `inventory_records.csv` |

To use a different folder, pass `--data-dir`.

### 3. Run the pipeline

```bash
python scripts/run_pipeline.py
```

This takes about a minute. It writes the results to `reports/` and exports the dashboard data to `dashboard/public/data/dashboard.json`. To also build the dashboard in the same step:

```bash
python scripts/run_pipeline.py --build-dashboard
```

### 4. Run the dashboard

```bash
cd dashboard
npm install
npm run dev          # development server with hot reload
```

To serve the production build:

```bash
npm run build
npm run preview
```

### 5. Run the notebook (optional)

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace Batch_Quality_Final_Notebook.ipynb
```

---

## Running the tests

```bash
python -m pytest -q          # Python tests, including one full pipeline run (about a minute)
cd dashboard
npm test                     # dashboard unit and component tests
```

The Python browser tests load every dashboard page, apply a filter and compare the displayed failure rate with the value in `reports/tables/failure_monthly.csv`. They need `npm run build` first and Playwright with Chromium. Without them, the browser tests are skipped and the rest still run.

---

## Dashboard pages

| Page | What it shows |
|---|---|
| Executive summary | Headline numbers, project targets, the master summary of all questions, recommendations and limits |
| Data exploration | Distributions, categories and missing values, correlations, the monthly trend and data checks |
| Batch alerts | Classifier fit: confusion matrix and ROC curve, failure by group, and the alert queue |
| Batch predictor | Predicted failure chance for one batch you describe |
| Equipment health | Machine comparison with intervals and maintenance effect |
| Process control | Control limits, capability and out-of-control readings |
| Receivables | Open credit by age and the balance field check |
| Margin and discounts | Profit under a discount cap in three volume cases |
| Inventory and stock | Stock range, reorder points, cancelled stock and the demand forecast |
| Cost and suppliers | Input cost trend, supplier premiums and pass-through |
| Printable summary | All questions and the master table on one page, for print or PDF |

Each question panel opens with its answer and folds the full method, target, findings and meaning under **Details**.

### Filters

The filter strip holds the section, dosage form and a production year range. Each page states which filters apply. Batch filters change the batch failure figures and the equipment and process pages. They do not change the commercial pages, because the sales and inventory files cannot be linked to batches.

### Printing and PDF

Use **Print / Save as PDF** in the top bar, or open the printable summary page. The print layout hides navigation and filters.

---

## Batch predictor

Describe a batch by its numeric parameters, dosage form, manufacturing section and maintenance status. The page shows:

- the predicted failure chance and the success rate,
- a risk level and a QA flag,
- the factors that raise or lower the risk, compared with a typical batch.

The scorer runs in the browser from `reports/predictor.json`. The model is the adopted logistic regression, calibrated on 2022 to 2023 data. It is a triage aid for review, not a release decision. Inputs outside the training range are flagged, because a prediction there is an extrapolation.

---

## Security and secrets

Operator, supplier, customer and batch identifiers are replaced with HMAC-SHA256 keyed hashes. The dashboard shows only the last four characters of any identifier.

- **Development (default):** a public development salt is used. Outputs are labelled "Development".
- **Production:** set both variables in the environment:

```bash
export BQP_ENV=production
export BQP_PSEUDONYM_SALT='<your secret value>'
```

The pipeline refuses to run in production mode with the development salt. Keep the production salt in your secret manager and never commit it.

---

## Method

- **Time-based split.** Train on 2007 to 2021, tune on 2022 to 2023, and score the test period (2024 to 2026) once. Thresholds are set on the validation period only.
- **No leakage.** Assay, impurity, dissolution time, actual yield, rejection reason and failure mode are measured at or after release, so they are not model inputs.
- **Model selection.** Logistic regression is the baseline. A challenger is adopted only if it beats it on validation ROC-AUC by at least 0.02.
- **Statistics.** Wilson intervals for rates, exact Poisson intervals for observed-to-expected ratios, chi-square tests, logistic odds ratios and permutation importance. The random seed is 42 throughout.
- **Forecasting.** The method is selected on 2024 and reported once on the 2025 holdout year.
- **Composite risk score.** Percentiles and the high-risk cut-off come from the validation period. The weights are assumptions and can be changed in the configuration.

---

## Assumptions

These values are not in the source data. They are set in `src/bqp/config.py`, and each should be confirmed by the business before it drives a decision.

| Assumption | Value used |
|---|---|
| Discount cap | 5% |
| Reorder lead times | 14, 30 and 60 days |
| Service level | z = 1.65 (about 95%) |
| Assay specification | 95% to 105% |
| Composite risk weights | quality 0.6, equipment 0.2, process 0.2 |
| High-risk tier | top 10% of validation batches |

---

## Known limits

- **Batch failure stands in for equipment failure.** There are no repair or downtime records, so true time between failures and repair time cannot be measured.
- **The classifier is weak at the top of its ranking.** Its ROC-AUC is 0.581 and its precision is 4.0%. It meets the recall target only by flagging about three quarters of batches.
- **Classifier metrics are full-period.** They are not refitted for each filter.
- **Commercial and forecasting outputs cannot be split by batch.** The sales and batch files share no key.
- **The inventory file has no opening balances.** Stock on hand cannot be verified, so opening stock is shown as a range.
- **Cancelled orders still show stock leaving** in the inventory file. These movements are corrected before stock ranges and demand are calculated.
- **The credit balance field fails its reliability check.** Most of the open balance is more than a year old, so no collection list is produced until the finance ledger is reconciled.
- **Risk weights and assumptions are not yet signed off.** The weight sensitivity table shows how much the choice matters.

---

## Consistency checks

Two kinds of check are run:

- The notebook recomputes eight headline figures and the dashboard is compared with it. All eight agree.
- Nine key figures are recomputed independently and compared with the pipeline values. Six match. The three differences are reported in the report, not adjusted to fit.

Results are in `reports/notebook_check.json` and `reports/tables/reconciliation.csv`.

---

## Troubleshooting

- **Dashboard shows "no data yet".** Run `python scripts/run_pipeline.py` first. The dashboard reads `public/data/dashboard.json`, which the pipeline writes.
- **The pipeline refuses to run.** You are in production mode without a production salt. Set `BQP_PSEUDONYM_SALT`, or unset `BQP_ENV` for a development run.
- **Browser tests are skipped.** Run `npm run build` in `dashboard/` and install Playwright with Chromium.
- **Missing module errors in Python.** Activate the virtual environment, then run `pip install -r requirements.txt`.
- **Dashboard looks blank on GitHub Pages.** Set `base: '/<repo-name>/'` in `dashboard/vite.config.ts` and rebuild.

---

## Licence and citation

This project is submitted as a capstone for the Data Science programme at Moringa School. If you reuse the code or the method, please cite the project report and credit the author.
