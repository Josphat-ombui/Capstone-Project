# Batch Quality and Commercial Platform: download folder

## What is here

| File | What it is |
|---|---|
| Capstone_Final_Project_Report.docx | Written report, 24 pages, version 2.0 of 12 November 2026: findings, achievement against project targets, discussion, limits, recommendations and appendices. Editable. |
| Capstone_Final_Project_Report.pdf | The same report as a PDF. |
| Batch_Quality_Final_Notebook.ipynb | Final analysis notebook, executed top to bottom, with 66 charts. |
| notebook_comparison.ipynb | Comparison of the earlier notebooks, ending with a recommendation. |
| batch_quality_platform_react.zip | The React dashboard only (source, tests, built copy in dist/). |
| batch_quality_platform.zip | The full project: pipeline, tests, notebooks, reports, dashboard. |

## Running the dashboard

Requirements: Node.js 22 or later. The unzipped folder is `batch_quality_platform_react/`.

    cd batch_quality_platform_react
    npm install
    npm run dev        # development view
    npm run build      # production build in dist/
    npm test           # unit tests

The data file is already in `public/data/dashboard.json`. To rebuild it from the raw files, run the pipeline in the full project (`python scripts/run_pipeline.py` then `python scripts/export_dashboard_data.py`).

Pages: Executive summary (with the master table and the link to the printable full summary), Data exploration, Batch alerts, Batch predictor, Equipment health, Process control, Receivables, Margin and discounts, Inventory and stock, Cost and suppliers, and the printable full summary.

Printing: the top-bar button prints the open page. On the printable full summary it prints all eleven questions and the master table. Charts are on their own pages and are not repeated in the printout.

## Limits (also shown on the Executive summary)

1. Batch failure is a stand-in for equipment failure. There are no repair or downtime records, so time between failures and repair time cannot be measured.
2. The inventory file has no opening balances. Opening stock is shown as a range.
3. Cancelled orders still show stock leaving. These are corrected before stock ranges and demand are calculated.
4. The credit balance check fails. No collection list is produced until the finance ledger is reconciled.
5. The batch classifier misses the precision and ROC-AUC targets (precision 4.0% against 19%; ROC-AUC 0.581 against 0.80). Use it to order QA review, not as a release gate.
6. Risk weights and the 5% discount cap are assumptions to confirm before decisions rely on them.
7. Predictions for settings outside the training range are extrapolations and are marked on the predictor page.

## Not in this folder

- The slides are in the Claude Slides artifact and are not in this folder.
