# Predicting Financial Reporting Misstatements in U.S. Public Companies

**Temporal validation of machine-learning models, XBRL-based data integration and risk-based audit resource allocation**

Author: Tosin Bello

This repository contains the code, executed notebooks and result tables for the study. Each notebook answers one research objective and runs end to end from the raw public data.

## Research aim

To develop and empirically evaluate machine-learning models for predicting financial reporting misstatement risk among U.S. listed companies, with emphasis on temporal generalisability, financial reporting indicators, and the efficient allocation of limited forensic audit resources.

## Headline results

| Objective | Result |
|---|---|
| 1 Replication | RUSBoost, Bao et al. data, 2003–08: AUC **0.719** with serial misstatements recoded vs 0.779 without; AAER firm-years in top 1%: 9 vs 18. Matches the 2022 erratum (0.723; 10). |
| 2 Models | Boosted ensembles on raw items AUC 0.716–0.724; Dechow ratio logit 0.691 (no significant difference, firm-cluster bootstrap); size only 0.641. |
| 3 XBRL | 94,229 as-first-reported firm-years, fiscal 2009–2025, from 70 FSDS quarters. AAER labels end in fiscal 2015 (52 positives in 2011–14). Compustat-trained models: AUC 0.76→0.72 (logit) on XBRL inputs. New dated restatement label (1,028 firm-years): XBRL-native models AUC 0.68–0.69 out of time, 2013–2024. |
| 4 Budgets | Top 1% captures 2.7–3.4% of AAER firm-years; top 5% captures 11–15% (AAER) and 13–18% (restatements). Frozen models decay; fixed probability cut-offs drift. |
| 5 SHAP | Soft assets ranks 1st in all 12 years; adjacent-year rank correlation median 0.94, 2003 vs 2014 0.69. Several drivers proxy for firm size. |

## Repository layout

```
├── notebooks/
│   ├── 01_objective1_replication_serial_fraud.ipynb   Bao et al. (2020) replication, serial-fraud treatments
│   ├── 02_objective2_model_comparison.ipynb           Logit vs RF / GBM / XGBoost / RUSBoost, size baseline
│   ├── 03_objective3_xbrl_extension.ipynb             SEC FSDS panel, mapping fidelity, AAER + restatement labels
│   ├── 04_objective4_out_of_time_budgets.ipynb        Top-1% / top-5% budgets, frozen vs retrained, threshold drift
│   └── 05_objective5_shap_stability.ipynb             SHAP importance, temporal stability, size confound
├── src/misstatement/
│   ├── config.py      paths, seeds, test windows
│   ├── data.py        Bao et al. and AAER loaders, variable lists and labels
│   ├── metrics.py     AUC, NDCG@k, top-k capture/precision, calibration, firm-cluster bootstrap, PSI
│   ├── models.py      model zoo, train-only preprocessing, serial-fraud treatments, rolling-origin evaluator
│   ├── fsds.py        SEC Financial Statement Data Sets: discovery, extraction, panel, restatement labels
│   └── plots.py       figure style
├── data/
│   ├── raw/           inputs you download (not tracked; see below)
│   ├── interim/       per-quarter 10-K extracts from the FSDS (generated, not tracked)
│   └── processed/     xbrl_panel.parquet, xbrl_restatements.parquet (generated)
├── results/
│   ├── tables/        every table reported in the paper (CSV)
│   ├── figures/       every figure (PNG and PDF)
│   └── cache/         saved out-of-sample predictions (generated, not tracked)
└── manuscript/        manuscript (Word)
```

## Data

| Input | Source | Put in |
|---|---|---|
| `data_FraudDetection_JAR2020.csv` (Bao et al. 2020 public data; Compustat items, fiscal 1990–2014) | https://github.com/JarFraud/FraudDetection | `data/raw/` |
| `AAER_firm_year.csv` (AAER firm-years, AAERs to #4012, Dec 2018) | same repository / JAR online supplement | `data/raw/` |
| SEC Financial Statement Data Sets, 2009 Q2 – 2026 Q3 (`sub.txt`, `num.txt` per quarter) | https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets | `data/raw/` (any sub-folder layout) |

Set `MISSTATE_DATA` to use another folder. Quarter folders may be nested and files may carry download suffixes such as `num (3).txt`: notebook 03 identifies each quarter from the filing dates in `sub.txt`, drops duplicate downloads and checks every `sub`/`num` pairing before use.

The Bao et al. data are derived from Compustat and are redistributed by their authors for research use; cite their paper. SEC data are public.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_objective1_replication_serial_fraud.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_objective2_model_comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_objective3_xbrl_extension.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_objective4_out_of_time_budgets.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_objective5_shap_stability.ipynb
```

Run the notebooks in order: 04 reads the predictions saved by 02 and 03, and 05 reads those of 02. Predictions are cached in `results/cache/`; delete a cache file to recompute it. On a two-core laptop a full run takes roughly 3–4 hours, most of it the random forest and RUSBoost fits and the FSDS extraction (about 1–2 hours for 70 quarters). Set `MISSTATE_NJOBS` to give each model more threads when notebooks are not run in parallel.

## Design choices that matter

* **Rolling origin with a two-year gap.** Models are trained on fiscal years up to T−2 and tested on year T, as in Bao et al. (2020).
* **Serial misstatements.** Training firm-years of an AAER that is still running in the test year are recoded to 0 (`recode`); a stricter variant drops all training years of firms that misstate in the test year (`firm_excl`). Keeping them (`naive`) leaks future information.
* **Label timing in the XBRL sample.** A training firm-year counts as restated only if the restating 10-K/A was filed by 31 December of the test year.
* **Train-only preprocessing.** Winsorising, imputation and scaling are fitted on the training window only.
* **Firm-cluster inference.** Confidence intervals and paired tests resample firms with all their years.
* **Budgets as percentiles.** Top-1% and top-5% lists are formed within each test year, not from fixed probability cut-offs.

## XBRL-derived restatement label

For each original 10-K in the FSDS, the study looks for a later 10-K/A for the same company and fiscal period. The firm-year is labelled *restated* if the amendment reports net income, total assets or stockholders' equity more than 0.5% different from the original. Amendments that only add Part III information carry no financial data and are not flagged. The label is public, dated (the 10-K/A filing date) and available through fiscal 2024. It is broader and noisier than AAER enforcement; see the manuscript for its validation and limitations.

## Citation

If you use this code, please cite the accompanying paper (Bello, *forthcoming*) and Bao, Y., Ke, B., Li, B., Yu, Y. J., & Zhang, J. (2020). Detecting accounting fraud in publicly traded U.S. firms using a machine learning approach. *Journal of Accounting Research*, 58(1), 199–235.

## Licence

Code: MIT (add a `LICENSE` file before publishing). Data: subject to the terms of their providers.
