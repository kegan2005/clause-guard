# Report Notes - what to capture for the college project report

Capture screenshots at a browser width of about 1400 px. Take the **demo-mode** screenshots first
(they work with no training), then re-take the Analyze screens after training so the report shows real models.
Never edit numbers in a screenshot; every metric must come from `models/*/metrics.json`.

## A. Application screens

| # | Screen | What to show | Report section |
|---|---|---|---|
| 1 | Analyze - input | Sample contract loaded, model selector, threshold slider, `RUN ANALYSIS` button | System design / UI |
| 2 | Analyze - demo mode | `DEMO MODE` banner above the summary cards | Graceful fallback |
| 3 | Analyze - summary cards | Total, `[HIGH]`, `[MED]`, `[LOW]` counts and overall risk score | Results |
| 4 | Analyze - contract viewer | Highlighted text: red tint `[HIGH]`, hatched `[MED]`, dotted `[LOW]` and the legend | Risk highlighting; accessibility |
| 5 | Analyze - clause table | Table with filters applied (e.g. only HIGH) | Results |
| 6 | Analyze - clause inspector | One high-risk clause with full text and explanation | Explainability |
| 7 | Analyze - export | Download buttons, and the opened CSV/JSON file | Outputs |
| 8 | Model Performance | Summary cards + per-category F1 bar chart (trained models only) | Evaluation |
| 9 | Model Performance - table | Precision / recall / F1 / support per category | Evaluation |
| 10 | Model Performance - confusion matrix | Counts and row-normalised versions | Error analysis |
| 11 | Model Performance - comparison | SVM vs Legal-BERT grouped bars and overall metric table | Model comparison |
| 12 | Compare Models | Agreement cards + disagreement table (`Show disagreements only` ticked) | Discussion |
| 13 | Compare Models - missing model | The warning when only one model is trained (shows graceful handling) | Robustness |
| 14 | About | SDG 16 section and the prominent disclaimer | Introduction / ethics |
| 15 | Sidebar + status bar | Active-page highlight, model status panel, bottom status bar | UI design |

## B. Charts and tables to include

* Per-category F1 bar chart for each model (page 2, one tab per model).
* Confusion matrix heatmap for each model; discuss the most confused pairs (e.g. `Non-Compete` vs `Exclusivity`, and any class mistaken for `None`).
* Grouped F1 comparison chart (page 2, *Comparison* tab).
* Class distribution: open `data/processed/stats.json` and make a table of `label_counts` per split, plus `n_contracts` per split (shows split-by-contract).
* The risk map table from the About page (category, level, reason).

## C. Metrics to quote (all from `metrics.json`)

* Overall: accuracy, **macro F1** (main metric because classes are imbalanced), weighted F1, test sentence count.
* Per category: precision, recall, F1, support. Point out low-support categories whose scores are unreliable.
* Validation macro F1 for the SVM (`val_macro_f1`) and the hyperparameters stored in each file.
* Training sentences (`n_train`) and the training date.
* State clearly that the test set is held-out *by contract* and that `None` was down-sampled (`--max-none-ratio`), so precision on a full real contract will be lower. If you ran `--full-test`, say so.

## D. Other outputs to attach (appendix)

* Terminal output of `python -m data.prepare_cuad ...`, `python -m ml.train_baseline` and `python -m ml.train_legalbert` (final epochs and the "Saved model" lines).
* Terminal output of `pytest`.
* A CSV and JSON export from the Analyze page.
* `data/sample_contract.txt` and the highlighted result for it.

## E. Suggested discussion points

* Why split by contract (avoid leakage: boilerplate repeats within a contract).
* Why class weights, and why macro F1 rather than accuracy.
* SVM vs Legal-BERT: accuracy gain vs training cost, model size and inference speed.
* Risk map: a transparent, editable heuristic - discuss that real risk depends on which party the reader is.
* Confidence: the SVM score is a softmax of margins and is not calibrated; the demo score is a rule strength.
* Limitations (see the About page) and the disclaimer: **This tool assists review and is not legal advice.**
* SDG 16: widening access to understandable contract review for people who cannot afford a lawyer.
