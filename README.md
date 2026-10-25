# ClauseGuard - Contract Clause Classifier and Risk Highlighter

ClauseGuard splits a contract into sentences, classifies each sentence into a legal clause category
(trained on the CUAD dataset), maps the category to a **HIGH / MEDIUM / LOW** risk level with a plain-English
reason, and highlights the risky text. It is a college NLP mini-project aligned with
**UN SDG 16: Peace, Justice and Strong Institutions**.

> **This tool assists review and is not legal advice.**

The app works **before any model is trained**: it falls back to a rule-based classifier and shows a
visible `DEMO MODE` banner. Nothing on the *Model Performance* page is invented - it only reads saved
`metrics.json` files.

## 1. Installation

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt            # everything, including PyTorch + Transformers
# or, for a lightweight demo / SVM-only setup:
pip install -r requirements-light.txt
```

## 2. Try the app immediately (demo mode)

```bash
streamlit run app.py
```

Open **Analyze**, choose *Sample contract* and press **RUN ANALYSIS**.

## 3. Prepare the CUAD dataset

1. Download **CUAD v1** (`CUAD_v1.json`) from the Atticus Project (https://www.atticusprojectai.org/cuad).
   The Hugging Face copy `theatticusproject/cuad-qa` is also supported.
2. Run (from the project root):

```bash
python -m data.prepare_cuad --cuad-json path/to/CUAD_v1.json
# or: python -m data.prepare_cuad --hf
```

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--num-categories N` | 15 | Keep the N most frequent clause categories (counted on training contracts only) |
| `--include-risk-categories` | off | Also keep every category in `ml/risk_map.py` (recommended: several high-risk ones are rare in CUAD) |
| `--max-none-ratio R` | 3.0 | Keep at most R `None` sentences per labelled sentence in each split (`-1` keeps all) |
| `--full-test` | off | Keep all `None` sentences in the test split (more realistic, much larger) |
| `--seed` | 42 | Random seed for the contract split and down-sampling |

Output in `data/processed/`: `train.csv`, `val.csv`, `test.csv`, `labels.json`, `stats.json`.
**Contracts are split, not sentences**, so no contract appears in more than one split.

## 4. Train the models

```bash
python -m ml.train_baseline        # TF-IDF + LinearSVC, takes minutes on a laptop CPU
python -m ml.train_legalbert       # Legal-BERT fine-tuning; a GPU is strongly recommended
```

Legal-BERT options: `--epochs 3` (use 3-4), `--batch-size 16`, `--max-length 256`,
`--max-train-samples 3000` (quick smoke test). fp16 mixed precision is used automatically on CUDA GPUs.
The first run downloads `nlpaueb/legal-bert-base-uncased` from Hugging Face.

Artefacts are written to:

```
models/svm/model.joblib        models/svm/metrics.json
models/legalbert/              models/legalbert/metrics.json   (+ tokenizer, config, weights)
```

Reload the app (or press **R** in the browser) and the *Model Status* panel in the sidebar turns to `TRAINED`.

## 5. Run the tests

```bash
pytest
```

Tests cover the risk map, sentence splitting and character offsets, inference structure and the
demo-mode fallback, HTML escaping, data preparation (including leakage checks) and an SVM
train-and-load round trip on synthetic data. No downloads are needed.

## 6. Launch the app

```bash
streamlit run app.py
```

## Project layout

```
app.py                 entry point (page config, navigation, status bar)
pages/                 1_Analyze, 2_Model_Performance, 3_Compare_Models, 4_About
ui.py                  CSS injection, windows, badges, tables, status bar, chart styling
assets/style.css       all CSS ("Windows 2000 meets Neo-Brutalism")
ml/risk_map.py         category -> risk level + explanation
ml/text_utils.py       sentence splitting with character offsets (regex, optional spaCy)
ml/inference.py        SVM / Legal-BERT / demo classifiers, thresholding, structured results
ml/highlight.py        safe HTML highlighting using exact offsets
ml/train_baseline.py   ml/train_legalbert.py   ml/metrics.py
data/prepare_cuad.py   data/sample_contract.txt
tests/                 pytest suite
```

## Troubleshooting

* **"DEMO MODE" banner** - the selected model has not been trained (or could not load, e.g. PyTorch is missing). The banner text says which.
* **Scanned PDFs** - only selectable text is extracted; OCR is not included.
* **Legal-BERT is slow on CPU** - use `--max-train-samples` for a trial run, or train on a free Colab/Kaggle GPU and copy the `models/legalbert/` folder back.
