# Zepto Data + Analytics + Support Assistant Assignment

This repository contains the three requested modules.

## Structure
- `data_pipeline/` — BooksToScrape scraping, cleaning, GBP→INR conversion, SQLite schema, SQL/pandas.
- `analytics/` — Titanic EDA, cleaning, visualization, classification, imbalance handling, tuning, regression.
- `support_assistant/` — exact 8-document policy corpus, SentenceTransformer embeddings, ChromaDB, LangGraph and FastAPI.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run order
```bash
python data_pipeline/pipeline.py
python analytics/01_eda.py
python analytics/02_modeling.py
cd support_assistant
uvicorn main:app --reload --port 7860
```

## Assignment-specific notes
- Fixed conversion: **1 GBP = 105.50 INR**.
- Data pipeline handles malformed numeric fields with median imputation and drops rows
  whose title/category is missing.
- Titanic is loaded from Seaborn once in `analytics/01_eda.py`, immediately saved as
  `analytics/titanic.csv`, and later processing uses the saved/cleaned data.
- Modeling preprocessing is inside scikit-learn pipelines and is fit only on the
  training split.
- SMOTE is applied only to training data through an imbalanced-learn pipeline.
- The Random Forest grid-search estimator has `oob_score=True`.
- The saved classifier artifact is a complete preprocessing + estimator pipeline.
- Support Assistant defaults to `MOCK_LLM=1` and does not require an API key.
