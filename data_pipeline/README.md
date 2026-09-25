# Module 1 — Data Pipeline

This module scrapes `books.toscrape.com`, cleans the fields, converts GBP to INR,
loads a normalized SQLite database, and demonstrates SQL + pandas queries.

## Required conversion
**1 GBP = 105.50 INR**. This is the fixed project-defined baseline; no currency API is used.

## Install
```bash
pip install -r ../requirements.txt
```

## Run
```bash
python pipeline.py
```

The script:
1. Scrapes at least 60 books.
2. Uses book detail pages to capture categories.
3. Parses price/rating/availability.
4. Median-imputes malformed numeric price/rating values.
5. Drops rows with missing category/title because category cannot be safely inferred.
6. Computes `price_inr = price_gbp * 105.50`.
7. Creates `books.db`.
8. Runs five SQL queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, and JOIN.
9. Reads query results with `pd.read_sql` and reproduces the JOIN with `pd.merge`.

The SQLite schema uses `categories.category_id` as the primary key referenced by
`books.category_id` as a foreign key.
