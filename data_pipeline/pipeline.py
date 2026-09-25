import re
import sqlite3
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
DB_PATH = Path(__file__).parent / "books.db"
OUTPUT_CSV = Path(__file__).parent / "books_cleaned.csv"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def get_soup(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def scrape_all_books(min_books=60):
    rows = []
    url = BASE_URL
    seen = set()

    while url and len(rows) < min_books:
        soup = get_soup(url)
        for card in soup.select("article.product_pod"):
            a = card.select_one("h3 a")
            title = a.get("title", "").strip() if a else ""
            relative = a.get("href") if a else None
            detail_url = requests.compat.urljoin(url, relative) if relative else None

            price_text = card.select_one(".price_color").get_text(strip=True) if card.select_one(".price_color") else ""
            rating_text = ""
            rating_el = card.select_one(".star-rating")
            if rating_el:
                classes = rating_el.get("class", [])
                rating_text = next((x for x in classes if x in RATING_MAP), "")

            availability = card.select_one(".availability")
            availability_text = availability.get_text(" ", strip=True) if availability else ""

            # Category is available on the detail page breadcrumb.
            category = ""
            if detail_url:
                try:
                    detail_soup = get_soup(detail_url)
                    crumbs = detail_soup.select("ul.breadcrumb li")
                    if len(crumbs) >= 3:
                        category = crumbs[-1].get_text(strip=True)
                except requests.RequestException:
                    category = ""

            if title and title not in seen:
                seen.add(title)
                rows.append({
                    "title": title,
                    "price": price_text,
                    "star_rating": rating_text,
                    "availability": availability_text,
                    "category": category,
                })

            if len(rows) >= min_books:
                break

        next_link = soup.select_one("li.next a")
        url = requests.compat.urljoin(url, next_link.get("href")) if next_link else None

    return pd.DataFrame(rows)

def clean_books(df):
    out = df.copy()

    # Numeric parsing: malformed price/rating rows are median-imputed as required.
    out["price_gbp"] = pd.to_numeric(
        out["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce"
    )
    out["rating"] = out["star_rating"].map(RATING_MAP)

    # "In stock" is True when the listed availability contains "In stock".
    out["in_stock"] = out["availability"].astype(str).str.contains(
        r"\bIn stock\b", case=False, regex=True, na=False
    )

    # Unexpected/missing numeric values use medians.
    out["price_gbp"] = out["price_gbp"].fillna(out["price_gbp"].median())
    out["rating"] = out["rating"].fillna(out["rating"].median()).round().astype(int)

    # A missing category is non-numeric and cannot safely be inferred; drop such rows.
    out = out.dropna(subset=["category", "title"])
    out["category"] = out["category"].astype(str).str.strip()
    out["price_inr"] = (out["price_gbp"] * GBP_TO_INR).round(2)

    return out[
        ["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]
    ].reset_index(drop=True)

def create_database(df):
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY,
        category_name TEXT NOT NULL UNIQUE
    );

    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
        in_stock INTEGER NOT NULL CHECK(in_stock IN (0,1)),
        category_id INTEGER NOT NULL,
        FOREIGN KEY(category_id) REFERENCES categories(category_id)
    );
    """)

    categories = sorted(df["category"].unique())
    con.executemany(
        "INSERT INTO categories(category_name) VALUES (?)",
        [(x,) for x in categories]
    )
    mapping = dict(con.execute("SELECT category_name, category_id FROM categories"))

    book_rows = [
        (
            r.title, r.price_gbp, r.price_inr, int(r.rating),
            int(bool(r.in_stock)), mapping[r.category]
        )
        for r in df.itertuples(index=False)
    ]
    con.executemany("""
        INSERT INTO books
        (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, book_rows)
    con.commit()
    return con

QUERIES = {
    "select_where": """
        SELECT title, price_gbp, rating, in_stock
        FROM books
        WHERE rating >= 4
        ORDER BY price_gbp DESC;
    """,
    "order_by_limit": """
        SELECT title, price_gbp, rating
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,
    "distinct_categories": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,
    "between": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
    """,
    "join_top_rated": """
        SELECT c.category_name, b.title, b.rating, b.price_gbp
        FROM books b
        JOIN categories c ON c.category_id = b.category_id
        WHERE b.rating IN (4, 5)
        ORDER BY c.category_name, b.rating DESC, b.price_gbp DESC
        LIMIT 10;
    """
}

def run_queries(con):
    outputs = {}
    for name, sql in QUERIES.items():
        result = pd.read_sql_query(sql, con)
        outputs[name] = result
        print(f"\n--- {name} ---\n{result.to_string(index=False)}")
    return outputs

def pandas_equivalent(df):
    cats = df[["category"]].drop_duplicates().reset_index(drop=True)
    cats["category_id"] = range(1, len(cats) + 1)
    books = df.copy()
    books["category_id"] = books["category"].map(
        dict(zip(cats["category"], cats["category_id"]))
    )
    merged = pd.merge(
        books, cats, on="category_id", how="inner", suffixes=("_book", "_cat")
    )
    result = merged[merged["rating"].isin([4, 5])][
        ["category_cat", "title", "rating", "price_gbp"]
    ].rename(columns={"category_cat": "category_name"})
    return result.sort_values(
        ["category_name", "rating", "price_gbp"],
        ascending=[True, False, False]
    ).head(10).reset_index(drop=True)

def main():
    raw = scrape_all_books(60)
    if len(raw) < 60:
        raise RuntimeError(f"Only scraped {len(raw)} books; need at least 60.")
    clean = clean_books(raw)
    if len(clean) < 60 or clean["category"].nunique() < 3:
        raise RuntimeError("Final dataset does not meet 60-book/3-category requirement.")

    clean.to_csv(OUTPUT_CSV, index=False)
    con = create_database(clean)
    sql_results = run_queries(con)

    # At least two pd.read_sql calls, plus SQL-vs-pandas JOIN equivalence.
    q1_df = pd.read_sql(QUERIES["select_where"], con)
    q2_df = pd.read_sql(QUERIES["order_by_limit"], con)

    sql_join = pd.read_sql(QUERIES["join_top_rated"], con)
    pd_join = pandas_equivalent(clean)

    print("\nSQL JOIN result:")
    print(sql_join.to_string(index=False))
    print("\nPandas merge result:")
    print(pd_join.to_string(index=False))
    print("\nJOIN equivalent:",
          sql_join.reset_index(drop=True).equals(pd_join.reset_index(drop=True)))

    con.close()

if __name__ == "__main__":
    main()
