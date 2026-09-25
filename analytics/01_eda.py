from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum()), lo, hi

def main():
    # The only network/cache load in the project.
    df = sns.load_dataset("titanic")
    df.to_csv(BASE / "titanic.csv", index=False)

    print("INFO")
    df.info()
    print("\nDESCRIBE\n", df.describe(include="all").T)
    print("\nSHAPE:", df.shape)

    missing = df.isna().mean().mul(100)
    print("\nMISSING %")
    print(missing[missing > 0])

    # Apply threshold rule.
    cleaned = df.copy()
    strategies = {}
    for col in list(cleaned.columns):
        pct = cleaned[col].isna().mean() * 100
        if pct == 0:
            continue
        if pct < 5:
            cleaned = cleaned.dropna(subset=[col])
            strategies[col] = f"{pct:.2f}% -> dropped rows (<5%)"
        elif pct <= 30:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            else:
                cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
            strategies[col] = f"{pct:.2f}% -> imputed (5%-30%)"
        else:
            cleaned[col] = cleaned[col].fillna("Missing")
            strategies[col] = f"{pct:.2f}% -> encoded as Missing (>30%)"
    print("\nCLEANING STRATEGIES")
    for k,v in strategies.items():
        print(k, v)

    # Univariate charts.
    for col in ["age", "fare"]:
        plt.figure()
        sns.histplot(cleaned[col], kde=True)
        plt.title(f"{col.title()} distribution")
        plt.tight_layout()
        plt.savefig(FIG / f"{col}_hist.png", dpi=150)
        plt.close()

        plt.figure()
        sns.boxplot(x=cleaned[col])
        plt.title(f"{col.title()} box plot")
        plt.tight_layout()
        plt.savefig(FIG / f"{col}_box.png", dpi=150)
        plt.close()

    for col in ["age", "fare"]:
        n, lo, hi = iqr_outliers(cleaned[col])
        print(f"{col} IQR outliers: {n}; bounds=({lo:.3f}, {hi:.3f})")

    fare_mean = cleaned["fare"].mean()
    fare_median = cleaned["fare"].median()
    fare_mode = cleaned["fare"].mode().iloc[0]
    print("\nFare mean/median/mode:", fare_mean, fare_median, fare_mode)
    if fare_mean > fare_median > fare_mode:
        skew_text = "right-skewed"
    elif fare_mean < fare_median < fare_mode:
        skew_text = "left-skewed"
    else:
        skew_text = "approximately symmetric/mixed"
    print("Fare skew conclusion:", skew_text)

    # Bivariate survival rates using boolean masks.
    sex_rate = cleaned.groupby("sex")["survived"].mean()
    pclass_rate = cleaned.groupby("pclass")["survived"].mean()
    sex_class_rate = cleaned.groupby(["sex", "pclass"])["survived"].mean()
    print("\nSurvival by sex:\n", sex_rate)
    print("\nSurvival by pclass:\n", pclass_rate)
    print("\nSurvival by sex+pclass:\n", sex_class_rate)

    # Explicit boolean masking examples.
    female = cleaned[cleaned["sex"] == "female"]
    male_or_first = cleaned[(cleaned["sex"] == "male") | (cleaned["pclass"] == 1)]
    print("\nBoolean mask female survival:", female["survived"].mean())
    print("Boolean mask male OR first-class survival:", male_or_first["survived"].mean())

    numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = cleaned[numeric_cols].corr()
    print("\nExact 6-column correlation matrix:\n", corr)

    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Titanic numeric correlation matrix")
    plt.tight_layout()
    plt.savefig(FIG / "correlation_heatmap.png", dpi=150)
    plt.close()

    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            pairs.append((numeric_cols[i], numeric_cols[j], corr.iloc[i,j], abs(corr.iloc[i,j])))
    top2 = sorted(pairs, key=lambda x: x[3], reverse=True)[:2]
    print("\nTwo strongest absolute off-diagonal correlations:")
    for p in top2:
        print(p)

    # Multivariate data story: 4 charts.
    plt.figure()
    sns.barplot(data=cleaned, x="sex", y="survived", hue="pclass")
    plt.title("Survival rate by sex and class")
    plt.tight_layout()
    plt.savefig(FIG / "survival_sex_pclass.png", dpi=150)
    plt.close()

    plt.figure()
    sns.boxplot(data=cleaned, x="survived", y="fare")
    plt.title("Fare by survival")
    plt.tight_layout()
    plt.savefig(FIG / "fare_survival.png", dpi=150)
    plt.close()

    plt.figure()
    sns.scatterplot(data=cleaned, x="age", y="fare", hue="survived")
    plt.title("Age vs fare by survival")
    plt.tight_layout()
    plt.savefig(FIG / "age_fare_survival.png", dpi=150)
    plt.close()

    plt.figure()
    sns.barplot(data=cleaned, x="pclass", y="survived", hue="embarked")
    plt.title("Survival by class and embarkation")
    plt.tight_layout()
    plt.savefig(FIG / "survival_class_embarked.png", dpi=150)
    plt.close()

    # EDA-only standardization on the full cleaned DataFrame.
    standardized = cleaned[["age", "fare"]].copy()
    for col in standardized:
        standardized[col] = (standardized[col] - standardized[col].mean()) / standardized[col].std()
    print("\nStandardization before/after")
    print("Before:\n", cleaned[["age","fare"]].agg(["mean","std"]))
    print("After:\n", standardized.agg(["mean","std"]))

    cleaned.to_csv(BASE / "titanic_cleaned.csv", index=False)

if __name__ == "__main__":
    main()
