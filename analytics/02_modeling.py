from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.base import clone

BASE = Path(__file__).parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

def make_preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

def classifier_metrics(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    cm = confusion_matrix(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": auc,
        "confusion_matrix": cm,
        "fpr": roc_curve(y_test, proba)[0],
        "tpr": roc_curve(y_test, proba)[1]
    }

def build_model(estimator, numeric, categorical):
    return Pipeline([
        ("preprocessor", make_preprocessor(numeric, categorical)),
        ("model", estimator)
    ])

def main():
    df = pd.read_csv(BASE / "titanic_cleaned.csv")

    target = "survived"
    # Exclude target and derived boolean flags; use the remaining useful predictors.
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    X = df[features]
    y = df[target]

    print("Class balance:\n", y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical = ["sex", "embarked"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        )
    }

    fitted = {}
    metrics = []
    for name, estimator in models.items():
        pipe = build_model(estimator, numeric, categorical)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        metrics.append(classifier_metrics(name, pipe, X_test, y_test))

    metrics_df = pd.DataFrame([
        {k:v for k,v in m.items() if k not in ["confusion_matrix","fpr","tpr"]}
        for m in metrics
    ])
    print("\nCLASSIFICATION COMPARISON\n", metrics_df.to_string(index=False))

    for m in metrics:
        print(f"\n{m['model']} confusion matrix:\n{m['confusion_matrix']}")
        plt.figure()
        sns.heatmap(m["confusion_matrix"], annot=True, fmt="d")
        plt.title(f"{m['model']} confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(FIG / (m["model"].lower().replace(" ","_") + "_cm.png"), dpi=150)
        plt.close()

    plt.figure()
    for m in metrics:
        plt.plot(m["fpr"], m["tpr"], label=f"{m['model']} AUC={m['auc']:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "roc_curves.png", dpi=150)
    plt.close()

    # Decision tree visualization with transformed feature names.
    tree_pipe = fitted["Decision Tree"]
    prep = tree_pipe.named_steps["preprocessor"]
    tree = tree_pipe.named_steps["model"]
    feature_names = prep.get_feature_names_out()
    plt.figure(figsize=(24,12))
    plot_tree(tree, feature_names=feature_names, class_names=["0","1"],
              filled=True, max_depth=3)
    plt.tight_layout()
    plt.savefig(FIG / "decision_tree.png", dpi=150)
    plt.close()

    # Imbalance comparison: Logistic Regression.
    variants = {
        "baseline": LogisticRegression(max_iter=2000),
        "class_weight_balanced": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        )
    }
    imbalance_rows = []
    for name, est in variants.items():
        pipe = build_model(est, numeric, categorical)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        imbalance_rows.append({
            "variant": name,
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0)
        })

    # SMOTE is applied after preprocessing and only to the training fold.
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE

        smote_pipe = ImbPipeline([
            ("preprocessor", make_preprocessor(numeric, categorical)),
            ("smote", SMOTE(random_state=42)),
            ("model", LogisticRegression(max_iter=2000))
        ])
        smote_pipe.fit(X_train, y_train)
        pred = smote_pipe.predict(X_test)
        imbalance_rows.append({
            "variant": "SMOTE_training_only",
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0)
        })
    except ImportError:
        print("Install imbalanced-learn to run the SMOTE comparison.")

    imbalance_df = pd.DataFrame(imbalance_rows)
    print("\nIMBALANCE COMPARISON\n", imbalance_df.to_string(index=False))

    # RF grid search. OOB is requested explicitly in the estimator.
    rf = RandomForestClassifier(
        random_state=42, oob_score=True, bootstrap=True
    )
    rf_pipe = Pipeline([
        ("preprocessor", make_preprocessor(numeric, categorical)),
        ("model", rf)
    ])
    grid = GridSearchCV(
        rf_pipe,
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__max_features": ["sqrt", "log2"]
        },
        cv=5, scoring="f1", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print("\nRF BEST PARAMS:", grid.best_params_)
    print("RF OOB SCORE:", grid.best_estimator_.named_steps["model"].oob_score_)

    # Regression side-task: fare from all other available features.
    reg_features = ["survived", "pclass", "sex", "age", "sibsp", "parch", "embarked"]
    Xr = df[reg_features]
    yr = df["fare"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.20, random_state=42
    )
    reg_numeric = ["survived","pclass","age","sibsp","parch"]
    reg_categorical = ["sex","embarked"]
    reg_pipe = Pipeline([
        ("preprocessor", make_preprocessor(reg_numeric, reg_categorical)),
        ("model", LinearRegression())
    ])
    reg_pipe.fit(Xr_train, yr_train)
    rpred = reg_pipe.predict(Xr_test)
    mae = mean_absolute_error(yr_test, rpred)
    rmse = np.sqrt(mean_squared_error(yr_test, rpred))
    r2 = r2_score(yr_test, rpred)
    n, p = len(yr_test), Xr_test.shape[1]
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    residuals = yr_test - rpred
    print("\nREGRESSION")
    print("MAE:", mae, "RMSE:", rmse, "R2:", r2, "Adjusted R2:", adj_r2)

    plt.figure()
    sns.scatterplot(x=rpred, y=residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted fare")
    plt.ylabel("Residual")
    plt.title("Regression residual plot")
    plt.tight_layout()
    plt.savefig(FIG / "regression_residuals.png", dpi=150)
    plt.close()

    # Save the complete best classifier pipeline, including preprocessing.
    best_name = metrics_df.sort_values("f1", ascending=False).iloc[0]["model"]
    full_pipeline = fitted[best_name]
    joblib.dump(full_pipeline, BASE / "best_classifier_pipeline.joblib")

    # Reload and test on raw, unprocessed input.
    loaded = joblib.load(BASE / "best_classifier_pipeline.joblib")
    print("\nReloaded pipeline prediction:",
          loaded.predict(X_test.head(5)).tolist())

    # Separate metric groups as required.
    comparison = metrics_df.drop(columns=["model"]).copy()
    comparison.insert(0, "classifier", metrics_df["model"])
    comparison["reg_MAE"] = mae
    comparison["reg_RMSE"] = rmse
    comparison["reg_R2"] = r2
    comparison["reg_Adjusted_R2"] = adj_r2
    comparison.to_csv(BASE / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON\n", comparison.to_string(index=False))

if __name__ == "__main__":
    main()
