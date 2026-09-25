# Module 2 — Analytics Pipeline

Run the EDA first, then modeling:

```bash
python 01_eda.py
python 02_modeling.py
```

`01_eda.py` performs the single `sns.load_dataset("titanic")` load and immediately
writes `titanic.csv`. `02_modeling.py` reads `titanic_cleaned.csv`, so the raw dataset
is not independently loaded a second time.

## Cleaning decisions
The script measures missing percentages before applying the assignment threshold:
under 5% -> drop affected rows; 5%-30% -> impute; above 30% -> encode `"Missing"`.
The exact percentages are printed at runtime because they are properties of the loaded
dataset.

## Modeling
A stratified train/test split is performed before preprocessing. Numeric features use
median imputation + StandardScaler; categorical features use most-frequent imputation
+ OneHotEncoder. These are inside a ColumnTransformer and Pipeline so fitting occurs
only on the training split.

The script trains Logistic Regression, Decision Tree, and Random Forest; reports
confusion matrices, accuracy, precision, recall, F1 and ROC/AUC; performs baseline /
class-weight / SMOTE comparison; runs Random Forest GridSearchCV with `oob_score=True`;
and performs the fare regression side-task.

Four EDA charts plus histograms/box plots and a correlation heatmap are saved under
`figures/`. Each required numeric result is printed by the script and can be copied
into the submission notebook/README with the runtime values.

The final classifier is saved as a complete preprocessing+estimator pipeline in
`best_classifier_pipeline.joblib` and reloaded to verify raw-input prediction.
