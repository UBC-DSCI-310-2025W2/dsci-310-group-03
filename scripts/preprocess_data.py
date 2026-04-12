# preprocess_data.py
# Validates and preprocesses the raw wine quality data.
# Runs data validation checks, then splits into train/test sets.

import click
import pandas as pd
import os
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from winepredictor.data_utils import extract_features_and_target


@click.command()
@click.option(
    "--input-path",
    type=str,
    required=True,
    help="Path to the raw wine quality data CSV (e.g., data/wine_quality_raw.csv)",
)
@click.option(
    "--train-output-path",
    type=str,
    required=True,
    help="Path to save the training data CSV (e.g., data/train.csv)",
)
@click.option(
    "--test-output-path",
    type=str,
    required=True,
    help="Path to save the test data CSV (e.g., data/test.csv)",
)
def main(input_path, train_output_path, test_output_path):
    """Split the wine quality dataset into training (70%) and test (30%) sets."""
    wine_df = pd.read_csv(input_path)

    # ------------------------------------------
    # PRE-SPLIT DATA VALIDATION CHECKS
    # ------------------------------------------
    # Check 2: Correct column names (match expected schema)
    assert "quality" in wine_df.columns, "Validation Hard Fail: Target column 'quality' is missing."

    # Check 3: No empty observations (no fully-null rows)
    assert not wine_df.isnull().all(axis=1).any(), "Validation Hard Fail: Dataset contains completely empty rows."

    # Check 4: Missingness not beyond expected threshold (< 5% per column)
    max_missing_pct = wine_df.isnull().mean().max()
    assert max_missing_pct < 0.05, f"Validation Hard Fail: Missing data exceeds 5% threshold (Max: {max_missing_pct:.1%})"

    # Check 5: Correct data types in each column (Wine data should be 100% numeric)
    non_numeric_cols = wine_df.select_dtypes(exclude=[np.number]).columns
    assert len(non_numeric_cols) == 0, f"Validation Hard Fail: Non-numeric columns detected: {list(non_numeric_cols)}"

    # Check 6: Correct category levels/Expected range (Quality must be between 0 and 10)
    assert wine_df["quality"].between(0, 10).all(), "Validation Hard Fail: 'quality' values found outside the expected 0-10 range."

    # ------------------------------------------
    # PRE-SPLIT DATA VALIDATION CHECKS (Warnings)
    # ------------------------------------------
    # Check 7: No duplicate observations
    if wine_df.duplicated().any():
        num_dupes = wine_df.duplicated().sum()
        warnings.warn(f"Validation Warning: {num_dupes} duplicate observations detected in the dataset.")

    # Check 8: No outlier or anomalous values (Using extreme 3x IQR rule)
    numeric_features = wine_df.drop(columns=["quality"])
    q1 = numeric_features.quantile(0.25)
    q3 = numeric_features.quantile(0.75)
    iqr = q3 - q1
    extreme_outliers = ((numeric_features < (q1 - 3 * iqr)) | (numeric_features > (q3 + 3 * iqr)))
    if extreme_outliers.any().any():
        warnings.warn("Validation Warning: Extreme outliers (> 3x IQR) detected in feature columns.")

    # Check 9: Target/response variable follows expected distribution
    if wine_df["quality"].nunique() < 3:
        warnings.warn(f"Validation Warning: Target 'quality' distribution is unexpectedly narrow (Only {wine_df['quality'].nunique()} unique values).")

    # ------------------------------------------
    # DATA SPLIT
    # ------------------------------------------
    X, y = extract_features_and_target(wine_df, "quality")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123
    )

    train_df = X_train.copy()
    train_df["quality"] = y_train
    train_df = train_df.reset_index(drop=True)

    test_df = X_test.copy()
    test_df["quality"] = y_test
    test_df = test_df.reset_index(drop=True)

    # ------------------------------------------
    # POST-SPLIT VALIDATION CHECKS - NO LEAKAGE (Warnings)
    # ------------------------------------------
    # Check 10: No anomalous correlations between target and features (training set)
    target_correlations = train_df.corr()["quality"].drop("quality").abs()
    if target_correlations.max() > 0.95:
        warnings.warn(f"Validation Warning: Suspiciously high feature-target correlation detected (Max: {target_correlations.max():.2f}). Check for data leakage.")

    # Check 11: No anomalous correlations between features (training set)
    feature_corr_matrix = X_train.corr().abs()
    upper_tri = feature_corr_matrix.where(np.triu(np.ones(feature_corr_matrix.shape), k=1).astype(bool))
    if upper_tri.max().max() > 0.98:
        warnings.warn(f"Validation Warning: Highly collinear features detected (Max correlation: {upper_tri.max().max():.2f}).")

    # ------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------
    for path in [train_output_path, test_output_path]:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"Training data ({X_train.shape[0]} samples) saved to {train_output_path}")
    print(f"Test data ({X_test.shape[0]} samples) saved to {test_output_path}")


if __name__ == "__main__":
    main()
