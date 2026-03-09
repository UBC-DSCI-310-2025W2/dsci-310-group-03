import click
import pandas as pd
import os
from sklearn.model_selection import train_test_split


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

    X = wine_df.drop(columns=["quality"])
    y = wine_df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123
    )

    train_df = X_train.copy()
    train_df["quality"] = y_train
    train_df = train_df.reset_index(drop=True)

    test_df = X_test.copy()
    test_df["quality"] = y_test
    test_df = test_df.reset_index(drop=True)

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
