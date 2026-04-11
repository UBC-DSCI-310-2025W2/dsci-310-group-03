import click
import pandas as pd
import os
from ucimlrepo import fetch_ucirepo


@click.command()
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="Path to save the raw wine quality data CSV (e.g., data/wine_quality_raw.csv)",
)
def main(output_path):
    """Download the Wine Quality dataset from the UCI ML Repository and save to disk."""
    wine_quality = fetch_ucirepo(id=186)

    wine_df = pd.concat([wine_quality.data.features, wine_quality.data.targets], axis=1)

    # ------------------------------------------
    # DATA VALIDATION CHECK 1
    # ------------------------------------------
    # Check 1: Correct data file format (ensures it's a non-empty df before saving)
    assert isinstance(wine_df, pd.DataFrame), "Validation Hard Fail: Downloaded data is not a DataFrame."
    assert not wine_df.empty, "Validation Hard Fail: Downloaded dataset is completely empty."
    # ------------------------------------------

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    wine_df.to_csv(output_path, index=False)
    print(f"Raw data saved to {output_path}")
    print(f"Dataset shape: {wine_df.shape}")

if __name__ == "__main__":
    main()