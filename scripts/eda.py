import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


@click.command()
@click.option(
    "--input-path",
    type=str,
    required=True,
    help="Path to the wine quality data CSV (e.g., data/wine_quality_raw.csv)",
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Directory to save EDA figures and summary table (e.g., results)",
)
def main(input_path, output_dir):
    """Generate EDA figures and summary statistics for the wine quality dataset."""
    os.makedirs(output_dir, exist_ok=True)

    wine_df = pd.read_csv(input_path)

    # Summary statistics table
    summary = wine_df.describe()
    summary.to_csv(os.path.join(output_dir, "summary_stats.csv"))
    print(f"Summary statistics saved to {output_dir}/summary_stats.csv")

    # Figure 1: Distribution of Wine Quality
    quality_counts = wine_df["quality"].value_counts().sort_index()
    plt.figure()
    plt.bar(quality_counts.index, quality_counts.values)
    plt.title("Distribution of Wine Quality")
    plt.xlabel("Quality")
    plt.ylabel("Count")
    plt.savefig(
        os.path.join(output_dir, "quality_distribution.png"), bbox_inches="tight"
    )
    plt.close()
    print(f"Quality distribution plot saved to {output_dir}/quality_distribution.png")

    # Figure 2: Correlation Heatmap
    plt.figure(figsize=(8, 7))
    sns.heatmap(wine_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap of Wine Features")
    plt.savefig(
        os.path.join(output_dir, "correlation_heatmap.png"), bbox_inches="tight"
    )
    plt.close()
    print(f"Correlation heatmap saved to {output_dir}/correlation_heatmap.png")

    # Figure 3: Alcohol Content by Wine Quality
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="quality", y="alcohol", data=wine_df)
    plt.title("Alcohol Content by Wine Quality")
    plt.xlabel("Wine Quality")
    plt.ylabel("Alcohol (% vol)")
    plt.savefig(os.path.join(output_dir, "alcohol_by_quality.png"), bbox_inches="tight")
    plt.close()
    print(f"Alcohol by quality plot saved to {output_dir}/alcohol_by_quality.png")


if __name__ == "__main__":
    main()
