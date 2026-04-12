import click
import pandas as pd
import matplotlib.pyplot as plt
import os
from winepredictor.data_utils import extract_features_and_target
from winepredictor.model_utils import run_knn_grid_search


@click.command()
@click.option(
    "--train-path",
    type=str,
    required=True,
    help="Path to the training data CSV (e.g., data/train.csv)",
)
@click.option(
    "--test-path",
    type=str,
    required=True,
    help="Path to the test data CSV (e.g., data/test.csv)",
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Directory to save model results (figures and tables) (e.g., results)",
)
def main(train_path, test_path, output_dir):
    """Train a KNN regression model on wine quality data and save results."""
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = extract_features_and_target(train_df, "quality")
    X_test, y_test = extract_features_and_target(test_df, "quality")

    # Run grid search
    results = run_knn_grid_search(X_train, y_train)
    best_k = results["best_k"]
    best_cv_score = results["best_cv_score"]
    best_model = results["best_model"]
    best_train_score = results["best_train_score"]
    cv_results = results["cv_results_df"]
    test_score = best_model.score(X_test, y_test)

    print(f"Best K: {best_k}")
    print(f"Train R²: {best_train_score:.5f}")
    print(f"CV R²:    {best_cv_score:.5f}")
    print(f"Test R²:  {test_score:.5f}")

    # Save CV results table
    cv_results.to_csv(os.path.join(output_dir, "cv_results.csv"), index=False)
    print(f"CV results table saved to {output_dir}/cv_results.csv")

    # Save final model scores table
    scores_df = pd.DataFrame(
        {
            "metric": ["best_k", "train_r2", "cv_r2", "test_r2"],
            "value": [best_k, best_train_score, best_cv_score, test_score],
        }
    )
    scores_df.to_csv(os.path.join(output_dir, "model_scores.csv"), index=False)
    print(f"Model scores table saved to {output_dir}/model_scores.csv")

    # Save CV score vs K figure
    k_values = list(range(1, 30))
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, cv_results["mean_test_score"])
    plt.scatter(best_k, best_cv_score, color="red", zorder=5, label=f"Best K={best_k}")
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Mean CV Score (R²)")
    plt.title("KNN Regression: Mean CV Score vs K")
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "cv_scores_vs_k.png"), bbox_inches="tight")
    plt.close()
    print(f"CV score vs K plot saved to {output_dir}/cv_scores_vs_k.png")


if __name__ == "__main__":
    main()

