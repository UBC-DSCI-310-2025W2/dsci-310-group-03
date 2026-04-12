# DSCI 310 Group 03 Project

## Contributors

- Arnav Gupta
- Ashley Chan
- Jacob Andersen-Lum
- Nathan Shack

## Summary

This project investigates whether the quality of wine can be predicted from its physicochemical properties. Using the [Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) from the UCI Machine Learning Repository (Cortez et al., 2009), we apply K-Nearest Neighbors (KNN) regression to predict wine quality scores based on features such as alcohol content, acidity, pH, and residual sugar. The dataset contains physicochemical measurements and sensory quality ratings for Portuguese "Vinho Verde" wine samples. We found that there is a measurable relationship between the chemical properties and quality, however it has a relatively modest predictive performance. These results are not entirely surprising as many of the physicochemical features had weak linear correlations with the quality score, suggesting that wine quality may depend on complex, nonlinear interactions among these properties.

## How to Run the Analysis

### Prerequisites

1. Install [Docker](https://www.docker.com/get-started) (recommended), **or** install [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Miniforge).

### Running With Docker (Recommended)

1. Clone this repository:

   ```bash
   git clone https://github.com/UBC-DSCI-310-2025W2/dsci-310-group-03.git
   cd dsci-310-group-03
   ```

2. Pull the image from DockerHub:

   ```bash
   docker pull jacoblum22/dsci-310-group-03@sha256:b88aaf638ca3379d2478c4c27d60a68d6cd50db9e732d1bba63a79c2da124f2d
   ```

3. Launch the container with Docker Compose:

   ```bash
   docker compose up
   ```

   This will start the container, launch Jupyter Lab at http://localhost:8888, and mount the project directory inside the container at `/home/jovyan/work`.

4. Run the full analysis. Once the container is running, open a terminal in Jupyter Lab and run:

   ```bash
   make all
   ```

   This will reproduce the full analysis pipeline, including data processing, modeling, and results.

5. After you are finished, you can stop the container.

    ```bash
   docker compose down
   ```

### Running Without Docker (Using Conda)

1. Clone this repository:

   ```bash
   git clone https://github.com/UBC-DSCI-310-2025W2/dsci-310-group-03.git
   cd dsci-310-group-03
   ```

2. Create and activate the conda environment from `environment.yml`:

   ```bash
   conda env create -f environment.yml
   conda activate dsci-310-group-03
   ```

3. Run the full analysis:

   ```bash
   make all
   ```

4. To clean all generated files and re-run from scratch:

   ```bash
   make clean
   make all
   ```

## Data

The analysis uses the [Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) from the UCI Machine Learning Repository (Cortez et al., 2009). The raw data is **not** committed to this repository — it is automatically downloaded when you run the pipeline (`make all`). The download script (`scripts/download_data.py`) fetches the dataset via the `ucimlrepo` package and saves it to `data/raw/wine_quality_raw.csv`. Processed train/test splits are saved to `data/processed/`.

If you need to access the raw data directly, you can download it from: https://archive.ics.uci.edu/dataset/186/wine+quality

## Dependencies

The project dependencies are managed via [environment.yml](environment.yml). Key packages and their versions:

| Package | Version |
|---|---|
| Python | 3.11 |
| JupyterLab | 4.5.6 |
| pandas | 3.0.1 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| pytest | 9.0.2 |
| ucimlrepo | 0.0.7 |
| click | 8.3.1 |
| tabulate | 0.10.0 |
| winepredictor | 0.1.0 |

## Testing

The utility functions used by this analysis (data processing, model training, and plotting) are packaged in [winepredictor](https://github.com/UBC-DSCI-310-2025W2/winepredictor). Tests live in that package repository and can be run with:

```bash
pip install "winepredictor[dev] @ git+https://github.com/UBC-DSCI-310-2025W2/winepredictor.git@v0.1.0"
pytest --pyargs winepredictor
```

Alternatively, clone the package repo and run tests directly:

```bash
git clone https://github.com/UBC-DSCI-310-2025W2/winepredictor.git
cd winepredictor
pip install -e ".[dev]"
pytest tests/ -v
```

## Licenses

This project is licensed under two licenses:

- **MIT License** — for the project code. See [LICENSE.md](LICENSE.md) for details.
- **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** — for the project report and associated written content. See [LICENSE.md](LICENSE.md) for details.
