# DSCI 310 Group 03 Project

## Contributors

- Arnav Gupta
- Ashley Chan
- Jacob Andersen-Lum
- Nathan Shack

## Summary

This project investigates whether the quality of wine can be predicted from its physicochemical properties. Using the [Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) from the UCI Machine Learning Repository (Cortez et al., 2009), we apply K-Nearest Neighbors (KNN) regression to predict wine quality scores based on features such as alcohol content, acidity, pH, and residual sugar. The dataset contains physicochemical measurements and sensory quality ratings for Portuguese "Vinho Verde" wine samples. We found that there is a measurable relationship between the chemical properties and quality, however it has a relatively modest predictive performance. These results are not entirely surprising as many of the physicochemical features had weak linear correlations with the quality score,

## How to Run the Analysis

### Prerequisites

1. Install [Docker](https://www.docker.com/get-started).

### Running With Docker

1. Clone this repository:

   ```bash
   git clone https://github.com/UBC-DSCI-310-2025W2/dsci-310-group-03.git
   cd dsci-310-group-03
   ```

2. Pull the image from DockerHub:

   ```bash
   docker pull jacoblum22/dsci-310-group-03@sha256:08be05f3f5feb5ab41a2f0a77031df070539edeb69df7aeb9f45d8ee6a6d60fa
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

## Dependencies

The project dependencies are listed in [environment.yml](environment.yml). Key packages include:

- Python 3.11
- JupyterLab
- pandas
- scikit-learn
- matplotlib
- seaborn
- ucimlrepo

## Licenses

This project is licensed under two licenses:

- **MIT License** — for the project code. See [LICENSE.md](LICENSE.md) for details.
- **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** — for the project report and associated written content. See [LICENSE.md](LICENSE.md) for details.
