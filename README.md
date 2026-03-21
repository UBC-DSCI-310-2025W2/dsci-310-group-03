# DSCI 310 Group 03 Project

## Contributors

- Arnav Gupta
- Ashley Chan
- Jacob Andersen-Lum
- Nathan Shack

## Summary

This project investigates whether the quality of wine can be predicted from its physicochemical properties. Using the [Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) from the UCI Machine Learning Repository (Cortez et al., 2009), we apply K-Nearest Neighbors (KNN) regression to predict wine quality scores based on features such as alcohol content, acidity, pH, and residual sugar. The dataset contains physicochemical measurements and sensory quality ratings for Portuguese "Vinho Verde" wine samples.

## How to Run the Analysis

### Prerequisites

1. Install [Docker](https://www.docker.com/get-started).

### Running the Analysis

1. Clone this repository:

   ```bash
   git clone https://github.com/UBC-DSCI-310-2025W2/dsci-310-group-03.git
   cd dsci-310-group-03
   ```

2. Pull and run the Docker container:

   ```bash
   docker pull <dockerhub-username>/dsci-310-group-03
   docker run --rm -p 8888:8888 <dockerhub-username>/dsci-310-group-03
   ```

3. Open the Jupyter notebook URL printed in the terminal and navigate to the analysis notebook.

*TODO: Update Docker image name and any additional steps once the environment is finalized.*
## Running With Docker 
1. Pull the image from DockerHub

Open the terminal and use the bash command:
 ```bash
 docker pull jacoblum22/dsci-310-group-03@sha256:ff021892fac1b14f2dfa9fe20b392ba29e1a76dce17624354533be63f92e9c8e
 ```

2. Launch the container with Docker Compose 

Make sure you are in the project root, then run:
```bash
docker compose up
```
This will start the container, launch Jupyter Lab at: http://localhost:8888, and mount the project directory inside the container at /home/jovyan/work 

3. Run the full analysis
Once the container is running, open a terminal in Jupyter Lab and run:
``` bash
make all
```
This will reproduce the full analysis pipeline, including data processing, modeling, and results.

## Dependencies

The project dependencies are listed in [environment.yml](environment.yml). Key packages include:

- Python 3.11
- JupyterLab
- pandas
- scikit-learn
- matplotlib
- ucimlrepo

## Licenses

This project is licensed under two licenses:

- **MIT License** — for the project code. See [LICENSE.md](LICENSE.md) for details.
- **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** — for the project report and associated written content. See [LICENSE.md](LICENSE.md) for details.
