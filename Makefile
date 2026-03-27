# Makefile for Wine Quality Prediction Analysis
# Root Directory: dsci-310-group-03

# Targets that are not files
.PHONY: all clean test help

# Rendered report
all: _output/notebooks/wine_quality_prediction.html

# Download Data
data/raw/wine_quality_raw.csv: scripts/download_data.py
	python scripts/download_data.py --output-path data/raw/wine_quality_raw.csv

# Preprocess Data
data/processed/train.csv data/processed/test.csv: data/raw/wine_quality_raw.csv scripts/preprocess_data.py
	python -m scripts.preprocess_data --input-path data/raw/wine_quality_raw.csv \
		--train-output-path data/processed/train.csv \
		--test-output-path data/processed/test.csv

# EDA Artifacts
results/summary_stats.csv results/quality_distribution.png results/correlation_heatmap.png results/alcohol_by_quality.png: data/raw/wine_quality_raw.csv scripts/eda.py
	python -m scripts.eda --input-path data/raw/wine_quality_raw.csv --output-dir results

# Modeling Artifacts
results/cv_results.csv results/model_scores.csv results/cv_scores_vs_k.png: data/processed/train.csv data/processed/test.csv scripts/train_model.py
	python -m scripts.train_model --train-path data/processed/train.csv --test-path data/processed/test.csv --output-dir results

# Render Report
_output/notebooks/wine_quality_prediction.html: notebooks/wine_quality_prediction.qmd \
                                        notebooks/references.bib \
                                        results/summary_stats.csv \
                                        results/quality_distribution.png \
                                        results/correlation_heatmap.png \
                                        results/alcohol_by_quality.png \
                                        results/cv_results.csv \
                                        results/model_scores.csv \
                                        results/cv_scores_vs_k.png
	quarto render notebooks/wine_quality_prediction.qmd

# Cleanup
clean:
	rm -f data/raw/*.csv
	rm -f data/processed/*.csv
	rm -f results/*.csv
	rm -f results/*.png
	rm -rf _output/

# Run all unit tests
test:
	python -m pytest tests/

# Helper to explain the Makefile
help:
	@echo "Usage:"
	@echo "  make all      : Run the full analysis pipeline"
	@echo "  make test     : Run unit tests with pytest"
	@echo "  make clean    : Remove generated files and build artifacts"