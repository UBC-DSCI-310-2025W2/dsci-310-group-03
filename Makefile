# Makefile for Wine Quality Prediction Analysis
# Root Directory: dsci-310-group-03

# Targets that are not files
.PHONY: all clean

# endered report
all: notebooks/wine_quality_prediction.html

# Download Data
data/wine_quality_raw.csv: scripts/download_data.py
	python scripts/download_data.py --output-path data/wine_quality_raw.csv

# Preprocess Data
data/train.csv data/test.csv: data/wine_quality_raw.csv scripts/preprocess_data.py
	python scripts/preprocess_data.py --input-path data/wine_quality_raw.csv \
		--train-output-path data/train.csv \
		--test-output-path data/test.csv

# EDA Artifacts
results/summary_stats.csv results/quality_distribution.png results/correlation_heatmap.png results/alcohol_by_quality.png: data/wine_quality_raw.csv scripts/eda.py
	python scripts/eda.py --input-path data/wine_quality_raw.csv --output-dir results

# Modeling Artifacts
results/cv_results.csv results/model_scores.csv results/cv_scores_vs_k.png: data/train.csv data/test.csv scripts/train_model.py
	python scripts/train_model.py --train-path data/train.csv --test-path data/test.csv --output-dir results

# Render Report
notebooks/wine_quality_prediction.html: notebooks/wine_quality_prediction.qmd \
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
	rm -f data/*.csv
	rm -f results/*.csv
	rm -f results/*.png
	rm -f notebooks/wine_quality_prediction.html