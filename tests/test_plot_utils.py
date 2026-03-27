import os
import pytest
import pandas as pd
import matplotlib.pyplot as plt

from src.plot_utils import plot_correlation_heatmap
# `plot_correlation_heatmap` should return a PNG of a heatmap figure

# to run tests
# python -m pytest tests/test_plot_utils.py


@pytest.fixture
def numeric_df():
    """Create a small synthetic dataset for heatmap correlation testing"""
    return pd.DataFrame({
        "alcohol": [8.0, 9.0, 10.0, 11.0],
        "pH": [3.1, 3.2, 3.3, 3.4],
        "sulphates": [0.5, 0.6, 0.7, 0.8],
    })

class TestRunPlotCorrelationHeatmap:
    """tests for plot_correlation_heatmap"""
    #test if output PNG file is created at the specified path
    def test_output_file_is_created_path(self, numeric_df, tmp_path):
        output = str(tmp_path / "correlation_heatmap.png")
        plot_correlation_heatmap(numeric_df, output)
        assert os.path.exists(output)

    #test if output file is a valid PNG
    def test_output_is_valid_png(self, numeric_df, tmp_path):
        output = str(tmp_path / "correlation_heatmap.png")
        plot_correlation_heatmap(numeric_df, output)
        with open(output, "rb") as f:
            magic = f.read(4)
        assert magic == b"\x89PNG"
        

    #test if works with numeric only dataframe
    def test_works_with_numeric_only_dataframe(self, numeric_df, tmp_path):
        output = str(tmp_path / "correlation_heatmap.png")
        plot_correlation_heatmap(numeric_df, output)  # should not raise

    #test if it raises error with no numeric columns
    def test_raise_with_no_numeric_columns(self, tmp_path):
        df = pd.DataFrame({"type": ["red", "white"], "region": ["napa", "sonoma"]})
        output = str(tmp_path / "correlation_heatmap.png")
        with pytest.raises(ValueError, match="numeric"):
            plot_correlation_heatmap(df, output)

    #test custom figsize is respected
    def test_custom_figsize_is_respected(self, numeric_df, tmp_path):
        output = str(tmp_path / "correlation_heatmap.png")
        plot_correlation_heatmap(numeric_df, output, figsize=(10, 9))
        img = plt.imread(output)
        assert img.shape[1] == 1000  # width: 10in * 100dpi
        assert img.shape[0] == 900   # height: 9in * 100dpi

    #creates parent directories if they do not exist
    def test_creates_parent_directories(self, numeric_df, tmp_path):
        output = str(tmp_path / "nested" / "dirs" / "correlation_heatmap.png")
        plot_correlation_heatmap(numeric_df, output)
        assert os.path.exists(output)
