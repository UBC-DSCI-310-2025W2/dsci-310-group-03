import pandas as pd

def extract_features_and_target(df, target_col):
    # Extracts features (X) and target (y) from a DataFrame.
    
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in the DataFrame.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y