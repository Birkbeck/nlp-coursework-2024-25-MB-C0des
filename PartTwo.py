import pandas as pd
from pathlib import Path


def load_and_filter_data(path=Path.cwd() / "hansard40000.csv"):
    # 1. Load
    df = pd.read_csv(path)

    
# Read in the hansard4000.csv into a pandas dataframe
# df = pd.read_csv()

print(df.describe())