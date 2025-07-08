import pandas as pd
from pathlib import Path # to use Path for file paths
from sklearn.feature_extraction.text import TfidfVectorizer


def load_and_filter_data(path=Path.cwd() / "hansard40000.csv"):

    df = pd.read_csv(path)
    print(df.head)
    return

    
# Read in the hansard4000.csv into a pandas dataframe
# df = pd.read_csv()


if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    load_and_filter_data()
    
