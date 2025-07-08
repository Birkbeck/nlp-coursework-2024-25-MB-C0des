import pandas as pd
from pathlib import Path # to use Path for file paths
from sklearn.feature_extraction.text import TfidfVectorizer


def load_and_filter_data(path=Path.cwd() / "hansard40000.csv"):

    df = pd.read_csv(path) # Read in the hansard4000.csv into a pandas dataframe
    print(df.head) # df = pd.read_csv()

    print(df.describe())
    
    # Rename the "Labour (Co-op)" value to "Labour" in the party column
    df.replace('Labour (Co-op)', 'Labour', inplace=True)
    print(df['party'].value_counts())

    # Remove any rows where the party column isn't one of the top four (not including speaker)
    df = df[df.party != "Speaker"]

    top_parties = df['party'].value_counts().nlargest(4).index
    df = df[df['party'].isin(top_parties)]

    #df.drop(party=["Speaker"],inplace=True)
    print(df['party'].value_counts())

    # Remove any rows where the 'speech_class' column is NOT "Speech"
    df = df[df.speech_class == "Speech"]
    print(df['speech_class'].value_counts())

    # Remove rows where text in the 'speech' column is <1000 characters

    df = df[df.speech.str.len() >= 1000]

    print(df.shape)

    return df




if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    load_and_filter_data()
    
