import pandas as pd
from pathlib import Path # to use Path for file paths
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split



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


    
def vectorise_speeches(df):
    """
    Vectorizes the speeches in the DataFrame using TF-IDF.
    Returns a DataFrame with the TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(df['speech'])
    
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']  # Predict party variable
    
    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.33,
        stratify=y,
        random_state=26
    )
    return X_train, X_test, y_train, y_test, vectorizer
   




if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    df = load_and_filter_data()
    X_train, X_test, y_train, y_test, vectorizer = vectorise_speeches(df)
    print(vectorizer)
     
    
