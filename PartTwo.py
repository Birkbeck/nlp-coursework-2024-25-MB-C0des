import pandas as pd
from pathlib import Path # to use Path for file paths
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report



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
   
def train_and_test(X_train, X_test, y_train, y_test):
        
    rf = RandomForestClassifier(n_estimators = 3000) # Random Forest Classifier
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    macro_f1_rf = f1_score(y_test, y_pred_rf, average="macro")
    print(f"The Random Forest macro average F1 score: {macro_f1_rf:.4f}")
    print(classification_report(y_test, y_pred_rf))
    
    
    svm = SVC(kernel='linear')   # Linear SVM Classifier
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    macro_f1_svm = f1_score(y_test, y_pred_svm, average="macro")
    print(f"The Linear SVM macro average F1 score: {macro_f1_svm:.4f}")
    print(classification_report(y_test, y_pred_svm))


def vectorise_speeches_ngrams(df):

    # Train and test with n-grams in range (1, 3)
    # This is the same as the vectorise_speeches function but with n-grams
    vectorizer_ngrams = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 3))
    X_ngram = vectorizer_ngrams.fit_transform(df['speech'])
    y = df['party']

    X_ngram = vectorizer_ngrams.fit_transform(df["speech"])
    y = df["party"]
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(
        X_ngram, y,
        test_size=0.33,
        stratify=y,
        random_state=26
    )
    return Xn_train, Xn_test, yn_train, yn_test, vectorizer


if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    df = load_and_filter_data()
    X_train, X_test, y_train, y_test, vectorizer = vectorise_speeches(df)
    print(vectorizer)
     
    train_and_test(X_train, X_test, y_train, y_test)


    

    Xn_train, Xn_test, yn_train, yn_test, vectorizer = vectorise_speeches_ngrams(df)
    print(vectorizer)
     
    train_and_test(Xn_train, Xn_test, yn_train, yn_test)
    
