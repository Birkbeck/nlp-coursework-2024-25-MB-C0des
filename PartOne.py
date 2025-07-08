#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.
import os   # to use os.path.join
import glob  # to use glob.glob

import nltk  # to use nltk.corpus.cmudict
nltk.download("punkt_tab")  # to download the punkt tokenizer
import spacy  # to use spacy for text processing
from pathlib import Path  # to use Path for file paths

import pandas as pd  # to use pandas for dataframes

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    
    # Calculate total words
    tokens = nltk.tokenize.word_tokenize(text.lower())
    new_tokens = [word for word in tokens if word.isalpha()]

    num_tokens = len(new_tokens)

    print(f"Number of tokens: {num_tokens}")

    # Calculate total sentences

    sentences = nltk.tokenize.sent_tokenize(text)
    num_sentences = len(sentences)
    print(f"Number of sentences: {num_sentences}")

    # Calculate total syllables
    total_syl = 0
    for token in new_tokens:
        total_syl = total_syl + count_syl(token, d)

    print(f"Number of syllables: {total_syl}")

    # Flesch-Kincaid Grade Level = 0.39 × ( Total Words / Total Sentences ) + 11.8 × ( Total Syllables / Total Words ) − 15.59

    fk = 0.39 * (num_tokens / num_sentences) + 11.8 * (total_syl /
                                                       num_tokens) - 15.59
    print(f"Flesch-Kincaid Grade Level: {fk}")

    return fk
    pass


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """

    total = 0

    if word in d:
        syllables = d[word]
        for syl in syllables[0]:
            if any(c.isdigit() for c in syl):
                total = total + 1

        return total
    else:
        total = estimate_syllables(word)
        #print(f"Word not in dictionary: {word}, estimated as {total}")
        return total
    pass


def estimate_syllables(word):
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if count == 0:
        count += 1
    return count
    pass

def read_novels(path=Path.cwd() / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    
    data = []

    for file in path.glob("*.txt"):
        parts = file.stem.split("-") 


        title = parts[0]
        author = parts[1]
        year = parts[2]

        year = year.split(".")
        year = year[0]

        title = title.replace("_", " ")

        # Read file from the filename
        file = open(file, "r", encoding='utf-8')
        text = file.read()

        # Add this record to the list
        data.append([text, title, author, year])

    df = pd.DataFrame(data)

    # Add the titles to the columns ["text", "title", "author", "year"]
    names = ["text", "title", "author", "year"]
    df.rename(columns={
        0: 'text',
        1: 'title',
        2: 'author',
        3: 'year'
    },
              inplace=True)

    # Sort data by year
    df = df.sort_values(by='year', ascending=False)
    #print(df)

    return df
    pass


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
   # Tokenise the text
    tokens = nltk.tokenize.word_tokenize(text.lower())
    new_tokens = [word for word in tokens if word.isalpha()]
    #print(new_tokens)

    types = set(new_tokens)
    #print(new_tokens)

    num_types = len(types)
    #print(f"Number of types: {num_types}")

    num_tokens = len(new_tokens)
    #print(f"Number of tokens: {num_tokens}")

    # ttr = num of types/ num of tokens
    ttr = num_types / num_tokens
    print(f"Type token ratio: {ttr}")

    return ttr
    pass

def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd()/ "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

