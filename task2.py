import pickle
import pandas as pd
from collections import defaultdict
from task1 import preprocess_text, remove_stop_words

INVERTED_INDEX = 'inverted_index.pkl'
DATA = "cw-data/candidate-passages-top1000.tsv"

#remove stop words because they do not indicate topic and would be very memory inefficient to store in the inverted index and then follow all the same text processing as task1
# store frequency of each term per passage they are in



def parse_passage(passage: str, inverted_index: dict, passage_id: int):
    processed_passage = preprocess_text(passage)
    tokens = remove_stop_words(processed_passage)
    for token in tokens:
        if token not in inverted_index:
            inverted_index[token] = {}
        if passage_id not in inverted_index[token]:
            inverted_index[token][passage_id] = 0
        inverted_index[token][passage_id] += 1
    

def gen_inverted_index():
    "create inverted index, save to file, and return it"
    with open(DATA, 'r') as f:
        df = pd.read_csv(
            DATA,
            sep="\t",
            header=None,
            names=["qid", "pid", "query", "passage"],
            dtype={"qid": str, "pid": str}, 
            encoding="utf-8",
            engine="python"  
    )

    inverted_index = {}
    for index, row in df.iterrows():
        parse_passage(row['passage'], inverted_index, row['pid'])
    
    with open(INVERTED_INDEX, 'wb') as f:
        pickle.dump(inverted_index, f)
    return inverted_index


def load_inverted_index():
    try:
        with open(INVERTED_INDEX, 'rb') as f:
            inverted_index = pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        inverted_index = gen_inverted_index()
    return inverted_index



def task2():
    inverted_index = load_inverted_index()
    return inverted_index



if __name__ == "__main__":
    task2()
