import pickle
import pandas as pd
from collections import defaultdict
from task1 import preprocess_text, remove_stop_words

INVERTED_INDEX = 'inverted_index.pkl'
DATA = "cw-data/candidate-passages-top1000.tsv"

#remove stop words because they do not indicate topic and would be very memory inefficient to store in the inverted index and then follow all the same text processing as task1
# store frequency of each term per passage they are in


def read_tsv_data(file_path: str, names: list[str], encoding: str = "utf-8", header: int | None = None, dtype: dict[str, type] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        file_path,
        sep="\t",
        header=header,
        names=names,
        dtype=dtype, 
        encoding=encoding,
        engine="python"  )

def parse_passage(passage: str, inverted_index: dict, passage_id: int):
    processed_passage = preprocess_text(passage)
    tokens = remove_stop_words(processed_passage)
    for token in tokens:
        if token not in inverted_index:
            inverted_index[token] = defaultdict(int)
        inverted_index[token][passage_id] += 1
    

def gen_inverted_index():
    "create inverted index, save to file, and return it"
    df = read_tsv_data(DATA, names=["qid", "pid", "query", "passage"], dtype={"qid": str, "pid": str})

    inverted_index = {}
    seen_passages = set()
    for _, row in df.iterrows():
        if row['pid'] not in seen_passages:
            seen_passages.add(row['pid'])
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
    # print("freq of definition:", inverted_index.get("definition"))
    # print("total freq of definition", sum(inverted_index.get("definition", {}).values()))
    return inverted_index



if __name__ == "__main__":
    task2()
