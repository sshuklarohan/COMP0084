import math
from task1 import preprocess_text
from task2 import load_inverted_index, read_tsv_data
from task3 import get_tf_dict, get_idf_dict, load_document_lengths
import pandas as pd
QUERIES_PATH = "cw-data/test-queries.tsv"
PASSAGES_PATH = "cw-data/candidate-passages-top1000.tsv"


def laplace_smoothing(query_tokens, pid, tf_dict, doc_length, vocab_size):
    score = 0
    for token in query_tokens:
        tf = tf_dict.get(token, {}).get(pid, 0)
        token_prob = (tf + 1) / (doc_length + vocab_size)
        score += math.log(token_prob)
    return score


def lidstone_correction(query_tokens, pid, tf_dict, doc_length, vocab_size, epsilon=0.1):
    score = 0
    for token in query_tokens:
        tf = tf_dict.get(token, {}).get(pid, 0)
        token_prob = (tf + epsilon) / (doc_length + epsilon * vocab_size)
        score += math.log(token_prob)
    return score


def dirichlet_smoothing(query_tokens, pid, tf_dict, doc_length, collection_freq, collection_size, mu=50):
    score = 0
    for token in query_tokens:
        if token not in collection_freq:
            continue
        tf = tf_dict.get(token, {}).get(pid, 0)
        collection_prob = collection_freq.get(token, 0) / collection_size
        token_prob = (tf + mu * collection_prob) / (doc_length + mu)
        score += math.log(token_prob)
    return score

def generate_laplace_csv(queries_df: pd.DataFrame, candidate_passages_df: pd.DataFrame, doc_lengths: dict[int, int], tf_dict: dict[str, dict[int, int]], vocab_size: int):
    results = []
    for index, query_row in queries_df.iterrows():
        qid = query_row['qid']
        query = query_row['query']
        query_tokens = preprocess_text(query)

        candidate_rows = candidate_passages_df[candidate_passages_df["qid"] == qid]

        scores = []

        for _, passage_row in candidate_rows.iterrows():
            pid = passage_row['pid']
            score = laplace_smoothing(query_tokens, pid, tf_dict, doc_lengths[pid], vocab_size)
            scores.append((pid, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        for pid, score in scores[:100]:
            results.append([qid,pid,score])
        
        print(f"finished processing row {index} with qid {qid}")
        print(f"completed {index + 1} out of {len(queries_df)} queries")

    output_df = pd.DataFrame(results)
    output_df.to_csv("laplace.csv", index=False, header=False)

def generate_lidstone_csv(queries_df: pd.DataFrame, candidate_passages_df: pd.DataFrame, doc_lengths: dict[int, int], tf_dict: dict[str, dict[int, int]], vocab_size: int, epsilon=0.1):
    results = []
    for index, query_row in queries_df.iterrows():
        qid = query_row['qid']
        query = query_row['query']
        query_tokens = preprocess_text(query)

        candidate_rows = candidate_passages_df[candidate_passages_df["qid"] == qid]

        scores = []

        for _, passage_row in candidate_rows.iterrows():
            pid = passage_row['pid']
            score = lidstone_correction(query_tokens, pid, tf_dict, doc_lengths[pid], vocab_size, epsilon)
            scores.append((pid, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        for pid, score in scores[:100]:
            results.append([qid,pid,score])
        
        print(f"finished processing row {index} with qid {qid}")
        print(f"completed {index + 1} out of {len(queries_df)} queries")

    output_df = pd.DataFrame(results)
    output_df.to_csv("lidstone.csv", index=False, header=False)

def generate_dirichlet_csv(queries_df: pd.DataFrame, candidate_passages_df: pd.DataFrame, doc_lengths: dict[int, int], tf_dict: dict[str, dict[int, int]], collection_freq: dict[str, int], collection_size: int, mu=50):
    results = []
    for index, query_row in queries_df.iterrows():
        qid = query_row['qid']
        query = query_row['query']
        query_tokens = preprocess_text(query)

        candidate_rows = candidate_passages_df[candidate_passages_df["qid"] == qid]

        scores = []

        for _, passage_row in candidate_rows.iterrows():
            pid = passage_row['pid']
            score = dirichlet_smoothing(query_tokens, pid, tf_dict, doc_lengths[pid], collection_freq, collection_size, mu)
            scores.append((pid, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        for pid, score in scores[:100]:
            results.append([qid,pid,score])
        
        print(f"finished processing row {index} with qid {qid}")
        print(f"completed {index + 1} out of {len(queries_df)} queries")

    output_df = pd.DataFrame(results)
    output_df.to_csv("dirichlet.csv", index=False, header=False)

def task_4():
    candidate_passages_df = read_tsv_data(PASSAGES_PATH, names=["qid", "pid", "query", "passage"], dtype={"qid": str, "pid": str})
    num_candidate_passages = candidate_passages_df['pid'].nunique()
    queries_df = read_tsv_data(QUERIES_PATH, names=["qid", "query"], dtype={"qid": str}) 
    
    document_lengths = load_document_lengths(candidate_passages_df) 
    inverted_index = load_inverted_index()
    tf_dict = get_tf_dict(inverted_index)
    cf_dict = {token: sum(pids.values()) for token, pids in tf_dict.items()}

    print("---- starting Laplace Smoothing ----")
    generate_laplace_csv(queries_df, candidate_passages_df, document_lengths, tf_dict, len(tf_dict))
    print("---- finished Laplace Smoothing ----")

    print("---- starting Lidstone Correction ----")
    generate_lidstone_csv(queries_df, candidate_passages_df, document_lengths, tf_dict, len(tf_dict), epsilon=0.1)
    print("---- finished Lidstone Correction ----")

    print("---- starting Dirichlet Smoothing ----")
    generate_dirichlet_csv(queries_df, candidate_passages_df, document_lengths, tf_dict, cf_dict, num_candidate_passages)
    print("---- finished Dirichlet Smoothing ----")




if __name__ == "__main__":
    task_4()