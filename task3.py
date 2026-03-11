

import math

from task1 import preprocess_text
from task2 import load_inverted_index, read_tsv_data
import pandas as pd
QUERIES_PATH = "cw-data/test-queries.tsv"
PASSAGES_PATH = "cw-data/candidate-passages-top1000.tsv"
#inverted_index = {Token: {passage_id: frequency}}
#idf_dict = {Token: idf_value = log(total_docs / (docs_with_token))}
#tf_dict = {Token: {passage_id: frequency}}

def get_tf_dict(inverted_index: dict[str, dict[int, int]]) -> dict[str, dict[int, int]]:
    return inverted_index
    #NOTE if we decide to make the inverted index more complex we will have to simplify this function to return the correct tf

def get_idf_dict(inverted_index: dict[str, dict[int, int]], total_docs: int) -> dict[str, float]:
    idf_dict = {}
    for word, passages in inverted_index.items():
        doc_frequency = len(passages)
        idf_dict[word] = math.log(total_docs / doc_frequency)
        if word == "definition":
            print(f"Word: {word}, Doc Frequency: {doc_frequency}, IDF: {idf_dict[word]}")  
    return idf_dict


def cosine_similarity(query, doc_id, tf_dict, idf_dict):
    sum = 0
    for word in query:
        if word in tf_dict and doc_id in tf_dict[word]:
            sum += (tf_dict[word][doc_id] * idf_dict[word])
    return sum

def calc_bm25_score(query_tokens, doc_id, tf_dict, idf_dict, k1, k2, b, doc_length, avg_doc_length):
    score = 0
    query_freq = {word: query_tokens.count(word) for word in query_tokens}
    for word in query_tokens:
        if word in tf_dict and doc_id in tf_dict[word]:
            tf = tf_dict[word][doc_id]
            idf = idf_dict.get(word, 0)
            K = k1 * ((1 - b) + b * (doc_length / avg_doc_length))
            score += idf * ((tf * (k1 + 1)) / (tf + K)) * ((k2 + 1) * query_freq[word]) / (k2 + query_freq[word])
    return score

def generate_tfidf_csv(queries: list[str], document_df):
    #process each query each should match to a max of 100 passages and add those matches to the output df
    #output_df = pd.DataFrame(columns=['qid', 'pid', 'score'])
    for query in queries:
        query_tokens = preprocess_text(query)
        matched_docs = 0
        for doc_id in document_df['passage_id']:
            cosine_sim = 0#cosine_similarity(query_tokens, doc_id, tf_dict, idf_dict, inverted_index)
            if cosine_sim > 0:
                #output_df = output_df.append({'qid': query_id, 'pid': doc_id, 'score': cosine_sim}, ignore_index=True)
                matched_docs += 1
            if matched_docs >= 100:
                break

    return 0
    # return output_df output should not have headers, rank the qid,pid pairs by similarity score in decending order



def generate_bm25_csv(queries_df: pd.DataFrame, candidate_passages_df: pd.DataFrame, doc_lengths: dict[int, int], tf_dict: dict[str, dict[int, int]], idf_dict: dict[str, float], k1: float = 1.2, k2: float = 100, b: float = 0.75):
    #K = k1((1-b) + b * (doc_length / avg_doc_length))
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths)
    results = []

    for index, query_row in queries_df.iterrows():
        qid = query_row['qid']
        query = query_row['query']
        query_tokens = preprocess_text(query)

        candidate_rows = candidate_passages_df[candidate_passages_df["qid"] == qid]

        scores = []

        for _, passage_row in candidate_rows.iterrows():
            pid = passage_row['pid']
            doc_length = doc_lengths.get(pid, 1)
            bm_25_score = calc_bm25_score(query_tokens, pid, tf_dict, idf_dict, k1, k2, b, doc_length, avg_doc_length)
            scores.append((pid, bm_25_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        for pid, score in scores[:100]:
            results.append([qid,pid,score])
        
        print(f"finished processing row {index} with qid {qid}")
        print(f"completed {index + 1} out of {len(queries_df)} queries")

    output_df = pd.DataFrame(results)
    output_df.to_csv("bm25_output.csv", index=False, header=False)

def load_document_lengths(candidate_passages_df: pd.DataFrame) -> dict[int, int]:
    doc_lengths = {}
    for _, row in candidate_passages_df.iterrows():
        pid = row['pid']
        if pid in doc_lengths:
            continue
        passage = row['passage']
        processed_passage = preprocess_text(passage)
        doc_lengths[pid] = len(processed_passage)
    return doc_lengths


#When done test the BM25 implementation against the dummy data from slides
#Check if output number is the same
#do we have to preprocess queries? Should be fine since the inverted index applies the same processing so we would look at the same tokens?



def task3(): # This function should return a dictionary of passage_id to document length

    candidate_passages_df = read_tsv_data(PASSAGES_PATH, names=["qid", "pid", "query", "passage"], dtype={"qid": str, "pid": str})
    num_candidate_passages = candidate_passages_df['pid'].nunique()
    queries_df = read_tsv_data(QUERIES_PATH, names=["qid", "query"], dtype={"qid": str}) 
    
    document_lengths = load_document_lengths(candidate_passages_df) 
    inverted_index = load_inverted_index()
    tf_dict = get_tf_dict(inverted_index)
    idf_dict = get_idf_dict(inverted_index, total_docs= num_candidate_passages)

    generate_bm25_csv(queries_df, candidate_passages_df, document_lengths, tf_dict, idf_dict)
    print("completed bm25 generation")

    generate_tfidf_csv()



    return



if __name__ == "__main__":
    task3()
