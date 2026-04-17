import numpy as np
import pandas as pd
from task1 import preprocess_text
from task2 import gen_inverted_index, load_inverted_index, read_tsv_data
from task3 import calc_bm25_score, get_idf_dict, get_tf_dict, load_document_lengths
import time
from sklearn.preprocessing import StandardScaler
TRAIN_PATH = "cw-data/train-data.tsv"
VALIDATION_PATH = "cw-data/validation-data.tsv"
NUM_QUERIES_TO_SAMPLE_BM25 = 1500
NUM_QUERIES_TO_SAMPLE_LR = 500
import os
import gensim.downloader as api




class LogisticRegression():

    def __init__(self, learning_rate=0.01, batch_size=32, epochs=1000):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def init_params(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, 0)
    
    def compute_loss(self, X, y):
        m = len(y)
        proba = np.clip(self.predict_proba(X), 1e-15, 1 - 1e-15)
        loss = - (1/m) * np.sum(y * np.log(proba) + (1 - y) * np.log(1 - proba))
        return loss
    

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.init_params(n_features)

        for epoch in range(self.epochs):
        # Shuffle data at the start of each epoch
            indices = np.random.permutation(n_samples)

            # Mini-batch updates
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                batch_size = len(X_batch)  # last batch may be smaller

                proba = self.predict_proba(X_batch)
                dw = (1 / batch_size) * np.dot(X_batch.T, (proba - y_batch))
                db = (1 / batch_size) * np.sum(proba - y_batch)

                self.weights -= self.learning_rate * dw
                self.bias    -= self.learning_rate * db
            
            loss = self.compute_loss(X, y)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")


def average_precision(ranked_relevance: list[int])-> float:
    
    num_relevant = sum(ranked_relevance)
    if num_relevant == 0:
        return 0.0
 
    score = 0.0
    hits = 0
    for rank, rel in enumerate(ranked_relevance, start=1):
        if rel:
            hits += 1
            score += hits / rank          
    return score / num_relevant


def dcg(ranked_relevance: list[int], k: int) -> float:
    dcg_score = 0.0
    for i, rel in enumerate(ranked_relevance[:k], start=1):
        dcg_score += rel / np.log2(i + 1)   # log base 2, position starts at 1
    return dcg_score

def ndcg(ranked_relevance: list[int], k: int) -> float:
    ideal_relevance = sorted(ranked_relevance, reverse=True)
    ideal_dcg = dcg(ideal_relevance, k)
    if ideal_dcg == 0:
        return 0.0
    actual_dcg = dcg(ranked_relevance, k)
    return actual_dcg / ideal_dcg

def bm25_rank(query_tokens: list[str], candidate_passages_df: pd.DataFrame, tf_dict: dict[str, dict[int, int]], idf_dict: dict[str, float], doc_lengths: dict[int, int], avg_doc_length: float, k1: float, k2: float, b: float) -> list[tuple[int, float]]:
    scores = []
    for _, passage_row in candidate_passages_df.iterrows():
        pid = passage_row['pid']
        doc_length = doc_lengths.get(pid, 1)
        bm_25_score = calc_bm25_score(query_tokens, pid, tf_dict, idf_dict, k1, k2, b, doc_length, avg_doc_length)
        scores.append((pid, bm_25_score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def create_rank_relevance_list(ranked: list[tuple[int, float]], relevance_df: pd.DataFrame) -> list[int]:
    relevance_dict = dict(zip(relevance_df['pid'], relevance_df['relevance']))
    return [float(relevance_dict.get(pid, 0)) for pid, _ in ranked]

def evaluate_bm25(df: pd.DataFrame, bm25_params: dict[str, float],tf_dict: dict[str, dict[int, int]], idf_dict: dict[str, float], doc_lengths: dict[int, int], avg_doc_length: float, k: int = 10) -> dict[str, float]:
    ap_scores = []
    ndcg_scores = []

    k1, k2, b = bm25_params['k1'], bm25_params['k2'], bm25_params['b']
    for qid, group in df.groupby('qid'):
        query_tokens = preprocess_text(group['query'].iloc[0])
        candidate_passages_df = group[['pid', 'relevance']]
        ranked = bm25_rank(query_tokens, candidate_passages_df, tf_dict, idf_dict, doc_lengths, avg_doc_length, k1, k2, b)
        ranked_relevance = create_rank_relevance_list(ranked, candidate_passages_df)

        ap_scores.append(average_precision(ranked_relevance))
        ndcg_scores.append(ndcg(ranked_relevance, k))
    
    return {
        "MAP": np.mean(ap_scores),
        f"NDCG@{k}": np.mean(ndcg_scores),
        "num_queries": len(ap_scores)
    }




def evaluate_lr(df, model: LogisticRegression, scaler, embeddings):
    ap_scores = []
    ndcg_scores = []

    for qid, group in df.groupby('qid'):
        X_val, y_val = build_features(group, embeddings, dim=100)
        X_val_scaled = scaler.transform(X_val)
        proba = model.predict_proba(X_val_scaled)

        ranked_relevance = [float(rel) for _, rel in sorted(zip(proba, y_val), key=lambda x: x[0], reverse=True)]

        ap_scores.append(average_precision(ranked_relevance))
        ndcg_scores.append(ndcg(ranked_relevance, k=10))

    return {
        "MAP": np.mean(ap_scores),
        "NDCG@10": np.mean(ndcg_scores),
        "num_queries": len(ap_scores)
    }
    

def logistic_learning_rates(embeddings, train_df,valid_df, X_train, y_train, scaler):
    learning_rates = [0.1,0.01,0.001,]
    models = {}
    results_list = []

    for lr in learning_rates:
        model = LogisticRegression(learning_rate=lr, batch_size=32, epochs=100)
        model.fit(X_train, y_train)
        models[lr] = model
    
    for lr, model in models.items():
            results = evaluate_lr(valid_df, model, scaler, embeddings)
            print(f"Training Metrics for learning rate {lr}: MAP={results['MAP']:.4f}, NDCG@10={results['NDCG@10']:.4f}")
            results_list.append((lr, results["MAP"], results["NDCG@10"]))
    
    res_df = pd.DataFrame(
        results_list, columns=["learning_rate", "MAP", "NDCG@10"]
    ).sort_values("MAP", ascending=False)
    best_row = res_df.iloc[0]
    best_params = {'lr': best_row['learning_rate']}

    print("\n Ranked performance (validation):")
    print(res_df.head(10).to_string(index=False))
    return best_params, res_df

def grid_search(train_df: pd.DataFrame, tf_dict: dict[str, dict[int, int]], idf_dict: dict[str, float], doc_lengths: dict[int, int], avg_doc_length: float, k: int = 10) -> dict[str, float]:
    k1_values = [1.2,1.5,1.75]
    k2 = 100
    b_values = [0.5,0.75, 0.8]
    results = []

    # Build all (k1, b) combos upfront

    for k1 in k1_values:
        for b in b_values:
            bm25_params = {'k1': k1, 'b': b, 'k2': k2}
            metrics = evaluate_bm25(train_df, bm25_params, tf_dict, idf_dict, doc_lengths, avg_doc_length)
            combined = metrics["MAP"] + metrics[f"NDCG@{k}"]
            results.append((k1, b, metrics["MAP"], metrics[f"NDCG@{k}"], combined))


    res_df = pd.DataFrame(
        results, columns=["k1", "b", "MAP", f"NDCG@{k}", "MAP+NDCG"]
    ).sort_values("MAP+NDCG", ascending=False)

    best_row = res_df.iloc[0]
    best_params = {'k1': best_row['k1'], 'b': best_row['b'], 'k2': k2}

    print("\nTop 10 parameter combinations (training):")
    print(res_df.head(10).to_string(index=False))

    return best_params, res_df

def sentence_embedding(tokens, embeddings, dim):
    vectors = []
    
    for t in tokens:
        if t in embeddings:
            vectors.append(embeddings[t])
    
    if not vectors:
        return np.zeros(dim)
    
    return np.mean(vectors, axis=0)


def build_features(df, embeddings, dim):
    X, y = [], []
    
    for _, row in df.iterrows():
        q_tokens = preprocess_text(str(row["query"]))
        p_tokens = preprocess_text(str(row["passage"]))
        
        q_emb = sentence_embedding(q_tokens, embeddings, dim)
        p_emb = sentence_embedding(p_tokens, embeddings, dim)
        
        # Stronger feature representation
        features = np.concatenate([
            q_emb,
            p_emb,
        ])
        
        X.append(features)
        y.append(float(row["relevance"]))
    
    return np.array(X), np.array(y)


def task_5():
    train_df = read_tsv_data(TRAIN_PATH, names=["qid", "pid", "query", "passage", "relevance"], dtype={"qid": str, "pid": str}, header=0)
    validation_df = read_tsv_data(VALIDATION_PATH, names=["qid", "pid", "query", "passage", "relevance"], dtype={"qid": str, "pid": str}, header=0)
    print(f"Total queries in training data: {train_df['qid'].nunique()}, Total rows: {len(train_df)}")
    sample_qids = train_df['qid'].drop_duplicates().sample(n=NUM_QUERIES_TO_SAMPLE_BM25, random_state=42)
    sampled_df = train_df[train_df['qid'].isin(sample_qids)]
    print(f"Sampled queries: {sampled_df['qid'].nunique()}, Total rows in sampled data: {len(sampled_df)}")


    print("Finished reading training data. Generating inverted index and calculating BM25 components...")
    
    num_candidate_passages = sampled_df['pid'].nunique()
    document_lengths = load_document_lengths(sampled_df)
    inverted_index = gen_inverted_index(sampled_df)
    tf_dict = get_tf_dict(inverted_index)
    idf_dict = get_idf_dict(inverted_index, total_docs= num_candidate_passages)
    avg_doc_length = sum(document_lengths.values()) / len(document_lengths)

    print("Finished generating inverted index and calculating BM25 components. Starting grid search for best parameters...")

    best_params, results_df = grid_search(sampled_df, tf_dict, idf_dict, document_lengths, avg_doc_length, k=10)

    print(f"\nBest parameters found: k1={best_params['k1']}, b={best_params['b']}")

    
    validation_metrics = evaluate_bm25(validation_df, best_params, tf_dict, idf_dict, document_lengths, avg_doc_length, k=10)
    print(f"\nValidation Metrics with best parameters: MAP={validation_metrics['MAP']:.4f}, NDCG@10={validation_metrics['NDCG@10']:.4f}")
    
    
    #----- D14: Logistic Regression with GloVe Embeddings -----#
    embeddings = api.load("glove-wiki-gigaword-100")
    sample_qids = train_df['qid'].drop_duplicates().sample(n=NUM_QUERIES_TO_SAMPLE_LR, random_state=42)
    sampled_df = train_df[train_df['qid'].isin(sample_qids)]
    
    scaler = StandardScaler()
    X_train, y_train = build_features(sampled_df, embeddings, 100)
    X_train = scaler.fit_transform(X_train)
    print("data ready for model training")

    best_params, results_df = logistic_learning_rates(embeddings, sampled_df,validation_df, X_train, y_train, scaler)

if __name__ == "__main__":
    task_5()

