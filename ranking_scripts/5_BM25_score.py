# 5_BM25_score.py
import pandas as pd
from rank_bm25 import BM25Okapi  # Correct import

def compute_bm25_score(group):
    """Compute BM25 scores for each passage in a query group"""
    passages_tokenized = [p.split() for p in group['passage_clean']]
    bm25 = BM25Okapi(passages_tokenized)  # Correct class name
    query_tokens = group['query_clean'].iloc[0].split()
    scores = bm25.get_scores(query_tokens)
    group = group.copy()
    group['bm25_score'] = scores
    return group

# Apply to dataframe (assuming df is loaded)
# df = df.groupby('qid', group_keys=False).apply(compute_bm25_score)