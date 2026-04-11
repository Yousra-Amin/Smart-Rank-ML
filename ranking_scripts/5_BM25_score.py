from config import df

# pip install rank-bm25 
from rank_bm25 import BM25Okapi 

# Build a BM25 index over all passages (grouped by query for efficiency) 
def compute_bm25_score(group): 
    passages_tokenized = [p.split() for p in group['passage_clean']] 
    bm25 = BM25Okapi(passages_tokenized) 
    query_tokens = group['query_clean'].iloc[0].split() 
    scores = bm25.get_scores(query_tokens) 
    group = group.copy() 
    group['bm25_score'] = scores 
    return group 

df = df.groupby('qid', group_keys=False).apply(compute_bm25_score)