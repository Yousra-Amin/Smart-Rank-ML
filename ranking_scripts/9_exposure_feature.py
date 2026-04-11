from config import df

feature_cols = [
    'qid', 'pid', 'relevance', 
    'query_len', 'query_avg_tf', 
    'doc_len', 'keyword_density',
    'tfidf_cosine', 'bm25_score', 
    'ctr', 'term_overlap', 'exact_match', 
    'dwell_time_proxy', # or dwell_time_norm 
]
features_df = df[feature_cols].dropna() 
features_df.to_csv('features_dataset.csv', index=False) 
print(f'Feature matrix shape: {features_df.shape}')