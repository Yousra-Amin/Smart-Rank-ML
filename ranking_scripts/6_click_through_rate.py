from config import df

# Compute CTR = clicks / impressions (if available in dataset) 
# For MS MARCO, use relevance as a proxy if raw clicks unavailable 
if 'click_count' in df.columns and 'impression_count' in df.columns: 
    df['ctr'] = df['click_count'] / (df['impression_count'] + 1e-9) 
else: 
# Proxy: proportion of relevant docs per query 
    df['ctr'] = df.groupby('qid')['relevance'].transform('mean')