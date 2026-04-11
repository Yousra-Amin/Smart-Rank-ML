from config import df

# Query length (number of tokens) 
df['query_len'] = df['query_clean'].apply(lambda x: len(x.split())) 

# Query term frequency — avg frequency of each query term in the query 
from collections import Counter 
def avg_tf(text): 
    tokens = text.split() 
    if not tokens: return 0 
    counts = Counter(tokens) 
    return sum(counts.values()) / len(tokens) 

df['query_avg_tf'] = df['query_clean'].apply(avg_tf)
