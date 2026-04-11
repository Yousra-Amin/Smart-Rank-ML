from config import df

# Document length
df['doc_len'] = df['passage_clean'].apply(lambda x: len(x.split())) 

# Keyword density — how many query terms appear in the document 
def keyword_density(row): 
    q_terms = set(row['query_clean'].split()) 
    d_tokens = row['passage_clean'].split() 
    if not d_tokens: return 0 
    overlap = sum(1 for t in d_tokens if t in q_terms) 
    return overlap / len(d_tokens) 
df['keyword_density'] = df.apply(keyword_density, axis=1)