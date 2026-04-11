from config import df

# Number of exact query terms appearing in the document 
def term_overlap(row): 
    q = set(row['query_clean'].split()) 
    d = set(row['passage_clean'].split()) 
    return len(q & d) 
df['term_overlap'] = df.apply(term_overlap, axis=1) 

# Binary: does the passage contain the entire query as a substring? 
df['exact_match'] = df.apply( 
    lambda r: int(r['query_clean'] in r['passage_clean']), axis=1 
)