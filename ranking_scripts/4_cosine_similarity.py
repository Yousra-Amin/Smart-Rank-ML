from config import df

from scikit-learn.feature_extraction.text import TfidfVectorizer
from scikit-learn.metrics.pairwise import cosine_similarity 
import scipy.sparse 

# Fit TF-IDF on all passages 
tfidf = TfidfVectorizer(max_features=50000) 
corpus = df['passage_clean'].tolist() 
doc_matrix = tfidf.fit_transform(corpus) # sparse matrix 
# Transform queries using the same vocab 
query_matrix = tfidf.transform(df['query_clean'].tolist()) 

# Cosine similarity row-by-row 
df['tfidf_cosine'] = [ 
    cosine_similarity(query_matrix[i], doc_matrix[i])[0][0] 
    for i in range(len(df)) 
]