import pandas as pd, numpy as np 
df = pd.read_csv('data/clean_search_dataset.csv')


# qid,pid,query_clean,passage_clean,relevance
# 188714,1000052,foods and supplements to lower blood sugar,watch portion sizes even healthy foods will cause high blood sugar if you eat too much make sure each of your meals has the same amount of chos avoid foods high in sugar some foods to avoid sugar honey candies syrup cakes cookies regular soda and,0
# 1082792,1000084,what does the golgi apparatus do to the proteins and lipids once they arrive,start studying bonding carbs proteins lipids learn vocabulary terms and more with flashcards games and other study tools,0
# 995526,1000094,where is the federal penitentiary in ind,it takes thousands of macys associates to bring the magic of macys to life our associate team is an invaluable part of who we are and what we do f ind the seasonal job thats right for you at holidaymacysjobscom,0
# 199776,1000115,health benefits of eating vegetarian,the good news is that you will discover what goes into action spurs narrowing of these foods not only a theoretical supposition there are diagnosed with great remedy is said that most people and more can be done duncan was a wonderful can eating chicken cause gout benefits of natural options with your health,0
# 660957,1000115,what foods are good if you have gout,the good news is that you will discover what goes into action spurs narrowing of these foods not only a theoretical supposition there are diagnosed with great remedy is said that most people and more can be done duncan was a wonderful can eating chicken cause gout benefits of natural options with your health,0
# 820267,1000130,what is the endocrine system responsible for,the pancreas secretes pancreatic enzyme which is responsible for the breakdown of protein but it also secretes insulin so that we can get energy from glucose making it a part of the endocrine system or glands the liver detoxifies all of the blood that carries absorbed nutrients from the digestive system,0
# 837202,1000252,what is the nutritional value of oatmeal,oats make an easy balanced breakfast one cup of cooked oatmeal contains about 150 calories four grams of fiber about half soluble and half insoluble and six grams of protein to boost protein further my favorite way to eat oatmeal is with a swirl of almond butter nestled within,0
# 130825,1000268,definition for daring,such a requirement would have three desirable consequences first it would tend to make bank executives more conservative and less daring in gambling with other peoples money second it would put this liability of financial decision makers ahead of any taxpayer bailout in case of insolvency and third it would create a potentially powerful diseconomy of scale within big conglomerate banks,0
# 408149,1000288,is dhgate a scam,if you think you ve been targeted by a counterfeit check scam report it to the following agencies 1 the federal trade commission or 1877ftchelp 18773824357 2 the us postal inspection service or call your local post office 3 the number is in the blue pages of your local telephone directoryere s how to avoid a counterfeit check scam 1 throw away any offer that asks you to pay for a prize or a gift 2 if it s free or a gift a promotion or a sweepstakes you shouldn t have to pay anything for it,0
# 345453,1000327,how to become a teacher assistant,top 10 amazing movie makeup transformations biological chemistry or biochemistry is the study of the chemical composition of living organisms at a cellular levelomeone seeking a career in biological chemistry will usually need at least a bachelors degree with a bachelors degree an individual can qualify for a job as a science teacher at the high school level a research assistant laboratory technician or a scientist in a testing environment,0


print("\n\nCompleted Step 1\n\n")


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


print("\n\nCompleted Step 2\n\n")


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


print("\n\nCompleted Step 3\n\n")


from config import df

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
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



print("\n\nCompleted Step 4\n\n")



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


print("\n\nCompleted Step 5\n\n")


from config import df

# Compute CTR = clicks / impressions (if available in dataset) 
# For MS MARCO, use relevance as a proxy if raw clicks unavailable 
if 'click_count' in df.columns and 'impression_count' in df.columns: 
    df['ctr'] = df['click_count'] / (df['impression_count'] + 1e-9) 
else: 
# Proxy: proportion of relevant docs per query 
    df['ctr'] = df.groupby('qid')['relevance'].transform('mean')


print("\n\nCompleted Step 6\n\n")  


from config import df

# If raw dwell_time is in the dataset, normalize it 
if 'dwell_time' in df.columns: 
    from sklearn.preprocessing import MinMaxScaler 
    df['dwell_time_norm'] = MinMaxScaler().fit_transform(df[['dwell_time']]) 
else: 
    # Proxy: longer passages tend to have higher dwell time 
    
    df['dwell_time_proxy'] = np.log1p(df['doc_len'])


print("\n\nCompleted Step 7\n\n")


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


print("\n\nCompleted Step 8\n\n")


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


print("\n\nCompleted Step 9\n\n")
