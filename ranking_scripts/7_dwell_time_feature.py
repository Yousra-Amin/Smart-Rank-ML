from config import df

# If raw dwell_time is in the dataset, normalize it 
if 'dwell_time' in df.columns: 
    from sklearn.preprocessing import MinMaxScaler 
    df['dwell_time_norm'] = MinMaxScaler().fit_transform(df[['dwell_time']]) 
else: 
    # Proxy: longer passages tend to have higher dwell time 
    
    df['dwell_time_proxy'] = np.log1p(df['doc_len'])