# 7_dwell_time_feature.py
from config import df
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # Correct import

# If raw dwell time is in the dataset, normalize it
if 'dwell_time' in df.columns:
    df['dwell_time_norm'] = MinMaxScaler().fit_transform(df[['dwell_time']])
else:
    # Proxy: longer passages tend to have higher dwell time
    df['dwell_time_proxy'] = np.log1p(df['doc_len'])