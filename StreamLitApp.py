# -*- coding: utf-8 -*-
"""
Spyder Editor
@ date : 21st August 2022


"""

#pip install streamlit

import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

st.title('Load and Explore a dataset')
st.write('Google Analytics Customer Revenue Prediction')


file = st.file_uploader("Upload file", type=['csv'])
st.write(file)

train_df = pd.read_csv(file, dtype={'fullVisitorId': 'str'})  # read a CSV file 

st.write(train_df)

JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

#for column in JSON_COLUMNS:
#    column_as_df = json_normalize(train_df[column])
#    column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
#    train_df = train_df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
#print(f"Loaded {os.path.basename(csv_path)}. Shape: {train_df.shape}")



target = train_df['totals.transactionRevenue'].fillna(0).astype(float)
train_df.nunique()[train_df.nunique() == 1]
train_df.drop(['device.browserSize'
,'device.browserVersion'
,'device.flashVersion'
,'device.language'
,'device.mobileDeviceBranding'
,'device.mobileDeviceInfo'
,'device.mobileDeviceMarketingName'
,'device.mobileDeviceModel'
,'device.mobileInputSelector'
,'device.operatingSystemVersion'
,'device.screenColors'
,'device.screenResolution'
,'geoNetwork.cityId'
,'geoNetwork.latitude'
,'geoNetwork.longitude'
,'geoNetwork.networkLocation'
,'trafficSource.adwordsClickInfo.criteriaParameters'],axis=1,inplace=True)

train_df['totals.bounces'] = train_df['totals.bounces'].fillna('0')
train_df['totals.newVisits'] = train_df['totals.newVisits'].fillna('0')
train_df['trafficSource.adwordsClickInfo.isVideoAd'] = train_df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True)
train_df['trafficSource.isTrueDirect'] = train_df['trafficSource.isTrueDirect'].fillna(False)

columns = [col for col in train_df.columns if train_df[col].nunique() > 1]

train_df = train_df[columns]

train_df['diff_visitId_time'] = train_df['visitId'] - train_df['visitStartTime']
train_df['diff_visitId_time'] = (train_df['diff_visitId_time'] != 0).astype(int)
del train_df['visitId']
del train_df['sessionId']

format_str = '%Y%m%d' 
train_df['formated_date'] = train_df['date'].apply(lambda x: datetime.strptime(str(x), format_str))
train_df['month'] = train_df['formated_date'].apply(lambda x:x.month)
train_df['quarter_month'] = train_df['formated_date'].apply(lambda x:x.day//8)
train_df['day'] = train_df['formated_date'].apply(lambda x:x.day)
train_df['weekday'] = train_df['formated_date'].apply(lambda x:x.weekday())

del train_df['date']
del train_df['formated_date']

train_df['totals.hits'] = train_df['totals.hits'].astype(int)
train_df['mean_hits_per_day'] = train_df.groupby(['day'])['totals.hits'].transform('mean')
del  train_df['day']

train_df['formated_visitStartTime'] = train_df['visitStartTime'].apply(
    lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
train_df['formated_visitStartTime'] = pd.to_datetime(train_df['formated_visitStartTime'])
train_df['visit_hour'] = train_df['formated_visitStartTime'].apply(lambda x: x.hour)

del train_df['visitStartTime']
del train_df['formated_visitStartTime']

for col in train_df.columns:
    if col in ['fullVisitorId', 'month', 'quarter_month', 'weekday', 'visit_hour', 'WoY']: continue
    if train_df[col].dtypes == object or train_df[col].dtypes == bool:
        train_df[col], indexer = pd.factorize(train_df[col])
		
numerics = [col for col in train_df.columns if 'totals.' in col]
numerics += ['visitNumber', 'mean_hits_per_day', 'fullVisitorId']
categorical_feats =  [col for col in train_df.columns if col not in numerics]

for col in categorical_feats:
    train_df[col] = train_df[col].astype(int)

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.fillna(999, inplace=True)


#st.write(train_df.isnull().sum())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['fullVisitorId'],axis=1), target, test_size=0.33, random_state=42)


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))

st.write('RMSE value is')
st.write(rms)

