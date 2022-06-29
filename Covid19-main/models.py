#!/usr/bin/env python
# coding: utf-8




import pandas as pd

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle 


from statsmodels.tsa.arima_model import ARIMA


# In[46]:


df=pd.read_csv('/Users/kasturid3/Desktop/vaccines_by_age.csv',parse_dates=['Date'])


# In[47]:


df=df[['Date','Agegroup','At least one dose_cumulative']]


# In[48]:


df=df[df['Date']>'2021-06-01'].reset_index(drop=True)


# In[49]:


df1=df.pivot_table(index=['Date'],columns='Agegroup',values=['At least one dose_cumulative']).reset_index()


# In[50]:


df1.columns=['Date','05-11yrs','12-17yrs','18-29yrs','30-39yrs','40-49yrs','50-59yrs','60-69yrs','70-79yrs','80','Adults_18plus','Ontario_12plus','Ontario_5plus','Undisclosed_or_missing']


# In[51]:


df2=df1[['Date','12-17yrs', '18-29yrs', '30-39yrs', '40-49yrs',
       '50-59yrs', '60-69yrs', '70-79yrs', '80','Adults_18plus', 'Ontario_12plus']]




df2=df2.set_index('Date')


# In[54]:


df2.index = pd.to_datetime(df2.index)



df2.index=df2.index.to_period("D")



y = pd.Series(df2['12-17yrs'])

train=df2['12-17yrs'][:280]
test=df2['12-17yrs'][280:]


# 1,1,2 ARIMA Model
model = ARIMA(train, order=(1,1,2))
model=model.fit()
#prediction=model.predict(start=1, end=24, exog=None, dynamic=False)


# In[60]:


#pred







