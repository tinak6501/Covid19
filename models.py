#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from fbprophet import Prophet
mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.datasets import load_airline
#from sktime.forecasting.model_selection import temporal_train_test_split
#from sktime.utils.plotting.forecasting import plot_ys
%matplotlib inline

from sktime.forecasting.base import ForecastingHorizon
#from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.utils.plotting import plot_series







#print("Hello")


# In[2]:


import pandas as pd

#create dataset
df=pd.read_csv('/Users/kasturid3/Desktop/vaccines_by_age.csv',parse_dates=['Date'])    
df=df[['Date','Agegroup','At least one dose_cumulative']]
df=df[df['Date']>'2021-06-01'].reset_index(drop=True)
df1=df.pivot_table(index=['Date'],columns='Agegroup',values=['At least one dose_cumulative']).reset_index()
df1.columns=['Date','05-11yrs','12-17yrs','18-29yrs','30-39yrs','40-49yrs','50-59yrs','60-69yrs','70-79yrs','80','Adults_18plus','Ontario_12plus','Ontario_5plus','Undisclosed_or_missing']
df2=df1[['Date','12-17yrs', '18-29yrs', '30-39yrs', '40-49yrs',
       '50-59yrs', '60-69yrs', '70-79yrs', '80','Adults_18plus', 'Ontario_12plus']]
df2=df2.set_index('Date')

df2.index = pd.to_datetime(df2.index)

df2.index=df2.index.to_period("D")

y = pd.Series(df2['12-17yrs'])

from sktime.forecasting.model_selection import temporal_train_test_split
y_train, y_test = temporal_train_test_split(y, test_size=24)
fh = ForecastingHorizon(y_test.index, is_relative=False)



# In[3]:


#define response variable


#define explanatory variable
#X = df[['hours']]

#add constant to predictor variables


from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sktime.datasets import load_macroeconomic
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.impute import Imputer

from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import ForecastingPipeline
forecaster = ForecastingPipeline(
    steps=[
        ("imputer", Imputer(method="mean")),
        ("scale", TabularToSeriesAdaptor(MinMaxScaler(feature_range=(1, 2)))),
        ("boxcox", TabularToSeriesAdaptor(PowerTransformer(method="box-cox"))),
        ("forecaster", AutoARIMA(suppress_warnings=True)),
    ]
)
forecaster.fit(y=y_train)

# In[4]:


pickle.dump(forecaster, open('model.pkl','wb'))


# In[ ]:





# In[ ]:



