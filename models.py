
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle 
import stan

from prophet import Prophet

df=pd.read_csv('vaccines_by_age.csv',parse_dates=['Date'])


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


train=df2['12-17yrs'][:280]
test=df2['12-17yrs'][280:]

df2['y']=df2['12-17yrs']

df2.index.names = ['ds']


df3=df2[['y']]


df3.reset_index(inplace=True)


df3['ds'] = df3['ds'].dt.to_timestamp('s').dt.strftime('%Y-%m-%d')


regressor = Prophet()
regressor.fit(df3[:280])





pickle.dump(regressor, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))
future = model.make_future_dataframe(periods=len(df3[280:]))
preds=model.predict(future)
print(preds['y_hat'])
