#Step 3
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import pandas as pd


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb')) 
@app.route('/')
def home():    
    return render_template('page.html') 
@app.route('/predict',methods=['POST'])



def predict():
    

    df=pd.read_csv('/Users/kasturid3/Desktop/vaccines_by_age.csv',parse_dates=['Date'])


    # In[32]:


    df


    # In[33]:


    df=df[['Date','Agegroup','At least one dose_cumulative']]


    # In[34]:


    df=df[df['Date']>'2021-06-01'].reset_index(drop=True)


    # In[35]:


    df1=df.pivot_table(index=['Date'],columns='Agegroup',values=['At least one dose_cumulative']).reset_index()


    # In[36]:


    df1.columns=['Date','05-11yrs','12-17yrs','18-29yrs','30-39yrs','40-49yrs','50-59yrs','60-69yrs','70-79yrs','80','Adults_18plus','Ontario_12plus','Ontario_5plus','Undisclosed_or_missing']


    # In[37]:


    df2=df1[['Date','12-17yrs', '18-29yrs', '30-39yrs', '40-49yrs',
           '50-59yrs', '60-69yrs', '70-79yrs', '80','Adults_18plus', 'Ontario_12plus']]


    # In[38]:


    df2


    # In[39]:


    df2=df2.set_index('Date')


    # In[40]:


    df2.index = pd.to_datetime(df2.index)


    # In[41]:


    df2.index=df2.index.to_period("D")


    # In[42]:


    y = pd.Series(df2['12-17yrs'])


    # In[43]:


    train=df2['12-17yrs'][:280]
    test=df2['12-17yrs'][280:]


    # In[44]:


    


    # In[28]:


    from statsmodels.tsa.arima_model import ARIMA

    # 1,1,2 ARIMA Model
    model = ARIMA(train, order=(1,1,2))
    model=model.fit()
    prediction=model.predict(start=1, end=24, exog=None, dynamic=False)
    
    output = round(prediction)
    print(output)
    pred_text=''
    for i in output:
        pred_text+= str(i)+'\n'
        pred_text+= ''
    return render_template('page.html', prediction_text=pred_text)
if __name__ == "__main__":    
    app.run(debug=True)
