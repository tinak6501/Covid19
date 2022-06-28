#Step 3
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sktime.forecasting.compose import ForecastingPipeline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from sktime.datasets import load_airline
#from sktime.forecasting.model_selection import temporal_train_test_split
#from sktime.utils.plotting.forecasting import plot_ys


from sktime.forecasting.base import ForecastingHorizon
#from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.utils.plotting import plot_series


import pandas as pd


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb')) 
@app.route('/')
def home():    
    return render_template('page.html') 
@app.route('/predict',methods=['POST'])



def predict():
    df = pd.read_csv('vaccines_by_age.csv', parse_dates=['Date'])
    df = df[['Date', 'Agegroup', 'At least one dose_cumulative']]
    df = df[df['Date'] > '2021-06-01'].reset_index(drop=True)
    df1 = df.pivot_table(index=['Date'], columns='Agegroup', values=['At least one dose_cumulative']).reset_index()
    df1.columns = ['Date', '05-11yrs', '12-17yrs', '18-29yrs', '30-39yrs', '40-49yrs', '50-59yrs', '60-69yrs',
                   '70-79yrs', '80', 'Adults_18plus', 'Ontario_12plus', 'Ontario_5plus', 'Undisclosed_or_missing']
    df2 = df1[['Date', '12-17yrs', '18-29yrs', '30-39yrs', '40-49yrs',
               '50-59yrs', '60-69yrs', '70-79yrs', '80', 'Adults_18plus', 'Ontario_12plus']]
    df2 = df2.set_index('Date')

    df2.index = pd.to_datetime(df2.index)

    df2.index = df2.index.to_period("D")

    y = pd.Series(df2['12-17yrs'])

    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    prediction = model.predict(fh=fh)     
    output = round(prediction)
    print(output)
    pred_text=''
    for i in output:
        pred_text+= str(i)+'\n'
    pred_text+= ''
    return render_template('page.html', prediction_text=pred_text)
if __name__ == "__main__":    
    app.run(debug=True)
