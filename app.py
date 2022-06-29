import numpy as np  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle 
from prophet import Prophet
from flask import Flask, request, jsonify, render_template
import pickle 

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb')) 
@app.route('/')
def home():
   return render_template('index.html') 
@app.route('/predict',methods=['POST'])
def predict(): 
   df3= pd.read_csv('/df3.csv')
   future = model.make_future_dataframe(periods=len(df3[280:]))
   forecast =model.predict(future)
   preds=forecast[['yhat']]
   
   
   output = round(prediction)
   return render_template('index.html', prediction_text=output)

if __name__ == "__main__":    
   app.run(debug=True)
  
