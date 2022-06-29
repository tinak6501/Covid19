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
   
   int_features = [int(x) for x in request.form.values()]
   final_features = [np.array(int_features)]
   prediction = model.predict(final_features)
   
   
   output = round(prediction)
   return render_template('index.html', prediction_text=output)

if __name__ == "__main__":    
   app.run(debug=True)
  
