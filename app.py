#Step 3
import numpy as np
from sktime.forecasting.compose import ForecastingPipeline
from flask import Flask, request, jsonify, render_template
import pickle 
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb')) 
@app.route('/')
def home():    
    return render_template('page.html') 
@app.route('/predict',methods=['POST'])
def predict(): 
    
    prediction = model.predict(fh=fh)     
    output = round(prediction)
    return render_template('page.html', prediction_text='Number of vaccines that will be requiered'.format(output))
if __name__ == "__main__":    
    app.run(debug=True)
