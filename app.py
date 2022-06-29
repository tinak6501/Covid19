import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle 
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb')) 
@app.route('/')
def home():    
   return render_template('page.html') 
@app.route('/predict',methods=['POST'])
def predict():    
   '''    
    For rendering results on HTML GUI    
   '''    
   
   prediction=model.predict(start=1, end=24, exog=None, dynamic=False)   
   output = round(prediction)
   return render_template('page.html', prediction_text=str(output))
if __name__ == "__main__":    
   app.run(debug=True)
