#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Step 3
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
   int_features = [int(x) for x in request.form.values()]
   final_features=[np.array(int_features)]
   prediction = model.predict(final_features)     
   output = round(prediction[0], 2)
   return render_template('page.html', prediction_text='Covid 19 cases would be {}'.format(output))
if __name__ == "__main__":     
   app.run(debug=True)


# In[ ]:




