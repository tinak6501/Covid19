#Step 3
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle 
app = Flask(__name__)

@app.route('/')
def home():    
   return "HEllo WORLD"

if __name__ == "__main__":     
   app.run()




