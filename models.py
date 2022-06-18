#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
#print("Hello")


# In[2]:


import pandas as pd

#create dataset
df = pd.DataFrame({'hours': [1, 2, 4, 5, 5],
                   'score': [64, 66, 76, 73, 74]})
df     


# In[3]:


#define response variable
Y = df['score']

#define explanatory variable
X = df[['hours']]

#add constant to predictor variables

regressor = LinearRegression() #Fitting model with training data
regressor.fit(X, Y)


# In[4]:


pickle.dump(regressor, open('model.pkl','wb'))


# In[ ]:





# In[ ]:




