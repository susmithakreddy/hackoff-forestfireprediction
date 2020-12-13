#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Importing the required libraries
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms
import math
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[13]:


#Reading the dataset
data = pd.read_csv('../input/forest-fires-data-set/forestfires.csv')


# In[14]:


data.head()


# In[15]:


data.info()


# In[16]:


#Converting strings into integers
data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),
                   (1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
data.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)


# In[17]:


#Displaying the new values
data.head()


# In[19]:


# Correlation analysis of the dataset
data.corr()


# In[18]:


data.describe()


# In[54]:


X = data.iloc[:, 0:12].values
y = data.iloc[:, 12].values
data


# In[57]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2]) #For month
labelencoder_X_2 = LabelEncoder()
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3]) #For weekday
onehotencoder = OneHotEncoder()#dummy variable for month
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
onehotencoder = OneHotEncoder()#dummy variable for week
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# In[58]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # **Linear Regression**

# In[63]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[66]:


y_pred = model.predict(X_test)


# In[67]:


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
print("MSE = ", mse(y_pred, y_test))
print("MAE =", mae(y_pred, y_test))
print("R2 Score =", r2_score(y_pred, y_test))


# # **Desicion Tree Regression**

# In[68]:


from sklearn.tree import DecisionTreeRegressor as dtr
reg = dtr(random_state = 42)
reg.fit(X_train, y_train)


# In[69]:


y_pred = reg.predict(X_test)


# In[71]:


print("MSE =", mse(y_pred, y_test))
print("MAE =", mae(y_pred, y_test))
print("R2 Score =", r2_score(y_pred, y_test))


# # **Random Forest**

# In[72]:


from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)


# In[73]:


y_pred = regr.predict(X_test)


# In[75]:


print("MSE =", mse(y_pred, y_test))
print("MAE =", mae(y_pred, y_test))
print("R2 Score =", r2_score(y_pred, y_test))


# In[ ]:




