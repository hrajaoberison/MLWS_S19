#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 02:17:43 2019

@author: hrajaoberison
"""
# Import libbraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

# Import estimators
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

df = pd.read_excel('HED - UV cal data.xls',sheet_name = 'Sheet2') # Import data from the excel file
X = df.iloc[:,0:12] # Input
Y = df.iloc[:,12:13] # Output

print(X.shape)
print(Y.shape)


# Data normalization Scale (0-1)
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X)

# Set an estimator for the model
model = MLPRegressor().fit(X_train, Y)

# Save model
filename = 'UV_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Load the saved model
loaded_model = pickle.load(open(filename, 'rb'))

# Set row vectors of zeros and 1 value for alpha we want to find
InputList = []
for n in range(12):
    Input = np.zeros((1,12))
    Input[:,n:n+1] = 1
    InputList.append(Input)

# Find the alpha values of each input
EuvcalList = []
for i in range(len(InputList)):
    Euvcal = loaded_model.predict(InputList[i])
    EuvcalList.append(Euvcal)
    
Cte = model.predict(np.zeros((1,12))) # Constant value generated during the training