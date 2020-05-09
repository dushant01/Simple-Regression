# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:29:18 2020

@author: jethi
"""
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

""" connect data """

data= pd.read_csv("salary_data.csv")

X= data.iloc[:,:-1].values
Y= data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2,random_state=0)  
    

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

""" Predicting test set"""

Y_Pred= regressor.predict(X_test)

""" VIZ the traning set"""

plt.scatter(x=X_train,y=Y_train, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.show()

""" VIZ test set """
plt.scatter(x=X_test,y=Y_test, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.show()



